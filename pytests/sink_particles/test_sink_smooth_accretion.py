"""
Smooth accretion at the default widths: everything a sink takes from the gas
vanishes with its activation. A massless seed off the centre of the clump
accretes once at activations 0, 10⁻⁶, 10⁻³ and 1; the accreted mass scales
with the activation, the position shift goes to 0 with it, and mass and
momentum are conserved at every activation.
"""

# general
from functools import cache

# numerics
import numpy as np

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_accretion import _accrete_gas
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
)

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    CLUMP_CENTER,
    _density_threshold,
    _setup,
    _with_linear_velocity,
)

pytestmark = pytest.mark.x64

#: The relative tolerance of the conservation checks. The new density of a
#: cell is rounded at its own size, about 10⁻¹⁰ of the density the seed takes
#: at activation 10⁻⁶.
CONSERVATION_TOLERANCE = 1e-8

#: The activations of the seed, the full sink last.
ACTIVATIONS = (0.0, 1e-6, 1e-3, 1.0)

#: The seed's offset from the clump centre, in cells: off the centre, so that
#: the gas it accretes has its centre of mass away from the seed, and within
#: ½ ℓ cells of a face along x, so that its cell weight is shared between two
#: cells there.
SEED_OFFSET_IN_CELLS = (0.3, 0.2, 0.0)

#: The bulk velocity of the gas and of the seed, on top of the infall of the
#: gas toward the clump centre.
BULK_VELOCITY = (1.0, -2.0, 0.5)


@cache
def _accrete_at_each_activation():
    """Accrete once onto a massless seed of each activation of
    ``ACTIVATIONS``, with the default widths.

    Returns:
        ``(primitive_state, seed, results, config, params, registered_variables)``,
        with ``results`` mapping each activation to the primitive state and
        the sink particles after accretion.
    """
    state, config, params, registered_variables = _setup()
    state = _with_linear_velocity(state, registered_variables, (-1.0, -1.0, -1.0))
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].add(BULK_VELOCITY[axis])

    seed = _empty_sink_particles(config, primitive_state.dtype)
    seed = seed._replace(
        position=seed.position.at[0].set(
            CLUMP_CENTER + jnp.asarray(SEED_OFFSET_IN_CELLS) * CELL_SIZE
        ),
        velocity=seed.velocity.at[0].set(jnp.asarray(BULK_VELOCITY)),
    )

    results = {}
    for activation in ACTIVATIONS:
        results[activation] = _accrete_gas(
            primitive_state,
            seed._replace(activation=seed.activation.at[0].set(activation)),
            jnp.asarray(1),
            config,
            params,
            registered_variables,
        )
    return primitive_state, seed, results, config, params, registered_variables


def test_accreted_mass_scales_with_activation():
    """The mass a seed accretes is its activation times the mass a full sink
    accretes at the same place; at activation 0 it takes nothing and leaves
    the gas unchanged bit for bit."""
    primitive_state, seed, results, _, _, _ = _accrete_at_each_activation()
    full_gain = float(results[1.0][1].mass[0])
    assert full_gain > 0.0, "the full sink accreted nothing"

    for activation in ACTIVATIONS[:-1]:
        new_primitive_state, new_seed = results[activation]
        gain = float(new_seed.mass[0])
        np.testing.assert_allclose(
            gain,
            activation * full_gain,
            rtol=1e-12,
            atol=0.0,
            err_msg=(
                f"at activation {activation} the seed accreted {gain}, not "
                f"{activation} × {full_gain} of the full sink"
            ),
        )

    unchanged_state, unchanged_seed = results[0.0]
    assert jnp.array_equal(unchanged_state, primitive_state), (
        "a seed of activation 0 changed the gas"
    )
    assert jnp.array_equal(unchanged_seed.position, seed.position) and jnp.array_equal(
        unchanged_seed.velocity, seed.velocity
    ), "a seed of activation 0 moved or changed its velocity"


def test_position_shift_vanishes_with_activation():
    """A seed of activation a moves by a S / (a M + ε ρ_res ΔV), with S and M
    the first moment and the mass of the gas a full sink takes: the shift goes
    to 0 with a, and the mass ε ρ_res ΔV keeps a seed that takes little gas
    from jumping to that gas' centre of mass."""
    _, seed, results, config, params, _ = _accrete_at_each_activation()
    position_update_mass = (
        config.sink_particle_config.position_update_mass
        * float(_density_threshold(config, params))
        * CELL_SIZE**3
    )
    full_gain = float(results[1.0][1].mass[0])
    full_shift = np.asarray(results[1.0][1].position[0] - seed.position[0])
    assert np.linalg.norm(full_shift) > 0.05 * CELL_SIZE, (
        f"the full sink moved by only {full_shift / CELL_SIZE} cells; the test needs "
        "accreted gas off the seed"
    )
    first_moment = full_shift * (full_gain + position_update_mass)

    for activation in ACTIVATIONS[1:-1]:
        shift = np.asarray(results[activation][1].position[0] - seed.position[0])
        np.testing.assert_allclose(
            shift,
            activation * first_moment / (activation * full_gain + position_update_mass),
            rtol=1e-10,
            err_msg=f"the shift at activation {activation} is not a S / (a M + ε ρ_res ΔV)",
        )

    smallest_shift = np.asarray(results[ACTIVATIONS[1]][1].position[0] - seed.position[0])
    assert np.linalg.norm(smallest_shift) < 1e-2 * np.linalg.norm(full_shift), (
        f"at activation {ACTIVATIONS[1]} the seed moved by {smallest_shift / CELL_SIZE} "
        f"cells, not less than 1 % of the full sink's {full_shift / CELL_SIZE}"
    )


def test_accretion_conserves_mass_and_momentum_at_any_activation():
    """At every activation the gas loses exactly the mass and the momentum the
    seed gains."""
    primitive_state, seed, results, config, _, registered_variables = (
        _accrete_at_each_activation()
    )
    cell_volume = CELL_SIZE**3
    density_index = registered_variables.density_index
    velocity_index = jnp.asarray(registered_variables.velocity_index)

    def mass_and_momentum_density(state):
        density = state[density_index]
        return density, density * state[velocity_index]

    density_before, momentum_before = mass_and_momentum_density(primitive_state)
    seed_momentum_before = seed.mass[0] * seed.velocity[0]
    for activation in ACTIVATIONS[1:]:
        new_primitive_state, new_seed = results[activation]
        density_after, momentum_after = mass_and_momentum_density(new_primitive_state)

        gas_mass_lost = float(jnp.sum(density_before - density_after) * cell_volume)
        seed_mass_gain = float(new_seed.mass[0] - seed.mass[0])
        np.testing.assert_allclose(
            gas_mass_lost,
            seed_mass_gain,
            rtol=CONSERVATION_TOLERANCE,
            err_msg=f"at activation {activation} the gas lost a different mass than the seed gained",
        )

        gas_momentum_lost = np.asarray(
            jnp.sum(momentum_before - momentum_after, axis=(1, 2, 3)) * cell_volume
        )
        seed_momentum_gain = np.asarray(
            new_seed.mass[0] * new_seed.velocity[0] - seed_momentum_before
        )
        np.testing.assert_allclose(
            gas_momentum_lost,
            seed_momentum_gain,
            rtol=CONSERVATION_TOLERANCE,
            atol=CONSERVATION_TOLERANCE * np.linalg.norm(seed_momentum_gain),
            err_msg=(
                f"at activation {activation} the gas lost a different momentum than the "
                "seed gained"
            ),
        )
