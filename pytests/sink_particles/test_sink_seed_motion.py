"""
A forming seed moves in proportion to its activation and keeps the velocity
of the gas around it; a sink of activation 1 moves as before. A Gaussian
clump below the density threshold, so that nothing forms or accretes, moves
with a uniform velocity for a few fixed steps, with a seed of activation
10⁻³ at rest at the clump centre and a sink of activation 1 at rest away
from it at the start.
"""

# general
from functools import cache

# numerics
import numpy as np

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix containers
from astronomix import SinkParticles

# astronomix functions
from astronomix import time_integration
from astronomix._modules._sink_particles._sink_particle_dynamics import _tsc_weights
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
    _pad_sink_particles,
)

# sink particle test helpers
from _sink_helpers import CELL_SIZE, CLUMP_CENTER, SETTINGS, _density_threshold, _setup

pytestmark = pytest.mark.x64

#: The peak overdensity of the clump, below the density threshold
#: ρ_res ≈ 129, so that no seed forms and none accretes.
PEAK_OVERDENSITY = 10.0

#: The uniform velocity of the gas.
GAS_VELOCITY = (1.0, 0.5, -0.25)

#: The duration of the run and its number of fixed steps.
END_TIME = 0.01
NUM_STEPS = 10

#: The activation and the mass of the forming seed (slot 1). The mass is
#: about 0.3 % of the gas of its TSC cells: small enough for the seed to take
#: the gas velocity, large enough for the momentum it takes from the gas,
#: about 10⁻⁶ of the total, to show in the conservation check.
SEED_ACTIVATION = 1e-3
SEED_MASS = 1e-6

#: The distance of the sink from the seed along x, in cells: beyond the
#: reach r_acc + ℓ Δx = 3 cells of the merging, so that the seed does not
#: merge into the sink, and 5 clump widths from the clump centre, where the
#: seed stays and the gas keeps its uniform velocity by symmetry.
SINK_OFFSET_IN_CELLS = 8.0

#: The mass of the sink of activation 1 (slot 0). The gravity between a sink
#: and the moving gas conserves momentum only up to terms that grow with the
#: sink mass, about 3 × 10⁻¹² of the total momentum for this mass.
SINK_MASS = 1e-5


def _gas_momentum(primitive_state, registered_variables):
    """The total momentum of the gas of an (unpadded, periodic) state."""
    density = primitive_state[registered_variables.density_index]
    velocity = primitive_state[jnp.asarray(registered_variables.velocity_index)]
    return np.asarray(jnp.sum(density * velocity, axis=(1, 2, 3)) * CELL_SIZE**3)


def _sink_momentum(sink_particles):
    """The total momentum of the sink particles."""
    return np.asarray(jnp.sum(sink_particles.mass[:, None] * sink_particles.velocity, axis=0))


@cache
def _run():
    """Run the moving clump with the seed and the sink.

    Returns:
        ``(initial_state, final_state, config, registered_variables)``.
    """
    state, config, params, registered_variables = _setup(
        settings=SETTINGS._replace(peak_overdensity=PEAK_OVERDENSITY, t_end=END_TIME),
    )
    config = config._replace(num_timesteps=NUM_STEPS)
    assert SETTINGS.background_density + PEAK_OVERDENSITY < float(
        _density_threshold(config, params)
    ), "the clump reaches the density threshold"

    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(GAS_VELOCITY[axis])

    empty = _empty_sink_particles(config, primitive_state.dtype)
    center = jnp.full((3,), CLUMP_CENTER, dtype=primitive_state.dtype)
    sink_position = center.at[0].add(SINK_OFFSET_IN_CELLS * CELL_SIZE)
    sink_particles = _pad_sink_particles(
        SinkParticles(
            mass=empty.mass.at[0].set(SINK_MASS).at[1].set(SEED_MASS),
            position=empty.position.at[0].set(sink_position).at[1].set(center),
            velocity=empty.velocity,
            activation=empty.activation.at[0].set(1.0).at[1].set(SEED_ACTIVATION),
        ),
        config,
    )
    initial_state = state._replace(
        primitive_state=primitive_state,
        sink_particles=sink_particles,
    )
    final_state = time_integration(initial_state, config, params, registered_variables)

    final_activation = final_state.sink_particles.activation
    assert jnp.array_equal(final_activation, sink_particles.activation), (
        f"the activations changed from {sink_particles.activation[:2]} to "
        f"{final_activation[:2]}, or new seeds formed: "
        f"{int(jnp.sum(final_activation > 0.0))} active slots"
    )
    assert jnp.array_equal(final_state.sink_particles.mass, sink_particles.mass), (
        "a seed accreted gas"
    )
    return initial_state, final_state, config, registered_variables


def test_weak_seed_moves_with_the_gas_in_proportion_to_its_activation():
    """The seed of activation a = 10⁻³ takes the velocity of its gas at the
    end of its first step and keeps it, while it drifts by a v Δt: it moves
    a times the clump's displacement over the steps after the first."""
    initial_state, final_state, config, registered_variables = _run()
    gas_velocity = np.asarray(GAS_VELOCITY)
    time_step = END_TIME / NUM_STEPS

    displacement = np.asarray(
        final_state.sink_particles.position[1] - initial_state.sink_particles.position[1]
    )
    expected_displacement = SEED_ACTIVATION * gas_velocity * (END_TIME - time_step)
    np.testing.assert_allclose(
        displacement,
        expected_displacement,
        rtol=1e-2,
        err_msg=(
            "the seed did not move a times the clump's displacement over the steps "
            "after its first"
        ),
    )

    # The mass-weighted gas velocity of the seed's TSC cells at the end.
    primitive_state = final_state.primitive_state
    cell_indices, weights = _tsc_weights(
        final_state.sink_particles.position[1:2],
        primitive_state.shape[1:],
        config,
    )
    weighted_density = weights[0] * primitive_state[registered_variables.density_index][
        tuple(index[0] for index in cell_indices)
    ]
    velocity_in_cells = jnp.stack(
        [
            primitive_state[velocity_index][tuple(index[0] for index in cell_indices)]
            for velocity_index in registered_variables.velocity_index
        ],
        axis=-1,
    )
    local_gas_velocity = np.asarray(
        jnp.sum(weighted_density[:, None] * velocity_in_cells, axis=0)
        / jnp.sum(weighted_density)
    )
    seed_velocity = np.asarray(final_state.sink_particles.velocity[1])
    difference = np.linalg.norm(seed_velocity - local_gas_velocity)
    assert difference <= 2.0 * SEED_ACTIVATION * np.linalg.norm(gas_velocity), (
        f"the seed's velocity {seed_velocity} differs from that of its gas "
        f"{local_gas_velocity} by {difference:.3e}, more than 2 a |v|"
    )


def test_sink_is_not_dragged_by_the_gas():
    """The sink of activation 1 starts at rest and is not coupled to the gas:
    only the gravity of the clump changes its velocity, which stays below 1 %
    of the gas velocity."""
    initial_state, final_state, _, _ = _run()
    gas_speed = np.linalg.norm(GAS_VELOCITY)
    sink_velocity = np.asarray(final_state.sink_particles.velocity[0])
    assert np.linalg.norm(sink_velocity) < 1e-2 * gas_speed, (
        f"the sink reached the velocity {sink_velocity}, against the gas velocity "
        f"{GAS_VELOCITY}: it was dragged by the gas"
    )
    displacement = np.asarray(
        final_state.sink_particles.position[0] - initial_state.sink_particles.position[0]
    )
    assert np.linalg.norm(displacement) < 1e-2 * gas_speed * END_TIME, (
        f"the sink moved by {displacement}, against the clump's displacement "
        f"{np.asarray(GAS_VELOCITY) * END_TIME}"
    )


def test_seed_motion_conserves_momentum():
    """The momentum of the gas and the sink particles together is the same
    at the end of the run as at the start: the momentum the seed takes from
    the gas when it is tied to the gas velocity, about 10⁻⁶ of the total,
    is taken from its cells."""
    initial_state, final_state, _, registered_variables = _run()
    initial_momentum = _gas_momentum(
        initial_state.primitive_state, registered_variables
    ) + _sink_momentum(initial_state.sink_particles)
    final_momentum = _gas_momentum(
        final_state.primitive_state, registered_variables
    ) + _sink_momentum(final_state.sink_particles)
    np.testing.assert_allclose(
        final_momentum,
        initial_momentum,
        rtol=1e-10,
        atol=1e-10 * np.linalg.norm(initial_momentum),
        err_msg="the momentum of the gas and the sink particles changed in the run",
    )
