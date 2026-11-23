"""
The smooth sink formation and accretion, with the default smoothing
parameters: the factors of the creation weight, the activation of a candidate,
the conservation of mass and momentum in the accretion, the motion of a
candidate with its gas, without overshoot for a candidate heavier than its
gas, and finite values and derivatives for a slot past a non-periodic edge.
"""

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import BoundarySettings

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_dynamics import _tsc_weights
from astronomix._modules._sink_particles._sink_particle_formation import (
    SlotChecks,
    _creation_factors,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
)
from astronomix._modules._sink_particles._sink_particle_update import (
    _update_sink_particles,
)
from astronomix.time_stepping.time_integration import _prepare_padded_state

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    CLUMP_CENTER,
    NUM_CELLS,
    SETTINGS,
    _density_threshold,
    _gas_mass_and_momentum,
    _setup,
    _sink_at_clump_center,
    _update_sinks_once,
    _with_linear_velocity,
    _with_sink,
)

#: A bulk velocity of the gas, so that momenta are not zero.
BULK_VELOCITY = jnp.array([1.0, -2.0, 0.5])

#: A step past the end of a band, small against every band.
PAST_BAND = 1e-3


def _with_bulk_velocity(state, registered_variables, velocity=BULK_VELOCITY):
    """Add a uniform velocity to the gas."""
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].add(velocity[axis])
    return state._replace(primitive_state=primitive_state)


def _total_mass_and_momentum(primitive_state, sink_particles, registered_variables):
    """Gas plus slot mass and momentum of an (unpadded, periodic) state."""
    gas_mass, gas_momentum = _gas_mass_and_momentum(primitive_state, registered_variables)
    return (
        gas_mass + jnp.sum(sink_particles.mass),
        gas_momentum + jnp.sum(sink_particles.mass[:, None] * sink_particles.velocity, axis=0),
    )


# -------------------------------------------------------------
# ==================== ↓ Creation factors ↓ ===================
# -------------------------------------------------------------


def _checks(**values):
    """Check quantities of 3 positions that pass every check by a wide
    margin, with the given fields replaced."""
    passing = dict(
        has_cells=jnp.ones(3, dtype=bool),
        density_threshold=jnp.ones(3),
        density_ratio=jnp.full(3, 10.0),
        converging_velocity=jnp.ones((3, 6)),
        potential_height=jnp.full(3, -1.0),
        jeans_ratio=jnp.full(3, 10.0),
        bound_ratio=jnp.full(3, 10.0),
        center_of_mass_velocity=jnp.zeros((3, 3)),
    )
    passing.update(values)
    return SlotChecks(**passing)


def _factors(checks, sink_particles=None, separation_squared=None):
    """The creation factors of three candidates far apart, with the default
    smoothing parameters, and the width ω of the checks."""
    _, config, _, _ = _setup()
    if sink_particles is None:
        sink_particles = SinkParticles(
            mass=jnp.zeros(3),
            position=jnp.zeros((3, 3)),
            velocity=jnp.zeros((3, 3)),
            activation=jnp.zeros(3),
            occupied=jnp.ones(3, dtype=bool),
        )
        separation_squared = jnp.ones((3, 3))
    factors = _creation_factors(checks, sink_particles, separation_squared, config)
    return factors, config.sink_particle_config.check_width


def test_density_factor_is_exact_outside_its_band():
    """The density factor is 0 for ρ / ρ_thr ≤ 1 and 1 for ρ / ρ_thr ≥ 1 + ω."""
    _, width = _factors(_checks())
    factors, _ = _factors(
        _checks(density_ratio=jnp.array([0.5, 1.0, 1.0 + width + PAST_BAND]))
    )
    np.testing.assert_array_equal(
        factors.density, [0.0, 0.0, 1.0], err_msg="the density factor is not 0, 0, 1"
    )


def test_converging_flow_factor_is_exact_outside_its_band():
    """The converging-flow factor is 0 when a velocity u_k ≤ −ω, and 1 when
    every u_k ≥ 0, also at u_k = 0, where the hard check passes."""
    _, width = _factors(_checks())
    converging_velocity = jnp.ones((3, 6))
    converging_velocity = converging_velocity.at[0, 3].set(-width - PAST_BAND)
    converging_velocity = converging_velocity.at[1, :].set(0.0)
    factors, _ = _factors(_checks(converging_velocity=converging_velocity))
    np.testing.assert_array_equal(
        factors.converging_flow,
        [0.0, 1.0, 1.0],
        err_msg="the converging-flow factor is not 0, 1, 1",
    )


def test_potential_minimum_factor_is_exact_outside_its_band():
    """The potential-minimum factor is 0 at a height ≥ ω above the soft
    minimum, and 1 at a height ≤ 0, where the hard check passes."""
    _, width = _factors(_checks())
    factors, _ = _factors(
        _checks(potential_height=jnp.array([width + PAST_BAND, 0.0, -1.0]))
    )
    np.testing.assert_array_equal(
        factors.potential_minimum,
        [0.0, 1.0, 1.0],
        err_msg="the potential-minimum factor is not 0, 1, 1",
    )


def test_jeans_factor_is_exact_outside_its_band():
    """The Jeans factor is 0 for |E_grav| / (2 E_th + E_mag) ≤ 1 and 1 beyond
    1 + ω."""
    _, width = _factors(_checks())
    factors, _ = _factors(
        _checks(jeans_ratio=jnp.array([0.5, 1.0, 1.0 + width + PAST_BAND]))
    )
    np.testing.assert_array_equal(
        factors.jeans_instability, [0.0, 0.0, 1.0], err_msg="the Jeans factor is not 0, 0, 1"
    )


def test_bound_factor_is_exact_outside_its_band():
    """The bound factor is 0 for |E_grav| / (E_th + E_kin + E_mag) ≤ 1 and 1
    beyond 1 + ω."""
    _, width = _factors(_checks())
    factors, _ = _factors(
        _checks(bound_ratio=jnp.array([0.5, 1.0, 1.0 + width + PAST_BAND]))
    )
    np.testing.assert_array_equal(
        factors.bound_state, [0.0, 0.0, 1.0], err_msg="the bound factor is not 0, 0, 1"
    )


def test_proximity_factor_is_exact_outside_its_band():
    """The proximity factor of a candidate is 0 within r_acc of a sink with
    activation 1 and 1 beyond r_acc + ℓ Δx; candidates with activation 0 do
    not affect each other."""
    _, config, _, _ = _setup()
    sink_particle_config = config.sink_particle_config
    accretion_radius = sink_particle_config.accretion_radius_in_cells * CELL_SIZE
    band = sink_particle_config.spatial_width_in_cells * CELL_SIZE

    # Slot 0 is a sink; slots 1 and 2 are candidates at r_acc and just
    # beyond r_acc + ℓ Δx from it, and far from each other.
    sink_particles = SinkParticles(
        mass=jnp.array([1.0, 0.0, 0.0]),
        position=jnp.zeros((3, 3)),
        velocity=jnp.zeros((3, 3)),
        activation=jnp.array([1.0, 0.0, 0.0]),
        occupied=jnp.ones(3, dtype=bool),
    )
    distance = jnp.array(
        [
            [0.0, accretion_radius, accretion_radius + band + PAST_BAND * CELL_SIZE],
            [accretion_radius, 0.0, 1.0],
            [accretion_radius + band + PAST_BAND * CELL_SIZE, 1.0, 0.0],
        ]
    )
    factors, _ = _factors(_checks(), sink_particles, distance**2)
    np.testing.assert_array_equal(
        factors.proximity[1:],
        [0.0, 1.0],
        err_msg="the proximity factor is not 0 at r_acc and 1 beyond r_acc + ℓ Δx",
    )


# -------------------------------------------------------------
# ==================== ↑ Creation factors ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Activation, accretion ↓ ===============
# -------------------------------------------------------------


def test_dense_clump_activates_candidate_which_takes_whole_excess():
    """A clump far past every creation threshold, with weak inflow, gets a
    candidate whose activation reaches 1 in the same step, and the new sink
    takes the whole excess of every cell where its radial weight is 1, within
    r_acc − ℓ Δx; the cells in the band of the radial weight keep part of it."""
    state, config, params, registered_variables = _setup()
    state = _with_linear_velocity(state, registered_variables, (-1.0, -1.0, -1.0))

    new_primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
    )

    num_occupied = int(jnp.sum(sink_particles.occupied))
    assert num_occupied == 1, f"{num_occupied} candidates were created instead of 1"
    activation = float(sink_particles.activation[0])
    assert activation == 1.0, f"the candidate's activation is {activation}, not 1"

    sink_particle_config = config.sink_particle_config
    full_weight_radius = (
        sink_particle_config.accretion_radius_in_cells
        - sink_particle_config.spatial_width_in_cells
    ) * CELL_SIZE
    cell_centers = (jnp.arange(NUM_CELLS) + 0.5) * CELL_SIZE
    x, y, z = jnp.meshgrid(cell_centers, cell_centers, cell_centers, indexing="ij")
    distance = jnp.sqrt(
        (x - CLUMP_CENTER) ** 2 + (y - CLUMP_CENTER) ** 2 + (z - CLUMP_CENTER) ** 2
    )
    density_index = registered_variables.density_index
    density = state.primitive_state[density_index]
    new_density = new_primitive_state[density_index]
    density_threshold = _density_threshold(config, params)
    above_threshold = density > density_threshold

    fully_taken = above_threshold & (distance <= full_weight_radius)
    assert int(jnp.sum(fully_taken)) > 0, "no cell above the threshold has radial weight 1"
    np.testing.assert_allclose(
        new_density[fully_taken],
        density_threshold,
        rtol=1e-5,
        err_msg="a cell with radial weight 1 kept gas above the threshold",
    )
    in_band = above_threshold & (distance > full_weight_radius)
    assert jnp.all(new_density[in_band] > density_threshold * (1.0 + 1e-5)), (
        "a cell in the band of the radial weight gave its whole excess"
    )


def _two_overlapping_sinks(state, config, dtype):
    """Two sinks at rest 2 cells on either side of the clump centre along x,
    whose accretion radii overlap, of masses 0.3 and 0.2."""
    sink_particles = _empty_sink_particles(config, dtype)
    sink_particles = _with_sink(
        sink_particles,
        0,
        0.3,
        [CLUMP_CENTER - 2 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER],
    )
    return _with_sink(
        sink_particles,
        1,
        0.2,
        [CLUMP_CENTER + 2 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER],
    )


def test_smooth_accretion_conserves_mass_and_momentum():
    """Two sinks with overlapping accretion radii, in gas falling toward the
    clump centre on top of a bulk motion, share the gas: gas plus sink mass
    and momentum are conserved, and both sinks gain mass."""
    state, config, params, registered_variables = _setup()
    state = _with_linear_velocity(state, registered_variables, (-1.0, -1.0, -1.0))
    state = _with_bulk_velocity(state, registered_variables)
    sink_particles = _two_overlapping_sinks(state, config, state.primitive_state.dtype)
    sink_particles = sink_particles._replace(
        velocity=sink_particles.velocity.at[:2].set(BULK_VELOCITY)
    )

    new_primitive_state, new_sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )

    mass_before, momentum_before = _total_mass_and_momentum(
        state.primitive_state,
        sink_particles,
        registered_variables,
    )
    mass_after, momentum_after = _total_mass_and_momentum(
        new_primitive_state,
        new_sink_particles,
        registered_variables,
    )
    mass_gain = new_sink_particles.mass[:2] - sink_particles.mass[:2]
    assert jnp.all(mass_gain > 0.0), f"a sink gained no mass: gains {mass_gain}"
    np.testing.assert_allclose(
        mass_after, mass_before, rtol=1e-6, err_msg="gas plus sink mass changed"
    )
    np.testing.assert_allclose(
        momentum_after,
        momentum_before,
        rtol=1e-5,
        err_msg="gas plus sink momentum changed",
    )


def test_smooth_accretion_leaves_no_cell_below_threshold():
    """Every cell gives at most its excess above the threshold, also where
    the accretion radii of two sinks overlap: no cell above the threshold ends
    below it, and no cell below it loses gas."""
    state, config, params, registered_variables = _setup()
    state = _with_linear_velocity(state, registered_variables, (-1.0, -1.0, -1.0))
    sink_particles = _two_overlapping_sinks(state, config, state.primitive_state.dtype)

    new_primitive_state, _ = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )

    density_index = registered_variables.density_index
    density = state.primitive_state[density_index]
    new_density = new_primitive_state[density_index]
    density_threshold = _density_threshold(config, params)
    above_threshold = density > density_threshold
    lowest = float(jnp.min(new_density[above_threshold]))
    assert lowest >= density_threshold * (1.0 - 1e-6), (
        f"a cell ended at {lowest}, below the threshold {float(density_threshold)}"
    )
    assert jnp.array_equal(new_density[~above_threshold], density[~above_threshold]), (
        "a cell below the threshold lost gas"
    )


# -------------------------------------------------------------
# =================== ↑ Activation, accretion ↑ ===============
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Motion of candidates ↓ ================
# -------------------------------------------------------------


#: The velocity of the candidates relative to the gas.
CANDIDATE_RELATIVE_VELOCITY = jnp.array([0.3, 0.2, -0.1])


def _update_with_inactive_candidate(candidate_mass, peak_overdensity):
    """One update of a clump that is not Jeans unstable, so that a candidate
    stays inactive, moving with ``BULK_VELOCITY``, with an inactive candidate
    of ``candidate_mass`` at its centre moving relative to the gas, and a sink
    far away in the thin gas.

    Returns:
        ``(state, sink_particles, new_primitive_state, new_sink_particles,
        config, registered_variables)``.
    """
    settings = SETTINGS._replace(peak_overdensity=peak_overdensity)
    state, config, params, registered_variables = _setup(settings=settings)
    state = _with_bulk_velocity(state, registered_variables)
    dtype = state.primitive_state.dtype

    sink_particles = _with_sink(
        _empty_sink_particles(config, dtype),
        0,
        candidate_mass,
        [CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER],
    )
    sink_particles = sink_particles._replace(
        activation=sink_particles.activation.at[0].set(0.0),
        velocity=sink_particles.velocity.at[0].set(
            BULK_VELOCITY + CANDIDATE_RELATIVE_VELOCITY
        ),
    )
    sink_particles = _with_sink(sink_particles, 1, 0.1, [0.1, 0.1, 0.1])
    sink_particles = sink_particles._replace(
        velocity=sink_particles.velocity.at[1].set(FAR_SINK_VELOCITY)
    )

    new_primitive_state, new_sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )
    activation = float(new_sink_particles.activation[0])
    assert activation == 0.0, f"the candidate's activation grew to {activation}"
    return (
        state,
        sink_particles,
        new_primitive_state,
        new_sink_particles,
        config,
        registered_variables,
    )


#: The velocity of the sink far from the clump.
FAR_SINK_VELOCITY = jnp.array([-0.5, 0.4, 0.3])


def _assert_momentum_conserved(state, sink_particles, new_primitive_state, new_sink_particles, registered_variables):
    """Fail if gas plus slot momentum changed."""
    _, momentum_before = _total_mass_and_momentum(
        state.primitive_state,
        sink_particles,
        registered_variables,
    )
    _, momentum_after = _total_mass_and_momentum(
        new_primitive_state,
        new_sink_particles,
        registered_variables,
    )
    np.testing.assert_allclose(
        momentum_after,
        momentum_before,
        rtol=1e-6,
        err_msg="gas plus slot momentum changed",
    )


def test_light_candidate_moves_with_gas_and_conserves_momentum():
    """An inactive candidate much lighter than the gas in its TSC cells takes
    the velocity of that gas, its momentum change goes to the gas, and a sink
    far away keeps its velocity."""
    (
        state,
        sink_particles,
        new_primitive_state,
        new_sink_particles,
        _,
        registered_variables,
    ) = _update_with_inactive_candidate(candidate_mass=1e-6, peak_overdensity=200.0)

    np.testing.assert_allclose(
        new_sink_particles.velocity[0],
        BULK_VELOCITY,
        rtol=0.0,
        atol=1e-4,
        err_msg="the light inactive candidate does not move with the gas",
    )
    np.testing.assert_array_equal(
        new_sink_particles.velocity[1],
        FAR_SINK_VELOCITY,
        err_msg="the velocity of the sink far away changed",
    )
    _assert_momentum_conserved(
        state, sink_particles, new_primitive_state, new_sink_particles, registered_variables
    )


def test_heavy_candidate_takes_center_of_mass_velocity_without_overshoot():
    """An inactive candidate heavier than the gas in its TSC cells takes the
    centre-of-mass velocity of itself and that gas, which moves by less than
    the relative velocity: it never overshoots the candidate's velocity.
    Momentum is conserved. The clump (peak 150) is less dense than in the
    light case, so that the candidate's own mass does not make it Jeans
    unstable."""
    candidate_mass = 0.02
    (
        state,
        sink_particles,
        new_primitive_state,
        new_sink_particles,
        config,
        registered_variables,
    ) = _update_with_inactive_candidate(candidate_mass, peak_overdensity=150.0)

    cell_indices, weights = _tsc_weights(
        sink_particles.position[:1],
        state.primitive_state.shape[1:],
        config,
    )
    density = state.primitive_state[registered_variables.density_index]
    gas_mass = float(jnp.sum(weights * density[cell_indices])) * CELL_SIZE**3
    assert candidate_mass > 2.0 * gas_mass, (
        f"the candidate ({candidate_mass}) is not heavier than its gas ({gas_mass:.3g})"
    )
    candidate_velocity = BULK_VELOCITY + CANDIDATE_RELATIVE_VELOCITY
    center_of_mass_velocity = (
        candidate_mass * candidate_velocity + gas_mass * BULK_VELOCITY
    ) / (candidate_mass + gas_mass)
    np.testing.assert_allclose(
        new_sink_particles.velocity[0],
        center_of_mass_velocity,
        rtol=1e-5,
        err_msg="the heavy candidate does not take the centre-of-mass velocity",
    )

    velocity_rows = jnp.array(registered_variables.velocity_index)
    gas_velocity_change = (
        new_primitive_state[velocity_rows] - state.primitive_state[velocity_rows]
    )
    largest_change = float(jnp.max(jnp.linalg.norm(gas_velocity_change, axis=0)))
    relative_speed = float(jnp.linalg.norm(CANDIDATE_RELATIVE_VELOCITY))
    assert 0.0 < largest_change < relative_speed, (
        f"the gas velocity changed by up to {largest_change:.3g}, not between 0 and the "
        f"relative speed {relative_speed:.3g}"
    )
    _assert_momentum_conserved(
        state, sink_particles, new_primitive_state, new_sink_particles, registered_variables
    )


# -------------------------------------------------------------
# =================== ↑ Motion of candidates ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ========================= ↓ Guards ↓ ========================
# -------------------------------------------------------------


def test_slot_past_open_edge_has_finite_values_and_derivatives():
    """A partly active candidate with some mass 0.6 cells past a non-periodic
    edge, next to a clump on the first cell, has no cell to read its values
    from, but reaches real cells with its accretion and its TSC weights: the
    update keeps every value finite, and the derivatives of the outputs with
    respect to the gas and the slots are finite."""
    settings = SETTINGS._replace(
        boundary_settings=BoundarySettings(),
        overdensity_center=(0.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
    )
    state, config, params, registered_variables = _setup(settings=settings)
    primitive_state = _prepare_padded_state(
        state.primitive_state,
        config,
        params,
        registered_variables,
    )
    sink_particles = _with_sink(
        _empty_sink_particles(config, primitive_state.dtype),
        0,
        1e-3,
        [-0.6 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER],
    )
    sink_particles = sink_particles._replace(
        activation=sink_particles.activation.at[0].set(0.5)
    )

    def outputs(primitive_state, mass, position, velocity, activation):
        new_primitive_state, new_sink_particles = _update_sink_particles(
            primitive_state,
            sink_particles._replace(
                mass=mass,
                position=position,
                velocity=velocity,
                activation=activation,
            ),
            config,
            params,
            registered_variables,
        )
        return (
            jnp.sum(new_primitive_state)
            + jnp.sum(new_sink_particles.mass)
            + jnp.sum(new_sink_particles.position)
            + jnp.sum(new_sink_particles.velocity)
            + jnp.sum(new_sink_particles.activation)
        )

    inputs = (
        primitive_state,
        sink_particles.mass,
        sink_particles.position,
        sink_particles.velocity,
        sink_particles.activation,
    )
    value, gradients = jax.value_and_grad(outputs, argnums=tuple(range(5)))(*inputs)
    assert jnp.isfinite(value), f"the outputs are not finite: {value}"
    names = ("gas", "mass", "position", "velocity", "activation")
    for name, gradient in zip(names, gradients):
        assert jnp.all(jnp.isfinite(gradient)), f"the derivative with respect to the {name} is not finite"


# -------------------------------------------------------------
# ========================= ↑ Guards ↑ ========================
# -------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
