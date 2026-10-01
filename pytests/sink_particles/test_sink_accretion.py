"""
Sink particle accretion pytest (fast).

Checks the gas accretion onto sink particles of Federrath et al. (2010),
Section 2.3: the mass, position and velocity of a new sink after its first
accretion, the conservation of mass and momentum (with and without a magnetic
field), the bound and radial velocity checks, and the choice of a single sink
for each accreted cell.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_formation import (
    _empty_sink_particles,
)

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    CLUMP_CENTER,
    SETTINGS,
    _density_threshold,
    _gas_mass_and_momentum,
    _num_accreted_cells,
    _num_sinks,
    _setup,
    _sink_at_clump_center,
    _update_sinks_once,
    _with_linear_velocity,
)


def test_new_sink_properties():
    """A new sink accretes the mass above the density threshold (Eq. 32)
    within its accretion radius, and ends at the centre of mass and with the
    centre-of-mass velocity of the accreted gas. The accreted cells are left at
    the threshold with an unchanged sound speed; the other cells are
    untouched."""
    state, config, params, registered_variables = _setup()

    # Give the gas a uniform bulk velocity, which the sink must inherit as its
    # centre-of-mass velocity.
    bulk_velocity = jnp.array([1.0, -2.0, 0.5])
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(bulk_velocity[axis])
    state = state._replace(primitive_state=primitive_state)

    new_primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
    )

    # One sink, from the centre cell, the only potential minimum.
    assert _num_sinks(sink_particles) == 1

    # The 57 cells above the threshold all lie within r_acc of the sink,
    # created at the centre cell, so it accretes the whole mass above the
    # threshold.
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    density = primitive_state[density_index]
    density_threshold = _density_threshold(config, params)
    cells_above_threshold = density > density_threshold
    assert int(jnp.sum(cells_above_threshold)) == 57
    mass_above_threshold = jnp.sum(
        jnp.where(cells_above_threshold, density - density_threshold, 0.0)
    ) * config.grid_spacing**3
    assert jnp.allclose(sink_particles.mass[0], mass_above_threshold, rtol=1e-5)

    # The accreted gas is symmetric about the clump centre, so its centre of
    # mass is the clump centre.
    assert jnp.allclose(sink_particles.position[0], CLUMP_CENTER, atol=1e-5)

    assert jnp.allclose(sink_particles.velocity[0], bulk_velocity, atol=1e-5)

    # The accreted cells end at the threshold with c_s² unchanged; the other
    # cells are untouched.
    assert jnp.allclose(
        new_primitive_state[density_index][cells_above_threshold],
        density_threshold,
        rtol=1e-6,
    )
    sound_speed_squared_before = (
        SETTINGS.gamma * primitive_state[pressure_index] / primitive_state[density_index]
    )
    sound_speed_squared_after = (
        SETTINGS.gamma
        * new_primitive_state[pressure_index]
        / new_primitive_state[density_index]
    )
    assert jnp.allclose(sound_speed_squared_before, sound_speed_squared_after, rtol=1e-5)
    assert jnp.array_equal(
        new_primitive_state[:, ~cells_above_threshold],
        primitive_state[:, ~cells_above_threshold],
    )


def test_accretion_conservation():
    """Accretion moves mass and momentum from the gas to the sink without
    creating or destroying any."""
    state, config, params, registered_variables = _setup()

    # Gas falling toward the clump centre on top of a bulk motion, and an
    # existing sink at the centre moving relative to the gas. The accreted gas
    # has a different mean velocity than the sink, so the sink's velocity
    # changes and the momentum check covers the velocity update.
    bulk_velocity = jnp.array([1.0, -2.0, 0.5])
    state = _with_linear_velocity(state, registered_variables, (-1.0, -1.0, -1.0))
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].add(bulk_velocity[axis])
    state = state._replace(primitive_state=primitive_state)

    sink_mass = 0.5
    sink_velocity = bulk_velocity + jnp.array([0.05, -0.02, 0.01])
    existing_sink = _sink_at_clump_center(sink_mass, config, primitive_state.dtype)
    existing_sink = existing_sink._replace(
        velocity=existing_sink.velocity.at[0].set(sink_velocity)
    )

    new_primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        existing_sink,
    )
    assert _num_sinks(sink_particles) == 1
    assert not jnp.allclose(sink_particles.velocity[0], sink_velocity)

    # What the gas loses, the sink gains.
    gas_mass_before, gas_momentum_before = _gas_mass_and_momentum(
        primitive_state,
        registered_variables,
    )
    gas_mass_after, gas_momentum_after = _gas_mass_and_momentum(
        new_primitive_state,
        registered_variables,
    )
    sink_mass_gain = sink_particles.mass[0] - sink_mass
    sink_momentum_gain = (
        sink_particles.mass[0] * sink_particles.velocity[0] - sink_mass * sink_velocity
    )
    assert sink_mass_gain > 0.0
    assert jnp.allclose(gas_mass_before, gas_mass_after + sink_mass_gain, rtol=1e-6)
    assert jnp.allclose(
        gas_momentum_before,
        gas_momentum_after + sink_momentum_gain,
        rtol=1e-6,
    )


def test_accretion_with_magnetic_field():
    """Accretion leaves the magnetic field unchanged, and still moves mass and
    momentum from the gas to the sink without creating or destroying any."""
    state, config, params, registered_variables = _setup(
        settings=SETTINGS._replace(magnetic_field_z=1.0),
        mhd=True,
    )

    # A bulk velocity makes the momentum transfer non-trivial.
    bulk_velocity = jnp.array([1.0, -2.0, 0.5])
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(bulk_velocity[axis])
    state = state._replace(primitive_state=primitive_state)

    new_primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
    )

    assert _num_sinks(sink_particles) == 1
    magnetic_index = jnp.array(registered_variables.magnetic_index)
    assert jnp.array_equal(
        new_primitive_state[magnetic_index],
        state.primitive_state[magnetic_index],
    )

    gas_mass_before, gas_momentum_before = _gas_mass_and_momentum(
        state.primitive_state,
        registered_variables,
    )
    gas_mass_after, gas_momentum_after = _gas_mass_and_momentum(
        new_primitive_state,
        registered_variables,
    )
    sink_mass = jnp.sum(sink_particles.mass)
    sink_momentum = jnp.sum(sink_particles.mass[:, None] * sink_particles.velocity, axis=0)
    assert jnp.allclose(gas_mass_before, gas_mass_after + sink_mass, rtol=1e-6)
    assert jnp.allclose(
        gas_momentum_before,
        gas_momentum_after + sink_momentum,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "contraction_rate, expected_num_accreted_cells",
    [
        # Slow infall: every cell above the threshold is bound, all 57 are
        # accreted.
        (1.0, 57),
        # Fast infall: every cell except the sink's own one is unbound; the
        # sink's own cell is accreted without checks.
        (1000.0, 1),
    ],
)
def test_accretion_bound(contraction_rate, expected_num_accreted_cells):
    """Gas is accreted only if it is bound to the sink, E_grav + E_kin < 0
    (Section 2.3), except in the cell containing the sink. The existing sink
    gains the mass above the threshold of the accreted cells."""
    state, config, params, registered_variables = _setup()

    # Gas falling toward the clump centre, where an existing sink of mass 0.5
    # sits at rest, v = −g (x − x_c). The infall keeps the radial velocity
    # negative, so only the bound check decides.
    state = _with_linear_velocity(
        state,
        registered_variables,
        (-contraction_rate, -contraction_rate, -contraction_rate),
    )
    sink_mass = 0.5
    existing_sink = _sink_at_clump_center(sink_mass, config, state.primitive_state.dtype)

    new_primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        existing_sink,
    )

    # No new sink: the existing sink blocks creation (proximity check).
    assert _num_sinks(sink_particles) == 1

    num_accreted_cells = _num_accreted_cells(
        state.primitive_state,
        new_primitive_state,
        registered_variables,
    )
    assert num_accreted_cells == expected_num_accreted_cells

    density_index = registered_variables.density_index
    density = state.primitive_state[density_index]
    accreted = new_primitive_state[density_index] < density
    accreted_mass = jnp.sum(
        jnp.where(accreted, density - _density_threshold(config, params), 0.0)
    ) * config.grid_spacing**3
    assert jnp.allclose(sink_particles.mass[0], sink_mass + accreted_mass, rtol=1e-5)


@pytest.mark.parametrize(
    "velocity_gradient, expected_num_accreted_cells",
    [
        # Slowly expanding away from the sink: bound (as slow as the infall
        # of test_accretion_bound), but moving away, so only the sink's own
        # cell is accreted.
        (1.0, 1),
        # Slowly falling toward the sink: all 57 cells above the threshold.
        (-1.0, 57),
    ],
)
def test_accretion_radial_velocity(velocity_gradient, expected_num_accreted_cells):
    """Gas is accreted only if it moves toward the sink, v_r ≤ 10⁻⁵ c_s
    (Section 2.3, with FLASH's tolerance), except in the cell containing the
    sink."""
    state, config, params, registered_variables = _setup()
    state = _with_linear_velocity(
        state,
        registered_variables,
        (velocity_gradient, velocity_gradient, velocity_gradient),
    )
    existing_sink = _sink_at_clump_center(0.5, config, state.primitive_state.dtype)

    new_primitive_state, _ = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        existing_sink,
    )

    num_accreted_cells = _num_accreted_cells(
        state.primitive_state,
        new_primitive_state,
        registered_variables,
    )
    assert num_accreted_cells == expected_num_accreted_cells


def test_accretion_most_bound_sink():
    """A cell within reach of several sinks is accreted once, by the sink it
    is most strongly bound to (Section 2.3)."""
    state, config, params, registered_variables = _setup()
    dtype = state.primitive_state.dtype

    # Two sinks at rest, 2 cells on either side of the clump centre along x:
    # their accretion radii overlap, and every cell in the plane x = x_c is
    # equally far from both. The heavy sink binds those cells more strongly.
    heavy_sink_mass = 5.0
    light_sink_mass = 0.05
    sink_particles = _empty_sink_particles(config, dtype)
    sink_particles = sink_particles._replace(
        mass=sink_particles.mass.at[0].set(heavy_sink_mass).at[1].set(light_sink_mass),
        position=sink_particles.position.at[0].set(
            jnp.array([CLUMP_CENTER - 2 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER])
        ).at[1].set(
            jnp.array([CLUMP_CENTER + 2 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER])
        ),
    )

    new_primitive_state, new_sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )

    # Each cell is accreted once: the gas lost equals the mass gained.
    gas_mass_before, _ = _gas_mass_and_momentum(
        state.primitive_state,
        registered_variables,
    )
    gas_mass_after, _ = _gas_mass_and_momentum(new_primitive_state, registered_variables)
    heavy_sink_gain = new_sink_particles.mass[0] - heavy_sink_mass
    light_sink_gain = new_sink_particles.mass[1] - light_sink_mass
    assert jnp.allclose(
        gas_mass_before - gas_mass_after,
        heavy_sink_gain + light_sink_gain,
        rtol=1e-5,
    )

    # The overlap goes to the heavy sink; the light sink still gains the cells
    # within its reach only.
    assert heavy_sink_gain > light_sink_gain > 0.0


def test_accretion_tie_break():
    """A cell for which several sinks tie exactly is accreted once, by the
    sink in the lowest slot."""
    state, config, params, registered_variables = _setup()
    dtype = state.primitive_state.dtype

    # Two equal sinks at rest at the same position: they share the inner cell
    # and bind every other cell with exactly the same energy.
    sink_mass = 0.5
    sink_particles = _sink_at_clump_center(sink_mass, config, dtype)
    sink_particles = sink_particles._replace(
        mass=sink_particles.mass.at[1].set(sink_mass),
        position=sink_particles.position.at[1].set(sink_particles.position[0]),
    )

    new_primitive_state, new_sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )

    # Each cell is accreted once: the gas lost equals the mass gained.
    gas_mass_before, _ = _gas_mass_and_momentum(
        state.primitive_state,
        registered_variables,
    )
    gas_mass_after, _ = _gas_mass_and_momentum(new_primitive_state, registered_variables)
    sink_gain = new_sink_particles.mass[:2] - sink_mass
    assert jnp.allclose(gas_mass_before - gas_mass_after, jnp.sum(sink_gain), rtol=1e-5)

    # Every cell goes to the sink in the lower slot.
    assert sink_gain[0] > 0.0
    assert sink_gain[1] == 0.0


if __name__ == "__main__":
    test_new_sink_properties()
    test_accretion_conservation()
    test_accretion_with_magnetic_field()
    test_accretion_bound(1.0, 57)
    test_accretion_radial_velocity(1.0, 1)
    test_accretion_most_bound_sink()
    test_accretion_tie_break()
