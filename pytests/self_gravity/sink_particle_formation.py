"""
Sink particle formation and accretion pytest (fast).

Checks the sink particle creation and gas accretion of Federrath et al. (2010)
on the Gaussian overdensity setup in
``astronomix/test_setups/self_gravity/sink_particle_formation3D.py``: each
creation criterion, the accretion criteria, the choice of a single sink for
each accreted cell, and the conservation of mass and momentum. Also checks the
configuration requirements enforced by ``finalize_config``.
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

# astronomix containers
from astronomix import (
    GravityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import StaticIntVector
from astronomix._modules._sink_particles._sink_particle_options import SinkParticleConfig

# astronomix functions
from astronomix import (
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    SinkFormationSettings,
    setup_sink_formation,
)
from astronomix._modules._sink_particles._sink_particle_formation import (
    _empty_sink_particles,
    _form_sink_particles,
)
from astronomix._modules._sink_particles._sink_particle_accretion import _accrete_gas


NUM_CELLS = 32
NUM_TIMESTEPS = 3
CELL_SIZE = 1.0 / NUM_CELLS

# The clump is centred on the centre of cell (16, 16, 16), so that this cell is
# the unique potential minimum. With c_s² = 1, G = 1 and r_acc = 2.5 cells on a
# 32³ unit box, the density threshold (Eq. 32) is ρ_res = π / (4 r_acc²) ≈ 129.
# A peak overdensity of 400 puts 57 cells above it, all within the centre
# cell's control volume, and makes the control volume Jeans unstable
# (|E_grav| ≈ 1.26 × 2 E_th).
CLUMP_CENTER = 0.5 + 0.5 * CELL_SIZE
SETTINGS = SinkFormationSettings(
    peak_overdensity=400.0,
    overdensity_center=(CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER),
)


def _setup(sink_particles: bool, settings=SETTINGS, mhd=False, **sink_particle_options):
    """Set up the Gaussian overdensity with the given sink options.

    Args:
        sink_particles: Whether sink particle formation is switched on.
        settings: The problem constants of the setup.
        mhd: Whether MHD is switched on (the field is ``settings.magnetic_field_z``).
        **sink_particle_options: Further ``SinkParticleConfig`` fields.

    Returns:
        ``(state, config, params, registered_variables)``.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        progress_bar=False,
        mhd=mhd,
        sink_particle_config=SinkParticleConfig(
            sink_particles=sink_particles,
            **sink_particle_options,
        ),
    )
    state, config, params = setup_sink_formation(config, SimulationParams(), settings)
    registered_variables = get_registered_variables(config)
    return state, config, params, registered_variables


def _density_threshold(config, params):
    """The density threshold of Eq. 32 for the setup's uniform c_s² = 1."""
    accretion_radius = (
        config.sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    )
    return jnp.pi * SETTINGS.sound_speed_squared / (
        4.0 * params.gravitational_constant * accretion_radius**2
    )


def _update_sinks_once(state, config, params, registered_variables, sink_particles=None):
    """Apply one formation and accretion step, as in a time step, by default
    starting from no sinks.

    Returns:
        ``(primitive_state, sink_particles)`` after the step.
    """
    if sink_particles is None:
        sink_particles = _empty_sink_particles(config, state.primitive_state.dtype)
    sink_particles, num_active_sinks = _form_sink_particles(
        state.primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    return _accrete_gas(
        state.primitive_state,
        sink_particles,
        num_active_sinks,
        config,
        params,
        registered_variables,
    )


def _form_sinks_once(state, config, params, registered_variables, sink_particles=None):
    """Apply one formation and accretion step; return only the sinks."""
    _, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )
    return sink_particles


def _one_sink_at(position, config, dtype):
    """Sink particles with a single (unit-mass) sink at ``position``."""
    sink_particles = _empty_sink_particles(config, dtype)
    return sink_particles._replace(
        mass=sink_particles.mass.at[0].set(1.0),
        position=sink_particles.position.at[0].set(jnp.array(position)),
    )


def _num_sinks(sink_particles):
    """The number of filled sink slots."""
    return int(jnp.sum(sink_particles.mass > 0.0))


def _gas_mass_and_momentum(primitive_state, registered_variables):
    """Total gas mass and momentum of an (unpadded, periodic) state."""
    cell_volume = CELL_SIZE**3
    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [primitive_state[index] for index in registered_variables.velocity_index]
    )
    mass = jnp.sum(density) * cell_volume
    momentum = jnp.sum(density * velocity, axis=(1, 2, 3)) * cell_volume
    return mass, momentum


def test_accretion_conservation():
    """Accretion moves mass and momentum from the gas to the sinks without
    creating or destroying any, and leaves accreted cells at the density
    threshold with an unchanged sound speed."""
    state, config, params, registered_variables = _setup(sink_particles=True)

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

    # One call: what the gas loses, the sink gains.
    gas_mass_before, gas_momentum_before = _gas_mass_and_momentum(
        primitive_state,
        registered_variables,
    )
    gas_mass_after, gas_momentum_after = _gas_mass_and_momentum(
        new_primitive_state,
        registered_variables,
    )
    sink_mass = jnp.sum(sink_particles.mass)
    sink_momentum = jnp.sum(sink_particles.mass[:, None] * sink_particles.velocity, axis=0)
    assert sink_mass > 0.0
    assert jnp.allclose(gas_mass_before, gas_mass_after + sink_mass, rtol=1e-6)
    assert jnp.allclose(
        gas_momentum_before,
        gas_momentum_after + sink_momentum,
        rtol=1e-6,
    )

    # Accreted cells end at the threshold with c_s² unchanged; the other cells
    # are untouched.
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    density_threshold = _density_threshold(config, params)
    accreted = primitive_state[density_index] > density_threshold
    assert jnp.allclose(
        new_primitive_state[density_index][accreted],
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
        new_primitive_state[:, ~accreted],
        primitive_state[:, ~accreted],
    )

    # A full run: the total mass of gas and sinks stays constant.
    initial_mass, _ = _gas_mass_and_momentum(state.primitive_state, registered_variables)
    final_state = time_integration(state, config, params, registered_variables)
    final_gas_mass, _ = _gas_mass_and_momentum(
        final_state.primitive_state,
        registered_variables,
    )
    final_sink_mass = jnp.sum(final_state.sink_particles.mass)
    assert final_sink_mass > 0.0
    assert jnp.allclose(initial_mass, final_gas_mass + final_sink_mass, rtol=1e-5)


def test_accretion_by_existing_sink():
    """An existing sink accretes the gas above the density threshold within
    its accretion radius."""
    state, config, params, registered_variables = _setup(sink_particles=True)
    sink_mass = 0.5
    existing_sink = _one_sink_at(
        (CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER),
        config,
        state.primitive_state.dtype,
    )
    existing_sink = existing_sink._replace(mass=existing_sink.mass.at[0].set(sink_mass))

    new_primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        existing_sink,
    )

    # No new sink: the existing sink blocks creation (proximity check).
    assert _num_sinks(sink_particles) == 1

    density = state.primitive_state[registered_variables.density_index]
    density_threshold = _density_threshold(config, params)
    cells_above_threshold = density > density_threshold
    mass_above_threshold = jnp.sum(
        jnp.where(cells_above_threshold, density - density_threshold, 0.0)
    ) * config.grid_spacing**3
    assert jnp.allclose(
        sink_particles.mass[0],
        sink_mass + mass_above_threshold,
        rtol=1e-5,
    )
    assert jnp.all(
        new_primitive_state[registered_variables.density_index]
        <= density_threshold * (1.0 + 1e-6)
    )


def test_new_sink_properties():
    """A new sink accretes the mass above the density threshold (Eq. 32)
    within its accretion radius, and ends at the centre of mass and with the
    centre-of-mass velocity of the accreted gas."""
    state, config, params, registered_variables = _setup(sink_particles=True)

    # Give the gas a uniform bulk velocity, which the sink must inherit as its
    # centre-of-mass velocity.
    bulk_velocity = jnp.array([1.0, -2.0, 0.5])
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(bulk_velocity[axis])
    state = state._replace(primitive_state=primitive_state)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    # One sink, from the centre cell, the only potential minimum.
    assert _num_sinks(sink_particles) == 1

    # The 57 cells above the threshold all lie within r_acc of the sink,
    # created at the centre cell, so it accretes the whole mass above the
    # threshold.
    density = primitive_state[registered_variables.density_index]
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


def test_proximity():
    """No sink forms within r_acc of an existing sink (Section 2.2.7), with
    distances taken across periodic boundaries."""
    state, config, params, registered_variables = _setup(sink_particles=True)
    dtype = state.primitive_state.dtype
    box_length = SETTINGS.box_length

    # The sink created by the first call blocks a second call on the same
    # state.
    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    assert _num_sinks(sink_particles) == 1
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, sink_particles
    )
    assert _num_sinks(sink_particles) == 1

    # A sink far from the clump does not block formation.
    far_sink = _one_sink_at((0.1, 0.1, 0.1), config, dtype)
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, far_sink
    )
    assert _num_sinks(sink_particles) == 1 + 1

    # A sink one box length away from the clump centre is its own periodic
    # copy, so it blocks formation.
    periodic_copy_sink = _one_sink_at(
        (CLUMP_CENTER + box_length, CLUMP_CENTER, CLUMP_CENTER),
        config,
        dtype,
    )
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, periodic_copy_sink
    )
    assert _num_sinks(sink_particles) == 1

    # A clump centred on the first cell along x, next to the x = 0 boundary.
    # An existing sink in the last cell along x is one cell away across the
    # boundary, so it blocks formation.
    boundary_settings = SETTINGS._replace(
        overdensity_center=(0.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
    )
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        settings=boundary_settings,
    )
    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    assert _num_sinks(sink_particles) == 1
    across_boundary_sink = _one_sink_at(
        (box_length - 0.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
        config,
        dtype,
    )
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, across_boundary_sink
    )
    assert _num_sinks(sink_particles) == 1


def test_position_wrapping():
    """A sink whose centre of mass falls past a periodic boundary is stored
    inside the box."""
    # A clump centred on the second cell along x (index 1), and an external
    # potential well in the last cell along x (index 31), two cells away across
    # the x = 0 boundary. The well makes the last cell the potential minimum,
    # so the sink forms from it, while most of the gas in its control volume
    # lies past the edge: the centre of mass comes out at x ≈ 1.011.
    boundary_settings = SETTINGS._replace(
        peak_overdensity=400.0,
        overdensity_center=(1.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
    )
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        settings=boundary_settings,
    )
    config = config._replace(
        gravity_config=config.gravity_config._replace(external_potential=True)
    )
    well_index = (NUM_CELLS - 1, NUM_CELLS // 2, NUM_CELLS // 2)
    params = params._replace(
        gravitational_potential=jnp.zeros((NUM_CELLS,) * 3).at[well_index].set(-10.0)
    )

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    # The stored x lies within the first cell: past the edge, wrapped back.
    assert _num_sinks(sink_particles) == 1
    assert 0.0 <= sink_particles.position[0, 0] < CELL_SIZE


def _with_linear_velocity(state, registered_variables, velocity_gradient):
    """Set the gas velocity to v_d = g_d (x_d − x_c) along each axis d, a flow
    that expands (g_d > 0) or contracts (g_d < 0) about the clump centre x_c."""
    cell_centers = (jnp.arange(NUM_CELLS) + 0.5) * CELL_SIZE
    coordinates = jnp.meshgrid(cell_centers, cell_centers, cell_centers, indexing="ij")
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(
            velocity_gradient[axis] * (coordinates[axis] - CLUMP_CENTER)
        )
    return state._replace(primitive_state=primitive_state)


@pytest.mark.parametrize(
    "velocity_gradient, expected_num_sinks",
    [
        # Contracting along every axis: a sink forms.
        ((-1.0, -1.0, -1.0), 1),
        # Expanding along every axis: no sink.
        ((1.0, 1.0, 1.0), 0),
        # Contracting along y and z but expanding along x. The divergence is
        # negative, but the flow must converge along each axis: no sink.
        ((1.0, -2.0, -2.0), 0),
    ],
)
def test_converging_flow(velocity_gradient, expected_num_sinks):
    """Sinks only form where the flow converges along every axis (Section
    2.2.3)."""
    state, config, params, registered_variables = _setup(sink_particles=True)
    state = _with_linear_velocity(state, registered_variables, velocity_gradient)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


def test_potential_minimum():
    """A sink forms only where the potential is lowest in the control volume
    (Section 2.2.4), which need not be the densest cell."""
    state, config, params, registered_variables = _setup(sink_particles=True)

    # Add a narrow density spike in a single cell, 2 cells from the clump
    # centre along x (within r_acc). The spike cell becomes the densest cell
    # (ρ ≈ 484 against ≈ 401 at the centre), but it holds too little mass to
    # move the potential minimum away from the clump centre. The pressure is
    # raised with the density, keeping c_s² = 1.
    spike_index = (NUM_CELLS // 2 + 2, NUM_CELLS // 2, NUM_CELLS // 2)
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    primitive_state = state.primitive_state
    primitive_state = primitive_state.at[(density_index, *spike_index)].add(300.0)
    primitive_state = primitive_state.at[(pressure_index, *spike_index)].set(
        primitive_state[(density_index, *spike_index)]
        * SETTINGS.sound_speed_squared
        / SETTINGS.gamma
    )
    state = state._replace(primitive_state=primitive_state)
    density = primitive_state[density_index]
    assert density[spike_index] == jnp.max(density)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    # A single sink, formed from the centre cell: its centre of mass is pulled
    # slightly toward the spike, but stays within half a cell of the centre.
    assert _num_sinks(sink_particles) == 1
    assert jnp.all(jnp.abs(sink_particles.position[0] - CLUMP_CENTER) < 0.5 * CELL_SIZE)


@pytest.mark.parametrize(
    "peak_overdensity, sound_speed_squared, expected_num_sinks",
    [
        # Cold and dense enough: |E_grav| ≈ 1.26 × 2 E_th, a sink forms.
        (400.0, 1.0, 1),
        # Above the density threshold, but not Jeans unstable:
        # |E_grav| ≈ 0.63 × 2 E_th.
        (200.0, 1.0, 0),
        # Hot: doubling c_s² doubles E_th (|E_grav| ≈ 0.63 × 2 E_th). The
        # threshold doubles too (ρ_res ≈ 257), but the centre (ρ ≈ 401) stays
        # above it, so only the Jeans check fails.
        (400.0, 2.0, 0),
    ],
)
def test_jeans_instability(peak_overdensity, sound_speed_squared, expected_num_sinks):
    """A sink forms only if the gas in the control volume is Jeans unstable,
    |E_grav| > 2 E_th + E_mag (Section 2.2.5)."""
    settings = SETTINGS._replace(
        peak_overdensity=peak_overdensity,
        sound_speed_squared=sound_speed_squared,
    )
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        settings=settings,
    )

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


@pytest.mark.parametrize(
    "angular_velocity, expected_num_sinks",
    [
        # At rest, |E_grav| − E_th ≈ 1.52 E_th, and the rotation becomes
        # unbound at Ω ≈ 25.5 (where E_kin reaches |E_grav| − E_th).
        # About half of that: bound, a sink forms.
        (12.5, 1),
        # About twice that: unbound, no sink.
        (50.0, 0),
    ],
)
def test_bound_state(angular_velocity, expected_num_sinks):
    """A sink forms only if the gas in the control volume is bound,
    E_grav + E_th + E_kin + E_mag < 0 (Section 2.2.6)."""
    state, config, params, registered_variables = _setup(sink_particles=True)

    # Solid-body rotation about the z axis through the clump centre. Along
    # each axis, the velocity component along that axis does not change, so
    # the converging-flow check still passes; the centre-of-mass velocity is
    # zero by symmetry, so all the motion counts as kinetic energy.
    cell_centers = (jnp.arange(NUM_CELLS) + 0.5) * CELL_SIZE
    x, y, _ = jnp.meshgrid(cell_centers, cell_centers, cell_centers, indexing="ij")
    velocity_index = registered_variables.velocity_index
    primitive_state = state.primitive_state
    primitive_state = primitive_state.at[velocity_index.x].set(
        -angular_velocity * (y - CLUMP_CENTER)
    )
    primitive_state = primitive_state.at[velocity_index.y].set(
        angular_velocity * (x - CLUMP_CENTER)
    )
    primitive_state = primitive_state.at[velocity_index.z].set(0.0)
    state = state._replace(primitive_state=primitive_state)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


@pytest.mark.parametrize(
    "magnetic_field, expected_num_sinks",
    [
        # The control volume stops being Jeans unstable between B = 9 and
        # B = 10, and E_mag grows as B².
        # About 1% of the threshold magnetic energy: a sink forms.
        (1.0, 1),
        # About 10 times the threshold magnetic energy: no sink.
        (30.0, 0),
    ],
)
def test_magnetic_energy(magnetic_field, expected_num_sinks):
    """The magnetic energy E_mag = ½ Σ |B|² ΔV of the control volume counts
    against sink formation in the Jeans check, |E_grav| > 2 E_th + E_mag
    (Section 2.2.5)."""
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        settings=SETTINGS._replace(magnetic_field_z=magnetic_field),
        mhd=True,
    )

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


def test_accretion_with_magnetic_field():
    """Accretion leaves the magnetic field unchanged, and still moves mass and
    momentum from the gas to the sink without creating or destroying any."""
    state, config, params, registered_variables = _setup(
        sink_particles=True,
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


def _sink_at_clump_center(mass, config, dtype):
    """Sink particles with a single sink of ``mass`` at the clump centre, at
    rest."""
    sink_particles = _one_sink_at(
        (CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER),
        config,
        dtype,
    )
    return sink_particles._replace(mass=sink_particles.mass.at[0].set(mass))


def _num_accreted_cells(primitive_state, new_primitive_state, registered_variables):
    """The number of cells that lost gas."""
    density_index = registered_variables.density_index
    return int(jnp.sum(new_primitive_state[density_index] < primitive_state[density_index]))


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
    (Section 2.3), except in the cell containing the sink."""
    state, config, params, registered_variables = _setup(sink_particles=True)

    # Gas falling toward the clump centre, where an existing sink of mass 0.5
    # sits at rest, v = −g (x − x_c). The infall keeps the radial velocity
    # negative, so only the bound check decides.
    state = _with_linear_velocity(
        state,
        registered_variables,
        (-contraction_rate, -contraction_rate, -contraction_rate),
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
    state, config, params, registered_variables = _setup(sink_particles=True)
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
    state, config, params, registered_variables = _setup(sink_particles=True)
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

    # The overlap goes to the heavy sink; the light sink keeps at least its
    # own cell.
    assert heavy_sink_gain > light_sink_gain > 0.0


def test_accretion_tie_break():
    """A cell for which several sinks tie exactly is accreted once, by the
    sink in the lowest slot."""
    state, config, params, registered_variables = _setup(sink_particles=True)
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


def test_sink_particle_slots_overflow(capfd):
    """When a new sink passes but every slot is taken, it is discarded and a
    warning is printed."""
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        max_num_sinks=1,
    )
    far_sink = _one_sink_at((0.1, 0.1, 0.1), config, state.primitive_state.dtype)

    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, far_sink
    )
    sink_particles.mass.block_until_ready()

    assert _num_sinks(sink_particles) == 1
    assert jnp.allclose(sink_particles.position[0], 0.1)
    assert "1 new sink particles discarded" in capfd.readouterr().out


@pytest.mark.parametrize(
    "unsupported_options, state_shape, expected_error",
    [
        (dict(dimensionality=2), (4, 16, 16), ValueError),
        (dict(gravity_config=GravityConfig(self_gravity=False)), (5, 16, 16, 16), ValueError),
        (dict(state_struct=False), (5, 16, 16, 16), ValueError),
        (dict(return_snapshots=True), (5, 16, 16, 16), NotImplementedError),
    ],
)
def test_sink_particle_config_requirements(unsupported_options, state_shape, expected_error):
    """``finalize_config`` must reject each configuration that sink particle
    formation does not support.

    Args:
        unsupported_options: The config fields that make the configuration
            unsupported, on top of an otherwise valid one.
        state_shape: The primitive state shape matching the dimensionality.
        expected_error: The exception ``finalize_config`` must raise.
    """
    supported_options = dict(
        dimensionality=3,
        gravity_config=GravityConfig(self_gravity=True),
        state_struct=True,
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    config = SimulationConfig(**{**supported_options, **unsupported_options})

    with pytest.raises(expected_error):
        finalize_config(config, state_shape)


if __name__ == "__main__":
    test_accretion_conservation()
    test_accretion_by_existing_sink()
    test_new_sink_properties()
    test_proximity()
    test_position_wrapping()
    test_converging_flow((-1.0, -1.0, -1.0), 1)
    test_potential_minimum()
    test_jeans_instability(400.0, 1.0, 1)
    test_bound_state(12.5, 1)
    test_magnetic_energy(30.0, 0)
    test_accretion_with_magnetic_field()
    test_accretion_bound(1.0, 57)
    test_accretion_radial_velocity(1.0, 1)
    test_accretion_most_bound_sink()
    test_accretion_tie_break()
