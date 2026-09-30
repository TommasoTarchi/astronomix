"""
Sink particle formation pytest (fast).

Checks the sink particle creation of Federrath et al. (2010) on the Gaussian
overdensity setup in ``astronomix/test_setups/self_gravity/sink_particle_formation3D.py``,
and the configuration requirements checked by ``finalize_config``.
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


def _setup(sink_particles: bool, settings=SETTINGS, **sink_particle_options):
    """Set up the Gaussian overdensity with the given sink options.

    Args:
        sink_particles: Whether sink particle formation is switched on.
        settings: The problem constants of the setup.
        **sink_particle_options: Further ``SinkParticleConfig`` fields.

    Returns:
        ``(state, config, params, registered_variables)``.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        progress_bar=False,
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


def _form_sinks_once(state, config, params, registered_variables, sink_particles=None):
    """Apply one formation step, by default starting from no sinks."""
    if sink_particles is None:
        sink_particles = _empty_sink_particles(config, state.primitive_state.dtype)
    return _form_sink_particles(
        state.primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )


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


def test_sink_particles_do_not_change_the_fluid():
    """Sinks form during a run, but the fluid is identical to a run with
    sinks off, since sinks do not act on the gas."""
    final_state_with_sinks = time_integration(*_setup(sink_particles=True))
    final_state_without_sinks = time_integration(*_setup(sink_particles=False))

    assert final_state_without_sinks.sink_particles is None
    assert jnp.any(final_state_with_sinks.sink_particles.mass > 0.0)
    assert jnp.array_equal(
        final_state_with_sinks.primitive_state,
        final_state_without_sinks.primitive_state,
    )


def test_density_threshold():
    """The sink carries the mass above the density threshold (Eq. 32) in its
    control volume, sits at the centre of mass and moves with the
    centre-of-mass velocity of the gas."""
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

    # The 57 cells above the threshold all lie in the centre cell's control
    # volume, so the sink carries the whole mass above the threshold.
    density = primitive_state[registered_variables.density_index]
    density_threshold = _density_threshold(config, params)
    cells_above_threshold = density > density_threshold
    assert int(jnp.sum(cells_above_threshold)) == 57
    mass_above_threshold = jnp.sum(
        jnp.where(cells_above_threshold, density - density_threshold, 0.0)
    ) * config.grid_spacing**3
    assert jnp.allclose(sink_particles.mass[0], mass_above_threshold, rtol=1e-5)

    # The control volume is symmetric about the clump centre, so the centre
    # of mass is the clump centre.
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
    test_sink_particles_do_not_change_the_fluid()
    test_density_threshold()
    test_proximity()
    test_position_wrapping()
    test_converging_flow((-1.0, -1.0, -1.0), 1)
    test_potential_minimum()
    test_jeans_instability(400.0, 1.0, 1)
    test_bound_state(12.5, 1)
