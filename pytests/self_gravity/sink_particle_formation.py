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

# With c_s² = 1, G = 1 and r_acc = 2.5 cells on a 32³ unit box, the density
# threshold (Eq. 32) is ρ_res = π / (4 r_acc²) ≈ 129. A peak overdensity of 200
# puts exactly the 8 central cells above it.
SETTINGS = SinkFormationSettings(peak_overdensity=200.0)


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
    """One sink forms per cell above the density threshold (Eq. 32), with the
    mass above the threshold in its control volume and the centre-of-mass
    velocity of the gas."""
    state, config, params, registered_variables = _setup(sink_particles=True)

    # Give the gas a uniform bulk velocity, which every sink must inherit as
    # its centre-of-mass velocity.
    bulk_velocity = jnp.array([1.0, -2.0, 0.5])
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(bulk_velocity[axis])
    state = state._replace(primitive_state=primitive_state)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    density = primitive_state[registered_variables.density_index]
    density_threshold = _density_threshold(config, params)
    cells_above_threshold = density > density_threshold
    num_sinks = int(jnp.sum(sink_particles.mass > 0.0))
    assert num_sinks == int(jnp.sum(cells_above_threshold)) == 8

    # The 8 central cells lie in each other's control volumes, so every sink
    # carries the whole mass above the threshold.
    mass_above_threshold = jnp.sum(
        jnp.where(cells_above_threshold, density - density_threshold, 0.0)
    ) * config.grid_spacing**3
    assert jnp.allclose(sink_particles.mass[:num_sinks], mass_above_threshold, rtol=1e-5)

    # The sinks are placed symmetrically around the box centre.
    mean_position = jnp.mean(sink_particles.position[:num_sinks], axis=0)
    assert jnp.allclose(mean_position, 0.5, atol=1e-5)

    assert jnp.allclose(sink_particles.velocity[:num_sinks], bulk_velocity, atol=1e-5)


def test_proximity():
    """No sink forms within r_acc of an existing sink (Section 2.2.7), with
    distances taken across periodic boundaries."""
    state, config, params, registered_variables = _setup(sink_particles=True)
    dtype = state.primitive_state.dtype
    box_length = SETTINGS.box_length

    # Every candidate of the first call lies within r_acc of the sinks it
    # created, so a second call on the same state adds none. The first call
    # creates 8 sinks, one per cell above threshold: the proximity check only
    # compares with sinks that already exist.
    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    assert _num_sinks(sink_particles) == 8
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, sink_particles
    )
    assert _num_sinks(sink_particles) == 8

    # A sink far from the clump does not block formation.
    far_sink = _one_sink_at((0.1, 0.1, 0.1), config, dtype)
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, far_sink
    )
    assert _num_sinks(sink_particles) == 1 + 8

    # A sink one box length away from the clump centre is its own periodic
    # copy, so it blocks formation.
    periodic_copy_sink = _one_sink_at((0.5 + box_length, 0.5, 0.5), config, dtype)
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, periodic_copy_sink
    )
    assert _num_sinks(sink_particles) == 1

    # A strong clump centred just across the x = 0 boundary. Candidates in the
    # last cells along x have centres of mass past the box edge (up to about
    # x = 1.008 for this clump) before they are wrapped; every sink must be
    # stored inside the box, and the sinks must block a second formation
    # across the boundary.
    boundary_settings = SETTINGS._replace(
        peak_overdensity=1000.0,
        overdensity_center=(0.04, 0.5, 0.5),
    )
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        settings=boundary_settings,
        max_num_sinks=256,
        max_num_candidates=256,
    )
    density = state.primitive_state[registered_variables.density_index]
    num_cells_above_threshold = int(jnp.sum(density > _density_threshold(config, params)))

    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    assert _num_sinks(sink_particles) == num_cells_above_threshold
    positions = sink_particles.position[:num_cells_above_threshold]
    assert jnp.all((positions >= 0.0) & (positions < box_length))
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, sink_particles
    )
    assert _num_sinks(sink_particles) == num_cells_above_threshold


def test_sink_particle_slots_overflow(capfd):
    """When more sinks pass than there are free slots, the slots are filled,
    the rest is discarded and a warning is printed."""
    state, config, params, registered_variables = _setup(
        sink_particles=True,
        max_num_sinks=4,
    )

    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    sink_particles.mass.block_until_ready()

    assert jnp.all(sink_particles.mass > 0.0)
    assert "4 new sink particles discarded" in capfd.readouterr().out


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
