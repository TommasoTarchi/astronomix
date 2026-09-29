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


def _setup(sink_particles: bool, **sink_particle_options):
    """Set up the Gaussian overdensity with the given sink options.

    Args:
        sink_particles: Whether sink particle formation is switched on.
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
    state, config, params = setup_sink_formation(config, SimulationParams(), SETTINGS)
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


def _form_sinks_once(state, config, params, registered_variables):
    """Apply one formation step, starting from no sinks."""
    return _form_sink_particles(
        state.primitive_state,
        _empty_sink_particles(config, state.primitive_state.dtype),
        config,
        params,
        registered_variables,
    )


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
