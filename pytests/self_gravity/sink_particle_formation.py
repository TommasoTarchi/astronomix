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
    setup_sink_formation,
)


NUM_CELLS = 32
NUM_TIMESTEPS = 3


def _run_sink_formation(sink_particles: bool):
    """Run the Gaussian overdensity setup for a few fixed steps.

    Args:
        sink_particles: Whether sink particle formation is switched on.

    Returns:
        The final state struct.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        progress_bar=False,
        sink_particle_config=SinkParticleConfig(sink_particles=sink_particles),
    )
    state, config, params = setup_sink_formation(config, SimulationParams())
    registered_variables = get_registered_variables(config)
    return time_integration(state, config, params, registered_variables)


def test_sink_particles_do_not_change_the_fluid():
    """Switching sinks on must not change the fluid, and (with no formation
    check implemented yet) must leave every sink slot empty."""
    final_state_with_sinks = _run_sink_formation(sink_particles=True)
    final_state_without_sinks = _run_sink_formation(sink_particles=False)

    assert final_state_without_sinks.sink_particles is None
    assert jnp.all(final_state_with_sinks.sink_particles.mass == 0.0)
    assert jnp.array_equal(
        final_state_with_sinks.primitive_state,
        final_state_without_sinks.primitive_state,
    )


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
