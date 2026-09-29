"""
Sink particle formation test (3D).

A periodic box of uniform gas with a Gaussian overdensity at its centre, evolved
with self-gravity. The setup is used to check the sink particle creation of
Federrath et al. (2010), one creation check at a time.

Run as a script to perform the checks:

    JAX_PLATFORMS=cpu PYTHONPATH=. python astronomix/test_setups/self_gravity/sink_particle_formation3D.py
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._modules._sink_particles._sink_particle_options import SinkParticleConfig

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables


class SinkFormationSettings(NamedTuple):
    """Problem constants for the sink particle formation test."""

    #: Number of cells per axis.
    num_cells: int = 32

    #: Background density.
    background_density: float = 1.0

    #: Peak density of the Gaussian overdensity, on top of the background.
    peak_overdensity: float = 100.0

    #: Width (standard deviation) of the Gaussian overdensity.
    overdensity_width: float = 0.05

    #: Sound speed squared of the gas.
    sound_speed_squared: float = 1.0

    #: Adiabatic index of the gas.
    gamma: float = 5.0 / 3.0

    #: Gravitational constant.
    gravitational_constant: float = 1.0

    #: Number of (fixed) time steps to run.
    num_timesteps: int = 3

    #: End time of the run.
    t_end: float = 1e-3


def setup_sink_formation(
    sink_particles: bool,
    settings: SinkFormationSettings = SinkFormationSettings(),
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Set up the Gaussian overdensity in a periodic, self-gravitating 3D box.

    Args:
        sink_particles: Whether sink particle formation is switched on.
        settings: Problem constants.

    Returns:
        state: The initial state struct (no sinks passed in).
        config: The finalized simulation configuration.
        params: The simulation parameters.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=3,
        num_cells=settings.num_cells,
        box_size=StaticFloatVector(1.0, 1.0, 1.0),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        gravity_config=GravityConfig(self_gravity=True),
        state_struct=True,
        fixed_timestep=True,
        num_timesteps=settings.num_timesteps,
        sink_particle_config=SinkParticleConfig(sink_particles=sink_particles),
    )
    params = SimulationParams(
        t_end=settings.t_end,
        gamma=settings.gamma,
        gravitational_constant=settings.gravitational_constant,
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    # Gaussian overdensity centred in the box, with the gas at rest and at a
    # uniform sound speed (pressure proportional to density).
    cell_centers = helper_data.geometric_centers
    distance_squared = jnp.sum((cell_centers - 0.5) ** 2, axis=-1)
    density = settings.background_density + settings.peak_overdensity * jnp.exp(
        -0.5 * distance_squared / settings.overdensity_width**2
    )
    pressure = density * settings.sound_speed_squared / settings.gamma
    zero_velocity = jnp.zeros_like(density)

    primitive_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero_velocity,
        velocity_y=zero_velocity,
        velocity_z=zero_velocity,
        gas_pressure=pressure,
    )

    config = finalize_config(config, primitive_state.shape)

    return StateStruct(primitive_state=primitive_state), config, params


def run(sink_particles: bool, settings: SinkFormationSettings = SinkFormationSettings()):
    """Set up and run the test; return the final state struct."""
    state, config, params = setup_sink_formation(sink_particles, settings)
    registered_variables = get_registered_variables(config)
    return time_integration(state, config, params, registered_variables)


def check_no_sinks_and_unchanged_fluid():
    """
    With the formation step still a no-op, switching sinks on must leave all
    slots empty and the fluid result bit-identical to a run with sinks off.
    """
    final_with_sinks = run(sink_particles=True)
    final_without_sinks = run(sink_particles=False)

    assert final_without_sinks.sink_particles is None
    assert final_with_sinks.sink_particles.mass.shape == (64,)
    assert jnp.all(final_with_sinks.sink_particles.mass == 0.0)
    assert jnp.array_equal(
        final_with_sinks.primitive_state,
        final_without_sinks.primitive_state,
    )
    print("check_no_sinks_and_unchanged_fluid: passed")


if __name__ == "__main__":
    check_no_sinks_and_unchanged_fluid()
