"""
The update of the sink particles at the end of a time step: the creation of new
sinks and the accretion of gas onto every occupied slot, following Federrath et
al. (2010), ApJ 713, 269.
"""

# general
from functools import partial

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_accretion import _accrete_gas
from astronomix._modules._sink_particles._sink_particle_formation import (
    _form_sink_particles,
)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _update_sink_particles(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles]:
    """
    Create new sinks where the gas passes the creation checks, then let every
    occupied slot, the new sinks included, accrete the gas above the density
    threshold within its accretion radius.

    New sinks are created massless, with activation 1, and get their mass from
    the accretion in the same call.

    Args:
        primitive_state: The (padded) primitive state after the hydro update.
        sink_particles: The sink particles after the leapfrog.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles)`` after creation and accretion.
    """
    sink_particles = _form_sink_particles(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    return _accrete_gas(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
