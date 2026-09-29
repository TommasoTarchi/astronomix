"""
Container for an extended simulation state.

Wraps the primitive fluid state in a small struct so simulations that follow
additional quantities (e.g. sink particles) can carry them alongside the fluid.
Selected via ``config.state_struct``.
"""

# typing
from types import NoneType
from typing import NamedTuple, Union

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE


class SinkParticles(NamedTuple):
    """
    Sink particle data in fixed-size arrays.

    JIT requires array sizes known at compile time, so the arrays always have
    ``config.sink_particle_config.max_num_sinks`` slots. Sinks are only ever
    appended, so the filled slots are contiguous at the front; a slot is
    empty when its mass is zero.
    """

    #: Sink masses, shape (max_num_sinks,).
    mass: jax.Array

    #: Sink positions, shape (max_num_sinks, 3).
    position: jax.Array

    #: Sink velocities, shape (max_num_sinks, 3).
    velocity: jax.Array


class StateStruct(NamedTuple):
    """
    Struct bundling the fluid state with any extra simulation quantities.
    """

    #: The fluid (primitive) state.
    primitive_state: Union[STATE_TYPE, NoneType] = None

    #: The sink particles, or ``None`` when sink particles are not used. When
    #: sink particles are active and this is ``None``, the simulation starts
    #: without any sinks.
    sink_particles: Union[SinkParticles, NoneType] = None
