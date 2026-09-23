"""
Container for an extended simulation state.

Wraps the primitive fluid state in a small struct so simulations that follow
additional quantities (e.g. star-particle positions) can carry them alongside
the fluid. Selected via ``config.state_struct``.
"""

# typing
from types import NoneType
from typing import Any, NamedTuple, Union

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE


class SinkParticles(NamedTuple):
    """
    Fixed-size buffer of sink-particle slots.

    The number of slots, ``n_slots``, is set by ``config.num_sink_slots``. It
    is fixed because ``jax.jit`` requires every array shape to be known in
    advance and to stay constant across calls.
    """

    #: Slot masses, shape (n_slots,).
    mass: Any

    #: Slot positions, shape (n_slots, 3).
    position: Any

    #: Slot velocities, shape (n_slots, 3).
    velocity: Any


class StateStruct(NamedTuple):
    """
    Struct bundling the fluid state with any extra simulation quantities.
    """

    #: The fluid (primitive) state.
    primitive_state: Union[STATE_TYPE, NoneType] = None

    #: The sink-particle buffer, or ``None`` when sink particles are disabled.
    sinks: Union[SinkParticles, NoneType] = None

    # Further fields can be added here as the simulation grows to follow
    # additional quantities.
