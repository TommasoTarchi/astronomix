"""
Shared parts of the isothermal collapse setups used to validate the sink
particles: the configuration of a self-gravitating, isothermal, cubic 3D box,
the cell centre at the middle of the box, and the assembly of the initial state.
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import (
    ISOTHERMAL,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import StateStruct, finalize_state
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.variable_registry.registered_variables import get_registered_variables

#: Open boundaries on every side; the Poisson equation is then solved with
#: isolated boundary conditions.
OPEN_BOX = BoundarySettings()

#: Periodic boundaries on every side.
PERIODIC_BOX = BoundarySettings(
    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def _isothermal_box(
    config: SimulationConfig,
    params: SimulationParams,
    box_length: float,
    boundary_settings: BoundarySettings,
    sound_speed,
    gravitational_constant,
    t_end: float,
) -> tuple[SimulationConfig, SimulationParams]:
    """
    Configure a self-gravitating, isothermal, cubic 3D box.

    Enforces the geometry (3D Cartesian), the cubic box and its boundaries,
    self-gravity, the isothermal equation of state, and the state struct
    (which carries the sink particles). The number of cells, the solver, the
    sink particle configuration and the time-stepping options are left to the
    caller; the number of cells must be the same along every axis.

    Args:
        config: Simulation configuration.
        params: Simulation parameters.
        box_length: Side length of the box.
        boundary_settings: Boundary conditions of the box.
        sound_speed: The isothermal sound speed.
        gravitational_constant: The gravitational constant.
        t_end: The end time of the run.

    Returns:
        ``(config, params)``, not yet finalized.
    """
    config = config._replace(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=StaticFloatVector(box_length, box_length, box_length),
        boundary_settings=boundary_settings,
        gravity_config=config.gravity_config._replace(self_gravity=True),
        equation_of_state=ISOTHERMAL,
        state_struct=True,
    )
    params = params._replace(
        t_end=t_end,
        isothermal_sound_speed=sound_speed,
        gravitational_constant=gravitational_constant,
    )
    return config, params


def _cell_centers_and_spacing(config: SimulationConfig) -> tuple[jnp.ndarray, float]:
    """
    The cell centres of the box and the grid spacing.

    Args:
        config: The (not yet finalized) simulation configuration of a cubic box.

    Returns:
        ``(cell_centers, grid_spacing)``: the centres, shape (n, n, n, 3), and
        the spacing.
    """
    cell_centers = get_helper_data(config).geometric_centers
    grid_spacing = config.box_size.x / cell_centers.shape[0]
    return cell_centers, grid_spacing


def _central_cell_center(config: SimulationConfig) -> jnp.ndarray:
    """
    The centre of the cell at the middle of the box: the box centre for an
    odd number of cells, otherwise the centre of the cell just above it along
    every axis. A collapse centred there has one cell as its unique potential
    minimum.

    Args:
        config: The (not yet finalized) simulation configuration of a cubic box.

    Returns:
        The point, shape (3,).
    """
    cell_centers, grid_spacing = _cell_centers_and_spacing(config)
    num_cells = cell_centers.shape[0]
    return jnp.full((3,), (num_cells // 2 + 0.5) * grid_spacing)


def _isothermal_state(
    config: SimulationConfig,
    params: SimulationParams,
    density,
    velocity,
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Assemble the initial state struct of an isothermal setup and finalize the
    configuration.

    Args:
        config: The configuration from ``_isothermal_box``.
        params: The parameters from ``_isothermal_box``.
        density: The gas density, shape (n, n, n).
        velocity: The gas velocity, shape (n, n, n, 3).

    Returns:
        ``(state, config, params)``: the state struct without sinks, the
        finalized configuration, and the parameters.
    """
    registered_variables = get_registered_variables(config)
    primitive_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity[..., 0],
        velocity_y=velocity[..., 1],
        velocity_z=velocity[..., 2],
    )
    config = finalize_config(config, primitive_state.shape)
    return finalize_state(config, primitive_state), config, params
