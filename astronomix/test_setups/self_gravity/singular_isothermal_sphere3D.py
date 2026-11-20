"""
# Singular isothermal sphere (3D)

An isothermal sphere at rest with the density

    ρ(r) = A c_s² / (2πG (r² + r_c²))   for r ≤ R,

in a box of thin gas, evolved with self-gravity. With A = 1 and r_c = 0 it is
the singular isothermal sphere, the equilibrium whose inside-out collapse
accretes onto the centre at the constant rate 0.975 c_s³ / G (Shu 1977); A > 1
makes it overdense and the accretion faster. The core radius r_c, a fraction
of a cell, replaces the singularity on the grid. The sphere is centred on a
cell centre, so that one cell is the unique potential minimum, and the
boundaries are periodic by default.

The overdensity factor A is the input with respect to which the sink particle
gradients are validated.

## References

- Shu (1977), ApJ 214, 488.
- Federrath, Banerjee, Clark & Klessen (2010), ApJ 713, 269.
"""

# typing
from types import NoneType
from typing import NamedTuple, Union

# jax
import jax.numpy as jnp

# astronomix containers
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import BoundarySettings
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.test_setups.self_gravity._isothermal_collapse import (
    PERIODIC_BOX,
    _cell_centers_and_spacing,
    _central_cell_center,
    _isothermal_box,
    _isothermal_state,
)

#: The accretion rate of the collapse of the singular isothermal sphere, in
#: units of c_s³ / G (Shu 1977).
SHU_ACCRETION_RATE = 0.975


class SingularIsothermalSphereSettings(NamedTuple):
    """Problem constants of the singular isothermal sphere."""

    #: Side length of the cubic box.
    box_length: float = 1.0

    #: Radius R of the sphere.
    cloud_radius: float = 0.45

    #: Overdensity factor A of ρ = A c_s² / (2πG r²); 1 is the equilibrium.
    #: At 32³ cells the equilibrium does not collapse: the grid softens the
    #: gravity of the centre, and the core stays pressure supported. It may be
    #: a traced value.
    overdensity_factor: float = 2.0

    #: Core radius r_c in units of the grid spacing.
    core_radius_in_cells: float = 0.5

    #: Density of the gas outside the sphere, in units of the density at its
    #: edge.
    ambient_density_ratio: float = 0.1

    #: Isothermal sound speed c_s.
    sound_speed: float = 1.0

    #: Gravitational constant G.
    gravitational_constant: float = 1.0

    #: End time of the run; the expansion wave of the collapse reaches
    #: c_s t_end, which should stay well inside R.
    t_end: float = 0.2

    #: Boundary conditions of the box. Periodic boundaries keep the mass of gas
    #: and sinks constant; with open boundaries (``OPEN_BOX``) the cloud also
    #: draws gas in through the edges of the box.
    boundary_settings: BoundarySettings = PERIODIC_BOX

    #: Centre (x, y, z) of the sphere; ``None`` places it on the cell at the
    #: middle of the box.
    center: Union[tuple, NoneType] = None


def setup_singular_isothermal_sphere(
    config: SimulationConfig,
    params: SimulationParams,
    settings: SingularIsothermalSphereSettings = SingularIsothermalSphereSettings(),
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Set up a singular isothermal sphere in an isothermal, self-gravitating
    box.

    Args:
        config: Simulation configuration; the number of cells, the solver and
            the sink particles are left to the caller.
        params: Simulation parameters.
        settings: Problem constants.

    Returns:
        state: Initial state struct of the simulation (no sinks passed in).
        config: Finalized simulation configuration.
        params: Updated simulation parameters (t_end, isothermal sound speed,
            gravitational constant).
    """
    config, params = _isothermal_box(
        config,
        params,
        settings.box_length,
        settings.boundary_settings,
        settings.sound_speed,
        settings.gravitational_constant,
        settings.t_end,
    )
    cell_centers, grid_spacing = _cell_centers_and_spacing(config)
    if settings.center is None:
        center = _central_cell_center(config)
    else:
        center = jnp.array(settings.center)
    radius_squared = jnp.sum((cell_centers - center) ** 2, axis=-1)

    core_radius = settings.core_radius_in_cells * grid_spacing
    density_scale = (
        settings.overdensity_factor
        * settings.sound_speed**2
        / (2.0 * jnp.pi * settings.gravitational_constant)
    )
    edge_density = density_scale / (settings.cloud_radius**2 + core_radius**2)
    density = jnp.where(
        radius_squared <= settings.cloud_radius**2,
        density_scale / (radius_squared + core_radius**2),
        settings.ambient_density_ratio * edge_density,
    )
    velocity = jnp.zeros(density.shape + (3,), dtype=density.dtype)

    return _isothermal_state(config, params, density, velocity)
