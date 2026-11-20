"""
# Bonnor–Ebert sphere (3D)

An isothermal sphere in hydrostatic equilibrium, truncated at a radius R, in a
box of thin gas, evolved with self-gravity. Its density follows the isothermal
Lane–Emden equation,

    ρ(r) = ρ_c exp(−ψ(ξ)),   ξ = r / r_0,   r_0 = c_s / √(4πG ρ_c),
    ψ'' + (2/ξ) ψ' = e^(−ψ),   ψ(0) = ψ'(0) = 0,

and the density contrast ρ_c / ρ(R) = exp(ψ(R / r_0)) sets r_0 for the given
radius. Spheres with a contrast above 14.1 (ξ(R) > 6.45) are unstable. The
density inside R is multiplied by an enhancement factor f > 1, which starts the
collapse within a few free-fall times of the centre. The gas starts at rest,
the sphere is centred on a cell centre, so that one cell is the unique
potential minimum, and the boundaries are periodic by default.

The density contrast is the input with respect to which the sink particle
gradients are validated; the profile is differentiable in it.

## References

- Bonnor (1956), MNRAS 116, 351; Ebert (1955), Z. Astrophys. 37, 217.
- Federrath, Banerjee, Clark & Klessen (2010), ApJ 713, 269.
"""

# general
from functools import lru_cache

# typing
from types import NoneType
from typing import NamedTuple, Union

# numerics
import numpy as np

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

#: The largest dimensionless radius of the tabulated Lane–Emden solution.
LARGEST_TABULATED_RADIUS = 30.0

#: The step of the tabulated Lane–Emden solution in ξ.
TABULATION_STEP = 1e-3


class BonnorEbertSettings(NamedTuple):
    """Problem constants of the Bonnor–Ebert sphere."""

    #: Side length of the cubic box.
    box_length: float = 1.0

    #: Radius R of the sphere.
    cloud_radius: float = 0.375

    #: Density contrast ρ_c / ρ(R) between the centre and the edge of the
    #: sphere; above 14.1 the sphere collapses. It may be a traced value.
    density_contrast: float = 20.0

    #: Factor f by which the density inside R exceeds the equilibrium.
    density_enhancement: float = 2.0

    #: Density of the gas outside the sphere, in units of the edge density of
    #: the equilibrium.
    ambient_density_ratio: float = 0.1

    #: Isothermal sound speed c_s.
    sound_speed: float = 1.0

    #: Gravitational constant G.
    gravitational_constant: float = 1.0

    #: End time of the run.
    t_end: float = 0.3

    #: Boundary conditions of the box. Periodic boundaries keep the mass of gas
    #: and sinks constant; with open boundaries (``OPEN_BOX``) the cloud also
    #: draws gas in through the edges of the box.
    boundary_settings: BoundarySettings = PERIODIC_BOX

    #: Centre (x, y, z) of the sphere; ``None`` places it on the cell at the
    #: middle of the box.
    center: Union[tuple, NoneType] = None


@lru_cache(maxsize=1)
def _lane_emden_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The isothermal Lane–Emden solution ψ(ξ) and its derivative ψ'(ξ),
    integrated with the classical Runge–Kutta method on a uniform grid of ξ.

    The first step starts from the series ψ = ξ²/6 − ξ⁴/120, which avoids the
    singular term 2ψ'/ξ at ξ = 0.

    Returns:
        ``(xi, psi, psi_derivative)``, each of shape (num_points,).
    """
    xi = np.arange(0.0, LARGEST_TABULATED_RADIUS + TABULATION_STEP, TABULATION_STEP)
    psi = np.zeros_like(xi)
    psi_derivative = np.zeros_like(xi)
    psi[1] = xi[1] ** 2 / 6.0 - xi[1] ** 4 / 120.0
    psi_derivative[1] = xi[1] / 3.0 - xi[1] ** 3 / 30.0

    def rates(radius, state):
        value, derivative = state
        return np.array([derivative, np.exp(-value) - 2.0 * derivative / radius])

    for index in range(1, xi.size - 1):
        state = np.array([psi[index], psi_derivative[index]])
        radius = xi[index]
        k1 = rates(radius, state)
        k2 = rates(radius + 0.5 * TABULATION_STEP, state + 0.5 * TABULATION_STEP * k1)
        k3 = rates(radius + 0.5 * TABULATION_STEP, state + 0.5 * TABULATION_STEP * k2)
        k4 = rates(radius + TABULATION_STEP, state + TABULATION_STEP * k3)
        state = state + TABULATION_STEP / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        psi[index + 1], psi_derivative[index + 1] = state
    return xi, psi, psi_derivative


def bonnor_ebert_radius(density_contrast) -> jnp.ndarray:
    """
    The dimensionless radius ξ(R) of a Bonnor–Ebert sphere with the given
    density contrast, from ψ(ξ(R)) = ln(ρ_c / ρ(R)), interpolated in the
    tabulated solution (ψ grows monotonically with ξ).

    Args:
        density_contrast: The contrast ρ_c / ρ(R), at most
            exp(ψ(LARGEST_TABULATED_RADIUS)).

    Returns:
        ξ(R).
    """
    xi, psi, _ = _lane_emden_table()
    return jnp.interp(jnp.log(density_contrast), psi, xi)


def bonnor_ebert_mass(
    density_contrast,
    cloud_radius,
    sound_speed,
    gravitational_constant,
) -> jnp.ndarray:
    """
    The mass of a Bonnor–Ebert sphere, M = 4π ρ_c r_0³ ξ² ψ'(ξ) at ξ = ξ(R).

    Args:
        density_contrast: The contrast ρ_c / ρ(R).
        cloud_radius: The radius R.
        sound_speed: The isothermal sound speed c_s.
        gravitational_constant: The gravitational constant G.

    Returns:
        The mass of the sphere.
    """
    xi, _, psi_derivative = _lane_emden_table()
    edge_radius = bonnor_ebert_radius(density_contrast)
    scale_radius = cloud_radius / edge_radius
    central_density = sound_speed**2 / (
        4.0 * jnp.pi * gravitational_constant * scale_radius**2
    )
    return (
        4.0
        * jnp.pi
        * central_density
        * scale_radius**3
        * edge_radius**2
        * jnp.interp(edge_radius, xi, psi_derivative)
    )


def setup_bonnor_ebert_sphere(
    config: SimulationConfig,
    params: SimulationParams,
    settings: BonnorEbertSettings = BonnorEbertSettings(),
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Set up a Bonnor–Ebert sphere in an isothermal, self-gravitating box.

    The radius R is fixed, so a larger density contrast gives a smaller scale
    radius r_0 = R / ξ(R) and a denser centre.

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
    cell_centers, _ = _cell_centers_and_spacing(config)
    if settings.center is None:
        center = _central_cell_center(config)
    else:
        center = jnp.array(settings.center)
    radius = jnp.linalg.norm(cell_centers - center, axis=-1)

    xi, psi, _ = _lane_emden_table()
    edge_radius = bonnor_ebert_radius(settings.density_contrast)
    scale_radius = settings.cloud_radius / edge_radius
    central_density = settings.sound_speed**2 / (
        4.0 * jnp.pi * settings.gravitational_constant * scale_radius**2
    )
    edge_density = central_density / settings.density_contrast

    density = jnp.where(
        radius <= settings.cloud_radius,
        settings.density_enhancement
        * central_density
        * jnp.exp(-jnp.interp(radius / scale_radius, xi, psi)),
        settings.ambient_density_ratio * edge_density,
    )
    velocity = jnp.zeros(density.shape + (3,), dtype=density.dtype)

    return _isothermal_state(config, params, density, velocity)
