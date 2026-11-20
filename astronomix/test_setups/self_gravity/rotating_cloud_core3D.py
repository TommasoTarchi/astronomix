"""
# Rotating cloud core (3D)

A uniform, isothermal sphere in solid-body rotation about the z axis, with an
azimuthal density perturbation of mode m = 2,

    ρ = ρ_0 (1 + a cos 2φ)   for r ≤ R,   v = Ω ẑ × (x − x_c),

in a box of thin gas at rest, evolved with self-gravity. The ratios of the
thermal and the rotational energy to the gravitational energy of the
unperturbed sphere are

    α = 15 c_s² / (8πG ρ_0 R²),   β = Ω² / (4πG ρ_0),

and the defaults, α ≈ 0.26, β ≈ 0.16 and a = 0.1, are those of the standard
isothermal test of Boss & Bodenheimer (1979), in which the core collapses into
a rotating bar that fragments into a binary. The boundaries are periodic by
default.

The rotation rate Ω is the input with respect to which the sink particle
gradients are validated.

## References

- Boss & Bodenheimer (1979), ApJ 234, 289.
- Bate & Burkert (1997), MNRAS 288, 1060.
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


class RotatingCloudCoreSettings(NamedTuple):
    """Problem constants of the rotating cloud core."""

    #: Side length of the cubic box.
    box_length: float = 1.0

    #: Radius R of the core.
    cloud_radius: float = 0.3

    #: Mean density ρ_0 of the core; with the other defaults α ≈ 0.26.
    cloud_density: float = 25.5

    #: Angular velocity Ω of the solid-body rotation about z; with the other
    #: defaults β ≈ 0.16. It may be a traced value.
    rotation_rate: float = 7.16

    #: Amplitude a of the m = 2 density perturbation.
    perturbation_amplitude: float = 0.1

    #: Density of the gas outside the core, in units of ρ_0.
    ambient_density_ratio: float = 0.01

    #: Isothermal sound speed c_s.
    sound_speed: float = 1.0

    #: Gravitational constant G.
    gravitational_constant: float = 1.0

    #: End time of the run; the free-fall time of the core is
    #: √(3π / (32 G ρ_0)) ≈ 0.107.
    t_end: float = 0.2

    #: Boundary conditions of the box. Periodic boundaries keep the mass of gas
    #: and sinks constant; with open boundaries (``OPEN_BOX``) the cloud also
    #: draws gas in through the edges of the box.
    boundary_settings: BoundarySettings = PERIODIC_BOX

    #: Centre (x, y, z) of the core; ``None`` places it on the cell at the
    #: middle of the box.
    center: Union[tuple, NoneType] = None


def setup_rotating_cloud_core(
    config: SimulationConfig,
    params: SimulationParams,
    settings: RotatingCloudCoreSettings = RotatingCloudCoreSettings(),
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Set up a rotating, perturbed cloud core in an isothermal,
    self-gravitating box.

    The perturbation is cos 2φ = (Δx² − Δy²) / (Δx² + Δy²), set to 0 on the
    rotation axis, where φ is undefined.

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
    separation = cell_centers - center
    inside = jnp.sum(separation**2, axis=-1) <= settings.cloud_radius**2

    cylindrical_radius_squared = separation[..., 0] ** 2 + separation[..., 1] ** 2
    on_axis = cylindrical_radius_squared == 0.0
    cos_two_phi = jnp.where(
        on_axis,
        0.0,
        (separation[..., 0] ** 2 - separation[..., 1] ** 2)
        / jnp.where(on_axis, 1.0, cylindrical_radius_squared),
    )
    density = jnp.where(
        inside,
        settings.cloud_density * (1.0 + settings.perturbation_amplitude * cos_two_phi),
        settings.ambient_density_ratio * settings.cloud_density,
    )

    rotation_velocity = settings.rotation_rate * jnp.stack(
        [-separation[..., 1], separation[..., 0], jnp.zeros_like(separation[..., 2])],
        axis=-1,
    )
    velocity = jnp.where(inside[..., None], rotation_velocity, 0.0)

    return _isothermal_state(config, params, density, velocity)
