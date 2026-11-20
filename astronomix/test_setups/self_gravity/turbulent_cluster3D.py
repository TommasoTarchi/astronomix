"""
# Turbulent cluster (3D)

A periodic box of uniform, isothermal gas with a random, Gaussian velocity
field, evolved with self-gravity and without driving. The box holds several
Jeans masses; the turbulence compresses the gas in shocks, and the dense
regions that are bound collapse into a small cluster of sinks.

Each velocity component is an independent Gaussian random field with the
Fourier amplitude |v_k| ∝ k^s between the smallest and largest wavenumber
(s = −2 gives the power spectrum k⁻⁴ of shock-dominated, Burgers turbulence),
scaled to the rms Mach number M. The field depends only on the seed, so M
scales the velocities without changing their pattern.

The Mach number is the input with respect to which the sink particle
gradients are validated.

## References

- Federrath, Banerjee, Clark & Klessen (2010), ApJ 713, 269.
"""

# typing
from typing import NamedTuple

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.initial_condition_generation.turbulent_ic_generator import create_turb_field
from astronomix.test_setups.self_gravity._isothermal_collapse import (
    PERIODIC_BOX,
    _cell_centers_and_spacing,
    _isothermal_box,
    _isothermal_state,
)


class TurbulentClusterSettings(NamedTuple):
    """Problem constants of the turbulent cluster."""

    #: Side length of the cubic box.
    box_length: float = 1.0

    #: Mean density; 16π c_s² / (G L²) makes the Jeans length
    #: √(π c_s² / (G ρ)) a quarter of the box, and the free-fall time
    #: √(3π / (32 G ρ)) ≈ 0.077.
    mean_density: float = 16.0 * jnp.pi

    #: Root-mean-square Mach number of the initial velocity field. It may be
    #: a traced value.
    mach_number: float = 3.0

    #: Exponent s of the Fourier amplitude |v_k| ∝ k^s.
    spectral_slope: float = -2.0

    #: Smallest and largest wavenumber of the velocity field, in units of
    #: 2π / L.
    smallest_wavenumber: float = 1.0
    largest_wavenumber: float = 8.0

    #: Seed of the random velocity field.
    seed: int = 0

    #: Isothermal sound speed c_s.
    sound_speed: float = 1.0

    #: Gravitational constant G.
    gravitational_constant: float = 1.0

    #: End time of the run.
    t_end: float = 0.1


def turbulent_velocity_pattern(num_cells: int, settings: TurbulentClusterSettings) -> jnp.ndarray:
    """
    The pattern of the initial velocity field: three independent Gaussian
    random fields, scaled to an rms speed of 1.

    Args:
        num_cells: The number of cells along each axis.
        settings: Problem constants.

    Returns:
        The velocity pattern, shape (num_cells, num_cells, num_cells, 3).
    """
    keys = jax.random.split(jax.random.key(settings.seed), 3)
    components = [
        create_turb_field(
            num_cells,
            1.0,
            settings.spectral_slope,
            settings.smallest_wavenumber,
            settings.largest_wavenumber,
            key,
        )
        for key in keys
    ]
    pattern = jnp.stack(components, axis=-1)
    root_mean_square = jnp.sqrt(jnp.mean(jnp.sum(pattern**2, axis=-1)))
    return pattern / root_mean_square


def setup_turbulent_cluster(
    config: SimulationConfig,
    params: SimulationParams,
    settings: TurbulentClusterSettings = TurbulentClusterSettings(),
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Set up a periodic box of uniform, isothermal, self-gravitating gas with a
    random velocity field of rms Mach number M.

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
        PERIODIC_BOX,
        settings.sound_speed,
        settings.gravitational_constant,
        settings.t_end,
    )
    cell_centers, _ = _cell_centers_and_spacing(config)
    num_cells = cell_centers.shape[0]

    density = jnp.full(cell_centers.shape[:-1], settings.mean_density)
    velocity = (
        settings.mach_number
        * settings.sound_speed
        * turbulent_velocity_pattern(num_cells, settings)
    )

    return _isothermal_state(config, params, density, velocity)
