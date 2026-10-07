"""
Two-fluid (gas + CR) shock tube of Pfrommer et al. (2017), runs "CR" and
"CR+inj" of their Table 1, against the analytic solution (CPU, ~10 s).

rho 1 | 0.125, P_th 17.172 | 0.05, P_cr 34.344 | 0.05, box [0, 10] with the
membrane at 5, t = 0.35, 801 cells, finite-volume HLL (the only CR-capable
solver), float64. With diffusive shock acceleration (zeta = 0.5) the shock
compression rises from 3.90 to 4.78 and ~90 % of the post-shock CR pressure
is freshly injected, so the injected-CR plateau is a direct test of the Mach
estimate, the dissipated-energy flux and the energy split of
``cr_injection``.

Measured region-2 errors in rho / P_th / P_cr / u: within 0.05 / 0.08 / 0.05 /
0.04 % without injection and within 0.13 / 0.32 / 0.66 / 0.00 % with it.
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# The plateaus are compared with the analytic solution at 0.2 % relative
# tolerance, which needs double precision.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# testing
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import BackendConfig
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    CosmicRayConfig,
    CosmicRayParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)
from astronomix.test_setups.reference_solutions.cr_shock_tube import (
    cosmic_ray_shock_tube_regions,
    get_cosmic_ray_analytic_solution,
)

FIGURE_DIR = Path(__file__).resolve().parent / "figures"

NUM_CELLS = 801
T_END = 0.35
ZETA = 0.5
BOX_SIZE = 10.0
MEMBRANE_POSITION = 5.0
DENSITY_RIGHT = 0.125

#: Relative tolerances on the region-2 (post-shock) and region-3
#: (post-rarefaction) plateaus, about twice the errors measured at 801 cells.
#: Without injection these are rho2 -0.05, P_th2 +0.08, P_cr2 -0.05, v -0.04,
#: rho3 +0.06, P_th3 +0.13 and P_cr3 +0.08 %; with injection rho2 +0.13,
#: P_th2 -0.32, P_cr2 +0.66, v -0.00, rho3 +0.00, P_th3 +0.04 and P_cr3 +0.01 %.
TOLERANCES = {
    False: dict(
        rho2=2e-3,
        P_th2=2e-3,
        P_cr2=2e-3,
        v3=2e-3,
        rho3=2e-3,
        P_th3=3e-3,
        P_cr3=2e-3,
    ),
    True: dict(
        rho2=3e-3,
        P_th2=6e-3,
        P_cr2=1.2e-2,
        v3=2e-3,
        rho3=2e-3,
        P_th3=2e-3,
        P_cr3=2e-3,
    ),
}


def run_cr_shock_tube(dsa, num_cells=NUM_CELLS, zeta=ZETA):
    """
    Run the two-fluid shock tube to t = 0.35.

    Args:
        dsa: Whether diffusive shock acceleration is switched on.
        num_cells: The number of cells.
        zeta: The injection efficiency (only used with ``dsa``).

    Returns:
        The cell centres, density, velocity, thermal pressure and cosmic-ray
        pressure of the final state, as a tuple
        ``(x, density, velocity, thermal_pressure, cosmic_ray_pressure)``.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        first_order_fallback=False,
        num_cells=num_cells,
        box_size=BOX_SIZE,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=dsa,
        ),
        riemann_solver=HLL,
        progress_bar=False,
        # Cosmic rays exist only in the finite-volume solver, and only natively.
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
    )
    params = SimulationParams(
        t_end=T_END,
        cosmic_ray_params=CosmicRayParams(
            diffusive_shock_acceleration_efficiency=zeta,
            diffusive_shock_acceleration_start_time=0.0,
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    x = helper_data.geometric_centers
    left = x < MEMBRANE_POSITION
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.where(left, 1.0, DENSITY_RIGHT),
        velocity_x=jnp.zeros_like(x),
        gas_pressure=jnp.where(left, 17.172, 0.05),
        cosmic_ray_pressure=jnp.where(left, 34.344, 0.05),
    )
    config = finalize_config(config, state.shape)
    final_state = np.asarray(time_integration(state, config, params, registered_variables))
    cosmic_ray_pressure = np.asarray(
        cosmic_ray_pressure_from_n(final_state[registered_variables.cosmic_ray_n_index])
    )
    thermal_pressure = final_state[registered_variables.pressure_index] - cosmic_ray_pressure
    return (
        np.asarray(x),
        final_state[registered_variables.density_index],
        final_state[registered_variables.velocity_index],
        thermal_pressure,
        cosmic_ray_pressure,
    )


def _plateau(x, quantity, lower, upper, trim=0.2):
    """Median of ``quantity`` over the central (1 - 2 trim) of ``lower < x < upper``."""
    width = upper - lower
    mask = (x > lower + trim * width) & (x < upper - trim * width)
    return float(np.median(quantity[mask]))


def _plot_overview(
    x,
    density,
    velocity,
    thermal_pressure,
    cosmic_ray_pressure,
    zeta,
    figure_path,
):
    """
    Plot the simulated profiles against the analytic solution.

    Args:
        x: The cell centres.
        density: The simulated density.
        velocity: The simulated velocity.
        thermal_pressure: The simulated thermal pressure.
        cosmic_ray_pressure: The simulated cosmic-ray pressure.
        zeta: The injection efficiency of the analytic solution.
        figure_path: Where the figure is written.
    """
    (
        analytic_x,
        analytic_density,
        analytic_velocity,
        analytic_thermal_pressure,
        analytic_cosmic_ray_pressure,
        _,
    ) = get_cosmic_ray_analytic_solution(injection_efficiency=zeta)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].plot(x, density, lw=1.2, label="astronomix FV")
    axes[0].plot(analytic_x, analytic_density, "k--", lw=0.9, label="analytic")
    axes[1].plot(x, velocity, lw=1.2)
    axes[1].plot(analytic_x, analytic_velocity, "k--", lw=0.9)
    axes[2].plot(x, thermal_pressure, lw=1.2, label="P_th")
    axes[2].plot(x, cosmic_ray_pressure, lw=1.2, label="P_cr")
    axes[2].plot(analytic_x, analytic_thermal_pressure, "k--", lw=0.9)
    axes[2].plot(analytic_x, analytic_cosmic_ray_pressure, "k:", lw=0.9)
    for ax, label in zip(axes, ("density", "velocity", "pressure")):
        ax.set(xlabel="x", ylabel=label, xlim=(2, 10))
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Pfrommer+17 CR shock tube, zeta = {zeta}")
    fig.savefig(figure_path, dpi=110)
    plt.close(fig)


@pytest.mark.parametrize("dsa", [False, True], ids=["CR", "CR+inj"])
def test_cr_shock_tube(dsa):
    """
    The post-shock and post-rarefaction plateaus and the shock position match
    the analytic two-fluid solution, with and without injection, and the
    thermal pressure stays positive. Also writes an overview figure.
    """
    zeta = ZETA if dsa else 0.0
    x, density, velocity, thermal_pressure, cosmic_ray_pressure = run_cr_shock_tube(dsa)
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(thermal_pressure))
    assert np.min(thermal_pressure) > 0.0

    regions = cosmic_ray_shock_tube_regions(injection_efficiency=zeta)
    rarefaction_tail_position = MEMBRANE_POSITION - regions["vt"] * T_END
    contact_position = MEMBRANE_POSITION + regions["v3"] * T_END
    shock_position = MEMBRANE_POSITION + regions["vs"] * T_END
    assert regions["xs"] == pytest.approx(4.78 if dsa else 3.90, abs=5e-3)

    measured = dict(
        rho2=_plateau(x, density, contact_position, shock_position),
        P_th2=_plateau(x, thermal_pressure, contact_position, shock_position),
        P_cr2=_plateau(x, cosmic_ray_pressure, contact_position, shock_position),
        v3=_plateau(x, velocity, rarefaction_tail_position, shock_position),
        rho3=_plateau(x, density, rarefaction_tail_position, contact_position),
        P_th3=_plateau(x, thermal_pressure, rarefaction_tail_position, contact_position),
        P_cr3=_plateau(x, cosmic_ray_pressure, rarefaction_tail_position, contact_position),
    )
    for key, tolerance in TOLERANCES[dsa].items():
        assert measured[key] == pytest.approx(regions[key], rel=tolerance), (
            key,
            measured[key],
            regions[key],
        )

    # The simulated shock sits where the density crosses the mean of rho2 and
    # rho1, linearly interpolated between the bracketing cells.
    half_density = 0.5 * (regions["rho2"] + DENSITY_RIGHT)
    right_of_contact = x > contact_position + 0.05
    crossing_index = np.flatnonzero(right_of_contact & (density < half_density))[0]
    inner_x = x[crossing_index - 1]
    outer_x = x[crossing_index]
    inner_density = density[crossing_index - 1]
    outer_density = density[crossing_index]
    simulated_shock_position = inner_x + (half_density - inner_density) * (
        outer_x - inner_x
    ) / (outer_density - inner_density)
    grid_spacing = BOX_SIZE / NUM_CELLS
    assert abs(simulated_shock_position - shock_position) < 3 * grid_spacing, (
        simulated_shock_position,
        shock_position,
    )

    FIGURE_DIR.mkdir(exist_ok=True)
    _plot_overview(
        x,
        density,
        velocity,
        thermal_pressure,
        cosmic_ray_pressure,
        zeta,
        FIGURE_DIR / f"cr_shock_tube_{'inj' if dsa else 'noinj'}.png",
    )
