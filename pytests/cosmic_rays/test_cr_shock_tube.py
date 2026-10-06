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

Measured 2026-09-25 (after the shock-finder / injection fixes):
    no injection: region-2 rho / P_th / P_cr / u within 0.05 / 0.08 / 0.05 / 0.04 %
    injection:    within 0.13 / 0.32 / 0.66 / 0.00 %  (before the fixes the
                  injected P_cr was 1.8 % low and P_th 1.0 % high)
"""

# ==== device ====
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 1D, 801 cells; no GPU needed
# ruff: noqa: E402
# =================

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    BackendConfig,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    CosmicRayConfig,
    CosmicRayParams,
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

#: relative tolerances on the region-2 (post-shock) and region-3 (post-
#: rarefaction) plateaus, ~2x the errors measured at 801 cells:
#:   no inj.: rho2 -0.05, P_th2 +0.08, P_cr2 -0.05, v -0.04, rho3 +0.06, P_th3 +0.13, P_cr3 +0.08 %
#:   inj.:    rho2 +0.13, P_th2 -0.32, P_cr2 +0.66, v -0.00, rho3 +0.00, P_th3 +0.04, P_cr3 +0.01 %
#: The pre-fix code (rho2 -0.38, P_th2 +1.01, P_cr2 -1.79 % with injection)
#: fails the injection row.
TOLERANCES = {
    False: dict(rho2=2e-3, P_th2=2e-3, P_cr2=2e-3, v3=2e-3, rho3=2e-3, P_th3=3e-3, P_cr3=2e-3),
    True: dict(rho2=3e-3, P_th2=6e-3, P_cr2=1.2e-2, v3=2e-3, rho3=2e-3, P_th3=2e-3, P_cr3=2e-3),
}


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Run this module in float64 and restore the previous setting afterwards."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def run_cr_shock_tube(dsa, num_cells=NUM_CELLS, zeta=ZETA):
    """Run the tube; return ``x, rho, u, P_th, P_cr`` at t = 0.35."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        first_order_fallback=False,
        num_cells=num_cells,
        box_size=10.0,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True, diffusive_shock_acceleration=dsa,
        ),
        riemann_solver=HLL,
        progress_bar=False,
        # CRs exist only in the finite-volume solver, and only natively
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
    rv = get_registered_variables(config)
    x = helper_data.geometric_centers
    left = x < 5.0
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.where(left, 1.0, 0.125),
        velocity_x=jnp.zeros_like(x),
        gas_pressure=jnp.where(left, 17.172, 0.05),
        cosmic_ray_pressure=jnp.where(left, 34.344, 0.05),
    )
    config = finalize_config(config, state.shape)
    out = np.asarray(time_integration(state, config, params, rv))
    p_cr = np.asarray(cosmic_ray_pressure_from_n(out[rv.cosmic_ray_n_index]))
    return (np.asarray(x), out[rv.density_index], out[rv.velocity_index],
            out[rv.pressure_index] - p_cr, p_cr)


def _plateau(x, q, lo, hi, trim=0.2):
    """Median of ``q`` over the central (1 - 2 trim) of ``lo < x < hi``."""
    width = hi - lo
    mask = (x > lo + trim * width) & (x < hi - trim * width)
    return float(np.median(q[mask]))


@pytest.mark.parametrize("dsa", [False, True], ids=["CR", "CR+inj"])
def test_cr_shock_tube(dsa):
    zeta = ZETA if dsa else 0.0
    x, rho, u, p_th, p_cr = run_cr_shock_tube(dsa)
    assert np.all(np.isfinite(rho)) and np.all(np.isfinite(p_th))
    assert np.min(p_th) > 0.0

    reg = cosmic_ray_shock_tube_regions(injection_efficiency=zeta)
    x_tail = 5.0 - reg["vt"] * T_END
    x_cd = 5.0 + reg["v3"] * T_END
    x_sh = 5.0 + reg["vs"] * T_END
    assert reg["xs"] == pytest.approx(4.78 if dsa else 3.90, abs=5e-3)

    measured = dict(
        rho2=_plateau(x, rho, x_cd, x_sh), P_th2=_plateau(x, p_th, x_cd, x_sh),
        P_cr2=_plateau(x, p_cr, x_cd, x_sh), v3=_plateau(x, u, x_tail, x_sh),
        rho3=_plateau(x, rho, x_tail, x_cd), P_th3=_plateau(x, p_th, x_tail, x_cd),
        P_cr3=_plateau(x, p_cr, x_tail, x_cd),
    )
    for key, tol in TOLERANCES[dsa].items():
        assert measured[key] == pytest.approx(reg[key], rel=tol), (key, measured[key], reg[key])

    # shock position: where the density crosses the mean of rho2 and rho1
    half = 0.5 * (reg["rho2"] + 0.125)
    right = x > x_cd + 0.05
    i = np.flatnonzero(right & (rho < half))[0]
    x_shock = x[i - 1] + (half - rho[i - 1]) * (x[i] - x[i - 1]) / (rho[i] - rho[i - 1])
    dx = 10.0 / NUM_CELLS
    assert abs(x_shock - x_sh) < 3 * dx, (x_shock, x_sh)

    # overview figure (not part of the assertion)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    xa, rhoa, ua, ptha, pcra, _ = get_cosmic_ray_analytic_solution(injection_efficiency=zeta)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].plot(x, rho, lw=1.2, label="astronomix FV")
    axes[0].plot(xa, rhoa, "k--", lw=0.9, label="analytic")
    axes[1].plot(x, u, lw=1.2)
    axes[1].plot(xa, ua, "k--", lw=0.9)
    axes[2].plot(x, p_th, lw=1.2, label="P_th")
    axes[2].plot(x, p_cr, lw=1.2, label="P_cr")
    axes[2].plot(xa, ptha, "k--", lw=0.9)
    axes[2].plot(xa, pcra, "k:", lw=0.9)
    for ax, lab in zip(axes, ("density", "velocity", "pressure")):
        ax.set(xlabel="x", ylabel=lab, xlim=(2, 10))
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Pfrommer+17 CR shock tube, zeta = {zeta}")
    fig.savefig(FIGURE_DIR / f"cr_shock_tube_{'inj' if dsa else 'noinj'}.png", dpi=110)
    plt.close(fig)
