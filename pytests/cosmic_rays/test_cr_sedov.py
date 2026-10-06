"""
1D spherical Sedov-Taylor blast with cosmic-ray shock acceleration (CPU, ~1 min).

E = 1 into rho = 1 (P_0 = 1e-4) at t = 0.1, 2001 radial cells, finite-volume
HLL, float64 -- the setup of Pfrommer et al. (2017, Sec. 4.2) in 1D. The
reference is the EXACT self-similar two-fluid solution for a constant
injection efficiency (``_cr_sedov_reference.py``): zeta = 0.5 gives
R = 0.42113 (between the gamma = 7/5 Sedov radius 0.41116 that Pfrommer et
al. compare with and the gamma = 5/3 radius 0.45849), a post-shock
P_cr/P_th = zeta/[2(1 - zeta)] = 0.5 and a CR share of the blast energy of
0.529.

Measured 2026-09-25 (zeta = 0.5, DSA from t = 0): r_sh -0.39 / -0.21 /
-0.02 % at 1001 / 2001 / 4001 cells, E_cr/E 0.562 / 0.539 / 0.525; zeta = 0.1
at 2001: +0.27 % (the zeta = 0 run's own bias is +0.15 %), E_cr/E 0.136 vs
0.143. Before the fixes this run was NaN (a first "shock" whose injection
weights summed to ~0); with the start delayed to t = 1e-3 it ran but gave
r = 0.4295 (+2 %).
"""

# ==== device ====
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 1D; no GPU needed
# ruff: noqa: E402
# =================

import sys
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
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    SPHERICAL,
    BackendConfig,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    CosmicRayConfig,
    CosmicRayParams,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cr_sedov_reference import sedov_radius, sedov_two_fluid  # noqa: E402

NUM_CELLS = 2001
T_END = 0.1
NUM_INJECTION_CELLS = 5


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Run this module in float64 and restore the previous setting afterwards."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def run_cr_sedov(zeta, escape_fraction=0.0, num_cells=NUM_CELLS, start_time=0.0):
    """Run the blast; return a dict of the shock radius and energy budget."""
    config = SimulationConfig(
        geometry=SPHERICAL,
        first_order_fallback=True,
        progress_bar=False,
        num_cells=num_cells,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True, diffusive_shock_acceleration=zeta > 0,
        ),
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        riemann_solver=HLL,
    )
    params = SimulationParams(
        t_end=T_END,
        dt_max=1e-5,
        cosmic_ray_params=CosmicRayParams(
            diffusive_shock_acceleration_start_time=start_time,
            diffusive_shock_acceleration_efficiency=zeta,
            escape_fraction=escape_fraction,
        ),
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    r = helper_data.geometric_centers
    r_inj = helper_data.outer_cell_boundaries[NUM_INJECTION_CELLS]
    p_inj = (params.gamma - 1) / (4 / 3 * np.pi * r_inj**3)
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.ones_like(r), velocity_x=jnp.zeros_like(r),
        gas_pressure=(1e-4 * jnp.ones_like(r)).at[: NUM_INJECTION_CELLS + 1].set(p_inj),
        cosmic_ray_pressure=jnp.zeros_like(r),
    )
    config = finalize_config(config, state.shape)
    vol = np.asarray(helper_data.cell_volumes)
    e0 = float(np.sum(np.asarray(state[rv.pressure_index]) / (params.gamma - 1) * vol))

    out = np.asarray(time_integration(state, config, params, rv))
    rr = np.asarray(r)
    rho, v = out[rv.density_index], out[rv.velocity_index]
    p_cr = np.asarray(cosmic_ray_pressure_from_n(out[rv.cosmic_ray_n_index]))
    p_th = out[rv.pressure_index] - p_cr

    # shock radius: outer crossing of (rho_max + rho_0)/2, linearly interpolated
    i_peak = int(np.argmax(rho))
    half = 0.5 * (rho[i_peak] + 1.0)
    j = i_peak + int(np.argmax(rho[i_peak:] < half))
    r_sh = rr[j - 1] + (half - rho[j - 1]) * (rr[j] - rr[j - 1]) / (rho[j] - rho[j - 1])

    e_cr = float(np.sum(3.0 * p_cr * vol))
    e_th = float(np.sum(1.5 * p_th * vol))
    e_kin = float(np.sum(0.5 * rho * v**2 * vol))
    behind = (rr > 0.96 * r_sh) & (rr < 0.99 * r_sh)
    return dict(
        finite=bool(np.all(np.isfinite(out))), r_sh=r_sh, e0=e0,
        e_tot=e_cr + e_th + e_kin, e_cr=e_cr,
        x_behind=float(np.mean(p_cr[behind] / p_th[behind])),
        min_p_th=float(np.min(p_th)),
    )


def test_sedov_without_injection_matches_gamma_five_thirds():
    res = run_cr_sedov(zeta=0.0)
    assert res["finite"]
    exact = sedov_radius(sedov_two_fluid(0.0)["alpha"], time=T_END)
    assert exact == pytest.approx(0.45849, abs=1e-5)
    assert res["r_sh"] == pytest.approx(exact, rel=5e-3)
    assert res["e_cr"] == 0.0
    assert res["e_tot"] == pytest.approx(res["e0"], rel=1e-10)


def test_sedov_with_injection_matches_two_fluid_self_similar_solution():
    """zeta = 0.5 from t = 0 (this NaN'd before the injection guards)."""
    zeta = 0.5
    ref = sedov_two_fluid(zeta)
    exact = sedov_radius(ref["alpha"], time=T_END)
    assert exact == pytest.approx(0.42113, abs=1e-5)
    assert ref["X0"] == pytest.approx(0.5)

    res = run_cr_sedov(zeta=zeta, start_time=0.0)
    assert res["finite"]
    assert res["min_p_th"] > 0.0
    assert res["r_sh"] == pytest.approx(exact, rel=1e-2)
    # clearly softer than the pure gas blast
    assert res["r_sh"] < 0.97 * sedov_radius(sedov_two_fluid(0.0)["alpha"], time=T_END)
    assert res["e_cr"] / res["e_tot"] == pytest.approx(ref["E_cr_fraction"], abs=0.04)
    # P_cr/P_th just behind the shock ~ zeta / [2 (1 - zeta)] (it rises inwards)
    assert res["x_behind"] == pytest.approx(ref["X0"], rel=0.15)
    # injection moves energy between the fluids; the total is conserved
    assert res["e_tot"] == pytest.approx(res["e0"], rel=1e-10)


def test_sedov_escape_removes_energy():
    """escape_fraction removes a fraction of the injected CR energy: the blast
    loses energy, holds less CR energy and is smaller than without escape."""
    no_esc = run_cr_sedov(zeta=0.5, num_cells=1001)
    esc = run_cr_sedov(zeta=0.5, escape_fraction=0.5, num_cells=1001)
    assert esc["finite"]
    assert esc["e_tot"] < 0.9 * esc["e0"]
    assert esc["e_cr"] < no_esc["e_cr"]
    assert esc["r_sh"] < no_esc["r_sh"]
