"""
Unit tests of the CR shock finder and the DSA injection step (CPU, seconds).

Guards the fixes of 2026-09-25 (audit ``cr_code.md`` section 1):

* the Mach estimate is exact for Rankine-Hugoniot jumps, with and without CRs
  (it used to return M = 2.17 for a true M = 1.3 and had a pole at
  P2/P1 = 1.5, so the M > 1.3 gate passed weak compressions);
* the WENO smoothness sensor is non-negative (its curvature term was
  unsquared);
* one injection step conserves energy exactly (gas loss = CR gain), removes
  exactly ``zeta * e_diss * v_2 * dt`` from the gas for a resolved shock, keeps
  ``(1 - escape_fraction)`` of it in the CRs, respects the thermal-energy cap,
  does nothing (instead of NaN) for an invalid zone, and is AD-safe in
  CR-free cells (``P_cr ** (3/4)`` had an infinite derivative at 0);
* ``OUTERMOST_SHOCK`` selects the forward shock when a stronger inner shock
  exists.
"""

# ==== device ====
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 1D unit tests; no GPU needed
# ruff: noqa: E402
# =================

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
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    SPHERICAL,
    BackendConfig,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    OUTERMOST_SHOCK,
    STRONGEST_SHOCK,
    CosmicRayConfig,
    CosmicRayParams,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_n_from_pressure,
    cosmic_ray_pressure_from_n,
)
from astronomix._modules._cosmic_rays.cr_injection import inject_crs_at_strongest_shock
from astronomix.shock_finder.shock_finder import (
    find_shock_zone,
    mach_number_squared,
    shock_criteria,
    shock_sensor,
)
from astronomix.test_setups.reference_solutions.cr_shock_tube import (
    cosmic_ray_shock_tube_regions,
)

GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
NUM_CELLS = 128


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Run this module in float64 and restore the previous setting afterwards."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


# -------------------------------------------------------------
# ===================== ↓ helpers ↓ ===========================
# -------------------------------------------------------------


def _rankine_hugoniot(mach, rho1, p1, gamma=GAMMA):
    """Downstream (rho2, p2) and lab-frame (u2, v_shock) for a right-moving shock
    into gas at rest."""
    pressure_ratio = (2 * gamma * mach**2 - (gamma - 1)) / (gamma + 1)
    compression = (gamma + 1) * mach**2 / ((gamma - 1) * mach**2 + 2)
    v_shock = mach * np.sqrt(gamma * p1 / rho1)
    return rho1 * compression, p1 * pressure_ratio, v_shock * (1 - 1 / compression), v_shock


def _setup(shock_selection=STRONGEST_SHOCK, num_cells=NUM_CELLS):
    config = SimulationConfig(
        geometry=CARTESIAN,
        num_cells=num_cells,
        box_size=1.0,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=True,
            shock_selection=shock_selection,
        ),
        riemann_solver=HLL,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    return config, helper_data, registered_variables


def _smooth_step(num_cells, centre, width):
    """1 left of ``centre``, 0 right of it, tanh-smoothed over ``width`` cells."""
    i = np.arange(num_cells)
    return 0.5 * (1.0 - np.tanh((i - centre) / width))


def _single_shock_state(mach=10.0, width=0.6, p_cr_up=0.0, centre=64.3):
    """A resolved right-moving gas shock (upstream rho = p = 1 at rest)."""
    config, helper_data, rv = _setup()
    rho2, p2, u2, v_shock = _rankine_hugoniot(mach, 1.0, 1.0)
    f = _smooth_step(NUM_CELLS, centre, width)
    rho = 1.0 + (rho2 - 1.0) * f
    p = 1.0 + (p2 - 1.0) * f
    u = u2 * f
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.asarray(rho), velocity_x=jnp.asarray(u),
        gas_pressure=jnp.asarray(p),
        cosmic_ray_pressure=jnp.full(NUM_CELLS, p_cr_up),
    )
    config = finalize_config(config, state.shape)
    return state, config, helper_data, rv, (rho2, p2, u2, v_shock)


def _energies(state, rv, helper_data):
    """(E_gas_thermal, E_cr) of a primitive state (1D)."""
    p_cr = cosmic_ray_pressure_from_n(state[rv.cosmic_ray_n_index])
    p_gas = state[rv.pressure_index] - p_cr
    vol = helper_data.cell_volumes
    return jnp.sum(p_gas / (GAMMA - 1) * vol), jnp.sum(p_cr / (GAMMA_CR - 1) * vol)


# -------------------------------------------------------------
# ===================== ↑ helpers ↑ ===========================
# -------------------------------------------------------------


@pytest.mark.parametrize("mach", [1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 10.0, 100.0])
def test_mach_estimate_exact_for_gas_shocks(mach):
    """Dubois Eq. 16 inverts a gamma = 5/3 Rankine-Hugoniot jump exactly."""
    _, p2, _, _ = _rankine_hugoniot(mach, 1.0, 1.0)
    m_sq = float(mach_number_squared(1.0, 0.0, p2, 0.0, GAMMA, GAMMA_CR))
    assert np.sqrt(m_sq) == pytest.approx(mach, rel=1e-10)


@pytest.mark.parametrize("zeta", [0.0, 0.5])
def test_mach_estimate_exact_for_cr_shocks(zeta):
    """...and the composite gas + CR jump of the Pfrommer+17 tube (M = 10.00 / 9.56),
    with and without injection (energy is conserved across the jump either way)."""
    reg = cosmic_ray_shock_tube_regions(injection_efficiency=zeta)
    p1, p1_cr, rho1 = 0.1, 0.05, 0.125
    p2 = reg["P_th2"] + reg["P_cr2"]
    gamma_eff1 = (GAMMA_CR * p1_cr + GAMMA * (p1 - p1_cr)) / p1
    mach_exact = reg["vs"] / np.sqrt(gamma_eff1 * p1 / rho1)
    m_sq = float(mach_number_squared(p1, p1_cr, p2, reg["P_cr2"], GAMMA, GAMMA_CR))
    assert np.sqrt(m_sq) == pytest.approx(mach_exact, rel=1e-6)
    assert mach_exact == pytest.approx(10.00 if zeta == 0 else 9.559, abs=2e-3)


def test_mach_estimate_rejects_reverse_jumps():
    """A jump with the upstream on the LEFT (P2 < P1) is never a right-moving shock."""
    for ratio in [0.9, 0.5, 0.1, 1e-3]:
        assert float(mach_number_squared(1.0, 0.0, ratio, 0.0)) < 1.0


def test_shock_sensor_non_negative():
    """The WENO-JS indicator is a sum of squares (the curvature term was unsquared)."""
    concave_peak = jnp.array([1.0, 1.0, 2.0, 1.0, 1.0])
    assert float(shock_sensor(concave_peak)[2]) == pytest.approx(13 / 12 * 4.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = jnp.asarray(rng.uniform(0.1, 10.0, 64))
        assert float(jnp.min(shock_sensor(p))) >= 0.0


def test_shock_criteria_flags_only_real_shocks():
    """The M > 1.3 gate: a resolved M = 1.2 compression is not flagged, M = 2 is."""
    for mach, expected in [(1.2, False), (2.0, True), (10.0, True)]:
        state, config, helper_data, rv, _ = _single_shock_state(mach=mach, width=0.3)
        flagged = bool(jnp.any(shock_criteria(state, config, rv, helper_data)))
        assert flagged == expected, mach


@pytest.mark.parametrize("escape_fraction", [0.0, 0.5])
def test_injection_energy_bookkeeping(escape_fraction):
    """One step: gas loses zeta * e_diss * v2 * dt; CRs keep (1 - f_esc) of it."""
    zeta, dt = 0.5, 1e-4
    state, config, helper_data, rv, (rho2, p2, u2, v_shock) = _single_shock_state()
    params = CosmicRayParams(
        diffusive_shock_acceleration_efficiency=zeta, escape_fraction=escape_fraction,
    )
    new = inject_crs_at_strongest_shock(state, GAMMA, helper_data, params, config, rv, dt)
    eg0, ec0 = _energies(state, rv, helper_data)
    eg1, ec1 = _energies(new, rv, helper_data)
    gas_loss, cr_gain = float(eg0 - eg1), float(ec1 - ec0)

    # exact dissipated energy flux of this Rankine-Hugoniot shock
    compression = rho2 / 1.0
    e_diss = p2 / (GAMMA - 1) - 1.0 / (GAMMA - 1) * compression**GAMMA
    expected = zeta * e_diss * (v_shock / compression) * dt
    assert gas_loss == pytest.approx(expected, rel=0.02)
    assert cr_gain == pytest.approx((1 - escape_fraction) * gas_loss, rel=1e-12)
    # only a few cells change, and none of them loses more than it has
    changed = np.flatnonzero(np.asarray(new[rv.pressure_index] != state[rv.pressure_index]))
    assert 1 <= changed.size <= 6
    p_gas_new = new[rv.pressure_index] - cosmic_ray_pressure_from_n(new[rv.cosmic_ray_n_index])
    assert float(jnp.min(p_gas_new)) > 0.0


def test_injection_linear_in_efficiency_and_capped():
    """Linear in zeta below the cap; the cap bounds the per-cell thermal removal."""
    state, config, helper_data, rv, _ = _single_shock_state()
    dt = 1e-4
    gains = []
    for zeta in (0.1, 0.2):
        new = inject_crs_at_strongest_shock(
            state, GAMMA, helper_data,
            CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta),
            config, rv, dt,
        )
        gains.append(float(_energies(new, rv, helper_data)[1]))
    assert gains[1] == pytest.approx(2 * gains[0], rel=1e-12)

    cap = 1e-4
    new = inject_crs_at_strongest_shock(
        state, GAMMA, helper_data,
        CosmicRayParams(diffusive_shock_acceleration_efficiency=0.5,
                        max_thermal_fraction_per_step=cap),
        config, rv, 1e-2,
    )
    e_th_old = state[rv.pressure_index] / (GAMMA - 1)
    p_cr_new = cosmic_ray_pressure_from_n(new[rv.cosmic_ray_n_index])
    e_th_new = (new[rv.pressure_index] - p_cr_new) / (GAMMA - 1)
    removed = e_th_old - e_th_new
    assert float(jnp.max(removed / e_th_old)) <= cap * (1 + 1e-9)
    assert float(jnp.max(removed)) > 0.0


def test_injection_invalid_zone_is_a_no_op():
    """No shock (a pure pressure pulse at rest / an expansion): nothing, and no NaN."""
    config, helper_data, rv = _setup()
    x = np.arange(NUM_CELLS)
    for p in (1.0 + 10.0 * (x == 64), 1.0 + 0.5 * np.tanh((x - 64) / 2.0)):
        state = construct_primitive_state(
            config=config, registered_variables=rv,
            density=jnp.ones(NUM_CELLS), velocity_x=jnp.zeros(NUM_CELLS),
            gas_pressure=jnp.asarray(p), cosmic_ray_pressure=jnp.zeros(NUM_CELLS),
        )
        cfg = finalize_config(config, state.shape)
        new = inject_crs_at_strongest_shock(
            state, GAMMA, helper_data, CosmicRayParams(0.0, 0.5), cfg, rv, 1e-3,
        )
        assert bool(jnp.all(jnp.isfinite(new)))
        np.testing.assert_array_equal(np.asarray(new), np.asarray(state))


def test_injection_is_ad_safe_in_cr_free_cells():
    """Tangents through an injection step are finite where P_cr = 0 (every
    CR-free cell); d E_cr / d zeta = E_cr / zeta."""
    state, config, helper_data, rv, _ = _single_shock_state(p_cr_up=0.0)
    dt = 1e-4

    def step(s, zeta):
        return inject_crs_at_strongest_shock(
            s, GAMMA, helper_data,
            CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta),
            config, rv, dt,
        )

    tangent = jnp.asarray(np.random.default_rng(1).normal(size=state.shape)) * 1e-3
    _, jvp_state = jax.jvp(lambda s: step(s, 0.3), (state,), (tangent,))
    assert bool(jnp.all(jnp.isfinite(jvp_state)))

    def cr_energy(zeta):
        return _energies(step(state, zeta), rv, helper_data)[1]

    value, grad = jax.value_and_grad(cr_energy)(0.3)
    assert np.isfinite(float(grad))
    assert float(grad) == pytest.approx(float(value) / 0.3, rel=1e-10)

    # the helpers themselves: zero tangent at P_cr = 0, exact elsewhere
    g0 = jax.grad(lambda p: cosmic_ray_n_from_pressure(p))(0.0)
    g1 = jax.grad(lambda p: cosmic_ray_n_from_pressure(p))(2.0)
    assert float(g0) == 0.0
    assert float(g1) == pytest.approx(0.75 * 2.0 ** (-0.25), rel=1e-12)


def test_outermost_selection_picks_the_forward_shock():
    """Strong inner shock + weaker outer one: STRONGEST -> inner, OUTERMOST -> outer."""
    # outer M = 3 shock into (1, 1, 0); inner M = 10 shock into the outer's downstream
    rho_b, p_b, u_b, _ = _rankine_hugoniot(3.0, 1.0, 1.0)
    rho_a, p_a, du_a, _ = _rankine_hugoniot(10.0, rho_b, p_b)
    f_out = _smooth_step(NUM_CELLS, 90.3, 0.6)
    f_in = _smooth_step(NUM_CELLS, 40.3, 0.6)
    rho = 1.0 + (rho_b - 1.0) * f_out + (rho_a - rho_b) * f_in
    p = 1.0 + (p_b - 1.0) * f_out + (p_a - p_b) * f_in
    u = u_b * f_out + du_a * f_in
    picks = {}
    for selection in (STRONGEST_SHOCK, OUTERMOST_SHOCK):
        config, helper_data, rv = _setup(shock_selection=selection)
        state = construct_primitive_state(
            config=config, registered_variables=rv,
            density=jnp.asarray(rho), velocity_x=jnp.asarray(u),
            gas_pressure=jnp.asarray(p), cosmic_ray_pressure=jnp.zeros(NUM_CELLS),
        )
        config = finalize_config(config, state.shape)
        idx, _, _ = find_shock_zone(
            state, config, rv, helper_data, shock_selection=selection,
        )
        picks[selection] = int(idx)
    assert abs(picks[STRONGEST_SHOCK] - 40) <= 2
    assert abs(picks[OUTERMOST_SHOCK] - 90) <= 2


def test_spherical_injection_near_the_origin():
    """A flagged M = 2 shock a few cells from r = 0 in spherical geometry gets
    its CRs, conservatively and with no negative per-cell share.

    Reviewer regression test (2026-09-25): the weights used to be
    ``e_i V_i - e_ref V_ref``; with V_ref > V_i near the origin they were all
    negative here, and after the zero clip the step silently injected nothing.
    The weights are now ``(e_i - e_ref) V_i``.
    """
    num_cells, zeta, dt = 64, 0.5, 1e-3
    config = SimulationConfig(
        geometry=SPHERICAL, num_cells=num_cells, box_size=1.0,
        cosmic_ray_config=CosmicRayConfig(cosmic_rays=True, diffusive_shock_acceleration=True),
        riemann_solver=HLL, solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX), progress_bar=False,
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    rho2, p2, u2, _ = _rankine_hugoniot(2.0, 1.0, 1.0)
    f = _smooth_step(num_cells, 4.3, 2.0)
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.asarray(1.0 + (rho2 - 1.0) * f), velocity_x=jnp.asarray(u2 * f),
        gas_pressure=jnp.asarray(1.0 + (p2 - 1.0) * f),
        cosmic_ray_pressure=jnp.zeros(num_cells),
    )
    config = finalize_config(config, state.shape)
    # the solver would call the injection for this state
    assert bool(jnp.any(shock_criteria(state, config, rv, helper_data)))

    new = inject_crs_at_strongest_shock(
        state, GAMMA, helper_data,
        CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta), config, rv, dt,
    )
    eg0, ec0 = _energies(state, rv, helper_data)
    eg1, ec1 = _energies(new, rv, helper_data)
    gas_loss, cr_gain = float(eg0 - eg1), float(ec1 - ec0)
    assert cr_gain > 0.0
    assert cr_gain == pytest.approx(gas_loss, rel=1e-12)
    p_cr_new = np.asarray(cosmic_ray_pressure_from_n(new[rv.cosmic_ray_n_index]))
    assert float(np.min(p_cr_new)) >= 0.0
    p_gas_new = np.asarray(new[rv.pressure_index]) - p_cr_new
    assert float(np.min(np.asarray(state[rv.pressure_index]) - p_gas_new)) >= 0.0
