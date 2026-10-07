"""
CPU checks of the exact similarity rescaling (``casa_rescale``) and its traced
twin (``casa_pluto_diff.transform_fields(sim=...)``).

    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ./run.sh -m pytest -q casa_rescale_test.py
"""

# ==== CPU only ====
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
# ==================

from pathlib import Path

import numpy as np
import pytest

#: 128^3: the remnant spans ~22 cells in radius (at 32^3 the remap loses ~4 % of E_kin)
IC = Path("/export/data/lstorcks/casa_orlando150/work/pluto146_n128.npz")
pytestmark = pytest.mark.skipif(not IC.exists(), reason=f"{IC} not available")


@pytest.fixture(scope="module")
def ic():
    return dict(np.load(IC))


def test_identity_is_exact(ic):
    import casa_rescale as CR
    new = CR.rescale_state(ic, 1.0, 1.0, 1.0, log=lambda *a: None)
    for k in ("rho", "press", "vx", "C_ej", "time_since_shock", "density_time"):
        np.testing.assert_allclose(new[k], ic[k], rtol=1e-6, atol=1e-30)
    assert float(new["age"]) == pytest.approx(float(ic["age"]))


@pytest.mark.parametrize("L,T,M", [(1.3, 1.2, 1.1), (0.85, 0.9, 0.8)])
def test_conservative_factors(ic, L, T, M):
    """Ejecta mass exactly x M; energy x M (L/T)^2 up to the sub-cell kinetic
    energy of the remap; the age label x T; the wind's rho r^2 x M / L."""
    import casa_rescale as CR
    new = CR.rescale_state(ic, L, T, M, log=lambda *a: None)
    b0, b1 = CR.state_budget(ic), CR.state_budget(new)
    assert b1["M_ej"] / b0["M_ej"] == pytest.approx(M, rel=1e-5)
    F = CR.factors(L, T, M)
    assert b1["E_th"] / b0["E_th"] == pytest.approx(F["energy"], rel=5e-3)
    assert b1["E_tot"] / b0["E_tot"] == pytest.approx(F["energy"], rel=3e-2)
    assert float(new["age"]) == pytest.approx(T * float(ic["age"]))
    assert CR.wind_nh3(new) / CR.wind_nh3(ic) == pytest.approx(M / L)
    assert float(new["ambient_r_sh"]) == pytest.approx(L * float(ic["ambient_r_sh"]))


def test_traced_map(ic):
    """sim = 0 reproduces the legacy transform; the traced map keeps the
    ejecta mass, has the exact log-derivatives of E, M_ej and the IC age."""
    import jax
    import jax.numpy as jnp
    from astropy import units as u
    import casa_pluto_diff as PD
    from _common import snr_code_units
    box, n = float(ic["box"]), int(ic["num_cells"])
    x = (np.arange(n) + 0.5) * box / n - box / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    geom = tuple(jnp.asarray(a, jnp.float32) for a in (np.sqrt(X ** 2 + Y ** 2 + Z ** 2), X, Y, Z))
    rho_c = float((1.0 * snr_code_units().code_density).to(u.g / u.cm ** 3).value)
    th = jnp.asarray(PD.THETA0, jnp.float32)
    hist = ("shocked_fraction", "time_since_shock", "density_time")
    f0 = PD.transform_fields(ic, th, geom, rho_c, extra=hist)
    fz = PD.transform_fields(ic, th, geom, rho_c, extra=hist, sim=(0.0, 0.0, 0.0))
    for k in ("rho", "press", "C_ej"):
        np.testing.assert_allclose(np.asarray(fz[k]), np.asarray(f0[k]), rtol=0.0,
                                   atol=2e-4 * float(jnp.max(jnp.abs(f0[k]))))
    cv = (box / n) ** 3
    b0 = PD.ic_budget(f0, cv)
    sim = tuple(np.log([1.3, 1.2, 1.1]))
    # the momentum-conserving velocity resample keeps E to < 3 %; the primitive one
    # (default: stable at 512^3) interpolates v linearly and keeps it to < 6 % at L 1.3
    for vel, tol in (("momentum", 3e-2), ("primitive", 6e-2)):
        fs = PD.transform_fields(ic, th, geom, rho_c, extra=hist, sim=sim, velocity=vel)
        bs = PD.ic_budget(fs, cv)
        assert float(bs["M_ej"] / b0["M_ej"]) == pytest.approx(1.1, rel=1e-2)
        assert float(bs["E_tot"] / b0["E_tot"]) == pytest.approx(1.1 * (1.3 / 1.2) ** 2, rel=tol), vel
    assert float(PD.ic_age(ic, sim)) == pytest.approx(1.2 * float(ic["age"]), rel=1e-6)
    assert float(PD.ballistic_convergence_date(1681.0, 0.0, PD.ic_age(ic, sim))) == pytest.approx(1681.0)

    def g(s):
        f = PD.transform_fields(ic, th, geom, rho_c, extra=hist, sim=(s[0], s[1], s[2]))
        b = PD.ic_budget(f, cv)
        return jnp.log(jnp.stack([b["E_th"], b["M_ej"], PD.ic_age(ic, (s[0], s[1], s[2]))]))
    J = jax.jacfwd(g)(jnp.asarray(sim, jnp.float32))
    # rows: E_th, M_ej, age; columns: ln_L, ln_T, ln_M. E_th is interpolated linearly,
    # so its exponents are exact up to the interpolation error of the stretch
    np.testing.assert_allclose(np.asarray(J[:, 1:]), [[-2, 1], [0, 1], [1, 0]], atol=2e-2)
    assert abs(float(J[1, 0])) < 0.1 and abs(float(J[2, 0])) < 1e-6
    assert abs(float(J[0, 0]) - 2.0) < 0.1


def test_shell_knobs_remove_the_resampled_shell(ic):
    """Review 2026-09-27: under the map, ln_fsh -> -inf must leave the scaled
    wind (the shell that is removed is the one the RESAMPLED state carries, not
    the narrower analytic L-scaled template; the old subtraction left 0.45 Msun
    of +- residual and created 0.28 Msun through the density floor)."""
    import jax.numpy as jnp
    from astropy import units as u
    import casa_pluto_diff as PD
    from _common import snr_code_units
    box, n = float(ic["box"]), int(ic["num_cells"])
    x = (np.arange(n) + 0.5) * box / n - box / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    geom = tuple(jnp.asarray(a) for a in (r, X, Y, Z))
    cu = snr_code_units()
    rho_c = float((1.0 * cu.code_density).to(u.g / u.cm ** 3).value)
    to_msun = float((1.0 * cu.code_density * (1 * u.pc) ** 3).to(u.Msun).value) * (box / n) ** 3
    L = 1.42
    th = dict(zip(PD.PARAM_NAMES, np.asarray(PD.THETA0, np.float64)))
    th["ln_fsh"] = -20.0
    th = jnp.asarray([th[k] for k in PD.PARAM_NAMES])
    f = PD.transform_fields(ic, th, geom, rho_c, extra=("shocked_fraction",),
                            sim=(np.log(L), np.log(1.21), 0.0))
    wind = (float(ic["ambient_rho_w"]) / rho_c / L
            * (float(ic["ambient_r_ref"]) / np.maximum(r, 1e-3)) ** 2)
    w_amb = np.clip((1 - np.asarray(f["C_ej"])) * (1 - np.asarray(f["shocked_fraction"])), 0, 1)
    zone = np.abs(r - L * float(ic["ambient_r_sh"])) < 0.35 * L
    res = (np.asarray(f["rho"]) - wind) * w_amb * zone
    assert abs(res.sum()) * to_msun < 0.08          # legacy path: 0.029; old sim path: 0.285
    assert np.abs(res).sum() * to_msun < 0.12       # legacy path: 0.063; old sim path: 0.452


SAVED = Path("/export/data/lstorcks/casa_orlando150/work/state2000_R2_n128.npz")


@pytest.mark.skipif(not SAVED.exists(), reason=f"{SAVED} not available")
@pytest.mark.parametrize("method", ["conservative", "linear"])
def test_save_state_file_roundtrip(method):
    """Review 2026-09-27: a casa_xfit / casa_4dvar save_state file (dual-energy
    ``internal_energy`` in its var_layout) must survive rescaling and reload,
    with the calendar epoch kept and t_expl moved (age' = T age)."""
    import casa_rescale as CR
    from casa_xfit_state import load_state, state_from_npz
    d = dict(np.load(SAVED))
    L, T, M = 1.1, 1.05, 0.9
    new = CR.rescale_state(d, L, T, M, method=method, log=lambda *a: None)
    out = SAVED.parent / f"_rescale_test_{method}.npz"
    try:
        np.savez_compressed(out, **new)
        fields, meta = load_state(out)
        assert state_from_npz(out).shape[0] == int(d["num_vars"])
        assert float(meta["age"]) == pytest.approx(T * float(d["age"]))
        assert float(meta["epoch_year"]) - float(meta["t_expl"]) == pytest.approx(float(meta["age"]))
        names = [str(x) for x in d["names"]]
        assert float(meta["theta"][names.index("t_expl")]) == pytest.approx(float(meta["t_expl"]))
        F = CR.factors(L, T, M)
        e0 = np.asarray(d["internal_energy"], np.float64).sum()
        e1 = np.asarray(fields["internal_energy"], np.float64).sum()
        p0 = np.asarray(d["press"], np.float64).sum()
        p1 = np.asarray(fields["press"], np.float64).sum()
        assert (e1 / e0) / (p1 / p0) == pytest.approx(1.0, rel=2e-2)
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.parametrize("sim", [None, (0.05, 0.02, -0.03)])
def test_coordinate_precision(ic, sim):
    """Review 2026-09-27 (ers/integ2): every matmul of the traced IC map (the
    rotation matrix, R^T x) runs at Precision.HIGHEST. The GPU default for an f32
    matmul is TF32, which misplaced the resampling points by ~5e-4 |x| and changed
    3-4 % of the rotated interior vs the CPU on every GPU run (the apparent
    "8-way sharded IC bug"); the CPU ignores the setting, so this checks the jaxpr
    (backend-independent) instead of comparing values."""
    import jax
    import jax.numpy as jnp
    from astropy import units as u
    import casa_pluto_diff as PD
    from _common import snr_code_units
    n, box = int(ic["num_cells"]), float(ic["box"])
    g = (np.arange(n) + 0.5) * box / n - 0.5 * box
    Xg, Yg, Zg = np.meshgrid(g, g, g, indexing="ij")
    geom = tuple(jnp.asarray(a, jnp.float32) for a in (np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2), Xg, Yg, Zg))
    rho_c = float((1.0 * snr_code_units().code_density).to(u.g / u.cm ** 3).value)
    th = np.array([PD.PRIOR[k][0] for k in PD.PARAM_NAMES], np.float32)
    th[PD.PARAM_NAMES.index("rot_z")] = 30.0
    jaxpr = jax.make_jaxpr(lambda t: PD.transform_fields(ic, t, geom, rho_c, sim=sim))(jnp.asarray(th))
    dots = []

    def walk(jx):
        for e in jx.eqns:
            if e.primitive.name == "dot_general":
                dots.append(e.params["precision"])
            for v in e.params.values():
                for sub in (v if isinstance(v, (list, tuple)) else (v,)):
                    if hasattr(sub, "jaxpr"):
                        walk(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub.jaxpr.jaxpr)
    walk(jaxpr.jaxpr)
    assert len(dots) >= 2, dots                     # K @ K and R^T x
    hi = jax.lax.Precision.HIGHEST
    assert all(p is not None and all(q == hi for q in (p if isinstance(p, tuple) else (p,))) for p in dots), dots
