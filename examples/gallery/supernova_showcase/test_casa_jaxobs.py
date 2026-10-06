"""
Regression tests of the differentiable observation model (``casa_jaxobs``).

CPU-only, on a 32^3 downsample of a real 256^3 state; needs the tables in
``/export/data/lstorcks/casa_orlando150/jaxobs``. The full-resolution check
against the pyXSIM chain is ``casa_jaxobs.py validate`` (bands within 3 %).

    JAX_PLATFORMS=cpu ./run.sh -m pytest test_casa_jaxobs.py -q
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _plasma as P
import casa_jaxobs as J

STATE = Path("/export/data/lstorcks/casa_orlando150/work/plC_n256_age380yr_solarcsm.npz")
D_KPC = 3.0496
pytestmark = pytest.mark.skipif(not STATE.exists() or not (J.TABLE_DIR / "emissivity_chandra_aciss_cy0.npz").exists(),
                                reason="state or emissivity tables not available")


@pytest.fixture(scope="module")
def setup():
    f, box, _ = J.load_fields(STATE)
    f = {k: v[::8, ::8, ::8] for k, v in f.items()}
    return f, box, J.load_tables("chandra_aciss_cy0", 1.2)


def test_electron_temperature_matches_numpy(setup):
    f, _, _ = setup
    X = {k: np.asarray(v, np.float64) for k, v in J.element_fractions(f).items()}
    m = J.moments({k: jnp.asarray(v) for k, v in X.items()})
    rho = np.asarray(f["rho"], np.float64)
    T = J._T_PER_CODE * np.asarray(m["mu"]) * np.asarray(f["press"]) / rho
    Te_np, _ = P.electron_ion_temperatures(T, rho, np.asarray(f["time_since_shock"]), X)
    Te_j = J.electron_temperature(jnp.asarray(T, jnp.float32), jnp.asarray(rho * J._NB_PER_CODE_RHO, jnp.float32),
                                  jnp.asarray(f["time_since_shock"]) * P.CODE_TIME, m)
    hot = T > 1e6
    np.testing.assert_allclose(np.asarray(Te_j)[hot], Te_np[hot], rtol=2e-4)


def test_dem_matmul_equals_direct_sum(setup):
    """The tent-matmul DEM contracted with the tables == the per-cell bilinear sum."""
    f, box, tab = setup
    rate = J.aperture_spectrum(f, tab, box_pc=box, distance_kpc=D_KPC, v_los_kms=False,
                               aperture_arcsec=1e4)
    n = f["rho"].shape[0]
    norm = J.flux_norm((box / n * J.PC_CM) ** 3, D_KPC)
    ref = 0.0
    for fc, ew in J.emitting_components(f):
        pl = J.plasma(fc)
        w = J.component_weights(pl, tab, norm, ew)
        ii, jj, ww = J.corners(tab, pl)
        for c in range(len(tab["names"])):
            ref = ref + jnp.einsum("qxyz,qxyzh->h", w[c][None] * ww, tab["C"][c][ii, jj])
    np.testing.assert_allclose(np.asarray(rate), np.asarray(ref), rtol=2e-4, atol=1e-6 * float(ref.max()))


def test_images_conserve_the_aperture_rate(setup):
    """Without halo and PSF, the image total equals the band rates of the spectrum."""
    f, box, tab = setup
    rate = J.aperture_spectrum(f, tab, box_pc=box, distance_kpc=D_KPC, aperture_arcsec=1e4)
    img = J.band_images(f, tab, box_pc=box, distance_kpc=D_KPC, npix=128, pix_arcsec=8.0)
    np.testing.assert_allclose(np.asarray(img.sum((1, 2))), np.asarray(J.band_rates(rate, tab)), rtol=2e-3)


def test_vjp_equals_jvp(setup):
    f, box, tab = setup
    g = lambda rho: jnp.sum(J.aperture_spectrum(dict(f, rho=rho), tab, box_pc=box,  # noqa: E731
                                                distance_kpc=D_KPC))
    d = jnp.asarray(np.random.default_rng(1).standard_normal(f["rho"].shape), jnp.float32) * f["rho"]
    vjp = float(jnp.vdot(jax.grad(g)(f["rho"]), d))
    _, jvp = jax.jvp(g, (f["rho"],), (d,))
    assert np.isfinite(vjp) and abs(vjp - float(jvp)) <= 1e-3 * abs(float(jvp))


def test_roll_and_offset_gradients_are_finite(setup):
    f, box, tab = setup
    cols = J.band_columns(f, tab, box_pc=box, distance_kpc=D_KPC)
    loss = lambda p: jnp.sum(J.project_columns(cols, box_pc=box, distance_kpc=D_KPC, npix=64,  # noqa: E731
                                               pix_arcsec=8.0, roll_deg=p[0],
                                               offset_arcsec=(p[1], p[2]))[1] ** 2)
    g = jax.grad(loss)(jnp.asarray([10.0, 3.0, -2.0]))
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).sum()) > 0


def test_synchrotron_matches_numpy(setup):
    """sync_columns with an energy-flux 'table' == _synchrotron.synchrotron_fields (hard gate)."""
    import _synchrotron as SY
    f, _, _ = setup
    pl = J.plasma(f)
    tss_yr = np.asarray(f["time_since_shock"]) * P.CODE_TIME / 3.155693e7
    _, _, rep = SY.synchrotron_fields(np.asarray(f["rho"], np.float64) * P.CODE_DENSITY,
                                      np.asarray(pl["T_i"], np.float64), np.asarray(pl["mu_i"], np.float64),
                                      tss_yr, np.asarray(pl["shocked"]), epoch=2000.0, band=(4.2, 6.0))
    lec = np.log(np.geomspace(0.01, 100, 64))
    tab = SY.band_shape_integral(4.2, 6.0, np.exp(lec))[:, None].astype(np.float32)
    cols = J.sync_columns(f, lecut=lec, band_table=tab, year=2000.0, gate_softness=0.02)
    assert rep["flux_band"] > 0
    np.testing.assert_allclose(float(cols.sum()), rep["flux_band"], rtol=0.05)


# =============================================================================
# ============ ↓ v2 observation model (audit 2026-09-25) ↓ ====================
# =============================================================================
import casa_jaxobs_nei as NEIH

V2 = J.TABLE_DIR / "emissivity_bins_chandra_aciss_cy0.npz"
needs_v2 = pytest.mark.skipif(not V2.exists(), reason="v2 binned tables not built")
needs_hist = pytest.mark.skipif(not NEIH.TABLE_PATH.exists(), reason="NEI-history table not built")
needs_halo2 = pytest.mark.skipif(not (J.TABLE_DIR / "dusthalo_v2_chandra_aciss_cy0.npz").exists(),
                                 reason="v2 halo not built")
Q2_EMISSION = dict(kT_e_shock_keV=float(np.exp(-2.224)), teq_scale=float(np.exp(-1.0262)))


def test_relax_history_leaves_te_unchanged(setup):
    """The history accumulator rides along the relaxation scan: T_e is bit-identical."""
    f, _, _ = setup
    X = J.element_fractions(f)
    m = J.moments(X)
    rho = jnp.maximum(f["rho"], 1e-30)
    n_b = rho * J._NB_PER_CODE_RHO
    T = J._T_PER_CODE * m["mu"] * f["press"] / rho
    t_s = f["time_since_shock"] * P.CODE_TIME
    Te_v1 = J._electron_temperature_v1(T, n_b, t_s, m, 0.11, teq_scale=0.36)
    Te, rh = J._relax_history(T, n_b, t_s, m, 0.11, teq_scale=0.36)
    np.testing.assert_array_equal(np.asarray(Te), np.asarray(Te_v1))
    sh = np.asarray(t_s) > 0
    r = np.asarray(rh)
    assert np.all(r[~sh] == 1.0) and np.all((r[sh] > 0.70) & (r[sh] <= 1.0 + 1e-5))


def test_rho_hist_matches_the_numpy_track():
    """rho_hist (the forward model's scan) == casa_jaxobs_nei.rho_of_track on _plasma's own track."""
    X = {"Si": 0.57, "S": 0.30, "Ar": 0.08, "Ca": 0.05}
    for n_e, age, kte0, teq in ((30.0, 100.0, 0.11, 0.36), (100.0, 250.0, 0.3, 1.0), (3.0, 50.0, 0.3, 1.0)):
        _, Te, rho_np = NEIH.plasma_track(X, n_e, 12.0, age, kte0, teq)
        m = J.moments({k: jnp.asarray(v) for k, v in X.items()})
        n_b = jnp.asarray([n_e * float(m["mu_e"])])
        _, rh = J._relax_history(jnp.asarray([12.0 / NEIH.KB_KEV]), n_b,
                                 jnp.asarray([age * NEIH.YR_S * teq]), m, kte0)
        np.testing.assert_allclose(float(rh[0]), rho_np, rtol=3e-3)


@needs_hist
def test_nei_history_table_rho1_is_legacy():
    import _nei
    kt, rg, net, tab = NEIH.load_table()
    kt0, net0, leg = _nei.load_table()
    assert rg[-1] == 1.0 and np.allclose(kt, kt0) and np.allclose(net, net0)
    for el in ("O", "Si", "Fe"):
        np.testing.assert_allclose(tab[el][:, -1], leg[el], atol=1e-6)


@needs_hist
def test_nei_history_against_direct_integration():
    """The table at a parcel's own rho reproduces step-by-step ionisation along its
    _plasma T_e(t) track (He-like Si/S, Fe XXV) to < 5 %; the legacy constant-T_e
    table misses it by 30-50 % (obs_model section 7)."""
    import _nei
    kt, rg, net_g, tab = NEIH.load_table()
    _, _, leg = _nei.load_table()
    X = {"Si": 0.57, "S": 0.30, "Ar": 0.08, "Ca": 0.05}
    u, Te, rho = NEIH.plasma_track(X, 30.0, 12.0, 250.0, **dict(kte0=0.11, teq=0.36))
    tau = 30.0 * 250.0 * NEIH.YR_S
    for el, q in (("Si", 12), ("S", 14)):
        direct = NEIH.ionise_along(el, 0.5 * (Te[1:] + Te[:-1]), np.diff(u), [tau])[0][q]
        new = NEIH.interpolate(tab[el], kt, rg, net_g, Te[-1], rho, tau)[q]
        old = _nei.interpolate_fractions(leg[el], kt, net_g, np.array([Te[-1]]), np.array([tau]))[q, 0]
        assert abs(new / direct - 1) < 0.05, (el, new, direct)
        assert old / direct < 0.8, (el, old, direct)


def test_csm_solar_split():
    """Pure solar CSM -> solar Ne/O, Mg/O; pure ejecta -> the legacy split; mass conserved."""
    x = J.solar_csm_tracers()
    csm = {"C_ej": jnp.asarray([0.0, 1.0, 0.5]), "C_O": jnp.asarray([x["O"], 0.6, 0.3 + 0.5 * x["O"]]),
           "C_Si": jnp.asarray([x["Si"], 0.1, 0.05 + 0.5 * x["Si"]]), "C_Fe": jnp.asarray([x["Fe"], 0.05, 0.03]),
           "C_He": jnp.asarray([x["He"], 0.2, 0.1 + 0.5 * x["He"]])}
    old = J.element_fractions(csm)
    new = J.element_fractions(csm, csm_solar=True)
    R = P.SOLAR_NUMBER_RATIO_TO_H
    for el in ("Ne", "Mg"):
        solar = R[el] * P.ATOMIC[el][0] / (R["O"] * P.ATOMIC["O"][0])
        assert abs(float(new[el][0] / new["O"][0]) / solar - 1) < 1e-4
        assert float(old[el][0] / old["O"][0]) / solar < 0.35          # the bug: Ne 0.10x, Mg 0.27x
    for el in old:
        assert abs(float(new[el][1]) - float(old[el][1])) < 1e-7        # ejecta untouched
    tot_o = sum(float(v[2]) for v in old.values()); tot_n = sum(float(v[2]) for v in new.values())
    assert abs(tot_o - tot_n) < 1e-6


@needs_v2
def test_v2_band_tables_equal_v1_up_to_edges():
    """At rho = 1 and N_H = 1.2 the v2 band tables are the v1 tables integrated
    over the exact band edges instead of 117-eV-channel quantised ones: equal to
    a few % for the H continuum, the quantised Si band 1.526-2.110 is gone."""
    v1 = J.band_tables_of(J.load_tables("chandra_aciss_cy0", 1.2))[0]
    v2 = J.load_binned_tables("chandra_aciss_cy0", 1.2, "band", history=False)
    assert v2["C"].shape == v1.shape
    for b in range(6):
        H1 = np.asarray(v1[0, 30, 0, b]); H2 = np.asarray(v2["C"][0, 30, 0, b])     # H at kT = 3.4 keV
        assert abs(H2 / H1 - 1) < 0.08, (b, H2 / H1)
    # the 1.5-2.1 band now stops exactly at 2.1: Si Ly-a (2.006) fully in, S not
    edges = np.asarray(J.load_tables("chandra_aciss_cy0", 1.2)["ch_edges"])
    lo = edges[np.searchsorted(edges, 1.5)]; hi = edges[np.searchsorted(edges, 2.1)]
    assert abs(lo - 1.5) > 0.01 or abs(hi - 2.1) > 0.01                # v1 quantisation exists


@needs_v2
def test_v2_spectral_artefact_removed():
    """The v1 channel tables spread uniformly onto the 0.2 keV bins put Si-rich
    NEI emission 6-15 % too low in the Si He-a bin (1.8) relative to 2.0, and
    Fe-K 6.6 too low vs 6.4 / 6.8 (obs_model section 4, t1b). v2 / v1 at a
    Si-rich cell shows exactly that ratio pattern."""
    t1 = J.load_tables("chandra_aciss_cy0", 1.2)
    ch = np.asarray(t1["ch_edges"]); e = J.SPEC_EDGES
    sel = np.nonzero((ch[1:] > e[0] - 0.1) & (ch[:-1] < e[-1] + 0.1))[0]
    lo, hi = ch[sel], ch[sel + 1]
    O = np.clip(np.minimum(e[1:, None], hi[None]) - np.maximum(e[:-1, None], lo[None]), 0, None) / (hi - lo)[None]
    v2 = J.load_binned_tables("chandra_aciss_cy0", 1.2, "spec", history=False)
    i = int(np.argmin(np.abs(10 ** np.asarray(t1["lkt"]) - 1.5)))
    j = int(np.argmin(np.abs(10 ** np.asarray(t1["lnet"]) - 2e11)))
    si = t1["names"].index("Si")
    old = O @ np.asarray(t1["C"][si, i, j, sel]); new = np.asarray(v2["C"][si, i, j])
    mids = 0.5 * (e[1:] + e[:-1]); r = new / old
    k18, k20 = int(np.argmin(abs(mids - 1.8))), int(np.argmin(abs(mids - 2.0)))
    assert 1.04 < r[k18] < 1.25 and r[k20] < 0.98, (r[k18], r[k20])
    # the line-free continuum is unchanged
    k45 = int(np.argmin(abs(mids - 4.5)))
    H = t1["names"].index("H")
    rH = (np.asarray(v2["C"][H, i, j]) / (O @ np.asarray(t1["C"][H, i, j, sel])))
    assert abs(rH[k45] - 1) < 0.02


@needs_v2
def test_v2_history_axis_and_gradients(setup):
    """History tables: images conserve the band rates, rho = 1 reproduces the
    no-history tables when every cell is forced to constant T_e, VJP == JVP."""
    f, box, _ = setup
    th = J.load_binned_tables("chandra_aciss_cy0", 1.2, "band", history=True)
    t0 = J.load_binned_tables("chandra_aciss_cy0", 1.2, "band", history=False)
    kw = dict(box_pc=box, distance_kpc=D_KPC, plasma_kw=Q2_EMISSION)
    c_h = J.band_columns(f, th, **kw)
    c_0 = J.band_columns(f, t0, **kw)
    # the history raises the He-like Si/S fraction: Si-band emission up, not by more than 2x
    r = np.asarray(c_h.sum((1, 2)) / c_0.sum((1, 2)))
    assert 1.0 < r[1] < 2.0, r
    # kT-only equilibrated plasma (kte0 >= T): rho = 1 everywhere -> identical
    hot = dict(kT_e_shock_keV=1e4)
    np.testing.assert_allclose(np.asarray(J.band_columns(f, th, box_pc=box, distance_kpc=D_KPC, plasma_kw=hot)),
                               np.asarray(J.band_columns(f, t0, box_pc=box, distance_kpc=D_KPC, plasma_kw=hot)),
                               rtol=1e-5, atol=1e-9)
    g = lambda rho: jnp.sum(J.band_columns(dict(f, rho=rho), th, kt_interp="log", **kw) ** 2)  # noqa: E731
    d = jnp.asarray(np.random.default_rng(2).standard_normal(f["rho"].shape), jnp.float32) * f["rho"]
    vjp = float(jnp.vdot(jax.grad(g)(f["rho"]), d))
    _, jvp = jax.jvp(g, (f["rho"],), (d,))
    assert np.isfinite(vjp) and abs(vjp - float(jvp)) <= 2e-3 * abs(float(jvp))


@needs_v2
def test_log_kt_interpolation(setup):
    """Geometric kT interpolation: never above linear (AM-GM) for the positive
    tables; the soft continuum barely moves, the low-kT Fe-K tail falls."""
    f, box, _ = setup
    t = J.load_binned_tables("chandra_aciss_cy0", 1.2, "band", history=False)
    kw = dict(box_pc=box, distance_kpc=D_KPC, v_los_kms=False, plasma_kw=Q2_EMISSION)
    lin = np.asarray(J.band_columns(f, t, **kw).sum((1, 2)))
    log = np.asarray(J.band_columns(f, t, kt_interp="log", **kw).sum((1, 2)))
    assert np.all(log <= lin * (1 + 1e-5))
    assert abs(log[0] / lin[0] - 1) < 0.02 and log[5] / lin[5] < 0.999


@needs_v2
def test_v2_doppler_moments_and_frame(setup):
    """doppler_from_columns at roll 0 / offset 0 / scale 1 with hard edges ==
    the legacy sector reduction; the v2 statistic uses the exact window."""
    f, box, tab = setup
    dop = J.load_binned_tables("chandra_aciss_cy0", 1.2, "dop", history=False)
    assert np.allclose(dop["ch_edges"], [1.78, 1.94]) and abs(dop["E0"] - 1.86) < 1e-9
    S = J.doppler_columns(f, dop, box_pc=box, distance_kpc=D_KPC)
    v_new, ok = J.doppler_from_columns(S, box_pc=box, distance_kpc=D_KPC, E0=dop["E0"])
    n = f["rho"].shape[0]
    WX, NZ, R = J.sky_geometry(box, n, D_KPC)
    pa = np.rad2deg(np.arctan2(NZ, WX)) % 360.0
    sec = np.where((R > 40) & (R < 170), (pa / 15.0).astype(int) % 24, 24)
    Sk = np.stack([np.asarray(S)[:, sec == k].sum(1) for k in range(24)], 1)
    dE = Sk[1] / Sk[0]
    v_ref = -J.C_KMS * (dE - dE.mean()) / dop["E0"]
    np.testing.assert_allclose(np.asarray(v_new), v_ref, atol=2.0)
    v_s, _ = J.doppler_sectors(f, tab, box_pc=box, distance_kpc=D_KPC, moment_tables=dop)
    np.testing.assert_allclose(np.asarray(v_s), np.asarray(v_new), atol=1e-3)
    # soft edges: differentiable in the roll
    gr = jax.grad(lambda r: jnp.sum(J.doppler_from_columns(S, box_pc=box, distance_kpc=D_KPC, roll_deg=r,
                                                           soft_deg=1.0, soft_arcsec=2.0)[0] ** 2))(5.0)
    assert np.isfinite(float(gr)) and float(gr) != 0.0


@needs_halo2
def test_v2_halo_kernels():
    """Spectrum-weighted kernels: the 0.5-1.5 keV core is brighter than v1's
    uniform-energy kernel (0.49), it dims with N_H, and later epochs (harder
    in-band spectra) have brighter cores (obs_model section 9). At N_H = 1.2
    the node's own observed spectrum, re-absorbed from 1.43 to 1.2, gives 0.54
    (2000) -> 0.57 (2019); unre-absorbed (weights at 1.43) 0.57 -> 0.59."""
    old = J.load_halo(1.2)
    h = J.load_halo_v2("chandra_aciss_cy0", 1.2)
    half = int(h["kernel_half"])
    c_old, c_new = float(old["kernel"][0, half, half]), float(h["kernel"][0, half, half])
    assert abs(c_old - 0.492) < 0.01 and 0.52 < c_new < 0.62, (c_old, c_new)
    assert float(J.load_halo_v2("chandra_aciss_cy0", 2.6)["kernel"][0, half, half]) < c_new
    late = float(J.load_halo_v2("chandra_aciss_cy22", 1.2)["kernel"][0, half, half])
    assert late > c_new
    assert np.all(np.asarray(h["kernel"]).sum((1, 2)) <= 1.0 + 1e-5)
    st = J.load_halo_stack("chandra_aciss_cy0")
    assert np.all(np.diff(st["aperture_keep"][:, 20, 0]) <= 1e-3)        # more dust, less kept


def test_doppler_moment_maps_is_doppler_sectors(setup):
    """``doppler_moment_maps`` (exposed for casa_xfit's sky-frame sectors) reduced
    into the sim-frame sectors reproduces ``doppler_sectors`` (v1 tables), and
    with v2 ``dop`` tables it is ``doppler_columns``."""
    f, box, tab = setup
    kw = dict(box_pc=box, distance_kpc=D_KPC, plasma_kw=Q2_EMISSION)
    v_ref, ok_ref = J.doppler_sectors(f, tab, **kw)
    S, E0 = J.doppler_moment_maps(f, tab, **kw)
    assert S.shape == (2,) + f["rho"].shape[::2] and abs(E0 - 1.86) < 1e-9
    n = f["rho"].shape[0]
    WX, NZ, R = J.sky_geometry(box, n, D_KPC)
    pa = np.rad2deg(np.arctan2(NZ, WX)) % 360.0
    sec = np.where((R > 40.0) & (R < 170.0), (pa / 15.0).astype(int) % 24, 24)
    Ssec = np.stack([[np.asarray(S[k])[sec == s].sum() for s in range(24)] for k in range(2)])
    ok = Ssec[0] > 0
    dE = np.where(ok, Ssec[1] / np.where(ok, Ssec[0], 1.0), 0.0)
    v = np.where(ok, -J.C_KMS * (dE - dE[ok].mean()) / E0, 0.0)
    np.testing.assert_array_equal(ok, np.asarray(ok_ref))
    np.testing.assert_allclose(v, np.asarray(v_ref), atol=0.5)
    if V2.exists():
        dop = J.load_binned_tables("chandra_aciss_cy0", 1.2, "dop", history=True)
        S2, E02 = J.doppler_moment_maps(f, dop, **kw)
        np.testing.assert_allclose(np.asarray(S2), np.asarray(J.doppler_columns(f, dop, **kw)), rtol=1e-6)
        assert abs(E02 - 1.86) < 1e-9
        with pytest.raises(ValueError):
            J.doppler_moment_maps(f, dop, band=(1.75, 1.95), **kw)


def test_csm_solar_tangent_at_pure_csm():
    """Pure solar CSM sits exactly on both kinks of the CSM/ejecta split (C_ej = 0
    and (1 - C_ej) X_sun == C_O): the tangent must be the one-sided derivative
    into the physical region, not jnp.clip/minimum's 0.5 x 0.5 (review 2026-09-25)."""
    x = J.solar_csm_tracers()
    cell = {"C_ej": jnp.asarray([0.0]), "C_O": jnp.asarray([x["O"]]), "C_Si": jnp.asarray([x["Si"]]),
            "C_Fe": jnp.asarray([x["Fe"]]), "C_He": jnp.asarray([x["He"]])}
    for key, h in (("C_ej", 1e-3), ("C_O", 1e-4)):
        g = lambda c: J.element_fractions(dict(cell, **{key: c}), csm_solar=True)["Ne"][0]  # noqa: E731
        jv = float(jax.jvp(g, (cell[key],), (jnp.ones(1),))[1])
        fd = (float(g(cell[key] + h)) - float(g(cell[key]))) / h
        assert abs(jv / fd - 1) < 0.02, (key, jv, fd)


@needs_v2
def test_v2_table_guards(setup):
    """Mismatched table sets raise instead of gathering with clamped indices."""
    f, box, _ = setup
    th = J.load_binned_tables("chandra_aciss_cy0", 1.2, "band", history=True)
    t0 = J.load_binned_tables("chandra_aciss_cy0", 1.2, "band", history=False)
    kw = dict(box_pc=box, distance_kpc=D_KPC)
    with pytest.raises(ValueError):
        J.band_columns(f, th, band_tables=(t0["C"], t0["D"]), **kw)
    with pytest.raises(ValueError):
        J.doppler_moment_maps(f, th, **kw)
    with pytest.raises(ValueError):
        J.band_tables_of(J.load_binned_tables("chandra_aciss_cy0", 1.2, "dop", history=False))
    Cb, Db = J.band_tables_of(t0)
    assert Cb is t0["C"] and Db is t0["D"]
