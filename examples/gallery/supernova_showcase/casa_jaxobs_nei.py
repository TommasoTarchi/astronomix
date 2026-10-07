"""
History-aware NEI ion fractions for the differentiable observation model.

``_nei`` tabulates the ion fractions of a parcel that was ionised from neutral
at ONE electron temperature, and the observation model reads that table at the
parcel's CURRENT ``T_e``. Behind a collisionless shock ``T_e`` is not constant:
it starts at ``kT_e0`` (Ghavamian) and rises by Coulomb collisions
(``_plasma.electron_ion_temperatures``). Ionising the whole ``n_e t`` at the
final, highest ``T_e`` over-ionises (audit 2026-09-25, obs_model section 7,
forward_physics F5): at Q2's ``kT_e0 = 0.11 keV`` and 0.36x Spitzer the
He-like Si fraction of Si-layer ejecta at ``n_e t ~ 3e11`` is x2 too low.

The relaxation track has a universal shape. With ``t_eq ~ T_e^{3/2}`` and
``T_e`` well above ``(m_e/m_i) T_i``, ``y = T_e / T_mean`` obeys
``dy/ds = (1 - y) / y^{3/2}`` in ``s = t / t_relax(T_mean)``, and from a cold
start (``y0 -> 0``) every track is one curve ``y(s)``. A parcel observed at
``y_now`` has had ``T_e(u) = T_now y(u s_now) / y_now`` for ``u = t'/t`` in
(0, 1]; at constant density ``u`` is also the fraction of ``n_e t``. The
one-parameter family is labelled here by the n_e t-weighted mean electron
temperature in units of the current one,

    rho = <T_e>_{n_e t} / T_e(now)  in  [5/7, 1],

(5/7: the early-time power law ``T_e ~ t^{2/5}``; 1: constant ``T_e``, i.e. the
legacy ``_nei`` table). ``rho`` rather than ``y_now`` because it is robust off
the family: a parcel still near its post-shock ``kT_e0`` has a nearly constant
history, ``rho ~ 1``, whatever ``y_now`` says. The forward model computes each
cell's ``rho`` from the relaxation scan it already runs (``casa_jaxobs.
_relax_history``), exactly as in :func:`rho_of_track`.

The table ``f[el]`` has shape ``(n_kT, n_rho, n_net, Z + 1)`` on ``_nei``'s
kT and n_e t grids and :data:`RHO_GRID`. Its ``rho = 1`` slice IS the legacy
``_nei`` table (copied, not recomputed), so the history axis can only change
parcels whose electrons actually heated.

Validation (:func:`validate`): parcels (Si-layer ejecta, O-layer ejecta, solar
CSM) at several n_e, ages and (kT_e0, Spitzer rate) are ionised step by step
along the track ``_plasma.electron_ion_temperatures`` gives them (400 substeps,
AtomDB eigen propagator), and compared with (A) the current model (constant
current T_e), (B) an effective ionisation temperature ``rho T_now`` at constant
T, and (C) this table at the parcel's own ``rho``.

Assumptions kept from the model: constant density along the history (the
relaxation is exact for an adiabatic parcel, the n_e t weighting is not:
ejecta that were denser in the past accumulated more of their n_e t early, when
T_e was lower -- so the correction here is, if anything, an underestimate);
single-shock history from neutral.

    ~/.local/share/mamba/envs/astx/bin/python casa_jaxobs_nei.py build
    ~/.local/share/mamba/envs/astx/bin/python casa_jaxobs_nei.py validate
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

import _nei

TABLE_PATH = Path("/export/data/lstorcks/casa_orlando150/jaxobs/nei_history_fractions.npz")
EIGEN_DIR = Path(os.environ.get("ATOMDB", "/export/data/lstorcks/atomdb")) / "APED" / "ionbal" / "eigen"
#: history axis, n_e t-weighted <T_e> / T_e(now); dense at the low end (where the
#: cold-start power law puts every parcel with y_now < 0.5)
RHO_MIN = 5.0 / 7.0
RHO_GRID = np.array([RHO_MIN, 0.74, 0.77, 0.80, 0.85, 0.90, 0.95, 1.0])
#: u = t' / t substeps along a track (geometric; the first 1e-5 carries no n_e t)
N_TRACK_STEPS = 200
KT_EIGEN = _nei.EIGEN_TE_GRID * _nei.KBOLTZ_KEV_PER_K


# =============================================================================
# ============ ↓ AtomDB eigen propagator (vectorised) ↓ =======================
# =============================================================================
_EIG = {}


def eigen_data(el):
    """(FEQB, EIG, VR, VL) of element ``el`` on AtomDB's 1251-point T grid."""
    if el not in _EIG:
        from astropy.io import fits
        Z = _nei.ELEMENTS[el]
        with fits.open(EIGEN_DIR / f"eigen{el.lower()}_v3.1.0.fits") as f:
            d = f[1].data
            _EIG[el] = (np.asarray(d["FEQB"], np.float64), np.asarray(d["EIG"], np.float64),
                        np.asarray(d["VR"], np.float64).reshape(-1, Z, Z),
                        np.asarray(d["VL"], np.float64).reshape(-1, Z, Z))
    return _EIG[el]


def propagate(el, pop, kT_keV, dtau):
    """One constant-T_e step of ``dn/d(n_e t) = A(T_e) n`` for a batch.

    ``pop`` (B, Z + 1), ``kT_keV`` scalar (nearest of AtomDB's grid, as
    ``_nei._eigen_solution``), ``dtau`` (B,) in cm^-3 s. Same clipping and
    renormalisation as ``_nei._eigen_solution``.
    """
    feqb, eig, vr, vl = eigen_data(el)
    k = int(np.argmin(np.abs(KT_EIGEN - kT_keV)))
    work = pop[:, 1:] - feqb[k][None, 1:]
    f = work @ vl[k].T                                               # (B, Z)
    decay = np.exp(np.clip(np.asarray(dtau, np.float64)[:, None] * eig[k][None], -700.0, 700.0))
    out = np.zeros_like(pop)
    out[:, 1:] = (f * decay) @ vr[k] + feqb[k][None, 1:]
    np.clip(out, 0.0, None, out=out)
    s = out[:, 1:].sum(1, keepdims=True)
    out[:, 1:] = np.where(s > 1.0, out[:, 1:] / np.maximum(s, 1e-300), out[:, 1:])
    out[:, 0] = np.clip(1.0 - out[:, 1:].sum(1), 0.0, None)
    return out


def ionise_along(el, kT_steps, du, net):
    """Ion fractions after the piecewise-constant history ``kT_steps`` (keV) with
    n_e t fractions ``du`` (sum 1), for every total ``net`` (array): (n_net, Z + 1)."""
    Z = _nei.ELEMENTS[el]
    net = np.atleast_1d(np.asarray(net, np.float64))
    pop = np.zeros((len(net), Z + 1)); pop[:, 0] = 1.0
    for kT, d in zip(kT_steps, du):
        if d > 0:
            pop = propagate(el, pop, float(kT), net * d)
    return pop
# =============================================================================
# ============ ↑ AtomDB eigen propagator ↑ ====================================
# =============================================================================


# =============================================================================
# ============ ↓ The universal relaxation family ↓ ============================
# =============================================================================
def _family_grid():
    """y, s(y) = int_0^y t^1.5/(1-t) dt and rho(y) on a fine grid in y in (0, 1)."""
    y = np.concatenate([np.geomspace(1e-8, 0.5, 6000), 1.0 - np.geomspace(0.5, 1e-12, 6000)[1:]])
    f1, f2 = y ** 1.5 / (1.0 - y), y ** 2.5 / (1.0 - y)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (f1[1:] + f1[:-1]) * np.diff(y))]) + y[0] ** 2.5 / 2.5
    i2 = np.concatenate([[0.0], np.cumsum(0.5 * (f2[1:] + f2[:-1]) * np.diff(y))]) + y[0] ** 3.5 / 3.5
    return y, s, i2 / (y * s)


_FAM = None


def family_track(rho, u_edges):
    """``T_e(u) / T_e(now)`` at the midpoints (in log u) of ``u_edges`` for the
    family member with n_e t-weighted mean ``rho`` (clipped to [5/7, 1])."""
    global _FAM
    if _FAM is None:
        _FAM = _family_grid()
    y, s, r = _FAM
    um = np.sqrt(np.maximum(u_edges[:-1], 1e-30) * u_edges[1:])
    um[0] = 0.5 * u_edges[1] if u_edges[0] == 0.0 else um[0]
    if rho >= 1.0 - 1e-9:
        return np.ones_like(um)
    # the cold-start limit: at y_now -> 0 the track is the power law u^0.4, and
    # y_now = 1e-3 is it to 1e-4 while keeping s(u) resolved on the grid (at the
    # grid's first point, 1e-8, the whole track collapsed onto one y value and
    # the rho = 5/7 node silently became a CONSTANT-T_e track)
    r_min = float(np.interp(1e-3, y, r))
    rho = max(float(rho), r_min)
    y_now = float(np.interp(rho, r, y))                  # rho(y) is monotone
    s_now = float(np.interp(y_now, y, s))
    return np.interp(um * s_now, s, y) / y_now


def track_u_edges(n=N_TRACK_STEPS):
    return np.concatenate([[0.0], np.geomspace(1e-5, 1.0, n)])


def rho_of_track(Te, u):
    """n_e t-weighted (= time-weighted at constant density) mean of ``Te`` sampled at
    ``u`` (from 0 to 1), over its final value: the trapezoid rule, as the forward
    model's relaxation scan does."""
    Te, u = np.asarray(Te, np.float64), np.asarray(u, np.float64)
    return np.sum(0.5 * (Te[1:] + Te[:-1]) * np.diff(u), axis=-1) / (u[-1] - u[0]) / Te[..., -1]
# =============================================================================
# ============ ↑ The universal relaxation family ↑ ============================
# =============================================================================


# =============================================================================
# ============ ↓ The table ↓ ==================================================
# =============================================================================
def build_table(path=TABLE_PATH, rho_grid=RHO_GRID, n_steps=N_TRACK_STEPS):
    """f[el] (n_kT, n_rho, n_net, Z + 1); the rho = 1 slice is ``_nei``'s table."""
    kt, net, legacy = _nei.load_table()
    u = track_u_edges(n_steps)
    du = np.diff(u)
    out = {}
    t0 = time.time()
    for el, Z in _nei.ELEMENTS.items():
        f = np.zeros((len(kt), len(rho_grid), len(net), Z + 1))
        for b, rho in enumerate(rho_grid):
            if rho >= 1.0 - 1e-9:
                f[:, b] = legacy[el]
                continue
            shape = family_track(rho, u)
            for i, kT in enumerate(kt):
                f[:, b][i] = ionise_along(el, kT * shape, du, net)
        out[el] = f
        print(f"[nei-hist] {el:2s} ({time.time() - t0:.0f} s)", flush=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, kt=kt, net=net, rho=np.asarray(rho_grid), n_track_steps=n_steps,
                        elements=np.array(sorted(_nei.ELEMENTS)),
                        **{f"f_{el}": v.astype(np.float32) for el, v in out.items()})
    print(f"[nei-hist] wrote {path}")
    return out


def load_table(path=TABLE_PATH):
    """``(kt, rho, net, {el: f (n_kT, n_rho, n_net, Z + 1)})``."""
    d = np.load(path)
    return (np.asarray(d["kt"]), np.asarray(d["rho"]), np.asarray(d["net"]),
            {k[2:]: np.asarray(d[k], np.float64) for k in d.files if k.startswith("f_")})


def interpolate(f, kt, rho_grid, net, kT_e, rho, tau):
    """Trilinear in (log kT, rho, log n_e t), clipped; returns (Z + 1,) per scalar input."""
    def ax(g, v, log):
        gg = np.log(g) if log else g
        x = np.clip(np.log(v) if log else v, gg[0], gg[-1])
        i = int(np.clip(np.searchsorted(gg, x) - 1, 0, len(gg) - 2))
        return i, (x - gg[i]) / (gg[i + 1] - gg[i])
    i, fi = ax(kt, kT_e, True)
    b, fb = ax(rho_grid, rho, False)
    j, fj = ax(net, tau, True)
    out = 0.0
    for di, wi in ((0, 1 - fi), (1, fi)):
        for db, wb in ((0, 1 - fb), (1, fb)):
            for dj, wj in ((0, 1 - fj), (1, fj)):
                out = out + wi * wb * wj * f[i + di, b + db, j + dj]
    return out
# =============================================================================
# ============ ↑ The table ↑ ==================================================
# =============================================================================


# =============================================================================
# ============ ↓ Validation along the _plasma relaxation ↓ ====================
# =============================================================================
KB_KEV = 8.617333e-8
YR_S = 3.155693e7
#: parcels: label, mass fractions, single-fluid kT (keV)
PARCELS = (
    ("Si-layer ejecta", {"Si": 0.57, "S": 0.30, "Ar": 0.08, "Ca": 0.05}, 12.0),
    ("O-layer ejecta", {"O": 0.90, "Ne": 0.02, "Mg": 0.03, "Si": 0.05}, 12.0),
    ("Fe-rich ejecta", {"Fe": 0.80, "Si": 0.15, "S": 0.05}, 20.0),
    ("solar CSM", {"H": 0.70, "He": 0.28, "O": 0.0096, "Ne": 0.0017, "Mg": 0.0007, "Si": 0.001,
                   "S": 0.0005, "Fe": 0.0013}, 3.5),
)
#: the ions whose change the report quotes: (element, charge, label)
IONS = (("O", 7, "O VIII"), ("Si", 12, "Si XIII (He)"), ("Si", 13, "Si XIV (H)"),
        ("S", 14, "S XV (He)"), ("Ar", 16, "Ar XVII (He)"), ("Fe", 24, "Fe XXV (He)"))


def plasma_track(X, n_e, T_kev, age_yr, kte0, teq, n_u=400):
    """T_e(u) (keV) along ``_plasma.electron_ion_temperatures`` at ``u`` in [0, 1],
    and the rho the forward model would compute (its own 48 geometric substeps)."""
    import _plasma as P
    m = P.composition_moments(X)
    rho_code = n_e * m["mu_e"] * P.M_P / P.CODE_DENSITY
    u = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, n_u)])
    tc = u * age_yr * YR_S / P.CODE_TIME * teq          # teq_scale == a rescaled clock
    Te, _ = P.electron_ion_temperatures(np.full_like(tc, T_kev / KB_KEV), np.full_like(tc, rho_code),
                                        tc, X, kT_e_shock_keV=kte0)
    Te = Te * KB_KEV
    # the forward model's rho: the 48 geometric substep edges of the relaxation scan
    ue = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, 48)])
    Te_e = np.interp(ue, u, Te)
    return u, Te, float(rho_of_track(Te_e, ue))


def validate(out_path=None, table_path=TABLE_PATH):
    kt, rg, net_g, tab = load_table(table_path)
    _, _, legacy = _nei.load_table()
    rows = []
    for lab, X, kTm in PARCELS:
        for n_e, age in ((3.0, 100.0), (10.0, 100.0), (30.0, 100.0), (30.0, 250.0), (100.0, 100.0)):
            for kte0, teq, tag in ((0.11, 0.36, "Q2"), (0.3, 1.0, "Ghav+Spitzer")):
                u, Te, rho = plasma_track(X, n_e, kTm, age, kte0, teq)
                tau = n_e * age * YR_S
                Tnow = float(Te[-1])
                du = np.diff(u)
                Tm = 0.5 * (Te[1:] + Te[:-1])
                line = dict(parcel=lab, n_e=n_e, age=age, tag=tag, net=tau, kTe=Tnow, rho=rho)
                for el, q, name in IONS:
                    if el not in X and not (el == "O" and "O" in X):
                        continue
                    direct = ionise_along(el, Tm, du, [tau])[0]
                    A = _nei.interpolate_fractions(legacy[el], kt, net_g, np.array([Tnow]), np.array([tau]))[:, 0]
                    B = _nei.interpolate_fractions(legacy[el], kt, net_g, np.array([rho * Tnow]),
                                                   np.array([tau]))[:, 0]
                    C = interpolate(tab[el], kt, rg, net_g, Tnow, rho, tau)
                    line[name] = (direct[q], A[q], B[q], C[q])
                rows.append(line)
    hdr = f"{'parcel':16s} {'n_e':>5s} {'t':>4s} {'model':12s} {'n_e t':>8s} {'kTe':>5s} {'rho':>5s}"
    print(hdr + "  ion: direct | A const-now | B const(rho T) | C table   (ratios to direct)")
    for r in rows:
        s = (f"{r['parcel']:16s} {r['n_e']:5.0f} {r['age']:4.0f} {r['tag']:12s} {r['net']:8.1e} "
             f"{r['kTe']:5.2f} {r['rho']:5.3f}")
        for _, _, name in IONS:
            if name in r:
                d, a, b, c = r[name]
                if max(d, a, c) < 3e-3:
                    continue
                s += (f"\n      {name:13s} {d:.3f} | {a:.3f} (x{a / max(d, 1e-9):.2f}) | "
                      f"{b:.3f} (x{b / max(d, 1e-9):.2f}) | {c:.3f} (x{c / max(d, 1e-9):.2f})")
        print(s)
    # summary: |ln(model/direct)| over entries with fraction > 0.02
    for k, lab in ((1, "A: constant current T_e (legacy)"), (2, "B: constant rho T_e"),
                   (3, "C: history table")):
        errs = [abs(np.log(max(r[n][k], 1e-9) / r[n][0])) for r in rows for _, _, n in IONS
                if n in r and r[n][0] > 0.02]
        print(f"[validate] {lab:36s}: median |ln ratio| {np.median(errs):.3f}, "
              f"90th pct {np.percentile(errs, 90):.3f}, max {np.max(errs):.3f} over {len(errs)} ion fractions > 0.02")
    if out_path:
        np.save(out_path, np.array(rows, dtype=object), allow_pickle=True)
    return rows
# =============================================================================
# ============ ↑ Validation ↑ =================================================
# =============================================================================


def state_impact(state, theta_json=None, stride=2, table_path=TABLE_PATH):
    """Emission-weighted ion fractions of a real state, legacy vs history table.

    Runs ``casa_jaxobs.plasma`` (JAX, CPU) with a fit's emission knobs (kT_e0,
    Coulomb rate, Fe scale; default Q2's) on the shocked cells, and weights
    each ion's fraction by its element's emission measure n_e n_el V (the
    line emissivity is ~ that x the ion fraction x a T_e factor both tables
    share), over the ejecta phases of the ``_subgrid`` split as casa_xfit does.
    """
    import json
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp
    import casa_jaxobs as J
    import _plasma as P
    P.set_tracer_split("xrism_bulk")
    th = {"ln_kte": -2.224, "ln_teq": -1.0262, "ln_fe": -0.1203, "lg_fmass": -1.7125}
    if theta_json:
        j = json.load(open(theta_json))
        th.update(dict(zip(j["names"], j["theta"])))
    pkw = dict(kT_e_shock_keV=float(np.exp(th["ln_kte"])), teq_scale=float(np.exp(th["ln_teq"])),
               fe_scale=float(np.exp(th["ln_fe"])))
    f, box, age = J.load_fields(state)
    f = {k: v[::stride, ::stride, ::stride] for k, v in f.items()}
    kt, rg, net_g, tab = load_table(table_path)
    _, _, leg = _nei.load_table()
    fm = float(1.0 / (1.0 + np.exp(-th["lg_fmass"])))
    sums = {}
    for fc, ew in J.emitting_components(f, chi=4.0, f_mass=fm):
        pl = J.plasma(fc, **pkw)
        ok = np.asarray(pl["shocked"] & (pl["kT"] >= J.KT_MIN_KEV))
        kT = np.asarray(pl["kT"])[ok]; tau = np.asarray(pl["net"])[ok]; rh = np.asarray(pl["rho_hist"])[ok]
        wgt = (np.asarray(pl["n_e"]) * np.asarray(ew if np.ndim(ew) else np.full(ok.shape, ew)))[ok]
        pop = "1-C_ej" if fc is f else "ejecta"         # component 0 = the unsplit (1 - C_ej) part
        for el, q, name in IONS:
            if el not in pl["dens"]:
                continue
            w = wgt * np.asarray(pl["dens"][el])[ok]
            old = _nei.interpolate_fractions(leg[el], kt, net_g, kT, tau)[q]
            new = _interp_many(tab[el], kt, rg, net_g, kT, rh, tau)[..., q]
            key = (pop, name)
            a = sums.setdefault(key, np.zeros(3))
            a += (np.sum(w * old), np.sum(w * new), np.sum(w))
        if fc is f:
            rho_csm = rh
    print(f"[state] {Path(state).name} at stride {stride} (age {age:.0f} yr), kT_e0 {pkw['kT_e_shock_keV']:.3f} "
          f"keV, Coulomb x{pkw['teq_scale']:.3f}; rho_hist of the CSM component: median "
          f"{np.median(rho_csm):.3f} (16-84 %: {np.percentile(rho_csm, 16):.3f}-{np.percentile(rho_csm, 84):.3f})")
    for (pop, name), (o, nw, w) in sorted(sums.items()):
        if w > 0 and o > 0:
            print(f"    {pop:6s} {name:13s}: EM-weighted fraction {o / w:.3f} (legacy) -> {nw / w:.3f} "
                  f"(history), x{nw / o:.2f}")
    return sums


def _interp_many(f, kt, rho_grid, net, kT, rho, tau):
    """Vectorised trilinear interpolation (log kT, rho, log n_e t): (n, Z + 1)."""
    def ax(g, v, log):
        gg = np.log(g) if log else np.asarray(g)
        x = np.clip(np.log(np.maximum(v, 1e-30)) if log else v, gg[0], gg[-1])
        i = np.clip(np.searchsorted(gg, x) - 1, 0, len(gg) - 2)
        return i, ((x - gg[i]) / (gg[i + 1] - gg[i]))[:, None]
    i, fi = ax(kt, kT, True)
    b, fb = ax(rho_grid, rho, False)
    j, fj = ax(net, tau, True)
    out = 0.0
    for di, wi in ((0, 1 - fi), (1, fi)):
        for db, wb in ((0, 1 - fb), (1, fb)):
            for dj, wj in ((0, 1 - fj), (1, fj)):
                out = out + wi * wb * wj * f[i + di, b + db, j + dj]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "validate", "state"))
    ap.add_argument("--state", default="/export/data/lstorcks/casa_orlando150/work/plH_n256_age364yr_solarcsm.npz")
    ap.add_argument("--theta", default=None, help="state: an xfit json with the emission knobs (default Q2's)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default=str(TABLE_PATH))
    ap.add_argument("--steps", type=int, default=N_TRACK_STEPS)
    ap.add_argument("--save", default=None, help="validate: save the rows (npy)")
    args = ap.parse_args()
    if args.cmd == "build":
        if Path(args.out).exists():
            raise SystemExit(f"{args.out} exists; refusing to overwrite (pass a new --out)")
        build_table(Path(args.out), n_steps=args.steps)
    elif args.cmd == "state":
        state_impact(args.state, args.theta, args.stride, Path(args.out))
    else:
        validate(args.save, Path(args.out))


if __name__ == "__main__":
    main()
