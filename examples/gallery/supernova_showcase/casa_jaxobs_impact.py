"""
How much each v2 observation-model fix moves the integrated spectrum (CPU).

Replicates casa_xfit's ``spectrum()`` (r < 200" aperture, N_H map with the
fitted gradient, synchrotron, halo aperture-keep, sub-grid split, the fitted
emission knobs) on a fixed state, once with the v1 chain -- 117 eV channel
tables at N_H 0.8-2.0 spread uniformly onto the 0.2 keV bins -- and then with
the v2 fixes switched on one at a time:

  bins     tables folded on the analysis bins through the native channels
           (the N_H map still clamped to v1's 0.8-2.0 grid)
  nh       N_H grid 0.5-4 (the fitted map is no longer clamped at 0.8 / 2.0)
  logkt    log-kT interpolation (Fe-K Boltzmann tail)
  csm      solar O:Ne:Mg for the circumstellar part of the O tracer
  hist     NEI with the T_e history (casa_jaxobs_nei)
  halo     aperture keep from the v2 halo at the local N_H
  broad    OPTIONAL, not part of the cumulative v2: first- plus second-order
           Doppler terms (LOS velocity shift + broadening). The Taylor series
           is only valid for beta E << the RMF width (|v| < ~2000 km/s at
           Fe-K); Cas A's Fe ejecta reach 5000 km/s, so this row is a warning
           of how large the velocity terms are, not a recommended model.

and prints, per 0.2 keV bin, model(step) / model(previous step) and the
cumulative v2 / v1, all median-normalised as casa_xfit.summarize normalises
the data/model shape. With ``--xfit-model`` (a casa_xfit --save-model npz) it
also prints that fit's observed data/model shape and the shape predicted after
the fixes, (data / model_v1) x (model_v1 / model_v2).

    JAX_PLATFORMS=cpu ./run.sh casa_jaxobs_impact.py \\
        --state /export/data/lstorcks/casa_orlando150/work/plH_n256_age364yr_solarcsm.npz \\
        --theta /export/data/lstorcks/casa_orlando150/work/xfit_Q2.json --stride 2 \\
        --xfit-model /export/data/lstorcks/casa_orlando150/work/xfit_Q2_n128.npz
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import _plasma as P
import casa_jaxobs as J

W = Path("/export/data/lstorcks/casa_orlando150/work")
NH_GRID_V1 = np.array([0.8, 1.0, 1.2, 1.5, 2.0])            # casa_xfit.NH_GRID
D_REF_KPC = 3.0
SPEC_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
STEPS = ("v1", "bins", "nh", "logkt", "csm", "hist", "halo", "broad")
CUMULATIVE = "halo"                  # the last step of the recommended v2 chain


def theta_params(path):
    j = json.load(open(path))
    return dict(zip(j["names"], j["theta"]))


def nh_tents(lnh_map, grid):
    lg = np.log(grid)
    return jnp.stack([jnp.interp(lnh_map, jnp.asarray(lg), jnp.asarray(np.eye(len(lg))[j]))
                      for j in range(len(lg))])


def fold(T):                        # (n_nh, ..., b) -> (..., n_nh * b), casa_xfit.fold
    return jnp.moveaxis(T, 0, -2).reshape(*T.shape[1:-1], T.shape[0] * T.shape[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", default=str(W / "plH_n256_age364yr_solarcsm.npz"))
    ap.add_argument("--theta", default=str(W / "xfit_Q2.json"))
    ap.add_argument("--stride", type=int, default=2, help="subsample the state (CPU cost)")
    ap.add_argument("--instrument", default="chandra_aciss_cy0")
    ap.add_argument("--year", type=float, default=2000.0)
    ap.add_argument("--xfit-model", default=None)
    ap.add_argument("--out", default=None, help="npz with every step's spectrum")
    ap.add_argument("--scan", action="store_true",
                    help="with --xfit-model: re-tune the cheap global knobs the fixes interact with "
                         "(mean ln N_H, Fe yield, Coulomb rate) on the v2 chain, by the shape rms")
    args = ap.parse_args()
    p = theta_params(args.theta)
    P.set_tracer_split("xrism_bulk")
    f, box, age = J.load_fields(args.state)
    s = args.stride
    f = {k: v[::s, ::s, ::s] for k, v in f.items()}
    n = f["rho"].shape[0]
    d_kpc = float(np.exp(p["ln_D"]))
    scale = D_REF_KPC / d_kpc
    WX0, NZ0, _ = J.sky_geometry(box, n, D_REF_KPC)
    ps = np.deg2rad(p["psi"])
    sky_w = scale * (np.cos(ps) * WX0 - np.sin(ps) * NZ0) + p["dw"]
    sky_n = scale * (np.sin(ps) * WX0 + np.cos(ps) * NZ0) + p["dn"]
    R_sky = jnp.asarray(np.hypot(sky_w, sky_n), jnp.float32)
    lnh_map = jnp.asarray(p["ln_nh"] + (p["g_nh_w"] * sky_w + p["g_nh_n"] * sky_n) / 100.0, jnp.float32)
    nh_map = np.exp(np.asarray(lnh_map))
    print(f"[impact] {Path(args.state).name} at stride {s} ({n}^3), theta {Path(args.theta).name}: "
          f"N_H map {nh_map.min():.2f}-{nh_map.max():.2f}e22 over the grid", flush=True)
    amp = float(np.exp(p["ln_A"]) * scale ** 2)
    sg = dict(chi=4.0, f_mass=float(1.0 / (1.0 + np.exp(-p["lg_fmass"]))))
    pkw0 = dict(kT_e_shock_keV=float(np.exp(p["ln_kte"])), teq_scale=float(np.exp(p["ln_teq"])),
                fe_scale=float(np.exp(p["ln_fe"])))
    edges = J.SPEC_EDGES
    mids = 0.5 * (edges[1:] + edges[:-1])

    # ---- v1: exactly casa_xfit.load_spectrum_data + spectrum() -------------
    def v1_spectrum():
        t0 = J.load_tables(args.instrument, 1.2)
        ch = np.asarray(t0["ch_edges"])
        sel = np.nonzero((ch[1:] > edges[0] - 0.1) & (ch[:-1] < edges[-1] + 0.1))[0]
        lo, hi = ch[sel], ch[sel + 1]
        O = np.clip(np.minimum(edges[1:, None], hi[None]) - np.maximum(edges[:-1, None], lo[None]),
                    0.0, None) / (hi - lo)[None]
        C = jnp.asarray(np.stack([np.asarray(J.load_tables(args.instrument, nh)["C"])[..., sel]
                                  for nh in NH_GRID_V1]))
        Sd = np.load(J.TABLE_DIR / f"sync_{args.instrument}.npz")
        S = jnp.asarray(Sd["S"][..., sel])
        lecut = np.log(Sd["ecut"])
        halo = J.load_halo(1.2)
        e_mid = 0.5 * (lo + hi)
        keep = np.stack([np.interp(e_mid, halo["e_grid"], halo["aperture_keep"][:, k])
                         for k in range(len(halo["r_grid"]))], 1)
        w_nh = nh_tents(lnh_map, NH_GRID_V1)
        cc = J.band_columns(f, t0, box_pc=box, distance_kpc=D_REF_KPC, subgrid=sg,
                            band_tables=(fold(C), fold(C)), plasma_kw=pkw0, v_los_kms=False)
        sc = J.sync_columns(f, lecut=lecut, band_table=fold(S), year=args.year, plasma_kw=pkw0)
        tot = amp * cc + np.exp(p["ln_sync"]) * sc
        tot = jnp.einsum("jhxz,jxz->hxz", tot.reshape(len(NH_GRID_V1), len(sel), n, n), w_nh)
        kp = jax.vmap(lambda kr: jnp.interp(R_sky, jnp.asarray(halo["r_grid"], jnp.float32), kr,
                                            right=0.0))(jnp.asarray(keep, jnp.float32))
        return np.asarray(jnp.asarray(O, jnp.float32) @ jnp.sum(tot * kp, (1, 2)))

    # ---- v2 --------------------------------------------------------------
    def v2_spectrum(*, history, kt_interp, csm, halo_v2, broad, nh_clamp, dlnh=0.0, fe_mult=1.0, teq_mult=1.0):
        st = J.load_binned_stack(args.instrument, "spec", history=history, second_order=broad)
        tabs = J.load_binned_tables(args.instrument, 1.2, "spec", history=history, stack=st)
        nh = st["nh"]
        lm = jnp.clip(lnh_map, np.log(NH_GRID_V1[0]), np.log(NH_GRID_V1[-1])) if nh_clamp else lnh_map + dlnh
        w_nh = nh_tents(lm, nh)
        pkw = dict(pkw0, csm_solar=csm, fe_scale=pkw0["fe_scale"] * fe_mult,
                   teq_scale=pkw0["teq_scale"] * teq_mult)
        bt = tuple(fold(jnp.asarray(st[k])) for k in (("C", "D", "D2") if broad else ("C", "C")))
        cc = J.band_columns(f, tabs, box_pc=box, distance_kpc=D_REF_KPC, subgrid=sg, band_tables=bt,
                            plasma_kw=pkw, v_los_kms=broad, kt_interp=kt_interp)
        lecut, S, _ = J.load_sync_binned(args.instrument, "spec")
        sc = J.sync_columns(f, lecut=lecut, band_table=fold(jnp.asarray(S)), year=args.year, plasma_kw=pkw)
        tot = (amp * cc + np.exp(p["ln_sync"]) * sc).reshape(len(nh), len(mids), n, n)
        if halo_v2:
            hs = J.load_halo_stack(args.instrument)
            keep = np.stack([np.stack([np.interp(mids, hs["e_grid"], hs["aperture_keep"][a, :, k])
                                       for k in range(len(hs["r_grid"]))], 1) for a in range(len(nh))])
            rg = jnp.asarray(hs["r_grid"], jnp.float32)
            kp = jnp.stack([jax.vmap(lambda kr: jnp.interp(R_sky, rg, kr, right=0.0))(
                jnp.asarray(keep[a], jnp.float32)) for a in range(len(nh))])      # (nh, bin, x, z)
            return np.asarray(jnp.einsum("jhxz,jxz->h", tot * kp, w_nh))
        halo = J.load_halo(1.2)
        keep = np.stack([np.interp(mids, halo["e_grid"], halo["aperture_keep"][:, k])
                         for k in range(len(halo["r_grid"]))], 1)
        kp = jax.vmap(lambda kr: jnp.interp(R_sky, jnp.asarray(halo["r_grid"], jnp.float32), kr,
                                            right=0.0))(jnp.asarray(keep, jnp.float32))
        return np.asarray(jnp.sum(jnp.einsum("jhxz,jxz->hxz", tot, w_nh) * kp, (1, 2)))

    cfg = dict(history=False, kt_interp="linear", csm=False, halo_v2=False, broad=False, nh_clamp=True)
    spectra = {}
    for step in STEPS:
        t0 = time.time()
        if step == "v1":
            spectra[step] = v1_spectrum()
        else:
            cfg.update({"bins": {}, "nh": dict(nh_clamp=False), "logkt": dict(kt_interp="log"), "csm": dict(csm=True),
                        "hist": dict(history=True), "halo": dict(halo_v2=True),
                        "broad": dict(broad=True)}[step])
            spectra[step] = v2_spectrum(**cfg)
        print(f"[impact] {step}: {time.time() - t0:.0f} s, total {spectra[step].sum():.2f} counts/s", flush=True)

    def norm(x):
        return x / np.median(x)

    show = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.2, 3.8, 4.4, 5.0, 6.0, 6.4, 6.6, 6.8]
    idx = [int(np.argmin(np.abs(mids - m))) for m in show]
    print("\n[impact] model(step) / model(previous step), median-normalised, per bin centre (keV):")
    print("        " + " ".join(f"{m:5.1f}" for m in show))
    prev = None
    for step in STEPS:
        if prev is not None:
            r = norm(spectra[step]) / norm(spectra[prev])
            print(f"{step:7s} " + " ".join(f"{r[i]:5.3f}" for i in idx))
        prev = step
    tot = norm(spectra[CUMULATIVE]) / norm(spectra["v1"])
    print(f"{'v2/v1':7s} " + " ".join(f"{tot[i]:5.3f}" for i in idx))
    if args.xfit_model:
        d = np.load(args.xfit_model)
        ep = [str(e) for e in d["epochs"]]
        rat = []
        for e, label in enumerate(ep):
            sp = SPEC_DIR / f"epoch_{label}_spectrum.npz"
            if not sp.exists():
                continue
            q = np.load(sp)
            eb = np.round(np.asarray(q["ebins"]), 3)
            ix = np.searchsorted(eb, edges)
            cs = np.concatenate([[0.0], np.cumsum(q["counts"])])
            cnt = cs[ix[1:]] - cs[ix[:-1]]
            lam = np.asarray(d["spectra"][e]) * float(q["exposure"])
            r = cnt / np.maximum(lam, 1e-30)
            rat.append(r / np.median(r))
        rat = np.mean(rat, 0)
        print(f"\n[impact] {Path(args.xfit_model).name}: data/model shape (epoch mean, as casa_xfit.summarize)")
        print(f"{'fit':7s} " + " ".join(f"{rat[i]:5.2f}" for i in idx))
        pred = norm(rat / tot)
        print(f"{'fixed':7s} " + " ".join(f"{pred[i]:5.2f}" for i in idx) +
              "   (data / model_v2 = (data / model_v1) x (model_v1 / model_v2))")
        for step in STEPS[1:]:
            c = norm(spectra[step]) / norm(spectra["v1"])
            pr = norm(rat / c)
            print(f"  up to {step:6s}" + " ".join(f"{pr[i]:5.2f}" for i in idx))

        def rms(x):
            return float(np.sqrt(np.mean(np.log(x) ** 2)))
        print(f"[impact] shape rms (ln, all {len(mids)} bins): Q2 fit {rms(rat):.3f}; "
              f"v2 at fixed theta {rms(norm(rat / tot)):.3f}")
        if args.scan:
            full = dict(cfg, broad=False, halo_v2=True, history=True, csm=True, kt_interp="log", nh_clamp=False)
            best = None
            for dl in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
                for fe in (1.0, 1.3):
                    for tq in (1.0, 2.0):
                        m2 = v2_spectrum(**full, dlnh=dl, fe_mult=fe, teq_mult=tq)
                        pr = norm(rat / (norm(m2) / norm(spectra["v1"])))
                        r = rms(pr)
                        print(f"[scan] d ln N_H {dl:+.1f}, Fe x{fe:.1f}, Coulomb x{tq:.0f}: shape rms {r:.3f}", flush=True)
                        if best is None or r < best[0]:
                            best = (r, dl, fe, tq, pr)
            r, dl, fe, tq, pr = best
            print(f"[scan] best: d ln N_H {dl:+.1f} (N_H x{np.exp(dl):.2f}), Fe x{fe:.1f}, Coulomb x{tq:.0f}: rms {r:.3f}")
            print(f"{'retuned':7s} " + " ".join(f"{pr[i]:5.2f}" for i in idx))
    if args.out:
        np.savez(args.out, mids=mids, **spectra)
        print(f"[impact] wrote {args.out}")


if __name__ == "__main__":
    main()
