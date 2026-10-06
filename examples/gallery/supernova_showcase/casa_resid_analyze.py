"""
Offline analysis of ``casa_resid_dump.py`` outputs (numpy only, CPU): what the
residuals that no fit removes are made of, and what minimal model changes buy.

* closure (components vs the fit's own images / spectra),
* component shares (thermal CSM / thermal ejecta / synchrotron FS / RS) per
  band and region, and their secular trends against the data's (image regions
  about the expansion centre; the r < 200" spectrum),
* the synchrotron's time-dependence bookkeeping (radio anchor, fresh weight,
  cutoff),
* chi2 scans of the spectral terms (casa_xfit's spectrum_static /
  spectrum_temporal, exactly) under: a secular time law per component, a split
  kT_e0 / equilibration / n_e t per component (the dump's variants), the
  synchrotron cutoff scale eta, a free Fe line scale.

    python casa_resid_analyze.py $W/stage3/physics/dump/dump_V3.npz [--json out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

SPEC_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
DATA = Path("/export/data/lstorcks/casa_orlando150/jaxobs/data")
EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)
SIG_T, SIG_S = 0.046, 0.13               # casa_4dvar.likelihood_args
COE = (-13.8, -4.2)
CCO = (-(350.866417 - 350.8583) * np.cos(np.deg2rad(58.8149)) * 3600, (58.811778 - 58.8149) * 3600)
BAND_NAMES = ("0.5-1.5", "1.5-2.1", "2.1-2.8", "2.8-4.2", "4.2-6", "6-7")
COMPS = ("th_csm", "th_ej", "sy_csm", "sy_ej")
GROUPS = {"soft 0.7-1.1": (0.7, 1.1), "Si 1.7-2.1": (1.7, 2.1), "S 2.3-2.7": (2.3, 2.7),
          "hard 4.1-6.1": (4.1, 6.1), "FeK 6.5-6.9": (6.5, 6.9)}


# =============================================================================
# ============ ↓ Data and the spectral likelihood (casa_xfit replica) ↓ =======
# =============================================================================
def spectrum_data(epochs):
    C, X, has, acisi = [], [], [], []
    for e in epochs:
        f = SPEC_DIR / f"epoch_{e}_spectrum.npz"
        if not f.exists():
            C.append(np.zeros(31)); X.append(1.0); has.append(False); acisi.append(False); continue
        d = np.load(f)
        eb = np.round(np.asarray(d["ebins"]), 3)
        idx = np.searchsorted(eb, EDGES)
        cs = np.concatenate([[0.0], np.cumsum(d["counts"])])
        C.append(cs[idx[1:]] - cs[idx[:-1]]); X.append(float(d["exposure"])); has.append(True)
        acisi.append(e == "2022")
    n, expo, has, acisi = np.array(C), np.array(X), np.array(has), np.array(acisi)
    use = has[:, None] & (n > 20) & ~(acisi[:, None] & (EDGES[1:] <= 1.5 + 1e-6)[None])
    return n, expo, use


def spec_chi2(model_spec, n, expo, use, detail=False):
    """(temporal chi2, static chi2[, residual matrix]) of casa_xfit.residual_parts."""
    lam = model_spec * expo[:, None]
    sig2 = SIG_T ** 2 + 1.0 / np.maximum(n, 1.0)
    d = np.where(use, np.log(np.maximum(n, 1.0)) - np.log(np.maximum(lam, 1e-30)), 0.0)
    w = np.where(use, 1.0 / sig2, 0.0)
    a = (w * d).sum(1) / np.maximum(w.sum(1), 1e-30)
    r = np.where(use, d - a[:, None], 0.0)
    static = r.sum(0) / np.maximum(use.sum(0), 1)
    rt = np.where(use, (r - static[None]) / np.sqrt(sig2), 0.0)
    rs = np.where(use.sum(0) > 0, static / SIG_S, 0.0)
    out = (float((rt ** 2).sum()), float((rs ** 2).sum()))
    return out + ((rt, static),) if detail else out


def trend_profile(model_spec, n, expo, use, years):
    """Per-bin weighted linear trend (%/yr) of the temporal residual, and the
    chi2 it carries (the most a smooth secular term can remove)."""
    _, _, (rt, _) = spec_chi2(model_spec, n, expo, use, detail=True)
    sig = np.sqrt(SIG_T ** 2 + 1.0 / np.maximum(n, 1.0))
    dr = rt * sig
    t = np.asarray(years) - 2010.0
    tr, rem = np.zeros(31), 0.0
    for b in range(31):
        u = use[:, b]
        if u.sum() < 3:
            continue
        w = 1.0 / sig[u, b] ** 2
        tt = t[u] - np.sum(w * t[u]) / w.sum()
        s = np.sum(w * tt * dr[u, b]) / np.sum(w * tt ** 2)
        tr[b] = 100 * s; rem += s ** 2 * np.sum(w * tt ** 2)
    return tr, rem


def load_images(epochs):
    dat, expo, pm = [], [], []
    for e in epochs:
        d = np.load(DATA / f"bands_{e}.npz")
        dat.append(np.asarray(d["counts"], np.float64)); expo.append(float(d["exposure"]))
        npix, pix = d["counts"].shape[-1], float(d["pixel_arcsec"])
        ax = (np.arange(npix) - 0.5 * (npix - 1)) * pix
        NN, WW = np.meshgrid(ax, ax, indexing="ij")
        pm.append((np.asarray(d["edge"]) <= 0.02) & (np.hypot(WW - CCO[0], NN - CCO[1]) >= 6.0))
    w, nn = WW - COE[0], NN - COE[1]
    return np.stack(dat), np.array(expo), np.stack(pm), np.hypot(w, nn), np.degrees(np.arctan2(-w, nn)) % 360
def particle_background(epochs, model_images, bkg_json):
    """Per-epoch particle background from ``bkg_json`` (off-remnant annulus
    215-245" SB per band, stage-3 background.py) minus the model's own halo / PSF
    there; flat per keV below 2.8 keV at the 2.8-4.2 keV level. Returns
    (image (E, 6, 256, 256) counts/s per pixel, spectrum (E, 31) counts/s in the
    r < 200" aperture)."""
    bk = json.loads(Path(bkg_json).read_text())
    bw = np.array([1.0, 0.6, 0.7, 1.4, 1.8, 1.0])
    bands = [(0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0)]
    pix, n = 1.968, model_images.shape[-1]
    ax = (np.arange(n) - 0.5 * (n - 1)) * pix
    NN, WW = np.meshgrid(ax, ax, indexing="ij")
    ann = (np.hypot(WW - COE[0], NN - COE[1]) >= 215) & (np.hypot(WW - COE[0], NN - COE[1]) < 245)
    sbp = []
    for k, e in enumerate(epochs):
        p = np.maximum(np.array(bk[e]["sb"]) - model_images[k][:, ann].mean(1) / pix ** 2, 0.0)
        p[:3] = p[3] / bw[3] * bw[:3]
        sbp.append(p)
    sbp = np.array(sbp)
    img = sbp[:, :, None, None] * pix ** 2 * np.ones((1, 1, n, n))
    spec = np.zeros((len(epochs), 31))
    for b in range(31):
        mid = 0.5 * (EDGES[b] + EDGES[b + 1])
        j = [i for i, (lo, hi) in enumerate(bands) if lo <= mid < hi][0]
        spec[:, b] = sbp[:, j] * np.pi * 200.0 ** 2 * 0.2 / bw[j]
    return img, spec


# =============================================================================
# ============ ↑ Data and the spectral likelihood ↑ ===========================
# =============================================================================


def wslope(y, v, wt):
    """Weighted slope (%/yr) of ln v vs y, and its error."""
    ok = np.isfinite(v) & (v > 0) & (wt > 0)
    A = np.stack([np.ones(ok.sum()), y[ok] - 2010.0], 1)
    W = wt[ok]
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    c = cov @ (A.T @ (W * np.log(v[ok])))
    return 100 * c[1], 100 * np.sqrt(cov[1, 1])


def gidx(lo, hi):
    return np.nonzero((EDGES[:-1] >= lo - 1e-6) & (EDGES[1:] <= hi + 1e-6))[0]


def scan_time_law(comp_spec, which, n, expo, use, years, grid=np.linspace(-0.06, 0.06, 61), amp=None):
    """Best secular law exp(gamma + beta (t - 2010)) on the components ``which``
    (all others fixed): (chi2_t, chi2_s, beta, gamma) at the joint optimum of
    temporal + static."""
    t = (np.asarray(years) - 2010.0)[:, None]
    amp = np.linspace(-1.2, 1.2, 25) if amp is None else amp
    other = comp_spec.sum(1) - comp_spec[:, which].sum(1)
    base = comp_spec[:, which].sum(1)
    best = (np.inf,)
    for g in amp:
        for b in grid:
            ct, cs = spec_chi2(other + base * np.exp(g + b * t), n, expo, use)
            if ct + cs < best[0]:
                best = (ct + cs, ct, cs, b, g)
    return best[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--json", default=None)
    ap.add_argument("--bkg", default=None, help="background.json: add the measured particle background to the model")
    a = ap.parse_args()
    D = np.load(a.dump, allow_pickle=True)
    lab = str(D["label"]) + (" +bkg" if a.bkg else "")
    eps = [str(e) for e in D["epochs"]]
    yrs = np.asarray(D["years"], np.float64)
    chi2 = json.loads(str(D["chi2"]))
    res = dict(label=lab, chi2=chi2)
    print(f"=== {lab}: {D['state']}\n  chi2 " + ", ".join(f"{k} {v:.1f}" for k, v in chi2.items()))
    spec_c = np.asarray(D["spec_comp"], np.float64)                    # (E, 4, 31)
    img_c = np.asarray(D["img_comp"], np.float64)                      # (E, 4, 6, 256, 256)
    ms, mi = np.asarray(D["model_spectra"], np.float64), np.asarray(D["model_images"], np.float64)
    n, expo, use = spectrum_data(eps)
    has = use.any(1)
    clo_s = spec_c.sum(1)[has] / ms[has]
    clo_i = img_c.sum(1).sum((-2, -1)) / mi.sum((-2, -1))
    print(f"  closure: spectrum {clo_s.min():.4f}-{clo_s.max():.4f}; image {clo_i.min():.4f}-{clo_i.max():.4f}")
    # the components rescaled per bin to the fit's exact spectrum (N_H geo-mixing is not linear)
    spec_c = spec_c * np.where(has[:, None, None], (ms / np.maximum(spec_c.sum(1), 1e-30))[:, None], 1.0)
    if a.bkg:        # the background as a 5th, fixed component (thermal-CSM slot untouched)
        bimg, bspec = particle_background(eps, mi, a.bkg)
        spec_c = np.concatenate([spec_c, np.where(has[:, None], bspec, 0.0)[:, None]], 1)
        img_c = np.concatenate([img_c, bimg[:, None]], 1)
        ms = spec_c.sum(1)
    ct0, cs0 = spec_chi2(ms, n, expo, use)
    print(f"  spectrum chi2 (replica): temporal {ct0:.1f}, static {cs0:.1f}")
    tr, rem = trend_profile(ms, n, expo, use, yrs)
    print(f"  temporal chi2 removable by per-bin linear trends: {rem:.1f}; trend %/yr per bin:\n    "
          + " ".join(f"{EDGES[b]:.1f}:{tr[b]:+.2f}" for b in range(31)))
    res.update(spec_temporal=ct0, spec_static=cs0, trend_removable=rem, trend=tr.tolist())

    # ---- component shares and trends (r < 200" spectrum) --------------------
    ok = has
    t_ok = yrs[ok]
    print("  component shares (spectrum r<200\") 2000 -> 2019 and trend of each (%/yr, S-array epochs, incl. "
          "response; total in brackets; data in braces):")
    sh = {}
    for g, (lo, hi) in GROUPS.items():
        ib = gidx(lo, hi)
        c = spec_c[:, :, ib].sum(-1)
        tot = c.sum(1)
        e0, e1 = eps.index("2000"), eps.index("2019")
        wt = np.where(ok & (np.array(eps) != "2022"), 1.0, 0.0)          # S-array epochs
        sl = [wslope(yrs, c[:, k], wt)[0] if np.all(c[ok, k] > 0) else np.nan for k in range(4)]
        st = wslope(yrs, tot, wt)[0]
        sd = wslope(yrs, np.where(ok, n[:, ib].sum(1) / expo, np.nan), np.where(ok & (np.array(eps) != "2022"), 1.0, 0))[0]
        sm = wslope(yrs, tot, np.where(ok & (np.array(eps) != "2022"), 1.0, 0))[0]
        sh[g] = dict(share2000=(c[e0] / tot[e0]).tolist(), share2019=(c[e1] / tot[e1]).tolist(), slope=sl,
                     slope_tot=st, data_slope_Sarray=sd, model_slope_Sarray=sm)
        print(f"    {g:13s}: " + "  ".join(f"{COMPS[k]} {100*c[e0,k]/tot[e0]:4.1f}->{100*c[e1,k]/tot[e1]:4.1f}% "
                                        f"({sl[k]:+.2f})" for k in range(4)) +
              f"  [tot {st:+.2f}] {{data {sd:+.2f} vs model {sm:+.2f} S-array}}")
    res["groups"] = sh

    # ---- image regions: data vs model components -----------------------------
    dat, iexp, pm, R, PA = load_images(eps)
    s_arr = np.array([e != "2022" for e in eps])
    common = pm.all(0)
    regions = {"inner r<140": (R < 140), "outer 140-215": (R >= 140) & (R < 215),
               "rim 150-175": (R >= 150) & (R < 175),
               "W rim PA225-315 140-215": (R >= 140) & (R < 215) & (PA >= 225) & (PA < 315),
               "N rim PA315-45 140-215": (R >= 140) & (R < 215) & ((PA >= 315) | (PA < 45)),
               "E rim PA45-135 140-215": (R >= 140) & (R < 215) & (PA >= 45) & (PA < 135),
               "S rim PA135-225 140-215": (R >= 140) & (R < 215) & (PA >= 135) & (PA < 225),
               "all r<215": R < 215}
    print("  image regions, bands 4.2-6 / 6-7 keV: data slope, model slope, ratio slope (%/yr, S-array); "
          "component shares 2000; component slopes; sync share needed:")
    reg_out = {}
    for rn, sel in regions.items():
        s = common & sel
        Dd = (dat * s).sum((-2, -1)) / iexp[:, None]                      # (E, 6)
        Nd = (dat * s).sum((-2, -1))
        Mc = (img_c * s).sum((-2, -1))                                     # (E, 4, 6)
        Mt = Mc.sum(1)
        row = {}
        for b in (1, 2, 4, 5):
            wt = np.where(s_arr, 1.0 / (1.0 / np.maximum(Nd[:, b], 1) + 0.01 ** 2), 0.0)
            sd_, _ = wslope(yrs, Dd[:, b], wt); sm_, _ = wslope(yrs, Mt[:, b], wt)
            sr_, er_ = wslope(yrs, Dd[:, b] / Mt[:, b], wt)
            shr = Mc[0, :, b] / Mt[0, b]
            slc = [wslope(yrs, Mc[:, k, b], wt)[0] if np.all(Mc[s_arr, k, b] > 0) else np.nan for k in range(4)]
            sync = shr[2] + shr[3]
            row[BAND_NAMES[b]] = dict(data=sd_, model=sm_, ratio=sr_, ratio_err=er_, share=shr.tolist(), comp_slope=slc,
                                      mean_ratio=float(np.mean(Dd[s_arr, b] / Mt[s_arr, b])))
            if b in (4, 5):
                print(f"    {rn:24s} {BAND_NAMES[b]:6s}: data {sd_:+.2f} model {sm_:+.2f} ratio {sr_:+.2f}±{er_:.2f} "
                      f"<d/m {row[BAND_NAMES[b]]['mean_ratio']:.2f}> | share " +
                      "/".join(f"{100*x:.0f}" for x in shr) + " | slopes " + "/".join(f"{x:+.2f}" for x in slc) +
                      f" | if all in sync: {sr_ / max(sync, 1e-3):+.2f} %/yr extra")
        reg_out[rn] = row
    res["regions"] = reg_out
    for rn in ("inner r<140", "outer 140-215", "all r<215"):
        r1 = reg_out[rn]
        print(f"    {rn:24s} Si 1.5-2.1: data {r1['1.5-2.1']['data']:+.2f} model {r1['1.5-2.1']['model']:+.2f} "
              f"ratio {r1['1.5-2.1']['ratio']:+.2f}; S 2.1-2.8: ratio {r1['2.1-2.8']['ratio']:+.2f}; shares Si "
              + "/".join(f"{100*x:.0f}" for x in r1['1.5-2.1']['share']) + "; comp slopes Si "
              + "/".join(f"{x:+.2f}" for x in r1['1.5-2.1']['comp_slope']))

    # ---- the model's forward-shock speed (cone-mean r_FS) vs the T_i-based v_s -----
    rfs = np.asarray(D["model_r_fs_pc"], np.float64).mean(1)
    c1 = np.polyfit(yrs, rfs, 1)
    v_fs = c1[0] * 3.0857e13 / 3.156e7                       # pc/yr -> km/s
    print(f"  model FS: <r_FS> {rfs[0]:.3f} -> {rfs[-1]:.3f} pc, dr/dt = {v_fs:.0f} km/s (lab frame; the wind is ~at rest)")
    res["v_fs_kms"] = v_fs
    # ---- synchrotron bookkeeping ----------------------------------------------
    t = yrs
    print("  synchrotron bookkeeping (slopes %/yr over all epochs):")
    sy = {}
    for k in ("k_anchor", "wsum", "wf_csm", "wf_ej", "xb_csm", "xb_ej", "n_fresh"):
        v = np.asarray(D[f"sync_{k}"], np.float64)
        sy[k] = wslope(t, v, np.ones_like(t))[0]
    lne = np.asarray(D["sync_lne_csm"]) / np.maximum(np.asarray(D["sync_xb_csm"]), 1e-30)
    lne_ej = np.asarray(D["sync_lne_ej"]) / np.maximum(np.asarray(D["sync_xb_ej"]), 1e-30)
    sy["Ecut_csm_xw_keV_2000"] = float(np.exp(lne[0])); sy["Ecut_csm_xw_keV_2022"] = float(np.exp(lne[-1]))
    sy["dlnEcut_csm_pct_per_yr"] = 100 * np.polyfit(t, lne, 1)[0]
    sy["Ecut_ej_xw_keV_2000"] = float(np.exp(lne_ej[0]))
    vb = np.asarray(D["v_bins"]); vc = 0.5 * (vb[1:] + vb[:-1])
    hr = np.asarray(D["sync_hv_radio_csm"]); hx = np.asarray(D["sync_hv_x_csm"])
    vr = (hr * vc).sum(1) / np.maximum(hr.sum(1), 1e-30); vx = (hx * vc).sum(1) / np.maximum(hx.sum(1), 1e-30)
    sy.update(vs_radio_w_2000=float(vr[0]), vs_radio_w_2022=float(vr[-1]), vs_x_w_2000=float(vx[0]),
              vs_x_w_2022=float(vx[-1]), dlnv_x_pct_per_yr=100 * np.polyfit(t, np.log(vx), 1)[0],
              radio_decline_pct_per_yr=100 * np.log(1 - 0.007))
    print("    " + ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in sy.items()))
    res["sync"] = sy
    for k in ("em_csm", "em_ej", "kTe_csm", "kTe_ej", "lnet_csm", "lnet_ej"):
        v = np.asarray(D[f"cell_{k}"], np.float64)
        print(f"    cell {k}: 2000 {v[0]:.4g} 2022 {v[-1]:.4g}" + (f" ({wslope(t, v, np.ones_like(t))[0]:+.2f} %/yr)" if "lnet" not in k else ""))

    # ---- chi2 scans --------------------------------------------------------------
    print("  spectral chi2 scans (temporal + static; secular law exp(g + b (t-2010)) on the named components):")
    scans = {}
    for nm, which in (("sync (FS+RS)", [2, 3]), ("sync FS", [2]), ("thermal CSM", [0]), ("thermal ejecta", [1]),
                      ("all thermal", [0, 1])):
        ct, cs, b, g = scan_time_law(spec_c, which, n, expo, use, yrs)
        scans[nm] = dict(temporal=ct, static=cs, beta_pct=100 * b, gamma=g, dchi2=ct + cs - ct0 - cs0)
        print(f"    {nm:15s}: temporal {ct:.1f} static {cs:.1f} (d {ct + cs - ct0 - cs0:+.1f}) at beta {100*b:+.2f} %/yr, "
              f"amp x{np.exp(g):.2f}")
    # two laws at once: sync fading + ejecta brightening
    best = (np.inf,)
    for bs in np.linspace(-0.03, 0.01, 21):
        for be in np.linspace(-0.01, 0.03, 21):
            for gs in (-0.3, -0.15, 0.0, 0.15, 0.3):
                tt = (yrs - 2010.0)[:, None]
                m = spec_c[:, 0] + spec_c[:, 1] * np.exp(be * tt) + (spec_c[:, 2] + spec_c[:, 3]) * np.exp(gs + bs * tt)
                ct, cs = spec_chi2(m, n, expo, use)
                if ct + cs < best[0]:
                    best = (ct + cs, ct, cs, bs, be, gs)
    scans["sync + ejecta"] = dict(temporal=best[1], static=best[2], beta_sync_pct=100 * best[3],
                                  beta_ej_pct=100 * best[4], gamma_sync=best[5], dchi2=best[0] - ct0 - cs0)
    print(f"    sync+ejecta    : temporal {best[1]:.1f} static {best[2]:.1f} (d {best[0]-ct0-cs0:+.1f}) at beta_sync "
          f"{100*best[3]:+.2f}, beta_ej {100*best[4]:+.2f} %/yr, sync amp x{np.exp(best[5]):.2f}")
    res["scans"] = scans
    if "th_var" in D.files:
        tv = np.asarray(D["th_var"], np.float64)                        # (E, n_var, 2, 31)
        var = np.asarray(D["th_variants"])
        fac = np.where(has[:, None], ms / np.maximum(spec_c.sum(1), 1e-30), 1.0)   # ~1 (already rescaled)
        base_th = spec_c[:, 0:2]
        sy_sum = spec_c[:, 2] + spec_c[:, 3]
        # per-bin correction of the raw variant spectra to the rescaled components (base variant = index 0)
        corr = np.where(tv[:, 0] > 0, base_th / np.maximum(tv[:, 0], 1e-30), 1.0)
        tvc = tv * corr[:, None]
        print("  split emission-physics variants (csm factor x ejecta factor; chi2 temporal/static, d = change):")
        vs = {}
        for kind, col in (("kT_e0", 0), ("teq", 1), ("n_e t", 2)):
            idx = [i for i in range(len(var)) if all(var[i][j] == 1.0 for j in range(3) if j != col)]
            facs = [var[i][col] for i in idx]
            tab = {}
            for ic, fc in zip(idx, facs):
                for ie, fe in zip(idx, facs):
                    m = tvc[:, ic, 0] + tvc[:, ie, 1] + sy_sum
                    ct, cs = spec_chi2(m, n, expo, use)
                    tab[(fc, fe)] = (ct, cs)
            (bc, be), (ct, cs) = min(tab.items(), key=lambda kv: sum(kv[1]))
            vs[kind] = dict(best_csm=bc, best_ej=be, temporal=ct, static=cs, dchi2=ct + cs - ct0 - cs0,
                            table={f"{k[0]}x{k[1]}": v for k, v in tab.items()})
            print(f"    {kind:6s}: best csm x{bc:g}, ejecta x{be:g}: temporal {ct:.1f} static {cs:.1f} "
                  f"(d {ct + cs - ct0 - cs0:+.1f}); same factor for both: " +
                  ", ".join(f"x{f:g} {sum(tab[(f, f)]):.1f}" for f in sorted(set(facs))))
            for ib_name in ("FeK 6.5-6.9", "Si 1.7-2.1", "soft 0.7-1.1"):
                ib = gidx(*GROUPS[ib_name])
                r = [(f, (tvc[:, i, 0, ib].sum() + tvc[:, i, 1, ib].sum()) / (tvc[:, idx[facs.index(1.0)], 0, ib].sum()
                      + tvc[:, idx[facs.index(1.0)], 1, ib].sum())) for i, f in zip(idx, facs)]
                print(f"        thermal {ib_name} flux vs factor: " + " ".join(f"x{f:g}:{v:.2f}" for f, v in r))
        res["variants"] = vs
        if "fe_var" in D.files:
            fe0 = np.asarray(D["fe_var"], np.float64)[:, 0] * corr    # (E, 2, 31) with Fe tracer x0
            fe_part = base_th - fe0
            best = min(((spec_chi2(ms + (s - 1.0) * fe_part.sum(1), n, expo, use), s)
                        for s in np.linspace(0.5, 2.5, 41)), key=lambda x: sum(x[0]))
            ib = gidx(6.5, 6.9)
            print(f"    Fe line share of 6.5-6.9 keV: {fe_part[:, :, ib].sum() / ms[:, ib].sum():.2f}; free Fe scale: "
                  f"best x{best[1]:.2f}: temporal {best[0][0]:.1f} static {best[0][1]:.1f} (d {sum(best[0]) - ct0 - cs0:+.1f})")
            res["fe_scale"] = dict(best=best[1], temporal=best[0][0], static=best[0][1])
    if "sy_var" in D.files:
        sv = np.asarray(D["sy_var"], np.float64)                       # (E, n_eta, 2, 31)
        eta = np.asarray(D["eta_variants"])
        i1 = int(np.argmin(np.abs(eta - 1.0)))
        c = np.where(sv[:, i1] > 0, spec_c[:, 2:4] / np.maximum(sv[:, i1], 1e-30), 1.0)
        out = []
        for k, e in enumerate(eta):
            best = (np.inf,)
            for g in np.linspace(-1.0, 1.0, 41):
                m = spec_c[:, 0] + spec_c[:, 1] + np.exp(g) * (sv[:, k] * c).sum(1)
                ct, cs = spec_chi2(m, n, expo, use)
                if ct + cs < best[0]:
                    best = (ct + cs, ct, cs, g)
            out.append((float(e), best[1], best[2], best[3]))
        print("    synchrotron cutoff scale eta (amplitude re-fitted): " +
              ", ".join(f"eta {e:g}: {ct:.1f}/{cs:.1f} (amp x{np.exp(g):.2f})" for e, ct, cs, g in out))
        res["eta_scan"] = out
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
