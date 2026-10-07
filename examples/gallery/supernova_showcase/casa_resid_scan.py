"""
Expected chi2 gains of minimal emission-model changes, at FIXED hydrodynamics
(stage-3 residual physics; numpy, CPU): rescale the four components of a
``casa_resid_dump.py`` model -- thermal CSM, thermal ejecta, synchrotron --
each by ``A exp(beta (t - 2010))``, and re-evaluate casa_xfit's image_static +
image_temporal + spectrum_static + spectrum_temporal terms exactly (numpy
replicas, checked against the dump's own chi2 at the identity).

    python casa_resid_scan.py $W/stage3/physics/dump/dump_V3.npz [--json out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import casa_resid_analyze as A

DATA = A.DATA
CALIB = np.array([0.10, 0.03, 0.03, 0.03, 0.03, 0.03])
SIG_T_IMG, SIG_S_IMG, W_IMG, BKG, B = 0.075, 0.5, 1.0 / 6.0, 1e-6, 16
PARAMS = ("lnA_csm", "b_csm", "lnA_ej", "b_ej", "lnA_sy", "b_sy", "lnA_rs", "b_rs")


class ImageTerms:
    """casa_xfit.image_residuals (numpy), with the model images block-summed once
    per component so a rescaling costs O(E x 6 x 16 x 16)."""

    def __init__(self, epochs, img_comp):
        cnt, pm, bm, expo, bw = [], [], [], [], []
        for e in epochs:
            d = np.load(DATA / f"bands_{e}.npz")
            npix, pix = d["counts"].shape[-1], float(d["pixel_arcsec"])
            ax = (np.arange(npix) - 0.5 * (npix - 1)) * pix
            NN, WW = np.meshgrid(ax, ax, indexing="ij")
            p = ((np.asarray(d["edge"]) <= 0.02) & (np.hypot(WW, NN) < 150.0)
                 & (np.hypot(WW - A.CCO[0], NN - A.CCO[1]) >= 6.0))
            nb = npix // B
            bs = lambda a: a.reshape(*a.shape[:-2], nb, B, nb, B).sum((-3, -1))  # noqa: E731
            cnt.append(bs(np.asarray(d["counts"], np.float64) * p)); bm.append(bs(p.astype(float)) >= 0.9 * B * B)
            pm.append(p); expo.append(float(d["exposure"]))
            bw.append(np.array([0.0 if (e == "2022" and k == 0) else 1.0 for k in range(6)]))
        self.n, self.expo = np.stack(cnt), np.array(expo)
        pm, bm, bw = np.stack(pm), np.stack(bm), np.stack(bw)
        E = len(epochs)
        x = self.expo[:, None, None, None, None]
        self.lam_c = (img_comp * x * pm[:, None, None]).reshape(E, img_comp.shape[1], 6, 16, B, 16, B).sum((-3, -1))
        self.lam_bkg = (BKG * self.expo[:, None, None, None] * pm[:, None]).reshape(E, 1, 16, B, 16, B).sum((-3, -1))
        self.ok = bm[:, None] & (bw[:, :, None, None] > 0)

    def chi2(self, amp):
        """amp (E, n_comp): per-epoch component factors -> (static, temporal)."""
        lam = np.einsum("ec,ecbxy->ebxy", amp, self.lam_c) + self.lam_bkg
        n, ok = self.n, self.ok
        okf = ok.astype(float)
        N_tot = (n * okf).sum(0); L_tot = (lam * okf).sum(0)
        good = (okf.sum(0) >= 3) & (N_tot > 25)
        Ls = np.where(good, L_tot, 1.0)
        r_s = np.where(good, (np.log(np.maximum(N_tot, 1)) - np.log(Ls)) / np.sqrt(SIG_S_IMG ** 2 + 1 / np.maximum(N_tot, 1)), 0)
        ex = self.expo[:, None, None, None]
        E_tot = (ex * okf).sum(0)
        with np.errstate(divide="ignore"):
            n_rel = np.log(np.maximum(n, 1) / ex * E_tot / np.maximum(N_tot, 1))
        use = ok & good[None] & (n > 10)
        l_rel = np.log(np.where(use, lam / ex * E_tot / Ls, 1.0))
        sig2 = SIG_T_IMG ** 2 + 1 / np.maximum(n, 1)
        d = np.where(use, n_rel - l_rel, 0)
        w = np.where(use, 1 / sig2, 0)
        a_hat = (w * d).sum((-2, -1)) / (w.sum((-2, -1)) + 1 / CALIB[None] ** 2)
        r_t = np.where(use, (d - a_hat[..., None, None]) / np.sqrt(sig2), 0)
        return W_IMG * float((r_s ** 2).sum()), W_IMG * float((r_t ** 2).sum() + ((a_hat / CALIB[None]) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--json", default=None)
    ap.add_argument("--bkg", default=None, help="background.json: the measured particle background as a fixed component")
    a = ap.parse_args()
    D = np.load(a.dump, allow_pickle=True)
    eps = [str(e) for e in D["epochs"]]
    yrs = np.asarray(D["years"], np.float64)
    tau = yrs - 2010.0
    chi_dump = json.loads(str(D["chi2"]))
    ms = np.asarray(D["model_spectra"], np.float64)
    sc = np.asarray(D["spec_comp"], np.float64)
    n, expo, use = A.spectrum_data(eps)
    has = use.any(1)
    sc = sc * np.where(has[:, None, None], (ms / np.maximum(sc.sum(1), 1e-30))[:, None], 1.0)
    ic = np.asarray(D["img_comp"], np.float64)
    mi = np.asarray(D["model_images"], np.float64)
    ic = ic * (mi / np.maximum(ic.sum(1), 1e-30))[:, None]            # exact closure per pixel
    # groups: csm thermal, ejecta thermal, synchrotron (FS + RS; "sy" scales both), RS synchrotron extra
    sc3 = np.stack([sc[:, 0], sc[:, 1], sc[:, 2], sc[:, 3]], 1)
    ic3 = np.stack([ic[:, 0], ic[:, 1], ic[:, 2], ic[:, 3]], 1)
    if a.bkg:
        bimg, bspec = A.particle_background(eps, mi, a.bkg)
        sc3 = np.concatenate([sc3, np.where(has[:, None], bspec, 0.0)[:, None]], 1)
        ic3 = np.concatenate([ic3, bimg[:, None]], 1)
    del ic
    IT = ImageTerms(eps, ic3)

    def terms(th):
        th = dict(zip(PARAMS, th))
        sy = np.exp(th["lnA_sy"] + th["b_sy"] * tau)
        amp = np.stack([np.exp(th["lnA_csm"] + th["b_csm"] * tau), np.exp(th["lnA_ej"] + th["b_ej"] * tau),
                        sy, sy * np.exp(th["lnA_rs"] + th["b_rs"] * tau)]
                       + ([np.ones_like(tau)] if sc3.shape[1] == 5 else []), 1)  # (E, 4 [+ bkg])
        s_st, s_t = IT.chi2(amp)
        ct, cs = A.spec_chi2(np.einsum("ec,ecb->eb", amp, sc3), n, expo, use)
        return dict(image_static=s_st, image_temporal=s_t, spectrum_static=cs, spectrum_temporal=ct)

    base = terms(np.zeros(len(PARAMS)))
    print(f"=== {D['label']}{' +bkg' if a.bkg else ''}: identity check vs the dump's chi2: " + ", ".join(
        f"{k} {v:.1f} ({chi_dump[k]:.1f})" for k, v in base.items()))
    tot0 = sum(base.values())
    cases = {
        "sync secular (A_sy, b_sy)": ["lnA_sy", "b_sy"],
        "sync amp + CSM amp (A_sy, A_csm)": ["lnA_sy", "lnA_csm"],
        "sync secular + CSM amp": ["lnA_sy", "b_sy", "lnA_csm"],
        "ejecta secular (A_ej, b_ej)": ["lnA_ej", "b_ej"],
        "CSM secular (A_csm, b_csm)": ["lnA_csm", "b_csm"],
        "sync secular + ejecta secular": ["lnA_sy", "b_sy", "lnA_ej", "b_ej"],
        "RS sync amp (A_rs)": ["lnA_rs"],
        "RS sync secular (A_rs, b_rs)": ["lnA_rs", "b_rs"],
        "RS sync amp + FS sync secular": ["lnA_rs", "lnA_sy", "b_sy"],
        "RS sync secular + FS sync secular + CSM amp": ["lnA_rs", "b_rs", "lnA_sy", "b_sy", "lnA_csm"],
        "all": list(PARAMS),
    }
    out = dict(base=base, cases={})
    for name, free in cases.items():
        idx = [PARAMS.index(k) for k in free]

        def f(x):
            th = np.zeros(len(PARAMS)); th[idx] = x
            return sum(terms(th).values())
        best = None
        for x0 in (np.zeros(len(idx)), np.full(len(idx), 0.01)):
            r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-3, maxiter=3000))
            best = r if best is None or r.fun < best.fun else best
        th = np.zeros(len(PARAMS)); th[idx] = best.x
        t = terms(th)
        out["cases"][name] = dict(params={k: float(v) for k, v in zip(free, best.x)}, terms=t, dchi2=best.fun - tot0)
        print(f"  {name:34s}: d chi2 {best.fun - tot0:+7.1f}  [" + ", ".join(
            f"{k} {t[k] - base[k]:+.1f}" for k in t) + "]  at " + ", ".join(
            f"{k} {v:+.4f}" if k.startswith("b_") else f"{k} {np.exp(v):.2f}x" for k, v in zip(free, best.x)))
    # physically anchored cases (fixed values, no fitting)
    fixed = {"sync x2, CSM x0.6 (HV08 54 % share)": dict(lnA_sy=np.log(2.0), lnA_csm=np.log(0.6)),
             "sync extra -0.5 %/yr": dict(b_sy=-0.005), "sync extra -1.0 %/yr": dict(b_sy=-0.010),
             "sync x2, CSM x0.6, sync extra -0.5 %/yr": dict(lnA_sy=np.log(2.0), lnA_csm=np.log(0.6), b_sy=-0.005),
             "no radio decline in the anchor (+0.70 %/yr)": dict(b_sy=-np.log(1 - 0.007))}
    for name, th_d in fixed.items():
        th = np.array([th_d.get(k, 0.0) for k in PARAMS])
        t = terms(th)
        out["cases"][name] = dict(params={k: float(v) for k, v in th_d.items()}, terms=t, dchi2=sum(t.values()) - tot0)
        print(f"  {name:44s}: d chi2 {sum(t.values()) - tot0:+7.1f}  [" + ", ".join(
            f"{k} {t[k] - base[k]:+.1f}" for k in t) + "]")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
