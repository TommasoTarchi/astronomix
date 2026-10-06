"""
Fit diagnostics for the Orlando-state fits: model vs Chandra, per observable.

Reads ``casa_pluto_diff.py --save-model`` npz files (one per fit) and plots,
against the data the likelihood uses: the forward-shock outline per 10-degree
cone in 2000 and 2022, the epoch-mean radius vs year, the per-cone proper
motion (registration, ``casa_expansion``), and the Si He-alpha Doppler pattern
per sky sector (2004), with each fit's roll ``psi`` and dilution scale
``ln_kdop`` applied exactly as in ``casa_pluto_diff.residuals``.

    python casa_pluto_fitplot.py --models H:model_H_n128.npz D:model_D_n128.npz \\
        --out figures/pluto146_fit_diagnostics.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOPPLER = "/export/data/lstorcks/casa_orlando150/work/doppler_si_2004.npz"


def model_doppler(m, dop):
    """Model Doppler per data sector with psi and ln_kdop applied (as the fit does)."""
    names = [str(x) for x in m["names"]]
    th = dict(zip(names, m["theta"]))
    n = len(dop["v_kms"]); w = 360.0 / n
    nodes = np.arange(n) * w
    v = m["vlos_kms"][int(np.argmin(np.abs(m["years"] - float(dop["year"]))))]
    q = (nodes - th.get("psi", 0.0)) % 360.0 / w
    i0 = np.floor(q).astype(int) % n; f = q - np.floor(q)
    vm = v[i0] * (1 - f) + v[(i0 + 1) % n] * f
    return np.exp(th.get("ln_kdop", 0.0)) * (vm - vm.mean())


def per_cone_pm(years, r, mask):
    yc = years - years.mean()
    rr = np.where(mask, r, np.nan)
    out = np.full(r.shape[1], np.nan)
    for k in range(r.shape[1]):
        ok = mask[:, k]
        if ok.sum() >= 8:
            out[k] = np.polyfit(years[ok], rr[ok, k], 1)[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="LABEL:path.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = [(s.split(":", 1)[0], np.load(s.split(":", 1)[1])) for s in args.models]
    dop = np.load(DOPPLER)
    m0 = models[0][1]
    yrs, ang, ro, msk = m0["years"], m0["angles"], m0["r_obs"], m0["mask"].astype(bool)
    colors = ["C3", "C0", "C2", "C1"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    ax = axs[0, 0]
    for e, ls in ((0, "-"), (len(yrs) - 1, "--")):
        ax.plot(ang, np.where(msk[e], ro[e], np.nan), "ko" if e == 0 else "ks", ms=4,
                mfc="none" if e else "k", label=f"Chandra {yrs[e]:.0f}")
        for (lab, m), c in zip(models, colors):
            ax.plot(ang, m["r_fs_arcsec"][e], ls, color=c, label=f"fit {lab} {yrs[e]:.0f}")
    ax.set_xlabel("position angle [deg] (0 = W, 90 = N)"); ax.set_ylabel("forward-shock radius [arcsec]")
    ax.set_title("outline per 10° cone"); ax.legend(fontsize=7, ncol=2)

    ax = axs[0, 1]
    ax.plot(yrs, np.nanmean(np.where(msk, ro, np.nan), 1), "ko", label="Chandra")
    for (lab, m), c in zip(models, colors):
        ax.plot(yrs, np.nanmean(np.where(msk, m["r_fs_arcsec"], np.nan), 1), "-", color=c,
                label=f"fit {lab}")
    ax.set_xlabel("year"); ax.set_ylabel("<r_FS> [arcsec]"); ax.set_title("mean radius vs epoch")
    ax.legend()

    ax = axs[1, 0]
    pm_obs = m0["pm_obs"]
    ax.plot(ang, pm_obs, "ko", label="Chandra (registration)")
    for (lab, m), c in zip(models, colors):
        pm = per_cone_pm(yrs, m["r_fs_arcsec"], np.ones_like(msk))
        ax.plot(ang, pm, "-", color=c, label=f"fit {lab}")
    ax.set_xlabel("position angle [deg]"); ax.set_ylabel("proper motion [arcsec/yr]")
    ax.set_title("forward-shock proper motion 2000–2022"); ax.legend()

    ax = axs[1, 1]
    cen = (np.arange(len(dop["v_kms"])) + 0.5) * 360.0 / len(dop["v_kms"])
    ax.errorbar(cen, dop["v_kms"], yerr=np.sqrt(dop["v_err_kms"] ** 2 + 100 ** 2), fmt="ko",
                ms=4, label="Chandra 2004 (Si He-α centroid)")
    for (lab, m), c in zip(models, colors):
        vm = model_doppler(m, dop)
        cc = np.corrcoef(vm, dop["v_kms"])[0, 1]
        ax.plot(cen, vm, "-", color=c, label=f"fit {lab} (corr {cc:+.2f})")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("position angle [deg]"); ax.set_ylabel("line-of-sight velocity [km/s] (+ = receding)")
    ax.set_title("Si Doppler pattern"); ax.legend(fontsize=8)
    fig.suptitle("Orlando's 146-yr Cas A state evolved with astronomix (128³), fits vs Chandra 2000–2022")
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
