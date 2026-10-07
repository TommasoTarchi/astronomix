"""
Diagnostics of a ``casa_xfit.py --save-model`` npz: the multi-epoch X-ray fit.

Top row: per band, the model and Chandra count rates over 2000-2023 (inside
the likelihood mask) -- the X-ray light curve the fit sees. Below, for the
first and last epoch: Chandra and the model in 0.5-1.5 keV, Si and Fe-K bands at
the likelihood's block resolution, and the standardised block residual.

    python casa_xfit_plot.py model.npz --out figures/xfit.png [--sigma-img 0.3]
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))
BKG_RATE = 1e-6


def block(a, b):
    nb = a.shape[-1] // b
    return a.reshape(*a.shape[:-2], nb, b, nb, b).sum((-3, -1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-img", type=float, default=0.3)
    args = ap.parse_args()
    d = np.load(args.model)
    b = int(d["block"]); exp = d["exposure"]
    lam = block((d["images"] + BKG_RATE) * exp[:, None, None, None], b)
    n = d["counts"]; bm = d["bmask"][:, None]
    years = d["years"]
    fig = plt.figure(figsize=(16, 13), layout="constrained")
    gs = fig.add_gridspec(3, 6)
    for k, (lo, hi) in enumerate(BANDS):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(years, (n[:, k] * bm[:, 0]).sum((1, 2)) / exp, "ko", ms=3, label="Chandra")
        ax.plot(years, (lam[:, k] * bm[:, 0]).sum((1, 2)) / exp, "C3-", label="model")
        ax.set_title(f"{lo}-{hi} keV"); ax.set_xlabel("year")
        if k == 0:
            ax.set_ylabel("counts/s in mask"); ax.legend(fontsize=8)
    for row, e in ((1, 0), (2, len(years) - 1)):
        for j, (k, lab) in enumerate(((0, "0.5-1.5"), (1, "Si"), (5, "Fe-K"))):
            vmax = np.percentile(n[e, k], 99.5)
            a1 = fig.add_subplot(gs[row, 2 * j]); a2 = fig.add_subplot(gs[row, 2 * j + 1])
            a1.imshow(np.sqrt(n[e, k]), origin="lower", cmap="inferno", vmin=0, vmax=np.sqrt(vmax))
            res = np.where(bm[e, 0], (n[e, k] - lam[e, k]) /
                           np.sqrt(lam[e, k] + (args.sigma_img * 0.5 * (lam[e, k] + n[e, k])) ** 2 + 1), np.nan)
            im = a2.imshow(res, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
            a1.set_title(f"{d['epochs'][e]} {lab}: Chandra"); a2.set_title(f"{lab}: residual")
            for a in (a1, a2):
                a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=fig.axes[-1], shrink=0.6, label="(n - model) / sigma")
    fig.suptitle("casa_xfit: the state evolved through every epoch and observed with the differentiable "
                 "X-ray model (west to the right, north up)")
    fig.savefig(args.out, dpi=90)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
