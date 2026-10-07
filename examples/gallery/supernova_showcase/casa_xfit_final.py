"""
The fit's final comparison with Chandra: images, spectra and light curves.

Reads a ``casa_xfit.py --save-model`` npz (the model at the fitted theta) and
the real data it was fitted to, and draws:

* a three-colour image (0.5-1.5 keV red, Si 1.5-2.1 keV green, 4.2-7 keV
  blue) of Chandra and of the model, in 2000 and 2022, on the same 1.97"
  grid (row = north, column = west), plus the Si-band residual in 31" blocks;
* the integrated spectra (r < 200") at three epochs, Chandra vs model, with
  the model normalised per epoch exactly as the likelihood does (the shape is
  what is fitted), and the data/model ratio;
* the band light curves inside the image mask.

    python casa_xfit_final.py model.npz --out figures/xfit_Q2_final.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs/data")
SPEC_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))
SPEC_EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)
LINES = {"Mg": 1.35, "Si": 1.86, "S": 2.45, "Ar": 3.13, "Ca": 3.9, "Fe-K": 6.65}


def rgb(img, pct=99.7, scale=None):
    """(6, n, n) band images -> RGB (soft, Si, hard = 4.2-7 keV), sqrt stretch."""
    ch = [img[0], img[1], img[4] + img[5]]
    if scale is None:
        scale = [np.percentile(c, pct) for c in ch]
    out = np.stack([np.sqrt(np.clip(c / s, 0, 1)) for c, s in zip(ch, scale)], -1)
    return out, scale


def smooth(a, s=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = np.load(args.model)
    epochs = [str(e) for e in m["epochs"]]
    years = m["years"]
    th = dict(zip([str(x) for x in m["names"]], m["theta"]))

    fig = plt.figure(figsize=(18, 17), layout="constrained")
    gs = fig.add_gridspec(4, 6, height_ratios=[1.15, 1.15, 1.0, 0.8])
    # ---- images: 2000 and 2022 ----
    for row, lab in enumerate(("2000", "2022")):
        e = epochs.index(lab)
        d = np.load(DATA_DIR / f"bands_{lab}.npz")
        n = np.asarray(d["counts"], float)
        expo = float(d["exposure"])
        lam = np.asarray(m["images"][e], float) * expo
        # match the model's total per band only for DISPLAY colour balance of
        # the model panel: use the data's stretch for both panels
        dn, scale = rgb(np.stack([smooth(x) for x in n]))
        mn, _ = rgb(np.stack([smooth(x) for x in lam]), scale=scale)
        ext = np.array([-1, 1, -1, 1]) * 0.5 * n.shape[-1] * float(d["pixel_arcsec"])
        for j, (im, t) in enumerate(((dn, f"Chandra {lab}"), (mn, f"model {lab} (same stretch)"))):
            ax = fig.add_subplot(gs[row, 2 * j:2 * j + 2])
            ax.imshow(im, origin="lower", extent=ext)
            ax.set_title(f"{t}   R: 0.5-1.5  G: Si 1.5-2.1  B: 4.2-7 keV")
            ax.set_xlabel("west [arcsec]"); ax.set_ylabel("north [arcsec]")
            ax.set_xlim(-250, 250); ax.set_ylim(-250, 250)
        ax = fig.add_subplot(gs[row, 4:6])
        b = 16
        nb = n.shape[-1] // b
        bs = lambda a: a.reshape(nb, b, nb, b).sum((1, 3))              # noqa: E731
        N, L = bs(n[1]), bs(lam[1])
        rr = np.hypot(*np.meshgrid((np.arange(nb) - nb / 2 + 0.5) * b * 1.968,
                                   (np.arange(nb) - nb / 2 + 0.5) * b * 1.968))
        res = np.where((rr < 150) & (N > 25), np.log(N / np.maximum(L, 1e-9)), np.nan)
        im = ax.imshow(res, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        ax.set_xlim(-250, 250); ax.set_ylim(-250, 250)
        ax.set_title(f"{lab} Si band: ln(data / model) in 31\" blocks (r < 150\", the likelihood region)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    # ---- spectra ----
    mids = 0.5 * (SPEC_EDGES[1:] + SPEC_EDGES[:-1])
    width = np.diff(SPEC_EDGES)
    sp_axes = [fig.add_subplot(gs[2, 2 * k:2 * k + 2]) for k in range(3)]
    for ax, lab in zip(sp_axes, ("2000", "2010", "2022")):
        if lab not in epochs:
            continue
        e = epochs.index(lab)
        d = np.load(SPEC_DIR / f"epoch_{lab}_spectrum.npz")
        eb = np.round(np.asarray(d["ebins"]), 3)
        idx = np.searchsorted(eb, SPEC_EDGES)
        cs = np.concatenate([[0.0], np.cumsum(d["counts"])])
        n = (cs[idx[1:]] - cs[idx[:-1]]) / float(d["exposure"])        # counts/s per bin
        lam = np.asarray(m["spectra"][e], float)
        # the likelihood profiles the per-epoch normalisation: show it that way,
        # and quote the absolute ratio it removes
        w = n > 0
        a = np.exp(np.sum(np.log(n[w] / lam[w]) * n[w]) / np.sum(n[w]))
        ax.step(mids, n / width, where="mid", color="k", label=f"Chandra {lab}")
        ax.step(mids, a * lam / width, where="mid", color="C3", label=f"model x {a:.2f} (profiled norm.)")
        ax.set_yscale("log"); ax.set_xlim(0.7, 7.0)
        ax.set_xlabel("energy [keV]"); ax.set_ylabel("counts/s/keV (r < 200\")")
        for nm, E in LINES.items():
            ax.axvline(E, color="0.8", lw=0.8, zorder=0)
            ax.text(E, ax.get_ylim()[1] * 0.6, nm, fontsize=7, ha="center", color="0.4")
        ax.legend(fontsize=8, loc="lower left")
        ax2 = ax.inset_axes([0.55, 0.62, 0.43, 0.3])
        ax2.step(mids, n / (a * lam), where="mid", color="C0")
        ax2.axhline(1, color="0.5", lw=0.7); ax2.set_ylim(0.5, 1.7); ax2.set_xlim(0.7, 7)
        ax2.set_title("data / model", fontsize=7); ax2.tick_params(labelsize=6)
    # ---- light curves ----
    lc = []
    for lab in epochs:
        d = np.load(DATA_DIR / f"bands_{lab}.npz")
        lc.append((np.asarray(d["counts"], float).sum((1, 2)) / float(d["exposure"])))
    lc = np.array(lc)
    lm = np.asarray(m["images"], float).sum((2, 3))
    for k, (lo, hi) in enumerate(BANDS):
        ax = fig.add_subplot(gs[3, k])
        ax.plot(years, lc[:, k], "ko", ms=3, label="Chandra")
        ax.plot(years, lm[:, k], "C3-", label="model")
        ax.set_title(f"{lo}-{hi} keV (whole field)", fontsize=9); ax.set_xlabel("year")
        if k == 0:
            ax.set_ylabel("counts/s"); ax.legend(fontsize=7)
    fig.suptitle(f"Differentiable Cas A (casa_xfit fit Q2, 128³): Orlando's 146-yr state evolved to every "
                 f"Chandra epoch — explosion {th['t_expl']:.0f}, D = {np.exp(th['ln_D']):.2f} kpc, "
                 f"kT_e,0 = {np.exp(th['ln_kte']):.2f} keV", fontsize=13)
    fig.savefig(args.out, dpi=85)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
