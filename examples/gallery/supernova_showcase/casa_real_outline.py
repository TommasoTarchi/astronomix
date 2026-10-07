"""
The forward-shock outline of the REAL remnant, per epoch, and of synthetic images.

Two things this settles that the directory had been assuming (CALIBRATION.md
Result 26):

  * the observed position-angle spread of r_FS -- quoted as "0.2-0.4 pc" since
    Result 5 without a source -- is MEASURED here on the Chandra epoch images
    with one detector, and the same detector is run on the synthetic
    ``*_synimg.npz`` so model and data are scored identically;
  * the forward-shock EXPANSION between epochs, i.e. the first genuinely
    dynamical observable the multi-epoch data offer (ROADMAP.md Stage 1).

Detector (deliberately simple, identical for both): in each 10-degree cone of
position angle in the plane of the sky, the exposure-corrected surface
brightness is binned radially in 2-pixel (0.98") steps and smoothed with a
5-bin running mean; the forward shock is the radius of STEEPEST logarithmic
decline, ``argmin d ln SB / dr``, searched between ``--r-min`` and ``--r-max``
(default 135-200"; with 110" a third of the cones lock onto the outer edge of
the bright ejecta shell at 113-130" and the spread doubles -- the per-cone
radii are printed so this is visible). A threshold-on-background definition was tried first and
landed at ~180" on the real images because the dust halo and the PSF wings keep
the profile above any background multiple well past the shock (Result 13's
r^-4..-7 tail); the blast wave is an EDGE, so its position is where the profile
drops fastest, which is also insensitive to the absolute normalisation and to
the exposure. Cones in which the steepest drop sits at a search boundary are
reported as not found.

Usage (CPU, any env with numpy)::

    python casa_real_outline.py --epochs 2000 2004 2019 2023 \
        --synthetic /export/data/lstorcks/supernova_showcase/figures_wa5_synimg.npz
"""

import argparse
from pathlib import Path

import numpy as np

REAL_EPOCH_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
PIX_ARCSEC = 0.492
DISTANCE_PC = 3400.0
ARCSEC_IN_PC = DISTANCE_PC * np.pi / (180.0 * 3600.0)


def load_image(path):
    d = np.load(path)
    counts = np.asarray(d["counts"], dtype=np.float64)
    exposure = float(d["exposure"])
    return counts, exposure


#: the Thorstensen et al. (2001) expansion centre, arcsec WEST / NORTH of the
#: image centre RA0/DEC0 (= casa_pluto_diff.COE_ARCSEC)
COE_ARCSEC = (-13.8, -4.2)


def outline(counts, exposure, *, n_angles=36, r_min_arcsec=135.0, r_max_arcsec=200.0,
            dr_pix=2, smooth_bins=5, centre_arcsec=(0.0, 0.0)):
    """Forward-shock radius (arcsec) per position angle; NaN where not found.

    Angles are theta = atan2(north, west) (PA = theta - 90), about the image
    centre RA0/DEC0 shifted by ``centre_arcsec`` = (west, north); the fits'
    data are about RA0/DEC0 (the default), and ``casa_xfit`` re-centres its
    model onto that point. (Columns increase to the WEST, rows to the NORTH.)
    """
    n = counts.shape[0]
    c = (n - 1) / 2.0
    y, x = np.indices(counts.shape)
    dx, dy = (x - c - centre_arcsec[0] / PIX_ARCSEC), (y - c - centre_arcsec[1] / PIX_ARCSEC)
    r = np.hypot(dx, dy) * PIX_ARCSEC
    pa = np.rad2deg(np.arctan2(dy, dx)) % 360.0
    sb = counts / exposure
    edges = np.arange(0.0, r_max_arcsec + 40.0, dr_pix * PIX_ARCSEC)
    rc = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(r, edges) - 1, 0, len(rc) - 1)
    angles = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    width = 360.0 / n_angles
    out = np.full(n_angles, np.nan)
    for i, a in enumerate(angles):
        cone = np.abs(((pa - a + 180.0) % 360.0) - 180.0) < width
        cnt = np.bincount(idx[cone], minlength=len(rc)).astype(float)
        tot = np.bincount(idx[cone], weights=sb[cone], minlength=len(rc))
        prof = np.divide(tot, cnt, out=np.zeros(len(rc)), where=cnt > 0)
        prof = np.convolve(prof, np.ones(smooth_bins) / smooth_bins, mode="same")
        lp = np.log(np.maximum(prof, 1e-12 * max(prof.max(), 1e-300)))
        slope = np.gradient(lp, rc)
        search = np.where((rc >= r_min_arcsec) & (rc <= r_max_arcsec))[0]
        k = search[np.argmin(slope[search])]
        if k == search[0] or k == search[-1]:
            continue                      # steepest drop at a boundary: not found
        # parabolic refinement of the minimum of the slope
        y0, y1, y2 = slope[k - 1], slope[k], slope[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        out[i] = rc[k] + np.clip(off, -1.0, 1.0) * (rc[1] - rc[0])
    return angles, out


def statistics(angles, r_arcsec):
    r = np.asarray(r_arcsec) * ARCSEC_IN_PC
    ok = np.isfinite(r)
    th = np.deg2rad(angles)[ok]
    r = r[ok]
    mean = r.mean()
    c1 = 2 * np.mean((r - mean) * np.cos(th)); s1 = 2 * np.mean((r - mean) * np.sin(th))
    c2 = 2 * np.mean((r - mean) * np.cos(2 * th)); s2 = 2 * np.mean((r - mean) * np.sin(2 * th))
    return dict(mean=mean, spread=r.max() - r.min(), std=r.std(), m1=np.hypot(c1, s1),
                m2=np.hypot(c2, s2), m1_pa=np.rad2deg(np.arctan2(s1, c1)) % 360.0,
                n_found=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", nargs="*", default=["2000", "2004", "2019", "2022"],
                    help="2023 (obsid 27099) is a CCO subarray that clips the remnant")
    ap.add_argument("--synthetic", nargs="*", default=[], help="*_synimg.npz files")
    ap.add_argument("--r-min", type=float, default=135.0, help="search window (arcsec)")
    ap.add_argument("--r-max", type=float, default=200.0)
    ap.add_argument("--n-angles", type=int, default=36)
    ap.add_argument("--per-cone", action="store_true", help="print r_FS per cone")
    ap.add_argument("--centre", choices=("image", "coe"), default="image",
                    help="measure about RA0/DEC0 (image, what the fits use) or about the "
                         "Thorstensen+01 expansion centre (coe)")
    args = ap.parse_args()
    centre = COE_ARCSEC if args.centre == "coe" else (0.0, 0.0)

    rows = []
    print(f"{'image':<30}{'<r_FS> pc':>10}{'spread':>8}{'std':>7}{'m=1':>7}{'m=2':>7}"
          f"{'th(m=1)':>9}{'found':>7}")  # theta = PA + 90
    for ep in args.epochs:
        counts, exp = load_image(REAL_EPOCH_DIR / f"epoch_{ep}.npz")
        ang, r = outline(counts, exp, n_angles=args.n_angles, r_min_arcsec=args.r_min,
                         r_max_arcsec=args.r_max, centre_arcsec=centre)
        s = statistics(ang, r)
        rows.append((float(ep), s["mean"], r))
        print(f"{'Chandra ' + ep:<30}{s['mean']:>10.3f}{s['spread']:>8.3f}{s['std']:>7.3f}"
              f"{s['m1']:>7.3f}{s['m2']:>7.3f}{s['m1_pa']:>9.0f}{s['n_found']:>7d}")
        if args.per_cone:
            print("   r_FS(PA) [arcsec]: " + " ".join(f"{v:4.0f}" for v in r))
    for path in args.synthetic:
        counts, exp = load_image(path)
        ang, r = outline(counts, exp, n_angles=args.n_angles, r_min_arcsec=args.r_min,
                         r_max_arcsec=args.r_max)
        s = statistics(ang, r)
        print(f"{Path(path).stem[:29]:<30}{s['mean']:>10.3f}{s['spread']:>8.3f}{s['std']:>7.3f}"
              f"{s['m1']:>7.3f}{s['m2']:>7.3f}{s['m1_pa']:>9.0f}{s['n_found']:>7d}")

    if len(rows) >= 2:
        yrs = np.array([r[0] for r in rows]); means = np.array([r[1] for r in rows])
        slope, icpt = np.polyfit(yrs, means, 1)
        v_kms = slope * 3.0857e13 / 3.15576e7
        print(f"\nangle-averaged r_FS expansion {yrs.min():.0f}-{yrs.max():.0f}: "
              f"{slope * 1e3:.2f} mpc/yr = {slope / ARCSEC_IN_PC:.3f} arcsec/yr "
              f"= {v_kms:.0f} km/s (plane-of-sky, at 3.4 kpc)")
        # per-cone expansion: the same cones across epochs
        r0, r1 = rows[0][2], rows[-1][2]
        ok = np.isfinite(r0) & np.isfinite(r1)
        dt = rows[-1][0] - rows[0][0]
        pm = (r1[ok] - r0[ok]) / dt
        print(f"per-cone proper motion {rows[0][0]:.0f}->{rows[-1][0]:.0f}: median "
              f"{np.median(pm):.3f} arcsec/yr, 10-90% {np.percentile(pm, 10):.3f}-"
              f"{np.percentile(pm, 90):.3f} ({ok.sum()} cones)")
        print("(a 350-yr model whose r_FS matches a 2000 image is 27 yr too old for it: "
              f"{27 * slope / ARCSEC_IN_PC:.1f} arcsec)")


if __name__ == "__main__":
    main()
