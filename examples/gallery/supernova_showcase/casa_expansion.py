"""
Forward-shock proper motions of the real remnant by profile REGISTRATION.

``casa_real_outline`` locates the forward shock per epoch as the steepest
logarithmic drop of the cone-averaged surface brightness, then differences
epochs. That estimator jumps between neighbouring edges from epoch to epoch
(per-cone proper motions from -0.7 to +1.0"/yr), and the two ways of
summarising it disagree: 0.294"/yr for the slope of the epoch-mean radius,
0.345"/yr for the median per-cone motion. Since the Orlando-state fit's
age/distance tension hinges on exactly this number (PLUTO150.md section 4),
it is re-measured here the way expansion is measured in the literature: not
by finding an edge, but by finding the SHIFT that best superposes one epoch's
rim profile on another's.

Per 10-degree cone, the exposure-corrected radial profile of each epoch in the
rim window (default 130-210") is compared with the reference epoch's; the
radial shift maximising the correlation of the two profiles' log-derivatives
(the edge structure, insensitive to the brightness decline and to the ACIS-S /
ACIS-I response difference) is found to sub-bin precision. Each epoch is first
re-centred on the central compact object (a point source that moves < 0.03"/yr),
which removes the ~0.5" absolute-astrometry scatter between observations --
but the CCO is not at rest: it sits 6.6" from the expansion centre at PA 169
(Fesen et al. 2006) after ~340 yr, i.e. it moves at ~0.019"/yr toward PA 169,
and locking every epoch onto it subtracts that motion from every cone
(v_CCO . n_hat: a 0.02"/yr dipole). ``--cco-pm`` (default on; 2026-09-25 audit)
adds it back: each epoch is re-centred on the CCO's centroid MINUS its
predicted displacement since the reference epoch, so the frame follows a fixed
sky point. ``--cco-pm off`` reproduces the CCO-locked registration.
Proper motion per cone = slope of shift vs time; the remnant-average is
reported both as the mean over cones and as a global radial scaling
(r -> r (1 + e)), the Vink et al. (2022) style expansion rate.

Usage (CPU)::

    python casa_expansion.py --out /export/data/lstorcks/casa_orlando150/work/expansion_ccopm.npz
    python casa_expansion.py --outline-window 12 \
        --out /export/data/lstorcks/casa_orlando150/work/expansion_fs12_ccopm.npz
"""

import argparse
from pathlib import Path

import numpy as np

REAL_EPOCH_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
PIX = 0.492
#: epochs covering the whole rim (2006/2020/2023 are offset or subarray pointings)
EPOCHS = ("2000", "2002", "2004", "2007", "2009", "2010", "2012", "2013", "2014",
          "2015", "2016", "2017", "2018", "2019", "2022")
#: central compact object (Tananbaum et al. 1999), relative to the grid centre
CCO_RA, CCO_DEC = 350.86642, 58.81178
RA0, DEC0 = 350.8583, 58.8149
#: the CCO's own motion, from its offset from the Thorstensen+01 expansion
#: centre (Fesen et al. 2006: 6.6 +- 1.5" at PA 169 +- 8 deg) over the remnant's
#: age at the registration epochs (~340 yr): mu = 6.6 / 340 = 0.019"/yr
CCO_SEP_ARCSEC, CCO_PA_DEG, CCO_AGE_YR = 6.6, 169.0, 340.0


def cco_motion(sep=CCO_SEP_ARCSEC, pa_deg=CCO_PA_DEG, age=CCO_AGE_YR):
    """The CCO's proper motion (arcsec/yr, (west, north)), assuming it left the
    expansion centre at the explosion and moved ballistically."""
    mu = sep / age
    pa = np.deg2rad(pa_deg)
    return -mu * np.sin(pa), mu * np.cos(pa)


def decimal_year(z, label):
    if "dates" in z.files:
        ys = []
        for s in (str(x) for x in z["dates"]):
            y, m, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
            ys.append(y + (m - 1) / 12.0 + (d - 1) / 365.0)
        return float(np.mean(ys))
    return float(label) + 0.5


def cco_centre(rate, guess, box=12):
    """Centroid of the CCO in a small box around ``guess`` (pixel x, y)."""
    gx, gy = int(round(guess[0])), int(round(guess[1]))
    sub = rate[gy - box:gy + box + 1, gx - box:gx + box + 1]
    sub = np.clip(sub - np.median(sub), 0.0, None)
    # iterate on a shrinking window so the diffuse emission does not pull it
    cy, cx = np.unravel_index(np.argmax(sub), sub.shape)
    for half in (5, 3):
        y0, x0 = max(cy - half, 0), max(cx - half, 0)
        w = sub[y0:cy + half + 1, x0:cx + half + 1]
        yy, xx = np.indices(w.shape)
        cy = y0 + float((w * yy).sum() / w.sum())
        cx = x0 + float((w * xx).sum() / w.sum())
        cy, cx = int(round(cy)), int(round(cx))
    yy, xx = np.indices(w.shape)
    return (gx - box + x0 + float((w * xx).sum() / w.sum()),
            gy - box + y0 + float((w * yy).sum() / w.sum()))


def cone_profiles(rate, centre, *, n_angles=36, r_min=100.0, r_max=240.0, dr=0.5):
    """Mean surface brightness per (cone, radial bin), centred on ``centre``."""
    y, x = np.indices(rate.shape)
    dx, dy = x - centre[0], y - centre[1]
    r = np.hypot(dx, dy) * PIX
    pa = np.rad2deg(np.arctan2(dy, dx)) % 360.0
    edges = np.arange(r_min, r_max + dr, dr)
    rc = 0.5 * (edges[:-1] + edges[1:])
    rb = np.digitize(r, edges) - 1
    ok = (rb >= 0) & (rb < len(rc))
    width = 360.0 / n_angles
    angles = np.arange(n_angles) * width
    prof = np.full((n_angles, len(rc)), np.nan)
    for i, a in enumerate(angles):
        cone = ok & (np.abs(((pa - a + 180.0) % 360.0) - 180.0) < 0.5 * width)
        cnt = np.bincount(rb[cone], minlength=len(rc)).astype(float)
        tot = np.bincount(rb[cone], weights=rate[cone], minlength=len(rc))
        prof[i] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return angles, rc, prof


def edge_signal(p, smooth=5):
    """Log-derivative of a smoothed profile: the edge structure, amplitude-free."""
    k = np.ones(smooth) / smooth
    ps = np.convolve(np.nan_to_num(p, nan=np.nanmedian(p)), k, mode="same")
    return np.gradient(np.log(np.maximum(ps, 1e-12 * np.nanmax(ps))))


def best_shift(ref, cur, rc, window, max_shift=12.0):
    """Radial shift (arcsec) of ``cur`` relative to ``ref`` maximising the
    correlation of their edge signals inside ``window``, sub-bin via parabola."""
    dr = rc[1] - rc[0]
    sel = (rc >= window[0]) & (rc <= window[1])
    a = ref[sel] - ref[sel].mean()
    lags = np.arange(-int(max_shift / dr), int(max_shift / dr) + 1)
    cc = np.empty(len(lags))
    idx = np.nonzero(sel)[0]
    for j, L in enumerate(lags):
        b = cur[np.clip(idx + L, 0, len(cur) - 1)]
        b = b - b.mean()
        cc[j] = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-300)
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        den = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        off = 0.5 * (cc[k - 1] - cc[k + 1]) / den if den < 0 else 0.0
    else:
        off = 0.0
    return (lags[k] + off) * dr, cc[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", default="2004", help="deepest epoch (143 ks)")
    ap.add_argument("--window", type=float, nargs=2, default=(130.0, 210.0),
                    help="fixed rim window (arcsec); ignored with --outline-window")
    ap.add_argument("--outline-window", type=float, default=None, metavar="HALF",
                    help="per-cone window of +-HALF arcsec around that cone's forward "
                         "shock in the reference epoch (casa_real_outline), so bright "
                         "ejecta inside the blast wave cannot dominate the registration")
    ap.add_argument("--no-cco", action="store_true", help="skip the CCO re-centring")
    ap.add_argument("--cco-pm", choices=("on", "off"), default="on",
                    help="add the CCO's own proper motion back (on) or lock the frame to "
                         "the CCO (off: the pre-2026-09-25 registration)")
    ap.add_argument("--cco-sep", type=float, default=CCO_SEP_ARCSEC, help="CCO - CoE offset (arcsec)")
    ap.add_argument("--cco-pa", type=float, default=CCO_PA_DEG, help="its position angle (deg)")
    ap.add_argument("--cco-age", type=float, default=CCO_AGE_YR, help="age it was acquired over (yr)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # CCO guess position on the common grid (RA increases to the left)
    npix = 1024
    xi = np.cos(np.deg2rad(CCO_DEC)) * (CCO_RA - RA0) * 3600.0 / PIX
    eta = (CCO_DEC - DEC0) * 3600.0 / PIX
    guess = (npix / 2 - xi, npix / 2 + eta)

    data = {}
    for ep in EPOCHS:
        z = np.load(REAL_EPOCH_DIR / f"epoch_{ep}.npz")
        rate = np.asarray(z["counts"], float) / float(z["exposure"])
        c = guess if args.no_cco else cco_centre(rate, guess)
        data[ep] = (decimal_year(z, ep), rate, c)
    cref = data[args.reference][2]
    # the expansion centre is fixed (the CCO position in the reference epoch,
    # offset per epoch by that epoch's CCO registration)
    centre0 = ((npix - 1) / 2.0, (npix - 1) / 2.0)
    print(f"[exp] CCO centroids (pix): " + ", ".join(
        f"{ep} ({c[0] - cref[0]:+.2f},{c[1] - cref[1]:+.2f})" for ep, (_, _, c) in data.items()))

    # the CCO's predicted displacement since the reference epoch (pixels; x
    # increases to the WEST, y to the NORTH on these grids): subtracted from the
    # measured centroid shift, which then carries only the astrometric offset
    v_w, v_n = cco_motion(args.cco_sep, args.cco_pa, args.cco_age) if (
        args.cco_pm == "on" and not args.no_cco) else (0.0, 0.0)
    yr_ref = data[args.reference][0]
    print(f"[exp] CCO proper motion added back: ({v_w:+.4f}, {v_n:+.4f}) arcsec/yr (west, north)"
          if (v_w or v_n) else "[exp] frame locked to the CCO (its motion is subtracted)")
    prof = {}
    for ep, (yr, rate, c) in data.items():
        dx = v_w * (yr - yr_ref) / PIX; dy = v_n * (yr - yr_ref) / PIX
        centre = (centre0[0] + c[0] - cref[0] - dx, centre0[1] + c[1] - cref[1] - dy)
        angles, rc, prof[ep] = cone_profiles(rate, centre)
    ref_sig = [edge_signal(p) for p in prof[args.reference]]
    years = np.array([data[ep][0] for ep in EPOCHS])
    shifts = np.full((len(EPOCHS), len(angles)), np.nan)
    corr = np.full_like(shifts, np.nan)
    windows = [tuple(args.window)] * len(angles)
    if args.outline_window:
        from casa_real_outline import outline
        _, r_ref = outline(data[args.reference][1], 1.0)
        windows = [(r - args.outline_window, r + args.outline_window) if np.isfinite(r)
                   else tuple(args.window) for r in r_ref]
    for e, ep in enumerate(EPOCHS):
        for k in range(len(angles)):
            shifts[e, k], corr[e, k] = best_shift(ref_sig[k], edge_signal(prof[ep][k]),
                                                  rc, windows[k])
    good = corr > 0.5
    pm = np.full(len(angles), np.nan); pm_err = np.full(len(angles), np.nan)
    for k in range(len(angles)):
        ok = good[:, k]
        if ok.sum() >= 6:
            c, cov = np.polyfit(years[ok], shifts[ok, k], 1, cov=True)
            pm[k], pm_err[k] = c[0], np.sqrt(cov[0, 0])
    # global expansion: shift(e, k) = e_rate * (t - t_ref) * r_ref(k), fitted jointly
    s_ep = np.array([np.nanmean(np.where(good[e], shifts[e], np.nan)) for e in range(len(EPOCHS))])
    c_all = np.polyfit(years, s_ep, 1)
    s_ep_acis_s = [e for e, ep in enumerate(EPOCHS) if ep != "2022"]
    c_s = np.polyfit(years[s_ep_acis_s], s_ep[s_ep_acis_s], 1)
    print(f"[exp] per-cone proper motion: mean {np.nanmean(pm):.3f}, median "
          f"{np.nanmedian(pm):.3f}\"/yr, 10-90% {np.nanpercentile(pm, 10):.3f}-"
          f"{np.nanpercentile(pm, 90):.3f}; typical per-cone error {np.nanmedian(pm_err):.3f}")
    print(f"[exp] epoch-mean shift slope: {c_all[0]:.3f}\"/yr (all), {c_s[0]:.3f}\"/yr "
          f"(ACIS-S only, 2000-2019)")
    print("[exp] shifts (\") relative to", args.reference, "by epoch:",
          " ".join(f"{ep}:{s:+.2f}" for ep, s in zip(EPOCHS, s_ep)))
    print("[exp] per-cone PM (\"/yr):", " ".join(f"{a:.0f}:{v:.2f}" for a, v in zip(angles, pm)))
    if args.out:
        np.savez(args.out, epochs=np.array(EPOCHS), years=years, angles=angles,
                 shifts=shifts, corr=corr, pm=pm, pm_err=pm_err, epoch_mean_shift=s_ep,
                 windows=np.array(windows), reference=args.reference,
                 cco_pm_arcsec_per_yr=np.array([v_w, v_n]), cco_pm=args.cco_pm)
        print(f"[exp] wrote {args.out}")


if __name__ == "__main__":
    main()
