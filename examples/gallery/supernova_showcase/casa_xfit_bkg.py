"""
Measured instrumental background of the Cas A epochs for casa_xfit
(``--background``; stage-4 item 1, 2026-09-26).

The fit's old background was a constant floor, ``casa_xfit.BKG_RATE`` = 1e-6
counts/s per 1.97" pixel per band (2.6e-7 counts/s/arcsec^2). The stage-3
residual-physics round (work/stage3/physics, ``background.json``) measured the
surface brightness in an off-remnant annulus (215-245" about the expansion
centre): 1.0-1.2e-6 counts/s/arcsec^2 at 4.2-6 keV over 2002-2019, x2.6 in
2000. Adding it to the model was worth Delta chi2 -27 (R) / -31 (V3). The
stage-3 review flagged that part of that annulus is the remnant's own
out-of-time (OOT, "readout streak") events. Those are remnant counts, so adding
them as background inside the r < 200" aperture double counts them.

This module separates the two. It works from the evt2 events, so it is CPU and
astropy only.

* **Out-of-time events.** Each ACIS frame is exposed for EXPTIME (3.2 s). The
  1024-row transfer to the frame store takes TIMEDEL - EXPTIME = 41 ms, i.e.
  t_row = 40 us per row. During the transfer every row passes over the source
  for t_row, so a column with n in-time counts collects n t_row / EXPTIME OOT
  counts in EVERY row of that chip column. That is 1.28 % of n in total, spread
  uniformly along the readout direction (CHIPY). Per ObsID and chip:
  * the chipx histogram per band gives n per column;
  * a least-squares affine map (chipx, chipy) -> model grid, fitted on the
    events themselves, gives the readout direction on the sky for that roll;
  * the uniform-in-CHIPY streak is deposited through that map and smoothed by
    the 16" dither box.
  The same map gives the chip COVERAGE of every grid pixel (dither-smeared
  on-chip fraction). Off-chip annulus pixels have zero counts but pass the
  ``edge <= 0.02`` test (edge = 0 / max(0, 1)), which biased the stage-3
  annulus low by 0-17 %.
* **Particle background** per epoch and band:
  (annulus counts - OOT events - the model's own halo / PSF there)
  / (annulus exposure area), clipped at 0. Below 2.8 keV the dust halo swamps
  the annulus, so the level there is flat per keV at the 2.8-4.2 keV value (the
  stage-3 recipe). In the model it is ``sb x pixel area x coverage``; it is not
  vignetted.

``build`` writes ``BKG_FILE`` once (``python casa_xfit_bkg.py --model <casa_xfit
--save-model npz>``: the model supplies the halo it predicts in the annulus).
``background_terms`` turns it into what casa_xfit's likelihood adds, per
``--background`` mode:

* ``measured`` (default): particle (OOT-corrected) x coverage + the OOT
  image (additive, data-derived) in the images. In the spectra the particle
  background is added and the model is multiplied by (1 + the in-aperture OOT
  fraction), since the OOT events inside r < 200" are the remnant's own counts
  displaced.
* ``particle``: the OOT-corrected particle background only (no OOT term).
* ``annulus``: the stage-3 recipe (``casa_resid_analyze.particle_background``:
  no OOT correction, flat image, pi 200"^2). This reproduces the stage-3
  numbers.
* ``rate``: the old floor (casa_xfit.BKG_RATE; nothing loaded).
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

EVT_DIR = Path("/export/data/lstorcks/chandra_casa/evt2")
DATA_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs/data")
BKG_FILE = DATA_DIR / "background_stage4.npz"
CACHE_DIR = Path("/export/data/lstorcks/casa_orlando150/work/stage4/xfit/streak_cache")
STAGE3_BKG_JSON = Path("/export/data/lstorcks/casa_orlando150/work/stage3/physics/background.json")
BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))
BAND_W = np.array([b - a for a, b in BANDS])
SPEC_EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)
#: epoch -> ObsIDs (casa_jaxobs_data.EPOCHS; kept here so the loader imports nothing heavy)
EPOCHS = {
    "2000": [114], "2002": [1952], "2004": [4636], "2006": [6690],
    "2007": [9117, 9773], "2009": [10935, 12020], "2010": [10936, 13177],
    "2012": [14229], "2013": [14480], "2014": [14481], "2015": [14482],
    "2016": [19903, 18344], "2017": [19604], "2018": [19605], "2019": [19606],
    "2020": [22426], "2022": [26248], "2023": [27099],
}
COE_ARCSEC = (-13.8, -4.2)                 # casa_pluto_diff.COE_ARCSEC
ANNULUS_ARCSEC = (215.0, 245.0)            # about the CoE (stage-3 background.py)
APERTURE_ARCSEC = 200.0                    # the spectra: r < 200" about RA0/DEC0
ROW_TIME_DEFAULT = 0.04104 / 1024          # s per row, if TIMEDEL / EXPTIME are missing
DITHER_ARCSEC = 16.0                       # ACIS dither, peak to peak
COVERAGE_MIN = 0.95                        # annulus pixels used: >= 95 % on chip over the dither
#: bands whose annulus level is measured; the others are flat per keV at SOFT_FROM's level
SOFT_FROM = 3
MODES = ("measured", "particle", "annulus", "rate")


# =============================================================================
# ============ ↓ Event-level pieces (build time) ↓ ============================
# =============================================================================
def obsid_cache(obsid, *, npix=256, rebin=4, refresh=False):
    """Per ObsID: header timing and, per chip, the affine map (chipx, chipy, 1) ->
    (col, row) on the model grid plus the per-band chipx histogram. Cached as npz
    under ``CACHE_DIR``, because reading the events takes minutes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"obs_{obsid}_n{npix}_r{rebin}.npz"
    if path.exists() and not refresh:
        return dict(np.load(path))
    from astropy.io import fits
    from casa_observe import read_events
    evt = glob.glob(str(EVT_DIR / f"acisf{obsid:05d}N*_evt2.fits.gz"))[0]
    t0 = time.time()
    px, py, e, exposure = read_events(evt)
    with fits.open(evt) as f:
        ev = f["EVENTS"]
        h = ev.header
        chipx = np.asarray(ev.data["chipx"], np.int32)
        chipy = np.asarray(ev.data["chipy"], np.int32)
        ccd = np.asarray(ev.data["ccd_id"], np.int32)
        timedel, exptime = h.get("TIMEDEL"), h.get("EXPTIME")
        readmode = str(h.get("READMODE", "")); datamode = str(h.get("DATAMODE", ""))
        roll = float(h.get("ROLL_NOM", np.nan))
    col = (px - 512.0) / rebin + 0.5 * npix
    row = (py - 512.0) / rebin + 0.5 * npix
    ccds, maps, hists, nev, rms = [], [], [], [], []
    rng = np.random.default_rng(obsid)
    for c in np.unique(ccd):
        m = ccd == c
        if m.sum() < 20000:
            continue
        idx = np.nonzero(m)[0]
        sub = rng.choice(idx, size=min(len(idx), 400000), replace=False)
        A = np.stack([chipx[sub], chipy[sub], np.ones(len(sub))], 1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, np.stack([col[sub], row[sub]], 1), rcond=None)   # (3, 2)
        res = np.stack([col[sub], row[sub]], 1) - A @ coef
        hb = np.zeros((len(BANDS), 1024))
        for b, (lo, hi) in enumerate(BANDS):
            mb = m & (e >= lo) & (e < hi)
            hb[b] = np.bincount(np.clip(chipx[mb] - 1, 0, 1023), minlength=1024)[:1024]
        ccds.append(int(c)); maps.append(coef); hists.append(hb); nev.append(int(m.sum()))
        rms.append(float(np.sqrt(np.mean(np.sum(res ** 2, 1)))))
    out = dict(obsid=obsid, exposure=float(exposure), timedel=float(timedel or np.nan),
               exptime=float(exptime or np.nan), readmode=readmode, datamode=datamode, roll=roll,
               ccds=np.array(ccds), maps=np.array(maps), hists=np.array(hists), nev=np.array(nev),
               map_rms_pix=np.array(rms), npix=npix, rebin=rebin)
    np.savez_compressed(path, **out)
    print(f"[bkg] obsid {obsid}: {len(px)} events, chips {ccds} (map rms {np.round(rms, 2)} px, dither), "
          f"TIMEDEL {timedel} EXPTIME {exptime} {readmode}/{datamode}, roll {roll:.1f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return dict(np.load(path))


def row_time(c):
    """Seconds per row of the frame transfer, and the frame exposure (s)."""
    td, ex = float(c["timedel"]), float(c["exptime"])
    if np.isfinite(td) and np.isfinite(ex) and td > ex:
        return (td - ex) / 1024.0, ex
    return ROW_TIME_DEFAULT, (ex if np.isfinite(ex) else 3.2)


def _box(a, width_pix):
    """Separable running box of ``width_pix`` (fractional: linear end weights)."""
    from scipy.ndimage import convolve1d
    h = 0.5 * width_pix
    k = int(np.ceil(h))
    x = np.arange(-k, k + 1, dtype=np.float64)
    w = np.clip(h + 0.5 - np.abs(x), 0.0, 1.0)
    w /= w.sum()
    return convolve1d(convolve1d(a, w, axis=-1, mode="constant"), w, axis=-2, mode="constant")


def streak_and_coverage(c, *, supersample=2):
    """(OOT counts per grid pixel (6, npix, npix), chip coverage (npix, npix)) of
    one ObsID from its cache ``c``."""
    npix, rebin = int(c["npix"]), int(c["rebin"])
    t_row, t_exp = row_time(c)
    f_row = t_row / t_exp                                  # OOT counts per row per in-time count
    s = supersample
    q = (np.arange(1024 * s) + 0.5) / s + 0.5              # sub-pixel centres in chip coords (1..1024 pixel centres at k)
    CX, CY = np.meshgrid(q, q, indexing="ij")              # CX: chipx (column), CY: chipy (row)
    streak = np.zeros((len(BANDS), npix * npix))
    cover = np.zeros(npix * npix)
    for k, ccd in enumerate(c["ccds"]):
        coef = c["maps"][k]
        colg = coef[0, 0] * CX + coef[1, 0] * CY + coef[2, 0]
        rowg = coef[0, 1] * CX + coef[1, 1] * CY + coef[2, 1]
        ci, ri = np.floor(colg).astype(np.int64), np.floor(rowg).astype(np.int64)
        ok = (ci >= 0) & (ci < npix) & (ri >= 0) & (ri < npix)
        flat = (ri * npix + ci)[ok]
        # one native pixel = 1 / rebin^2 of a grid pixel, split over s^2 samples
        cover += np.bincount(flat, weights=np.full(flat.size, 1.0 / (rebin * s) ** 2), minlength=npix * npix)
        n_obs = c["hists"][k]                               # (6, 1024) per chipx
        n_in = n_obs / (1.0 + 1024 * f_row)                  # in-time counts per column
        col_of = np.clip(np.floor(CX - 0.5).astype(np.int64), 0, 1023)[ok]
        for b in range(len(BANDS)):
            # per native pixel n_in * f_row counts, split over the s^2 sub-samples
            wts = n_in[b][col_of] * f_row / s ** 2
            streak[b] += np.bincount(flat, weights=wts, minlength=npix * npix)
    pix = rebin * 0.492
    wd = DITHER_ARCSEC / pix
    streak = _box(streak.reshape(len(BANDS), npix, npix), wd)
    cover = np.clip(_box(cover.reshape(npix, npix), wd), 0.0, 1.0)
    return streak, cover


def grid_radius(npix, pix, centre=(0.0, 0.0)):
    ax = (np.arange(npix) - 0.5 * (npix - 1)) * pix
    NN, WW = np.meshgrid(ax, ax, indexing="ij")
    return np.hypot(WW - centre[0], NN - centre[1]), np.arctan2(NN - centre[1], WW - centre[0])
# =============================================================================
# ============ ↑ Event-level pieces (build time) ↑ ============================
# =============================================================================


# =============================================================================
# ============ ↓ Build ↓ ======================================================
# =============================================================================
def build(epochs, *, model_path=None, out=BKG_FILE, n_sec=12):
    """Per epoch and band: the OOT image, the coverage, the annulus decomposition
    (raw / OOT / model halo / particle) and the spectral terms; writes ``out``.

    ``model_path``: a casa_xfit ``--save-model`` npz, whose ``images`` (counts/s
    per pixel, CIAO responses applied) supply the model's own halo / PSF in the
    annulus. Without it the halo is taken as 0."""
    halo_imgs = None
    if model_path:
        M = np.load(model_path, allow_pickle=True)
        lab = [str(e) for e in M["epochs"]]
        halo_imgs = {e: np.asarray(M["images"][lab.index(e)], np.float64) for e in epochs if e in lab}
    res = {k: [] for k in ("streak", "coverage", "sb_raw", "sb_raw_edge", "sb_oot", "sb_halo", "sb_particle",
                           "oot_frac_ap", "ap_area", "sector_raw", "sector_corr", "exposure", "oot_total_frac")}
    for e in epochs:
        d = np.load(DATA_DIR / f"bands_{e}.npz")
        C = np.asarray(d["counts"], np.float64)
        pix, npix = float(d["pixel_arcsec"]), C.shape[-1]
        expo = float(d["exposure"])
        st = np.zeros((len(BANDS), npix, npix)); cov = np.zeros((npix, npix)); ex_sum = 0.0
        for o in EPOCHS[e]:
            c = obsid_cache(o, npix=npix)
            s_, cv = streak_and_coverage(c)
            st += s_; cov += cv * float(c["exposure"]); ex_sum += float(c["exposure"])
        cov /= max(ex_sum, 1e-30)
        rate = st / expo                                     # OOT counts/s per pixel
        R, TH = grid_radius(npix, pix, COE_ARCSEC)
        R0, _ = grid_radius(npix, pix)
        edge_ok = np.asarray(d["edge"]) <= 0.02
        ann_old = edge_ok & (R >= ANNULUS_ARCSEC[0]) & (R < ANNULUS_ARCSEC[1])
        ann = ann_old & (cov >= COVERAGE_MIN)
        area = cov[ann].sum() * pix ** 2
        sb_raw = C[:, ann].sum(1) / expo / area
        sb_raw_edge = C[:, ann_old].sum(1) / expo / (ann_old.sum() * pix ** 2)     # the stage-3 definition
        sb_oot = rate[:, ann].sum(1) / area
        sb_halo = (halo_imgs[e][:, ann].sum(1) / area) if (halo_imgs and e in halo_imgs) else np.zeros(len(BANDS))
        p = np.maximum(sb_raw - sb_oot - sb_halo, 0.0)
        p[:SOFT_FROM] = p[SOFT_FROM] / BAND_W[SOFT_FROM] * BAND_W[:SOFT_FROM]
        # azimuthal structure of the annulus (the review: +-30-50 % between
        # sectors at 4.2-6 keV) before / after removing the OOT events
        sec = np.floor(((TH + np.pi) / (2 * np.pi)) * n_sec).astype(int) % n_sec
        sr, sc = np.zeros((n_sec, len(BANDS))), np.zeros((n_sec, len(BANDS)))
        for k in range(n_sec):
            m = ann & (sec == k)
            a = cov[m].sum() * pix ** 2
            if a > 0:
                sr[k] = C[:, m].sum(1) / expo / a
                sc[k] = sr[k] - rate[:, m].sum(1) / a
        ap = R0 < APERTURE_ARCSEC
        tot_ap = C[:, ap].sum(1) / expo
        res["streak"].append(rate.astype(np.float32)); res["coverage"].append(cov.astype(np.float32))
        res["sb_raw"].append(sb_raw); res["sb_raw_edge"].append(sb_raw_edge); res["sb_oot"].append(sb_oot)
        res["sb_halo"].append(sb_halo); res["sb_particle"].append(p)
        res["oot_frac_ap"].append(rate[:, ap].sum(1) / np.maximum(tot_ap, 1e-30))
        res["oot_total_frac"].append(rate.sum((1, 2)) / np.maximum(C.sum((1, 2)) / expo, 1e-30))
        res["ap_area"].append(cov[ap].sum() * pix ** 2)
        res["sector_raw"].append(sr); res["sector_corr"].append(sc); res["exposure"].append(expo)
        print(f"[bkg] {e}: annulus {ann.sum()} px (stage-3 def {ann_old.sum()}), 1e-7 cts/s/arcsec^2 per band:\n"
              f"    raw     " + " ".join(f"{1e7 * x:7.2f}" for x in sb_raw)
              + f"\n    stage3  " + " ".join(f"{1e7 * x:7.2f}" for x in sb_raw_edge)
              + f"\n    OOT     " + " ".join(f"{1e7 * x:7.2f}" for x in sb_oot)
              + f"\n    halo    " + " ".join(f"{1e7 * x:7.2f}" for x in sb_halo)
              + f"\n    particle" + " ".join(f"{1e7 * x:7.2f}" for x in p)
              + f"\n    OOT share of annulus (%) " + " ".join(f"{100 * a / max(b, 1e-30):5.1f}" for a, b in zip(sb_oot, sb_raw))
              + f"; sector rms/mean 4.2-6 keV raw {np.std(sr[:, 4]) / np.mean(sr[:, 4]):.2f} -> "
                f"OOT-corrected {np.std(sc[:, 4]) / np.mean(sc[:, 4]):.2f}"
              + f"\n    in-aperture OOT fraction (%) " + " ".join(f"{100 * x:5.2f}" for x in res["oot_frac_ap"][-1])
              + f"; aperture area {res['ap_area'][-1] / (np.pi * APERTURE_ARCSEC ** 2):.3f} x pi 200^2", flush=True)
    arr = {k: np.stack(v) if k not in ("exposure",) else np.array(v) for k, v in res.items()}
    meta = dict(annulus=ANNULUS_ARCSEC, coe=COE_ARCSEC, coverage_min=COVERAGE_MIN, dither=DITHER_ARCSEC,
                soft_from=SOFT_FROM, model=str(model_path), aperture=APERTURE_ARCSEC,
                built=time.strftime("%Y-%m-%d %H:%M"))
    np.savez_compressed(out, epochs=np.array(epochs), bands=np.array(BANDS), meta=json.dumps(meta), **arr)
    print(f"[bkg] wrote {out}", flush=True)
    return arr
# =============================================================================
# ============ ↑ Build ↑ ======================================================
# =============================================================================


# =============================================================================
# ============ ↓ Likelihood terms (fit time) ↓ ================================
# =============================================================================
def spec_band_index(edges=SPEC_EDGES):
    """Image band of every spectral bin (by its centre)."""
    mids = 0.5 * (edges[1:] + edges[:-1])
    return np.array([[i for i, (lo, hi) in enumerate(BANDS) if lo <= m < hi][0] for m in mids])


def stage3_annulus_terms(epochs, npix, pix, model_images, bkg_json=STAGE3_BKG_JSON):
    """The stage-3 recipe (``casa_resid_analyze.particle_background``) verbatim:
    annulus SB minus the model's halo there, flat per keV below 2.8 keV at the
    2.8-4.2 keV level, a flat image and pi 200"^2 x 0.2 keV per spectral bin."""
    bk = json.loads(Path(bkg_json).read_text())
    ax = (np.arange(npix) - 0.5 * (npix - 1)) * pix
    NN, WW = np.meshgrid(ax, ax, indexing="ij")
    rr = np.hypot(WW - COE_ARCSEC[0], NN - COE_ARCSEC[1])
    ann = (rr >= 215) & (rr < 245)
    sbp = []
    for k, e in enumerate(epochs):
        p = np.maximum(np.array(bk[e]["sb"]) - model_images[k][:, ann].mean(1) / pix ** 2, 0.0)
        p[:3] = p[3] / BAND_W[3] * BAND_W[:3]
        sbp.append(p)
    sbp = np.array(sbp)
    img = np.broadcast_to(sbp[:, :, None, None] * pix ** 2, (len(epochs), len(BANDS), npix, npix))
    j = spec_band_index()
    spec = sbp[:, j] * np.pi * 200.0 ** 2 * 0.2 / BAND_W[j]
    return img.astype(np.float32), spec, np.ones_like(spec)


def background_terms(epochs, mode="measured", *, npix=256, pix=1.968, path=BKG_FILE, model_images=None):
    """What the likelihood adds for ``--background mode``, per epoch in ``epochs``:
    dict(img (E, 6, npix, npix) counts/s per pixel [additive], spec_add (E, 31)
    counts/s in the aperture [additive], spec_mult (E, 31) [model factor],
    sb_particle (E, 6), mode). ``None`` for mode ``rate``."""
    if mode not in MODES:
        raise ValueError(f"background mode {mode!r} not in {MODES}")
    if mode == "rate":
        return None
    if mode == "annulus":
        if model_images is None:
            raise ValueError("--background annulus needs the model images of the stage-3 recipe")
        img, spec, mult = stage3_annulus_terms(epochs, npix, pix, model_images)
        return dict(img=img, spec_add=spec, spec_mult=mult, sb_particle=None, mode=mode)
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing: build it with `python casa_xfit_bkg.py --model <xfit model npz>`")
    B = np.load(path, allow_pickle=True)
    lab = [str(e) for e in B["epochs"]]
    missing = [e for e in epochs if e not in lab]
    if missing:
        raise ValueError(f"{path} has no background for epochs {missing}")
    idx = [lab.index(e) for e in epochs]
    sbp = np.asarray(B["sb_particle"], np.float64)[idx]                    # (E, 6) cts/s/arcsec^2
    cov = np.asarray(B["coverage"], np.float64)[idx]                       # (E, n, n)
    if cov.shape[-1] != npix:
        raise ValueError(f"{path} is on a {cov.shape[-1]}-pixel grid, the data on {npix}")
    img = sbp[:, :, None, None] * pix ** 2 * cov[:, None]
    j = spec_band_index()
    area = np.asarray(B["ap_area"], np.float64)[idx]                       # arcsec^2 on chip
    spec = sbp[:, j] * area[:, None] * 0.2 / BAND_W[j]
    mult = np.ones_like(spec)
    if mode == "measured":
        img = img + np.asarray(B["streak"], np.float64)[idx]
        mult = 1.0 + np.asarray(B["oot_frac_ap"], np.float64)[idx][:, j]
    return dict(img=img.astype(np.float32), spec_add=spec, spec_mult=mult, sb_particle=sbp, mode=mode)
# =============================================================================
# ============ ↑ Likelihood terms (fit time) ↑ ================================
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", nargs="*", default=["2000", "2002", "2004", "2007", "2009", "2010", "2012", "2013",
                                                    "2014", "2015", "2016", "2017", "2018", "2019", "2022"])
    ap.add_argument("--model", default="/export/data/lstorcks/casa_orlando150/work/xfit_Rp_n128.npz",
                    help="casa_xfit --save-model npz: the model's own halo / PSF in the annulus")
    ap.add_argument("--out", default=str(BKG_FILE))
    ap.add_argument("--cache-only", action="store_true", help="only read the events into the cache")
    a = ap.parse_args()
    if a.cache_only:
        for e in a.epochs:
            for o in EPOCHS[e]:
                obsid_cache(o)
        return
    build(a.epochs, model_path=a.model or None, out=a.out)


if __name__ == "__main__":
    main()
