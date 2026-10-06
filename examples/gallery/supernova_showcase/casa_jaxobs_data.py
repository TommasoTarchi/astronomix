"""
Real Chandra band images on the grid of the differentiable observation model.

``casa_jaxobs.band_images`` predicts counts/s per pixel in the six bands of
``casa_jaxobs.BANDS`` on an ``npix x npix`` grid of ``4 x 0.492"`` pixels,
row = north, column = west, pixel centre (npix - 1) / 2 at Cas A's centre
``(RA0, DEC0)``. This bins every epoch's evt2 events (through
``casa_observe.read_events``, the same projection the synthetic events get)
onto exactly that grid, per band, and writes per epoch:

* ``counts`` (n_band, npix, npix) -- summed over the epoch's obsids;
* ``exposure`` (s), ``dates``, ``obsids``, ``detnam`` (``ACIS-7`` = S3 only);
* ``edge`` (npix, npix) -- the fraction of a pixel's events that were detected
  within ``EDGE_PIX`` chip pixels of a CCD edge. The dither (16") smears chip
  gaps and edges over ~32 chip pixels, where the effective exposure is lower
  than the header value; without an exposure map those pixels must be masked
  (``edge > 0.02``) rather than modelled.

Run in the xrayobs env::

    /export/home/lstorcks/xrayobs/bin/python casa_jaxobs_data.py
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from casa_observe import read_events

EVT_DIR = Path("/export/data/lstorcks/chandra_casa/evt2")
OUT_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs/data")
BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))
EDGE_PIX = 40            # chip pixels: dither amplitude 16" = 32.5 px, plus margin

#: epoch -> obsids, as make_epoch_images.py
EPOCHS = {
    "2000": [114], "2002": [1952], "2004": [4636], "2006": [6690],
    "2007": [9117, 9773], "2009": [10935, 12020], "2010": [10936, 13177],
    "2012": [14229], "2013": [14480], "2014": [14481], "2015": [14482],
    "2016": [19903, 18344], "2017": [19604], "2018": [19605], "2019": [19606],
    "2020": [22426], "2022": [26248], "2023": [27099],
}


def bin_obsid(obsid, npix, rebin):
    from astropy.io import fits
    path = glob.glob(str(EVT_DIR / f"acisf{obsid:05d}N*_evt2.fits.gz"))[0]
    px, py, e, exposure = read_events(path)
    with fits.open(path) as f:
        ev = f["EVENTS"]
        chipx = np.asarray(ev.data["chipx"]); chipy = np.asarray(ev.data["chipy"])
        date = ev.header.get("DATE-OBS", ""); detnam = ev.header.get("DETNAM", "")
    # read_events' grid is 1024 native pixels with (512, 512) at the centre;
    # the model grid is npix pixels of `rebin` native pixels, same centre
    col = (px - 512.0) / rebin + 0.5 * npix
    row = (py - 512.0) / rebin + 0.5 * npix
    ok = (col >= 0) & (col < npix) & (row >= 0) & (row < npix)
    ci, ri = col.astype(int), row.astype(int)
    counts = np.zeros((len(BANDS), npix, npix))
    for b, (lo, hi) in enumerate(BANDS):
        m = ok & (e >= lo) & (e < hi)
        np.add.at(counts[b], (ri[m], ci[m]), 1.0)
    near = (chipx < EDGE_PIX) | (chipx > 1024 - EDGE_PIX) | (chipy < EDGE_PIX) | (chipy > 1024 - EDGE_PIX)
    m = ok & (e >= 0.5) & (e < 7.0)
    tot = np.zeros((npix, npix)); edge = np.zeros((npix, npix))
    np.add.at(tot, (ri[m], ci[m]), 1.0)
    np.add.at(edge, (ri[m & near], ci[m & near]), 1.0)
    return counts, edge, tot, float(exposure), date, detnam


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npix", type=int, default=256)
    ap.add_argument("--rebin", type=int, default=4, help="native 0.492\" pixels per model pixel")
    ap.add_argument("--epochs", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for label, obsids in EPOCHS.items():
        if args.epochs and label not in args.epochs:
            continue
        C = 0.0; E = 0.0; T = 0.0; exp = 0.0; dates = []; dets = []
        for o in obsids:
            c, e, t, x, d, dn = bin_obsid(o, args.npix, args.rebin)
            C, E, T, exp = C + c, E + e, T + t, exp + x
            dates.append(d); dets.append(dn)
        edge = E / np.maximum(T, 1.0)
        path = out / f"bands_{label}.npz"
        np.savez_compressed(path, counts=C.astype(np.float32), exposure=exp, dates=np.array(dates),
                            obsids=np.array(obsids), detnam=np.array(dets),
                            edge=edge.astype(np.float32), bands=np.array(BANDS),
                            pixel_arcsec=args.rebin * 0.492)
        print(f"[data] {label}: obsids {obsids} ({', '.join(dets)}), {exp / 1e3:.1f} ks, "
              f"band counts " + " ".join(f"{s:.3g}" for s in C.sum((1, 2))) +
              f", edge-masked pixels {(edge > 0.02).sum()}", flush=True)


if __name__ == "__main__":
    main()
