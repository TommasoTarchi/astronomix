"""
The measured Si Doppler pattern of 2004: the DATA side of ``casa_jaxobs``'
Doppler statistic, as a script (it used to exist only as a one-off inline).

Per sky sector (24 x 15 deg, PA from west through north) of an annulus about a
sky centre, the mean event energy in [1.78, 1.94] keV, converted to a
line-of-sight velocity v = -c (<E> - mean over sectors) / E0 (E0 = 1.86 keV;
+ = receding), and its error std(E) / sqrt(N) c / E0.

``--legacy`` reproduces ``work/doppler_si_2004.npz`` bit for bit: the inline
script of 2026-09-23 (obsid 4636 only, ``read_events`` grid centred on pixel
511.5 -- 0.35" SE of RA0/DEC0, no CCO mask). Its provenance, recovered from
the session log: centre = the grid centre, NOT the explosion centre; energies
= the evt2 ENERGY column (no extra gain correction); CCO not masked.

The model statistic that matches it is ``casa_jaxobs.doppler_columns`` (native-
channel M0 / M1 moments in exactly this window) reduced by
``casa_jaxobs.doppler_from_columns`` with the model placed on the sky
(roll, explosion-centre offset (dw, dn) from RA0/DEC0, scale) and the sectors
about RA0/DEC0 -- the data frame. The model uses PI-channel energies with
counts uniform within a 14.6 eV channel; the data use the (continuous) event
energies: the same up to the distribution inside one channel.

Default (non-legacy) output: centre exactly RA0/DEC0, the CCO masked within 6"
(the point source is not in the hydro model; <= 0.3 % of the window's counts),
all obsids of the epoch that are on disk (only 4636 of the 2004 VLP is).

Run in the xrayobs env::

    /export/home/lstorcks/xrayobs/bin/python casa_jaxobs_doppler_data.py \\
        --out /export/data/lstorcks/casa_orlando150/work/doppler_si_2004_v2.npz
"""

import argparse
import glob
from pathlib import Path

import numpy as np

EVT_DIR = Path("/export/data/lstorcks/chandra_casa/evt2")
LEGACY_OUT = Path("/export/data/lstorcks/casa_orlando150/work/doppler_si_2004.npz")
RA0, DEC0 = 350.8583, 58.8149                 # casa_observe / casa_jaxobs grid centre
CCO_RADEC = (350.866417, 58.811778)
PIX_ARCSEC = 0.492
C_KMS_LEGACY = 3e5                            # the inline script's c (kept for --legacy)
C_KMS = 2.99792458e5


def sector_statistic(px, py, e, *, centre_px, n_sec=24, annulus=(40.0, 170.0), window=(1.78, 1.94),
                     mask=None, c_kms=C_KMS):
    """(v, v_err, <E>, N) per sector from read_events pixels (px west, py north)."""
    dx, dy = px - centre_px[0], py - centre_px[1]
    r = np.hypot(dx, dy) * PIX_ARCSEC
    pa = np.rad2deg(np.arctan2(dy, dx)) % 360
    ring = (r > annulus[0]) & (r < annulus[1])
    k = (pa / (360 / n_sec)).astype(int) % n_sec
    lo, hi = window
    s = ring & (e > lo) & (e < hi)
    if mask is not None:
        s &= mask
    E0 = (lo + hi) / 2
    m = np.array([e[s & (k == i)].mean() for i in range(n_sec)])
    n = np.array([(s & (k == i)).sum() for i in range(n_sec)])
    err = np.array([e[s & (k == i)].std() / np.sqrt(n[i]) for i in range(n_sec)])
    return -c_kms * (m - m.mean()) / E0, c_kms * err / E0, m, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--obsids", type=int, nargs="+", default=None,
                    help="default: every 2004 obsid on disk (4636)")
    ap.add_argument("--legacy", action="store_true", help="reproduce doppler_si_2004.npz exactly")
    ap.add_argument("--centre-offset", type=float, nargs=2, default=(0.0, 0.0),
                    help="sector centre, arcsec west / north of RA0/DEC0")
    ap.add_argument("--cco-mask", type=float, default=6.0, help="arcsec; 0 = no mask")
    ap.add_argument("--window", type=float, nargs=2, default=(1.78, 1.94))
    ap.add_argument("--annulus", type=float, nargs=2, default=(40.0, 170.0))
    ap.add_argument("--n-sec", type=int, default=24)
    ap.add_argument("--year", type=float, default=2004.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from casa_observe import read_events

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    obsids = args.obsids or ([4636] if args.legacy else
                             sorted(int(Path(p).name[5:10]) for p in glob.glob(str(EVT_DIR / "acisf0463[4-9]N*_evt2.fits.gz"))
                                    + glob.glob(str(EVT_DIR / "acisf05196N*_evt2.fits.gz"))
                                    + glob.glob(str(EVT_DIR / "acisf0531[9]N*_evt2.fits.gz"))
                                    + glob.glob(str(EVT_DIR / "acisf05320N*_evt2.fits.gz"))))
    PX, PY, EE = [], [], []
    for o in obsids:
        f = glob.glob(str(EVT_DIR / f"acisf{o:05d}N*_evt2.fits.gz"))[0]
        px, py, e, _ = read_events(f)
        PX.append(px); PY.append(py); EE.append(e)
    px, py, e = np.concatenate(PX), np.concatenate(PY), np.concatenate(EE)
    if args.legacy:
        centre = (511.5, 511.5)
        mask = None
        c_kms = C_KMS_LEGACY
    else:
        # read_events: px = 512 - xi / scale (xi east), py = 512 + eta / scale
        centre = (512.0 + args.centre_offset[0] / PIX_ARCSEC, 512.0 + args.centre_offset[1] / PIX_ARCSEC)
        mask = None
        if args.cco_mask > 0:
            cw = -(CCO_RADEC[0] - RA0) * np.cos(np.deg2rad(DEC0)) * 3600.0
            cn = (CCO_RADEC[1] - DEC0) * 3600.0
            mask = np.hypot((px - 512.0) * PIX_ARCSEC - cw, (py - 512.0) * PIX_ARCSEC - cn) >= args.cco_mask
        c_kms = C_KMS
    v, ve, m, n = sector_statistic(px, py, e, centre_px=centre, n_sec=args.n_sec, annulus=tuple(args.annulus),
                                   window=tuple(args.window), mask=mask, c_kms=c_kms)
    N = args.n_sec
    extra = {} if args.legacy else dict(
        mean_energy_kev=m, counts=n, obsids=np.array(obsids), centre_offset_arcsec=np.array(args.centre_offset),
        centre="RA0/DEC0 + offset (west, north)", cco_mask_arcsec=args.cco_mask, E0_kev=0.5 * sum(args.window),
        statistic="v = -c (<E>_sector - mean_sectors <E>) / E0; <E> = mean event ENERGY in (lo, hi)")
    np.savez(out, angles=(np.arange(N) + 0.5) * 360 / N, v_kms=v, v_err_kms=ve, year=args.year,
             annulus_arcsec=np.array(args.annulus if not args.legacy else [40, 170]),
             band_kev=np.array(args.window), **extra)
    print(f"[doppler-data] obsids {obsids}, {int(n.sum())} events in the window: v = " + " ".join(f"{x:+.0f}" for x in v))
    print(f"[doppler-data] errors {np.round(ve).astype(int).tolist()}; wrote {out}")


if __name__ == "__main__":
    main()
