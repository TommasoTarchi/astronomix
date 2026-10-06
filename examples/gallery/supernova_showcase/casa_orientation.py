"""
Is the model oriented like the sky? Test by composition, not by outline.

The outline's m = 1 lopsidedness (PLUTO150.md section 3) cannot tell a mirror
image from the right one and is a ~10 % effect. Cas A's element-specific
emission is not: the Si-rich NE jet / SW counter-jet and the Fe-rich SE and NW
regions are the remnant's most distinctive large-scale features. So per band
(Si He-a, S He-a, Fe-K) the azimuthal profile of the band's FRACTION of the
broadband counts is built in an annulus, for the real remnant and for a
synthetic event file, through the same binning (``casa_observe.read_events``),
and the model is compared under every rotation (5-degree steps) and with and
without a mirror flip. The best transformation, and how much better it is than
the identity, says whether the orientation convention is right.

Usage (xrayobs env)::

    python casa_orientation.py SYN_evt.fits --obsid 4636 --annulus 60 170
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from casa_observe import read_events

EVT_DIR = Path("/export/data/lstorcks/chandra_casa/evt2")
BANDS = {"Si He-a": (1.75, 1.95), "S He-a": (2.35, 2.55), "Fe-K": (6.4, 6.8),
         "continuum 4.2-6": (4.2, 6.0)}


def azimuthal_fractions(evtfile, *, annulus, n_pa=36, centre=None):
    """Per band, the fraction of 0.5-7 keV counts in each PA bin of the annulus.

    PA is measured on the image as atan2(dy, dx) with dx to the RIGHT (west)
    and dy up (north), the convention ``casa_real_outline`` uses.
    """
    px, py, e, _ = read_events(str(evtfile))
    n = 1024
    c = ((n - 1) / 2.0, (n - 1) / 2.0) if centre is None else centre
    dx, dy = px - c[0], py - c[1]
    r = np.hypot(dx, dy) * 0.492
    pa = np.rad2deg(np.arctan2(dy, dx)) % 360.0
    ring = (r > annulus[0]) & (r < annulus[1])
    k = np.clip((pa / (360.0 / n_pa)).astype(int), 0, n_pa - 1)
    broad = np.bincount(k[ring & (e > 0.5) & (e < 7.0)], minlength=n_pa).astype(float)
    out = {}
    for name, (lo, hi) in BANDS.items():
        cnt = np.bincount(k[ring & (e > lo) & (e < hi)], minlength=n_pa).astype(float)
        out[name] = cnt / np.maximum(broad, 1.0)
    out["broad"] = broad / broad.mean()
    return out


def transform(profile, rot_bins, mirror):
    """Profile of the model image rotated by rot_bins PA bins and/or mirrored
    left-right (PA -> 180 - PA)."""
    p = profile
    if mirror:
        n = len(p)
        idx = (np.round((180.0 - (np.arange(n) + 0.5) * 360.0 / n) / (360.0 / n) - 0.5)
               .astype(int)) % n
        p = p[idx]
    return np.roll(p, rot_bins)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("synthetic", help="SOXS event file of the model")
    ap.add_argument("--obsid", type=int, default=4636, help="real evt2 (2004 = 4636)")
    ap.add_argument("--annulus", type=float, nargs=2, default=(60.0, 170.0))
    args = ap.parse_args()
    real_file = glob.glob(str(EVT_DIR / f"acisf{args.obsid:05d}N*_evt2.fits.gz"))[0]
    real = azimuthal_fractions(real_file, annulus=args.annulus)
    syn = azimuthal_fractions(args.synthetic, annulus=args.annulus)
    n = len(real["broad"])
    print(f"[orient] annulus {args.annulus[0]:.0f}-{args.annulus[1]:.0f}\", {n} PA bins; "
          f"real {Path(real_file).name}, model {Path(args.synthetic).name}")
    total = np.zeros((2, n))
    for name in list(BANDS) + ["broad"]:
        a = real[name] - real[name].mean()
        corr = np.zeros((2, n))
        for mirror in (0, 1):
            for s in range(n):
                b = transform(syn[name], s, mirror)
                b = b - b.mean()
                corr[mirror, s] = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-300)
        total += corr
        m, s = np.unravel_index(np.argmax(corr), corr.shape)
        print(f"  {name:16s} identity corr {corr[0, 0]:+.2f}; best {corr[m, s]:+.2f} at "
              f"{'mirror + ' if m else ''}rotation {s * 360 / n:.0f} deg")
    m, s = np.unravel_index(np.argmax(total), total.shape)
    print(f"[orient] summed over bands: identity {total[0, 0] / 5:+.2f}, best "
          f"{total[m, s] / 5:+.2f} at {'MIRROR + ' if m else ''}rotation {s * 360 / n:.0f} deg; "
          f"best without mirror {total[0].max() / 5:+.2f} at "
          f"{np.argmax(total[0]) * 360 / n:.0f} deg, best with mirror "
          f"{total[1].max() / 5:+.2f} at {np.argmax(total[1]) * 360 / n:.0f} deg")


if __name__ == "__main__":
    main()
