"""
Side-by-side X-ray timelapse: Chandra (left) against the model (right), 2000-2022.

The real side is the epoch images of ``/export/data/lstorcks/chandra_casa/
epoch_images`` (0.5-7 keV, 0.492" pixels, exposure-corrected); the model side
is the ``*_synimg.npz`` written by ``casa_observe.py --compare <epoch>`` for
the state evolved to that epoch's age (``casa_orlando.py --snapshot-ages``),
observed through that epoch's ACIS array and cycle, with the dust halo, the
sub-grid split and the synchrotron rim. Both sides go through IDENTICAL
display processing: counts -> counts/s, the same Gaussian smoothing, the same
per-frame asinh stretch anchored on each frame's own percentiles (so secular
fading and instrument changes read as morphology, not as exposure), the same
colour map, and a time-proportional crossfade between epochs. Nothing is
aligned, deconvolved or sharpened. The synthetic exposure is 20 ks against
50-143 ks real, so the model side is noisier at the same stretch; that is
stated on the frame rather than hidden.

Usage (CPU, xrayobs env; ffmpeg via imageio-ffmpeg or on PATH)::

    python casa_timelapse.py --epochs 2000 2002 2004 2007 2010 2013 2016 2019 2022 \
        --model-prefix /export/data/lstorcks/supernova_showcase/tl_a075 \
        --out figures/casa_timelapse.mp4
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

REAL_EPOCH_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
PIX_ARCSEC = 0.492


def decimal_year(label, npz):
    """Mean observation date of the epoch if the image carries one, else the label."""
    if "dates" in npz.files:
        ys = []
        for s in (str(x) for x in npz["dates"]):
            y, m, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
            ys.append(y + (m - 1) / 12.0 + (d - 1) / 365.0)
        return float(np.mean(ys))
    return float(label) + 0.5


def rate_image(path):
    z = np.load(path)
    counts = np.asarray(z["counts"], dtype=np.float64)
    exposure = float(z["exposure"])
    return counts / exposure, exposure, z


def stretch(rate, *, crop, smooth, cmap):
    n = rate.shape[0]
    c = n // 2
    img = gaussian_filter(rate[c - crop:c + crop, c - crop:c + crop], smooth)
    pos = img[img > 0]
    soft = np.percentile(pos, 60) if pos.size else 1.0
    vmax = np.percentile(img, 99.95)
    floor = np.percentile(img, 30)
    a = np.arcsinh(np.clip(img, 0, None) / soft)
    lo, hi = np.arcsinh(floor / soft), np.arcsinh(vmax / soft)
    s = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    return (cmap(s)[:, :, :3] * 255).astype(np.uint8)


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main():
    import matplotlib
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", nargs="+", required=True)
    ap.add_argument("--model-prefix", required=True,
                    help="model images are <prefix>_<epoch>_synimg.npz")
    ap.add_argument("--out", default="figures/casa_timelapse.mp4")
    ap.add_argument("--crop", type=int, default=300, help="half-size in pixels (300 = 148\")")
    ap.add_argument("--smooth", type=float, default=1.5, help="Gaussian sigma in pixels")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--years-per-second", type=float, default=2.5)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds held on first/last epoch")
    ap.add_argument("--cmap", default="afmhot")
    ap.add_argument("--gif", action="store_true", help="also write an animated GIF (every 4th frame)")
    args = ap.parse_args()
    cmap = matplotlib.colormaps[args.cmap]

    frames = []
    for ep in args.epochs:
        real, real_exp, z = rate_image(REAL_EPOCH_DIR / f"epoch_{ep}.npz")
        model_path = Path(f"{args.model_prefix}_{ep}_synimg.npz")
        if not model_path.exists():
            print(f"[timelapse] no model image for {ep} ({model_path}); skipping")
            continue
        model, model_exp, _ = rate_image(model_path)
        t = decimal_year(ep, z)
        frames.append(dict(ep=ep, t=t,
                           real=stretch(real, crop=args.crop, smooth=args.smooth, cmap=cmap),
                           model=stretch(model, crop=args.crop, smooth=args.smooth, cmap=cmap),
                           real_ks=real_exp / 1e3, model_ks=model_exp / 1e3,
                           real_rate=float(real.sum()), model_rate=float(model.sum())))
        print(f"[timelapse] {ep}: t = {t:.2f}, real {real_exp / 1e3:.0f} ks "
              f"{real.sum():.0f} c/s, model {model_exp / 1e3:.0f} ks {model.sum():.0f} c/s")
    if len(frames) < 2:
        raise SystemExit("need at least two epochs with model images")
    frames.sort(key=lambda f: f["t"])

    size = 2 * args.crop
    gap = size // 30
    top = size // 12                     # title band above the images
    W, H = 2 * size + gap, top + size + size // 4
    W += W % 2; H += H % 2              # libx264 with yuv420p needs even dimensions
    font_path = __import__("matplotlib.font_manager", fromlist=["findfont"]).findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, size // 16)
    mid = ImageFont.truetype(font_path, size // 26)
    small = ImageFont.truetype(font_path, size // 34)

    def compose(rgb_real, rgb_model, t, ep_note):
        im = Image.new("RGB", (W, H), (0, 0, 0))
        im.paste(Image.fromarray(rgb_real[::-1]), (0, top))          # north up
        im.paste(Image.fromarray(rgb_model[::-1]), (size + gap, top))
        d = ImageDraw.Draw(im)
        m = size // 40
        d.text((size // 2, top // 2), "Chandra ACIS, 0.5-7 keV", anchor="mm", font=mid,
               fill=(220, 220, 220))
        d.text((size + gap + size // 2, top // 2), "model (recalibrated Route B, A1 = 0.75)",
               anchor="mm", font=mid, fill=(220, 220, 220))
        y = top + size
        d.text((W // 2, y + size // 14), f"{t:.1f}", anchor="mm", font=font,
               fill=(255, 255, 255))
        d.text((W // 2, y + size // 7 - size // 34), ep_note, anchor="mm", font=small,
               fill=(150, 150, 150))
        d.text((W // 2, y + size // 7 + size // 34),
               "NEI + dust halo + ejecta sub-grid + synchrotron rim; identical smoothing "
               "and per-frame asinh stretch on both sides; model at 20 ks", anchor="mm",
               font=small, fill=(150, 150, 150))
        return im

    tmp = Path(tempfile.mkdtemp(prefix="casa_tl_"))
    times = [f["t"] for f in frames]
    t0, t1 = times[0], times[-1]
    n_hold = int(args.hold * args.fps)
    n_span = max(int((t1 - t0) / args.years_per_second * args.fps), 2)
    idx = 0

    def note(j):
        f = frames[j]
        return (f"epoch {f['ep']}: real {f['real_ks']:.0f} ks, {f['real_rate']:.0f} c/s; "
                f"model {f['model_rate']:.0f} c/s")

    def emit(im):
        nonlocal idx
        im.save(tmp / f"frame_{idx:05d}.png"); idx += 1

    for _ in range(n_hold):
        emit(compose(frames[0]["real"], frames[0]["model"], t0, note(0)))
    for k in range(n_span):
        t = t0 + (t1 - t0) * k / (n_span - 1)
        j = int(np.searchsorted(times, t, side="right"))
        j = min(max(j, 1), len(times) - 1)
        w = (t - times[j - 1]) / (times[j] - times[j - 1])
        blend = lambda key: ((1 - w) * frames[j - 1][key].astype(np.float32)
                             + w * frames[j][key].astype(np.float32)).astype(np.uint8)
        emit(compose(blend("real"), blend("model"), t, note(j - 1 if w < 0.5 else j)))
    for _ in range(n_hold):
        emit(compose(frames[-1]["real"], frames[-1]["model"], t1, note(len(frames) - 1)))
    print(f"[timelapse] {idx} frames in {tmp}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = find_ffmpeg()
    if exe:
        subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", str(tmp / "frame_%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
        print(f"[timelapse] wrote {out}")
    else:
        print("[timelapse] no ffmpeg found; frames left in", tmp)
    if args.gif or not exe:
        ims = [Image.open(p) for p in sorted(tmp.glob("frame_*.png"))[::4]]
        gif = out.with_suffix(".gif")
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(4000 / args.fps), loop=0)
        print(f"[timelapse] wrote {gif}")
    if exe:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
