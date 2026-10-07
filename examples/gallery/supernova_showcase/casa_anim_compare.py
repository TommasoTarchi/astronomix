"""
Side-by-side X-ray animation, 2000-2022: Chandra against one or more 4D-Var
reconstructions, in a three-band false-colour composite.

Every panel shows the SAME quantity on the SAME grid: the count rate per
1.97" pixel of the fit's band images (``jaxobs/data/bands_<epoch>.npz`` for the
data, the model npz ``images`` written by ``stage3/validation/eval_model.py``
or ``casa_xfit --save-model`` for the models). Colour: red 0.5-1.5 keV (Fe-L,
O, Ne), green 1.5-2.1 keV (Si), blue 4.2-6 keV (continuum, synchrotron).

Display processing is identical for every panel: a light Gaussian smoothing,
an asinh stretch per band whose scale is FIXED over the 22 years (anchored on
each panel's own 2000 image, so the fading is visible and calibration offsets
between model and data do not hide morphology), and a time-proportional
crossfade between epochs. The models are noise-free; the data carry Poisson
noise. Each model panel says whether the frame's epoch was fitted
("fit"), lies after its fitting window ("forecast"), or is held out of every
fit ("held out").

    PYTHONPATH=/export/data/lstorcks/pylib_ffmpeg python casa_anim_compare.py \\
        --model "4D-Var 256^3, fine-scale (fit <=2009.9)=M1.npz:2009.9" \\
        --model "4D-Var 128^3, production (fit <=2018.5)=M2.npz:2018.5" \\
        --out figures/casa_anim_compare
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

DATA_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs/data")
HELD_OUT = ("2019", "2022")
RGB_BANDS = (0, 1, 4)          # 0.5-1.5, 1.5-2.1, 4.2-6 keV
PIX = 1.968                     # arcsec per pixel of the band images


def decimal_year(label):
    d = np.load(DATA_DIR / f"bands_{label}.npz")
    s = str(d["dates"][0])
    y, m, day = int(s[:4]), int(s[5:7]), int(s[8:10])
    return y + (m - 1) / 12 + (day - 1) / 365.25


def data_rates(epochs):
    out = []
    for e in epochs:
        d = np.load(DATA_DIR / f"bands_{e}.npz")
        out.append(d["counts"] / float(d["exposure"]))
    return np.stack(out)                                    # (E, 6, 256, 256)


def model_rates(path, epochs):
    d = np.load(path)
    ep = [str(x) for x in d["epochs"]]
    return np.stack([d["images"][ep.index(e)] for e in epochs])


def status(label, fit_end):
    if label in HELD_OUT:
        return "held out", (255, 120, 120)
    return ("fit", (150, 230, 150)) if decimal_year(label) <= fit_end else ("forecast", (255, 210, 120))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True,
                    help='"LABEL=path.npz:FIT_END_YEAR" (repeatable)')
    ap.add_argument("--epochs", nargs="*", default=["2000", "2002", "2004", "2007", "2009", "2010", "2012",
                                                    "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2022"])
    ap.add_argument("--zoom", type=float, default=None,
                    help="half-width of a square crop about the image centre, arcsec (default: r < 190\")")
    ap.add_argument("--centre", type=float, nargs=2, default=(0.0, 0.0),
                    help="crop centre offset in pixels (columns = west, rows = north) for --zoom")
    ap.add_argument("--scale", type=int, default=3, help="pixel upsampling for display")
    ap.add_argument("--smooth", type=float, default=0.6, help="Gaussian sigma in pixels (all panels)")
    ap.add_argument("--stretch", choices=("per-frame", "fixed"), default="per-frame",
                    help="per-frame: each band scaled to each frame's own percentiles (morphology; the ACIS "
                         "contamination's soft-band loss does not recolour the frames); fixed: anchored on 2000 "
                         "(shows the fading)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--frames-per-year", type=int, default=5)
    ap.add_argument("--out", required=True, help="output path without extension (writes .gif and .mp4)")
    args = ap.parse_args()

    epochs = list(args.epochs)
    years = np.array([decimal_year(e) for e in epochs])
    panels = [("Chandra ACIS", data_rates(epochs), None)]
    for spec in args.model:
        label, rest = spec.rsplit("=", 1)
        path, fit_end = rest.rsplit(":", 1)
        panels.append((label, model_rates(path, epochs), float(fit_end)))

    n = panels[0][1].shape[-1]
    half = int(round((args.zoom if args.zoom else 190.0) / PIX))
    cx, cy = n // 2 + int(args.centre[0]), n // 2 + int(args.centre[1])
    crop_y, crop_x = slice(max(cy - half, 0), min(cy + half, n)), slice(max(cx - half, 0), min(cx + half, n))
    # the band images' row index runs NORTH and the column index WEST (casa_jaxobs_data):
    # crop as (row, column), then flip rows for display so north is up (west right, as on the sky)
    crop = (crop_y, crop_x)

    def rgb_of(rates, soft, white, k):
        r = rates[k]
        ch = []
        for b in RGB_BANDS:
            x = r[b][crop]
            x = gaussian_filter(x, args.smooth) if args.smooth > 0 else x
            v = np.arcsinh(np.maximum(x, 0) / soft[b]) / np.arcsinh(white[b] / soft[b])
            ch.append(np.clip(v, 0, 1))
        return np.stack(ch, -1)[::-1]

    def band_scales(img):
        sm = [gaussian_filter(img[b], args.smooth)[crop] for b in range(img.shape[0])]
        return (np.array([np.percentile(x, 60) for x in sm]), np.array([np.percentile(x, 99.7) for x in sm]))

    # scales[j][k] = (soft, white) of panel j at epoch k
    scales = [[band_scales(rates[0 if args.stretch == "fixed" else k]) for k in range(len(epochs))]
              for _, rates, _ in panels]

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()

    # time axis: crossfade between bracketing epochs, frames proportional to time
    t_all = np.linspace(years[0], years[-1], int((years[-1] - years[0]) * args.frames_per_year) + 1)
    t_all = np.concatenate([np.full(args.fps, years[0]), t_all, np.full(2 * args.fps, years[-1])])
    h = crop_y.stop - crop_y.start
    w = crop_x.stop - crop_x.start
    pw, ph = w * args.scale, h * args.scale
    head, foot = 30, 44
    frames = []
    for t in t_all:
        k = int(np.clip(np.searchsorted(years, t, side="right") - 1, 0, len(years) - 2))
        a = float(np.clip((t - years[k]) / (years[k + 1] - years[k]), 0, 1))
        near = k if a < 0.5 else k + 1
        canvas = Image.new("RGB", (len(panels) * pw + (len(panels) - 1) * 6, ph + head + foot), (10, 10, 14))
        draw = ImageDraw.Draw(canvas)
        for j, (label, rates, fit_end) in enumerate(panels):
            img = (1 - a) * rgb_of(rates, *scales[j][k], k) + a * rgb_of(rates, *scales[j][k + 1], k + 1)
            im = Image.fromarray((255 * np.clip(img, 0, 1) ** 0.9).astype(np.uint8))
            im = im.resize((pw, ph), Image.LANCZOS)
            x0 = j * (pw + 6)
            canvas.paste(im, (x0, head))
            draw.text((x0 + 6, 6), label, fill=(235, 235, 235), font=font)
            if fit_end is not None:
                s, col = status(epochs[near], fit_end)
                draw.text((x0 + 6, head + ph + 4), s, fill=col, font=font)
            else:
                draw.text((x0 + 6, head + ph + 4), f"ACIS {epochs[near]}", fill=(200, 200, 200), font=font)
        draw.text((canvas.width - 120, head + ph + 4), f"{t:7.1f}", fill=(255, 255, 255), font=font)
        draw.text((6, head + ph + 24), "red 0.5-1.5 keV  green 1.5-2.1 keV (Si)  blue 4.2-6 keV;  "
                  f"stretch {args.stretch};  1.97\" pixels;  models noise-free",
                  fill=(160, 160, 160), font=small)
        frames.append(canvas)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"[anim] wrote {out.with_suffix('.gif')} ({len(frames)} frames)")
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
    if exe:
        with tempfile.TemporaryDirectory() as tmp:
            for i, fr in enumerate(frames):
                fr.save(f"{tmp}/f{i:04d}.png")
            subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{tmp}/f%04d.png",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "18", str(out.with_suffix(".mp4"))], check=True)
        print(f"[anim] wrote {out.with_suffix('.mp4')}")
    # a still of the first and last epochs for reports
    still = Image.new("RGB", (frames[0].width, 2 * frames[0].height), (10, 10, 14))
    still.paste(frames[args.fps], (0, 0))
    still.paste(frames[-1], (0, frames[0].height))
    still.save(out.with_name(out.name + "_2000_2022.png"))
    print(f"[anim] wrote {out.with_name(out.name + '_2000_2022.png')}")


if __name__ == "__main__":
    main()
