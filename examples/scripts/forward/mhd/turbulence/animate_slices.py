"""
Animate the mid-plane magnetic-energy slices extracted by
``turb_slices_series.py``.

Renders one frame per checkpoint: the full x-y mid-plane on the left and
a zoomed view of a small section of it on the right, so the fine folded
structure of the field stays visible even though the full 2048^2 plane is
heavily downsampled on screen.  The zoom region is outlined on the full
view in the same accent color as the zoom panel's frame.  The color scale
is logarithmic and FIXED across the whole animation (robust global
percentiles), so the secular brightening of the frames is the small-scale
dynamo amplifying the seed field, not a rescaling artifact.  Magnetic
energy is magnitude data, so it gets a perceptually uniform sequential
colormap (magma, the same ramp as the still figures from
``plot_slices.py``).

    python examples/scripts/forward/mhd/turbulence/animate_slices.py \
        --slices-dir /e/project1/astronomix/storcks1/runs/slices_series_icm2048 \
        --tcross 0.5 --fps 8 \
        --out /e/project1/astronomix/storcks1/runs/figures/EB_slices_icm2048.mp4
"""

# general
import argparse
from pathlib import Path

# third-party
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument("--slices-dir", type=str, required=True,
                    help="directory of slices_step_*.npz from turb_slices_series.py")
parser.add_argument("--tcross", type=float, default=0.5,
                    help="crossing time (for the title)")
parser.add_argument("--fps", type=int, default=8, help="animation frame rate")
parser.add_argument("--zoom-center", type=float, nargs=2, default=(0.5, 0.5),
                    metavar=("X", "Y"),
                    help="center of the zoom region, in units of L")
parser.add_argument("--zoom-frac", type=float, default=0.125,
                    help="side length of the zoom region, in units of L")
parser.add_argument("--out", type=str, required=True, help="output MP4 path")
args = parser.parse_args()

frame_paths = sorted(Path(args.slices_dir).glob("slices_step_*.npz"))
if not frame_paths:
    raise SystemExit(f"no slices_step_*.npz files in {args.slices_dir}")
print(f"[animate] {len(frame_paths)} frames from {args.slices_dir}")

# ------------------------------------------------------------------
# Global color limits
# ------------------------------------------------------------------
# One fixed log scale for the whole run: the low end comes from the seed
# field of the earliest frames, the high end from the saturated final
# frames, with robust percentiles so single extreme cells do not crush
# the ramp.  Every 8th pixel is plenty for stable percentiles and keeps
# the pre-pass cheap.
samples = []
for path in frame_paths:
    data = np.load(path)
    samples.append(data["EB_xy"][::8, ::8].ravel())
samples = np.concatenate(samples)
eb_limits = (np.percentile(samples, 0.5), np.percentile(samples, 99.9))
del samples
print(f"[animate] global log scale: [{eb_limits[0]:.3e}, {eb_limits[1]:.3e}]")

# ------------------------------------------------------------------
# Zoom region, in cell indices
# ------------------------------------------------------------------
N = int(np.load(frame_paths[0])["N"])
zoom_x0 = args.zoom_center[0] - args.zoom_frac / 2
zoom_y0 = args.zoom_center[1] - args.zoom_frac / 2
ix0 = int(round(zoom_x0 * N))
iy0 = int(round(zoom_y0 * N))
ix1 = ix0 + int(round(args.zoom_frac * N))
iy1 = iy0 + int(round(args.zoom_frac * N))
if not (0 <= ix0 < ix1 <= N and 0 <= iy0 < iy1 <= N):
    raise SystemExit("zoom region leaves the domain; adjust --zoom-center/--zoom-frac")

# ------------------------------------------------------------------
# Figure (built once, image data swapped per frame)
# ------------------------------------------------------------------
ink = "#333333"
# An accent that magma never produces, so the zoom marker stays visible
# on both the dark early frames and the bright saturated ones.
accent = "#00b8d4"
plt.rcParams.update({"text.color": ink})

# 12.8 x 6.4 in at dpi 100 gives 1280 x 640 — even pixel dimensions,
# which the h264 encoder requires.
fig, (ax_full, ax_zoom) = plt.subplots(
    1, 2, figsize=(12.8, 6.4), constrained_layout=True
)
norm = LogNorm(vmin=eb_limits[0], vmax=eb_limits[1])

# Arrays are indexed [x, y]; imshow wants the first axis vertical, so
# transpose and set origin so x runs rightward.
image_full = ax_full.imshow(
    np.full((2, 2), eb_limits[0]), origin="lower", extent=(0, 1, 0, 1),
    interpolation="nearest", cmap="magma", norm=norm,
)
image_zoom = ax_zoom.imshow(
    np.full((2, 2), eb_limits[0]), origin="lower",
    extent=(zoom_x0, zoom_x0 + args.zoom_frac, zoom_y0, zoom_y0 + args.zoom_frac),
    interpolation="nearest", cmap="magma", norm=norm,
)

ax_full.set_title("x-y plane at z = L/2", fontsize=12)
ax_zoom.set_title(f"zoom, {args.zoom_frac:g} L", fontsize=12)

# Purely pictorial panels: no ticks or axis labels anywhere.
for ax in (ax_full, ax_zoom):
    ax.set_xticks([])
    ax.set_yticks([])

# The zoom region marker on the full view and the zoom panel's frame share
# the accent color, which is what links the two panels visually.
ax_full.add_patch(Rectangle(
    (zoom_x0, zoom_y0), args.zoom_frac, args.zoom_frac,
    fill=False, edgecolor=accent, linewidth=1.5,
))
for spine in ax_full.spines.values():
    spine.set_color("#bbbbbb")
for spine in ax_zoom.spines.values():
    spine.set_color(accent)
    spine.set_linewidth(1.5)

colorbar = fig.colorbar(
    image_zoom, ax=(ax_full, ax_zoom), shrink=0.85, pad=0.02,
    label=r"magnetic energy $B^2/2$",
)
colorbar.outline.set_edgecolor("#bbbbbb")
colorbar.ax.tick_params(colors=ink)

suptitle = fig.suptitle("", fontsize=14)

# ------------------------------------------------------------------
# Encode
# ------------------------------------------------------------------
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
writer = FFMpegWriter(
    fps=args.fps,
    codec="h264",
    extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with writer.saving(fig, out_path, dpi=100):
    for index, path in enumerate(frame_paths):
        eb_xy = np.load(path)["EB_xy"]
        image_full.set_data(eb_xy.T)
        image_zoom.set_data(eb_xy[ix0:ix1, iy0:iy1].T)
        suptitle.set_text(
            f"Driven MHD turbulence, $2048^3$, "
            f"$M_\\mathrm{{turb}} \\approx 0.44$, $\\beta_0 = 10^6$ — "
            f"magnetic energy, "
            f"$t/t_\\mathrm{{cross}} = "
            f"{float(np.load(path)['time']) / args.tcross:.2f}$"
        )
        writer.grab_frame()
        if (index + 1) % 10 == 0 or index + 1 == len(frame_paths):
            print(f"[animate] frame {index + 1}/{len(frame_paths)}", flush=True)

print(f"wrote {out_path}")
