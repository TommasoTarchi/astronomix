"""Side-by-side animation of the dynamo: two codes' fields, spectra and E_B/E_K.

Four panels in a 2x2 grid sharing one clock:

* top left: mid-plane magnetic energy of AthenaPK PLM+VL2 (2nd order, GLM
  cleaning),
* top right: the same for astronomix WENO5+CT (5th order, constrained
  transport),
* bottom left: both magnetic spectra, with the kinetic spectra behind them for
  scale, and
* bottom right: E_B/E_K against time for both runs, with the current time
  marked.

The two slices share a single colour scale at every frame, so the panels are
directly comparable, and that scale follows the CT run's instantaneous maximum
-- the point of the animation is the *structure* and the relative amplitude, and
a fixed scale over six decades of dynamo growth would show a black frame
followed by a white one. The E_B/E_K panel carries the absolute growth that the
slice normalisation hides.

    python make_dynamo_movie.py --data data/anim
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import E_MAG, E_V, load_runs, spectra_of

HERE = Path(__file__).resolve().parent

#: Decades of magnetic energy shown in the slice panels below the frame maximum.
SLICE_DECADES = 3.0

#: The box starts at rest, so E_B/E_K is unbounded until the driving has built
#: the flow (~1 crossing time); the ratio panel starts here.
RATIO_T_MIN = 1.0


def _resample(series, t_src, t_dst):
    """Nearest-in-time frame of ``series`` for each time in ``t_dst``.

    The two codes are dumped on their own cadences and neither is guaranteed to
    hit the other's times, so the animation runs on a common clock and each
    panel shows its own nearest snapshot. Nearest-neighbour rather than
    interpolation: averaging two turbulent fields half a crossing time apart
    would produce a structure neither code ever had.
    """
    idx = np.abs(np.asarray(t_dst)[:, None] - np.asarray(t_src)[None, :]).argmin(1)
    return series[idx], np.asarray(t_src)[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "anim"))
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--out", default="dynamo_side_by_side.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth frame (1 = every dump)")
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--colors", type=int, default=96,
                   help="palette size for the final quantisation (0 disables)")
    p.add_argument("--n", type=int, default=256)
    args = p.parse_args()

    runs = {("astronomix" if str(r["code"]) == "astronomix"
             else str(r["scheme_key"])): r
            for r in load_runs(args.data, skip=("smoke",))
            if int(r["N"]) == args.n and "EB_slice_series" in r}
    missing = {"astronomix", "plm"} - set(runs)
    if missing:
        raise SystemExit(f"no {args.n}^3 run with a slice series for: "
                         f"{sorted(missing)} (run with --slice-series)")

    left, right = runs["plm"], runs["astronomix"]
    # Common clock: the coarser of the two cadences, over the overlap.
    t_lo = max(float(r["t_over_tc"][0]) for r in (left, right))
    t_hi = min(float(r["t_over_tc"][-1]) for r in (left, right))
    n_frames = min(len(left["t_over_tc"]), len(right["t_over_tc"]))
    clock = np.linspace(t_lo, t_hi, n_frames)[::max(1, args.stride)]

    panels = []
    for run in (left, right):
        t_run = np.asarray(run["t_over_tc"])
        sl, t_act = _resample(np.asarray(run["EB_slice_series"]), t_run, clock)
        spec = spectra_of(run, deconvolve=False)
        sp, _ = _resample(spec, t_run, clock)
        ratio_full = (np.asarray(run["E_B"])
                      / np.maximum(np.asarray(run["E_K"]), 1e-30))
        ratio, _ = _resample(ratio_full, t_run, clock)
        keep = t_run >= RATIO_T_MIN
        panels.append(dict(slices=sl, spectra=sp, t=t_act, ratio=ratio,
                           t_full=t_run[keep], ratio_full=ratio_full[keep],
                           label=str(run["label"]),
                           n_shell=np.asarray(run["n_shell"], dtype=float)))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6), dpi=args.dpi)
    ims, curves_B, curves_v, traces, dots = [], [], [], [], []
    colours = ("#d62728", "#1f77b4")

    # ----- top row: the two slices -----
    for ax, panel, colour in zip(axes[0], panels, colours):
        im = ax.imshow(np.log10(np.maximum(panel["slices"][0], 1e-300)).T,
                       origin="lower", cmap="inferno",
                       extent=(0, 1, 0, 1), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["label"], fontsize=11, color=colour)
        ims.append(im)
    time_text = axes[0][0].text(0.03, 0.96, "", transform=axes[0][0].transAxes,
                                ha="left", va="top", fontsize=10, color="w")

    # ----- bottom left: spectra -----
    ax = axes[1][0]
    for panel, colour in zip(panels, colours):
        n = panel["n_shell"]
        curves_B.append(ax.loglog(n[1:], np.maximum(panel["spectra"][0][E_MAG][1:],
                                                    1e-300),
                                  color=colour, lw=2.0,
                                  label=panel["label"])[0])
        curves_v.append(ax.loglog(n[1:], np.maximum(panel["spectra"][0][E_V][1:],
                                                    1e-300),
                                  color=colour, lw=1.0, ls=":", alpha=0.55)[0])
    ax.set_xlabel(r"mode number $n = kL/2\pi$")
    ax.set_ylabel(r"$E(n)$")
    ax.set_title("magnetic (solid) and kinetic (dotted) spectra", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(1, args.n / 2)
    # Fixed over the whole animation: the growth is the story, so the spectrum
    # panel must not rescale under it. Nine decades is enough to hold the whole
    # dynamo -- the seed itself is a single shell and would otherwise stretch the
    # axis over five decades of empty space.
    all_B = np.concatenate([p["spectra"][:, E_MAG, 1:].ravel() for p in panels])
    all_v = np.concatenate([p["spectra"][:, E_V, 1:].ravel() for p in panels])
    top = max(all_B.max(), all_v.max())
    ax.set_ylim(top * 1e-9, top * 3.0)

    # ----- bottom right: E_B / E_K against time -----
    ax = axes[1][1]
    for panel, colour in zip(panels, colours):
        # The whole history faintly, so the eye knows where the curve is going,
        # and the part already reached drawn solid.
        ax.semilogy(panel["t_full"], np.maximum(panel["ratio_full"], 1e-300),
                    color=colour, lw=1.0, alpha=0.25)
        traces.append(ax.semilogy([], [], color=colour, lw=2.0,
                                  label=panel["label"])[0])
        dots.append(ax.semilogy([], [], "o", color=colour, ms=6)[0])
    t_line = ax.axvline(clock[0], color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"$t / t_{\rm cross}$")
    ax.set_ylabel(r"$E_B / E_K$")
    ax.set_title(r"$E_B / E_K$", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.0, max(p["t_full"][-1] for p in panels))
    all_r = np.concatenate([np.maximum(p["ratio_full"], 1e-300) for p in panels])
    ax.set_ylim(all_r.min() / 3.0, all_r.max() * 3.0)

    def update(i):
        # One colour scale for both panels, following the CT run, so the two
        # slices can be compared at a glance rather than each auto-scaling.
        vmax = np.log10(max(panels[1]["slices"][i].max(), 1e-300))
        for im, panel in zip(ims, panels):
            im.set_data(np.log10(np.maximum(panel["slices"][i], 1e-300)).T)
            im.set_clim(vmax - SLICE_DECADES, vmax)
        for cB, cv, panel in zip(curves_B, curves_v, panels):
            cB.set_ydata(np.maximum(panel["spectra"][i][E_MAG][1:], 1e-300))
            cv.set_ydata(np.maximum(panel["spectra"][i][E_V][1:], 1e-300))
        for tr, dot, panel in zip(traces, dots, panels):
            m = panel["t_full"] <= clock[i] + 1e-9
            tr.set_data(panel["t_full"][m],
                        np.maximum(panel["ratio_full"][m], 1e-300))
            if panel["t"][i] >= RATIO_T_MIN:
                dot.set_data([panel["t"][i]], [max(panel["ratio"][i], 1e-300)])
            else:
                dot.set_data([], [])
        t_line.set_xdata([clock[i], clock[i]])
        time_text.set_text(f"$t / t_{{\\rm cross}} = {clock[i]:4.1f}$")
        return ims + curves_B + curves_v + traces + dots + [t_line, time_text]

    fig.tight_layout(pad=0.6)
    anim = FuncAnimation(fig, update, frames=len(clock), blit=False)
    out = Path(args.figures) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)

    if args.colors:
        # Matplotlib writes full-colour frames; the slices only ever use one
        # colormap, so a small palette is lossless in practice and much smaller.
        from PIL import Image
        src = Image.open(out)
        frames = []
        for i in range(src.n_frames):
            src.seek(i)
            frames.append(src.convert("RGB").quantize(colors=args.colors,
                                                      dither=Image.NONE))
        src.close()
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out}  ({len(clock)} frames, {args.fps} fps, "
          f"t/t_cross {clock[0]:.2f} to {clock[-1]:.2f}, "
          f"{out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
