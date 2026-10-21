"""The numerical Reynolds and Prandtl numbers of astronomix's two MHD schemes.

``Re``, ``Rm`` and ``Pm`` against resolution for the finite-difference scheme
(WENO-Z + constrained transport), the same with the upwind dissipation of the
CT edge EMFs at its default strength 0.03 (the setup of the turbulence runs,
drawn when its runs are among ``--data``) and the finite-volume scheme (VL2 +
PLM + HLLD + GLM cleaning), all run by ``dynamo_convergence.py`` on the same driven,
subsonic, isothermal dynamo box with the same forcing and reduced by the same
estimator. The numbers are those of ``make_mechanism_table.py --summary``
(``summary_rows``): the diffusivities from the spectral energy budget,
averaged over ``n / n_Nyq = 0.2-0.7`` at matched ``E_B/E_K = 0.01``, with
``L = 0.5``. Error bars combine a moving-block bootstrap with the spread over
the defensible bands and windows; open markers are runs whose implied
Kolmogorov shell lies beyond 0.8 of Nyquist, where the band-mean ``nu`` is
partly an extrapolation.

    python make_fd_fv_reynolds_figure.py --data data/fd_fv data/fd_fv_uct
    python make_fd_fv_reynolds_figure.py --data data/fd_fv --with-pp
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
from matplotlib.lines import Line2D

# the dynamo study's measurement
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_mechanism_table import RESOLVED_MAX, collect, summary_rows

HERE = Path(__file__).resolve().parent

#: The series of the figure, by scheme key: colour, marker and legend label.
#: Colours are the first three slots of the categorical palette (validated for
#: colour-vision deficiency as a set); the marker shape repeats the identity so
#: it never rests on colour alone.
SERIES = {
    "wenoz": ("#2a78d6", "o", "FD: WENO-Z + CT (5th order)"),
    "wenoz_uct0.03": ("#8a4fbf", "D", "FD: WENO-Z + CT + upwind EMF"),
    "vl2": ("#eb6834", "s", "FV: VL2 + PLM + GLM (2nd order)"),
    "wenoz_pp": ("#1baf7a", "^", "FD: WENO-Z + CT, positivity preserving"),
}

#: Series whose fitted exponent is written at the end of the Re and Rm lines;
#: the upwind series runs within a few percent of the plain FD one, where a
#: second label would overlap.
ANNOTATED_SERIES = ("wenoz", "vl2", "wenoz_pp")


def fit_power_law(resolutions, values):
    """Exponent ``a`` of ``values ~ N^a``, by least squares in log-log."""
    return float(np.polyfit(np.log(resolutions), np.log(values), 1)[0])


def draw(table, series_keys, figures, stem):
    """The three-panel figure, written as SVG, PDF and PNG.

    Args:
        table: The rows of ``summary_rows``.
        series_keys: Scheme keys to draw, in legend order.
        figures: Output directory.
        stem: Output file name without extension.

    Returns:
        The list of written paths.
    """
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10.5,
        "axes.titlesize": 10.5, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "axes.linewidth": 0.8,
        "axes.edgecolor": "0.25",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "mathtext.fontset": "dejavuserif",
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35))
    panels = (("Re", "Re_err", r"$Re = v_{\rm rms} L / \nu_{\rm eff}$"),
              ("Rm", "Rm_err", r"$Rm = v_{\rm rms} L / \eta_{\rm eff}$"),
              ("Pm", "Pm_err", r"$Pm = \nu_{\rm eff} / \eta_{\rm eff}$"))
    all_resolutions = sorted({row["N"] for row in table})

    legend_handles = []
    for key in series_keys:
        rows = sorted((row for row in table if row["scheme"] == key), key=lambda row: row["N"])
        if not rows:
            continue
        colour, marker, label = SERIES[key]
        resolutions = np.array([row["N"] for row in rows], dtype=float)
        resolved = np.array([row["resolvedness"] <= RESOLVED_MAX for row in rows])
        for ax, (quantity, error, _) in zip(axes, panels):
            values = np.array([row[quantity] for row in rows])
            errors = np.array([row[error] for row in rows])
            ax.plot(resolutions, values, "-", color=colour, lw=1.4, zorder=2)
            ax.errorbar(resolutions, values, yerr=errors, fmt="none", ecolor=colour,
                        elinewidth=1.0, capsize=2.5, zorder=2)
            for mask, face in ((resolved, colour), (~resolved, "white")):
                if mask.any():
                    ax.plot(resolutions[mask], values[mask], marker, color=colour,
                            mfc=face, ms=6, mew=1.3, ls="", zorder=3)
            # The fitted exponent as a direct label at the end of the line, for
            # the two panels whose scaling with N is the statement.
            if quantity in ("Re", "Rm") and len(rows) >= 2 and key in ANNOTATED_SERIES:
                exponent = fit_power_law(resolutions, values)
                ax.annotate(rf"$\propto N^{{{exponent:.2f}}}$",
                            (resolutions[-1], values[-1]), xytext=(6, -2),
                            textcoords="offset points", fontsize=9, color="0.2",
                            va="center", ha="left")
        legend_handles.append(Line2D([], [], color=colour, marker=marker, mfc=colour,
                                     ms=6, lw=1.4, label=label))

    for ax, (quantity, _, title) in zip(axes, panels):
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_resolutions)
        ax.set_xticklabels([f"${N}^3$" for N in all_resolutions])
        ax.minorticks_off()
        ax.set_xlim(all_resolutions[0] / 1.18, all_resolutions[-1] * 1.55)
        ax.set_xlabel("resolution")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="major", lw=0.5, color="0.7")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$Re$")
    axes[1].set_ylabel(r"$Rm$")
    axes[2].set_ylabel(r"$Pm$")
    pm_top = max(row["Pm"] + row["Pm_err"] for row in table if row["scheme"] in series_keys)
    axes[2].set_ylim(0.0, max(1.45, 1.15 * pm_top))
    axes[2].axhline(1.0, color="0.45", lw=0.8, ls="--", zorder=1)
    axes[2].text(all_resolutions[-1] * 1.5, 1.02, r"$Pm = 1$", fontsize=8.5, color="0.4",
                 va="bottom", ha="right")

    legend_handles.append(Line2D([], [], color="0.3", marker="o", mfc="white", ls="",
                                 ms=6, label=rf"open: $n_K / n_{{\rm Nyq}} > {RESOLVED_MAX:g}$"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 3),
               frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.09 if len(legend_handles) <= 3 else 0.15, 1, 1))

    figures.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in ("svg", "pdf", "png"):
        path = figures / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", default=[str(HERE / "data" / "fd_fv")],
                        help="directories of dynamo_convergence.py --transfer runs")
    parser.add_argument("--figures", default=str(HERE / "figures"))
    parser.add_argument("--stem", default="fd_fv_reynolds_prandtl")
    parser.add_argument("--with-pp", action="store_true",
                        help="also draw the FD runs with the positivity-preserving WENO")
    args = parser.parse_args()

    rows = [row for row in collect(args.data) if row["code"] == "astronomix"]
    if not rows:
        raise SystemExit(f"no astronomix run with transfer spectra in {args.data}")
    table = summary_rows(rows)
    series_keys = ["wenoz", "wenoz_uct0.03", "vl2"] + (["wenoz_pp"] if args.with_pp else [])

    print(f"{'scheme':42s} {'N':>4s} {'Re':>12s} {'Rm':>12s} {'Pm':>13s} "
          f"{'G t_cross':>9s} {'n_K/n_Nyq':>9s}")
    for key in series_keys:
        for row in sorted((r for r in table if r["scheme"] == key), key=lambda r: r["N"]):
            print(f"{SERIES[key][2]:42s} {row['N']:4d} {row['Re']:6.0f}+-{row['Re_err']:4.0f} "
                  f"{row['Rm']:6.0f}+-{row['Rm_err']:4.0f} {row['Pm']:6.3f}+-{row['Pm_err']:5.3f} "
                  f"{row['gamma']:9.3f} {row['resolvedness']:9.2f}")
    for path in draw(table, series_keys, Path(args.figures), args.stem):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
