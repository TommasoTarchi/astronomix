"""The three-panel summary of the numerical Reynolds and Prandtl numbers.

``Re``, ``Rm`` and ``Pm`` against resolution, one panel each, all measured from
the spectral energy budget at matched ``E_B/E_K`` (see
``make_mechanism_table.py`` for the definitions and the audit behind them).

The point of putting them side by side is that the first two panels look the
same for every scheme up to a prefactor -- both diffusivities fall together as
the order rises, so ``Re`` and ``Rm`` both scale as ``N^1.2`` -- while the third
splits the schemes into two groups that do not converge. Open markers are runs
the resolvedness check flags, where the Kolmogorov scale implied by the measured
``nu`` lies beyond Nyquist and ``Re`` is partly an extrapolation.

    python make_mechanism_figure.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_convergence_figures import SERIES, series_of
from make_mechanism_table import (MATCH_RATIO, RESOLVED_MAX, bootstrap, collect,
                                  measure_at_ratio, systematic, _reload)

HERE = Path(__file__).resolve().parent


def _laplacian_figure(per_scheme, figures):
    """One row -- ``Re``, ``Rm``, ``Pm`` from the Laplacian band mean -- for print.

    Same numbers as the top row of the full figure and as the ``--summary``
    table: the diffusivities are the band means of ``D / 2 k^2 E`` over
    ``n / n_Nyq = 0.2-0.7``, taken at the matched state ``E_B/E_K = 0.01``, and
    ``Re``, ``Rm`` use ``L = 0.5``. Open markers are the runs whose implied
    Kolmogorov shell lies beyond ``RESOLVED_MAX`` of Nyquist, where the
    band-mean reading of ``nu`` is an extrapolation rather than a measurement.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    rows = []
    for key, pts in sorted(per_scheme.items()):
        pts.sort()
        N = np.array([q[0] for q in pts], dtype=float)
        colour, label = SERIES[key]
        resolved = np.array([q[5] <= RESOLVED_MAX for q in pts])
        for ax, col in zip(axes, (1, 2, 3)):
            y = np.array([q[col] for q in pts])
            ax.plot(N, y, "-", color=colour, lw=1.6, alpha=0.95, zorder=2,
                    label=label if col == 1 else None)
            for mask, face in ((resolved, colour), (~resolved, "white")):
                if mask.any():
                    ax.plot(N[mask], y[mask], "o", color=colour, mfc=face,
                            ms=5.5, mew=1.4, zorder=3)
        axes[2].errorbar(N, [q[3] for q in pts], yerr=[q[4] for q in pts],
                         fmt="none", ecolor=colour, elinewidth=1.1, capsize=2.5,
                         zorder=2)
        for q in pts:
            rows.append((label, int(q[0]), q[1], q[2], q[3], q[4],
                         q[5] <= RESOLVED_MAX))

    guide = np.array([64.0, 256.0])
    for ax, anchor in zip(axes[:2], (600.0, 320.0)):
        ax.plot(guide, anchor * (guide / 64.0) ** 1.2, color="0.45", ls=":",
                lw=1.1, zorder=1)
        ax.text(150, anchor * (150 / 64.0) ** 1.2 * 0.52, r"$\propto N^{1.2}$",
                fontsize=9, color="0.4")

    for ax, ttl, ylab in zip(
            axes,
            (r"$Re = v_{\rm rms} L / \nu_{\rm eff}$",
             r"$Rm = v_{\rm rms} L / \eta_{\rm eff}$",
             r"$Pm = \nu_{\rm eff} / \eta_{\rm eff}$"),
            (r"$Re$", r"$Rm$", r"$Pm$")):
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256])
        ax.set_xticklabels(["$64^3$", "$128^3$", "$256^3$"])
        ax.set_xlim(56, 293)
        ax.set_xlabel("resolution")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.2, which="both", lw=0.5)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_ylim(0.0, 1.45)
    axes[2].axhline(1.0, color="0.45", lw=0.9, ls="--")
    axes[2].text(68, 1.05, r"$Pm = 1$", fontsize=9, color="0.4")

    handles, labels = axes[0].get_legend_handles_labels()
    order = np.argsort([l.split("(")[-1] for l in labels])
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    handles.append(Line2D([], [], color="0.3", marker="o", mfc="white", ls=""))
    labels.append(f"open: $n_K/n_{{\\rm Nyq}} > {RESOLVED_MAX:g}$")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.6,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.135, 1, 1))

    figures.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("pdf", "png"):
        out = figures / f"dynamo_reynolds_prandtl.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.02)
        outs.append(out)
    plt.close(fig)

    print(f"Numerical Re, Rm and Pm at matched E_B/E_K = {MATCH_RATIO:g}, "
          f"band mean of D/2k^2E over n/n_Nyq = 0.2-0.7, L = 0.5\n")
    print(f"{'scheme':28s} {'N':>4s} {'Re':>7s} {'Rm':>7s} {'Pm':>15s}  resolved")
    for label, N, Re, Rm, Pm, err, ok in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{label:28s} {N:4d} {Re:7.0f} {Rm:7.0f} "
              f"{Pm:7.2f} +- {err:4.2f} {'yes' if ok else 'no':>9s}")
    for out in outs:
        print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="*",
                   default=[str(HERE / "data" / "dissipation"),
                            str(HERE / "data" / "dissipation_mech"),
                            str(HERE / "data" / "dissipation_wenoz")])
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--laplacian-only", action="store_true",
                   help="publication figure: the Laplacian band-mean row alone "
                        "(Re, Rm, Pm), written as PDF and PNG")
    args = p.parse_args()

    rows = collect(args.data)
    per_scheme = {}
    for r in rows:
        run = _reload(r)
        m = measure_at_ratio(run)
        if m is None:
            continue
        err = float(np.hypot(bootstrap(run)["Pm"], systematic(run)))
        per_scheme.setdefault(series_of(run), []).append(
            (r["N"], m["Re"], m["Rm"], m["Pm"], err,
             r["n_kolmogorov"] / r["n_nyquist"],
             m["Re_D"], m["Rm_D"], m["Pm_D"]))

    # Two rows, same matched state. Top: the Laplacian band mean, which is the
    # table's definition and is like-for-like only between schemes whose
    # nu_eff(n) is flat (filled markers). Bottom: the dissipation-weighted shell
    # of the measured D(n), which assumes no form and is the comparison to use
    # once WENO-Z-type schemes with steep nu_eff(n) are in the set.
    if args.laplacian_only:
        # Publication figure: one row, the band-mean definition only. Serif
        # text at a size that survives a two-column reduction, and vector
        # output alongside the raster.
        plt.rcParams.update({
            "font.family": "serif", "font.size": 11,
            "axes.titlesize": 11, "axes.labelsize": 11,
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 9.5, "axes.linewidth": 0.9,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "mathtext.fontset": "dejavuserif",
        })
        return _laplacian_figure(per_scheme, Path(args.figures))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
    for key, pts in per_scheme.items():
        pts.sort()
        N = np.array([q[0] for q in pts], dtype=float)
        colour, label = SERIES[key]
        resolved = np.array([q[5] <= RESOLVED_MAX for q in pts])
        for row, cols in ((0, (1, 2, 3)), (1, (6, 7, 8))):
            for ax, col in zip(axes[row], cols):
                y = np.array([q[col] for q in pts])
                # The line carries the legend entry, so a scheme with no
                # resolved point (PPM, WENO-Z) still appears.
                ax.plot(N, y, "-", color=colour, lw=1.8, alpha=0.9, zorder=2,
                        label=label if (row, col) == (0, 1) else None)
                # Filled = the Laplacian reading is resolved on this grid. The
                # bottom row does not need the distinction but keeps it, so the
                # rows can be compared point by point.
                for mask, face in ((resolved, colour), (~resolved, "white")):
                    if mask.any():
                        ax.plot(N[mask], y[mask], "o", color=colour, mfc=face,
                                ms=7, mew=1.6, zorder=3)
        err = np.array([q[4] for q in pts])
        axes[0][2].errorbar(N, [q[3] for q in pts], yerr=err, fmt="none",
                            ecolor=colour, elinewidth=1.4, capsize=3, zorder=2)

    guide = np.array([64.0, 256.0])
    for ax, anchor in zip(axes[0][:2], (600.0, 320.0)):
        ax.plot(guide, anchor * (guide / 64.0) ** 1.2, "k:", lw=1.2, zorder=1)
        ax.text(150, anchor * (150 / 64.0) ** 1.2 * 0.55, r"$\propto N^{1.2}$",
                fontsize=9, color="0.35")
    for ax, anchor in zip(axes[1][:2], (7.0, 7.0)):
        ax.plot(guide, anchor * (guide / 64.0) ** (4.0 / 3.0), "k:", lw=1.2,
                zorder=1)
        ax.text(150, anchor * (150 / 64.0) ** (4.0 / 3.0) * 0.55,
                r"$\propto N^{4/3}$", fontsize=9, color="0.35")

    titles = (
        (r"Laplacian band mean: $Re = v_{\rm rms} L / \nu_{\rm eff}$",
         r"Laplacian band mean: $Rm = v_{\rm rms} L / \eta_{\rm eff}$",
         r"Laplacian band mean: $Pm = \nu_{\rm eff} / \eta_{\rm eff}$"),
        (r"dissipation shell: $Re_D = (n_{D_v} / n_{\rm inj})^{4/3}$",
         r"dissipation shell: $Rm_D = (n_{D_B} / n_{\rm inj})^{4/3}$",
         r"dissipation shell: $Pm_D = (n_{D_B} / n_{D_v})^{4/3}$"),
    )
    ylabels = ((r"$Re$", r"$Rm$", r"$Pm$"), (r"$Re_D$", r"$Rm_D$", r"$Pm_D$"))
    for row in (0, 1):
        for ax, ttl, ylab in zip(axes[row], titles[row], ylabels[row]):
            ax.set_xscale("log", base=2)
            ax.set_xticks([64, 128, 256])
            ax.set_xticklabels(["$64^3$", "$128^3$", "$256^3$"])
            ax.set_xlabel("resolution")
            ax.set_ylabel(ylab)
            ax.set_title(ttl, fontsize=10)
            ax.grid(alpha=0.25, which="both")
        axes[row][0].set_yscale("log")
        axes[row][1].set_yscale("log")
        axes[row][2].axhline(1.0, color="0.4", lw=1.0, ls="--")
        axes[row][2].text(70, 1.03, r"$Pm = 1$", fontsize=8, color="0.4")
    axes[0][2].set_ylim(0.0, 1.45)
    # The shell-based ratio is weighted by where E_B lives, which in the
    # kinematic phase is at small scales for every scheme, so it sits above 1
    # for all of them; the axis follows the data rather than the Pm panel above.
    top = max(q[8] for pts in per_scheme.values() for q in pts)
    axes[1][2].set_ylim(0.0, 1.15 * top)
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = np.argsort([l.split("(")[-1] for l in labels])
    # One legend for all three panels, in a row below them. The open-marker
    # convention is spelled out there too, since the title that used to carry it
    # is gone (the caption in DYNAMO_MECHANISM.md says what the panels show).
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    handles.append(Line2D([], [], color="0.3", marker="o", mfc="white", ls=""))
    labels.append(f"open: $n_K / n_{{\\rm Nyq}} > {RESOLVED_MAX:g}$ "
                  f"(unresolved)")
    fig.legend(handles, labels, fontsize=9, loc="lower center",
               ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = Path(args.figures) / "dynamo_mechanism.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
