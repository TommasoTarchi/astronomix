"""Dissipation spectra of the calibration runs: an imposed Laplacian, found.

The only runs in the study whose answer is known in advance are AthenaPK PLM at
``64^3`` with an explicit Laplacian coefficient added (``--ohm-diff`` for the
resistivity ladder, ``--mom-diff`` for the viscous one). A Laplacian is a
*flat* line in ``nu_eff(n) = D_v / 2k^2 E_v`` and ``eta_eff(n) = D_B / 2k^2 E_B``,
so each imposed coefficient should lift the whole curve by that amount above the
run with nothing imposed. This plots exactly that, with the additive expectation
(numerical at zero plus imposed) drawn for each run, and the cross-talk on the
other field beside it.

    python make_calibration_figure.py --data data/calibration
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import load_runs
from make_convergence_figures import SERIES, series_of
from make_dissipation_figure import dissipation
from make_mechanism_table import BAND, SAT_START

HERE = Path(__file__).resolve().parent

#: Colours along each ladder, nothing imposed first.
LADDER_COLOURS = ("#444444", "#1f77b4", "#ff7f0e", "#d62728")


def curves(run, sat_start):
    """``(x, nu_eff(n), eta_eff(n))`` over the saturated window, ``x = n/n_Nyq``."""
    d = dissipation(run, sat_start)
    if d is None:
        return None
    n = d["n"]
    k = 2.0 * np.pi * n
    keep = n >= 1
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = d["D_v"] / (2.0 * k ** 2 * d["E_v"])
        eta = d["D_B"] / (2.0 * k ** 2 * d["E_B"])
    x = n / (int(run["N"]) / 2)
    return x[keep], nu[keep], eta[keep]


def band_mean(x, q, band=BAND):
    m = (x >= band[0]) & (x <= band[1]) & np.isfinite(q) & (q > 0)
    return float(np.mean(q[m])) if m.any() else np.nan


def subtracted_figure(runs, sat_start, out, title=""):
    """The explicit part alone: ``eff(imposed) - eff(nothing imposed)`` per shell.

    Rows are the two fields, columns the resolutions. If the estimator were
    exactly additive every curve would be flat at its dashed line (the imposed
    coefficient); what it actually shows is how much of the numerical part the
    imposed Laplacian displaces, and at which scales.
    """
    Ns = sorted({int(r["N"]) for r in runs})
    fig, axes = plt.subplots(2, len(Ns), figsize=(6.0 * len(Ns), 8.4),
                             squeeze=False)
    if title:
        fig.suptitle(title, fontsize=11, y=0.995)
    summary = []
    for col, N in enumerate(Ns):
        here = [r for r in runs if int(r["N"]) == N]
        for row, (key, other, idx, sym) in enumerate(
                (("ohm_diff", "mom_diff", 2, r"$\eta$"),
                 ("mom_diff", "ohm_diff", 1, r"$\nu$"))):
            ladder = sorted((r for r in here if float(r[other]) == 0.0),
                            key=lambda r: float(r[key]))
            if len(ladder) < 2:
                continue
            base = curves(ladder[0], sat_start)
            ax = axes[row][col]
            lo_y, hi_y = np.inf, 0.0
            for colour, run in zip(LADDER_COLOURS[1:], ladder[1:]):
                c = curves(run, sat_start)
                imposed = float(run[key])
                diff = c[idx] - base[idx]
                ax.semilogx(c[0], diff / imposed, color=colour, lw=1.9,
                            label=f"imposed {sym} = {imposed:.1e}")
                m = (c[0] >= BAND[0]) & np.isfinite(diff)
                lo_y = min(lo_y, (diff[m] / imposed).min())
                hi_y = max(hi_y, (diff[m] / imposed).max())
                summary.append((N, sym, imposed, band_mean(c[0], diff) / imposed))
            ax.axhline(1.0, color="0.3", lw=1.0, ls="--")
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.axvspan(*BAND, color="0.5", alpha=0.08, lw=0)
            ax.set_xlim(BAND[0] / 2, 1.0)
            ax.set_ylim(min(-0.1, lo_y - 0.1), max(1.3, hi_y + 0.1))
            ax.set_xlabel(r"$n / n_{\rm Nyquist}$")
            ax.set_ylabel(fr"[{sym}$_{{\rm eff}}(n)$ - numerical] / imposed")
            ax.set_title(fr"${N}^3$: explicit {sym} recovered, shell by shell "
                         f"(dashed = exact)", fontsize=10)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{'N':>4s} {'field':>6s} {'imposed':>9s} {'band-mean recovered':>20s}")
    for N, sym, imposed, frac in summary:
        print(f"{N:4d} {sym:>6s} {imposed:9.1e} {frac:20.2f}")
    print(f"wrote {out}")


def publication_figure(runs, N, sat_start, figures, stem):
    """Two panels for print: the imposed field only, no titles, no cross-talk.

    Left ``eta_eff(n)`` for the resistive ladder, right ``nu_eff(n)`` for the
    viscous one, each with the run that has nothing imposed and the additive
    expectation (that run's band mean plus the imposed coefficient) as a dashed
    line. Everything the panels mean goes in the caption, so the figure itself
    carries only axes, curves and a legend.
    """
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "mathtext.fontset": "dejavuserif",
    })
    here = [r for r in runs if int(r["N"]) == N]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    summary = []
    for ax, (key, other, idx, sym) in zip(axes, (
            ("ohm_diff", "mom_diff", 2, r"\eta"),
            ("mom_diff", "ohm_diff", 1, r"\nu"))):
        ladder = sorted((r for r in here if float(r[other]) == 0.0),
                        key=lambda r: float(r[key]))
        if len(ladder) < 2:
            continue
        base = curves(ladder[0], sat_start)
        base_mean = band_mean(base[0], base[idx])
        lo_y, hi_y = np.inf, 0.0
        for colour, run in zip(LADDER_COLOURS, ladder):
            c = curves(run, sat_start)
            imposed = float(run[key])
            if imposed > 0:
                exp = int(np.floor(np.log10(imposed)))
                man = f"{imposed / 10.0 ** exp:g}"
                label = rf"${sym}_{{\rm imp}} = {man} \times 10^{{{exp}}}$"
            else:
                label = "none"
            ax.loglog(c[0], c[idx], color=colour, lw=1.7, label=label)
            if imposed > 0:
                ax.axhline(base_mean + imposed, color=colour, lw=0.9, ls="--",
                           alpha=0.85)
            m = (c[0] >= BAND[0]) & (c[0] <= 1.0) & np.isfinite(c[idx]) & (c[idx] > 0)
            if m.any():
                lo_y = min(lo_y, c[idx][m].min())
                hi_y = max(hi_y, c[idx][m].max())
            summary.append((sym, imposed,
                            (band_mean(c[0], c[idx]) - base_mean) / imposed
                            if imposed > 0 else np.nan))
        ax.axvspan(*BAND, color="0.5", alpha=0.09, lw=0)
        ax.set_xlim(0.05, 1.0)
        # Headroom for the legend, which sits top-left where no curve goes.
        ax.set_ylim(0.62 * lo_y, 4.0 * hi_y)
        ax.set_xlabel(r"$n / n_{\rm Nyquist}$")
        ax.set_ylabel(rf"${sym}_{{\rm eff}}(n)$")
        ax.grid(alpha=0.2, which="both", lw=0.5)
        ax.legend(loc="upper left", frameon=False, handlelength=1.5,
                  borderaxespad=0.5, labelspacing=0.35)
    fig.tight_layout(pad=0.6)
    figures.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("pdf", "png"):
        out = figures / f"{stem}.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.02)
        outs.append(out)
    plt.close(fig)
    print(f"N = {N}, band {BAND[0]}-{BAND[1]} of Nyquist, "
          f"t/t_cross >= {sat_start}")
    print(f"{'field':>7s} {'imposed':>9s} {'band-mean rise / imposed':>25s}")
    for sym, imposed, frac in summary:
        if imposed > 0:
            print(f"{sym:>7s} {imposed:9.1e} {frac:25.2f}")
    for out in outs:
        print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--publication", action="store_true",
                   help="two panels (imposed field only), no titles, PDF + PNG")
    p.add_argument("--subtracted", action="store_true",
                   help="instead: the explicit part alone, eff(imposed) - "
                        "eff(nothing imposed), normalised by the imposed "
                        "coefficient, all resolutions in one figure")
    p.add_argument("--data", default=str(HERE / "data" / "calibration"))
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--out", default="dynamo_dissipation_calibration.png")
    p.add_argument("--sat-start", type=float, default=SAT_START)
    p.add_argument("--n", type=int, default=None,
                   help="resolution to plot; default: the largest present")
    p.add_argument("--series", default=None,
                   help="scheme to plot (a key of make_convergence_figures.SERIES, "
                        "e.g. plm, astronomix, astronomix_wenoz); default: the "
                        "only one present, else required")
    args = p.parse_args()

    runs = load_runs(args.data, skip=("smoke",))
    runs = [r for r in runs if "ohm_diff" in r and "mom_diff" in r]
    present = sorted({series_of(r) for r in runs})
    if args.series is None:
        if len(present) != 1:
            raise SystemExit(f"several schemes in {args.data}: {present}; pass --series")
        args.series = present[0]
    runs = [r for r in runs if series_of(r) == args.series]
    # PLM keeps the historical file names; other schemes get a suffix.
    sfx = "" if args.series == "plm" else f"_{args.series}"
    if args.subtracted:
        out = Path(args.figures) / f"dynamo_dissipation_calibration_subtracted{sfx}.png"
        return subtracted_figure(runs, args.sat_start, out, SERIES[args.series][1])
    Ns = sorted({int(r["N"]) for r in runs})
    if not Ns:
        raise SystemExit(f"no calibration runs in {args.data}")
    N = args.n if args.n is not None else Ns[-1]
    if args.publication:
        return publication_figure(
            runs, N, args.sat_start, Path(args.figures),
            f"dynamo_calibration{sfx}_N{N}")
    runs = [r for r in runs if int(r["N"]) == N]
    if args.out == "dynamo_dissipation_calibration.png":
        args.out = f"dynamo_dissipation_calibration{sfx}_N{N}.png"
    eta_ladder = sorted((r for r in runs if float(r["mom_diff"]) == 0.0),
                        key=lambda r: float(r["ohm_diff"]))
    nu_ladder = sorted((r for r in runs if float(r["ohm_diff"]) == 0.0),
                       key=lambda r: float(r["mom_diff"]))
    if not eta_ladder or not nu_ladder:
        raise SystemExit(f"no calibration runs in {args.data}")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    fig.suptitle(f"{SERIES[args.series][1]}, ${N}^3$", fontsize=11, y=0.995)
    rows = (
        # (runs, imposed key, measured field index in curves(), title, cross-talk)
        (eta_ladder, "ohm_diff", 2, r"$\eta$", 1, r"$\nu$"),
        (nu_ladder, "mom_diff", 1, r"$\nu$", 2, r"$\eta$"),
    )
    summary = []
    for row, (ladder, key, idx, sym, other, other_sym) in enumerate(rows):
        base = curves(ladder[0], args.sat_start)
        base_mean = band_mean(base[0], base[idx])
        for colour, run in zip(LADDER_COLOURS, ladder):
            c = curves(run, args.sat_start)
            if c is None:
                continue
            imposed = float(run[key])
            label = (f"imposed {sym} = {imposed:.0e}" if imposed > 0
                     else "nothing imposed")
            # Measured field: the curve, and the additive expectation
            # (numerical at zero, band mean, plus the imposed coefficient).
            ax = axes[row][0]
            ax.loglog(c[0], c[idx], color=colour, lw=1.9, label=label)
            if imposed > 0:
                ax.axhline(base_mean + imposed, color=colour, lw=1.0, ls="--",
                           alpha=0.8)
            # The other field, for the cross-talk.
            axes[row][1].loglog(c[0], c[other], color=colour, lw=1.9, label=label)
            summary.append((sym, imposed, band_mean(c[0], c[idx]) - base_mean,
                            band_mean(c[0], c[other])))

        axes[row][0].set_title(
            fr"{sym}$_{{\rm eff}}(n)$ with imposed {sym}: dashed = numerical "
            f"(nothing imposed, band mean) + imposed", fontsize=10)
        axes[row][1].set_title(
            fr"cross-talk: {other_sym}$_{{\rm eff}}(n)$ of the same runs",
            fontsize=10)
        for ax, s in zip(axes[row], (sym, other_sym)):
            # Range from the in-band values: below the band the budget is the
            # forcing and flips sign, which on a log axis is a spike to nothing.
            lo, hi = np.inf, 0.0
            for line in ax.get_lines():
                x, y = line.get_xdata(), line.get_ydata()
                if len(x) < 3:
                    continue
                m = (x >= BAND[0]) & (x <= 1.0) & np.isfinite(y) & (y > 0)
                if m.any():
                    lo, hi = min(lo, y[m].min()), max(hi, y[m].max())
            ax.set_ylim(0.5 * lo, 2.0 * hi)
            ax.axvspan(*BAND, color="0.5", alpha=0.08, lw=0)
            ax.set_xlabel(r"$n / n_{\rm Nyquist}$")
            ax.set_ylabel(fr"{s}$_{{\rm eff}}(n)$")
            ax.grid(alpha=0.25, which="both")
            ax.set_xlim(0.03, 1.0)
        axes[row][0].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = Path(args.figures) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"N = {N}, band {BAND[0]}-{BAND[1]} of Nyquist, "
          f"t/t_cross >= {args.sat_start}")
    print(f"{'field':>5s} {'imposed':>9s} {'measured rise':>14s} "
          f"{'rise/imposed':>13s} {'other field':>12s}")
    for sym, imposed, rise, other in summary:
        frac = rise / imposed if imposed > 0 else float("nan")
        print(f"{sym:>5s} {imposed:9.1e} {rise:14.3e} {frac:13.2f} {other:12.3e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
