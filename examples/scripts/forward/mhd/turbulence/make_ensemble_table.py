"""Realisation-averaged kinematic growth rates under matched forcing statistics.

The single-realisation ladder showed the growth rate at ``64^3`` scattering by
~20% between forcing realisations, wider than the scheme differences being
fitted. This reads the matched-forcing ensemble (``run_wenoz_and_ensemble.sh``,
track 2): astronomix WENO5 and WENO-Z driven with AthenaPK's own 30-mode
forcing statistics, and AthenaPK PLM, four realisations each, in the
kinematic-eigenmode setup of ``data/reynolds/``. For every (scheme, N) it
reports the mean and standard deviation of the eigenmode growth rate, the flow
it was measured in, and the scheme ratios with the realisation scatter
propagated -- which is the number the single-seed ratios could not carry.

    python make_ensemble_table.py --data data/ensemble_matched
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
from _mhd_metrics import dynamo_time_series, eigenmode_growth_rate, load_runs
from make_convergence_figures import SERIES, series_of

HERE = Path(__file__).resolve().parent

#: Window over which the flow is characterised (spun up, field still passive).
FLOW_WINDOW = (2.5, 5.0)


def measure(run):
    """Growth rate, its per-decade spread and the flow, for one realisation."""
    tc, E_B, E_K = dynamo_time_series(run)
    gamma, spread, per_decade, n = eigenmode_growth_rate(tc, E_B, E_K)
    t_snap = np.asarray(run["t_over_tc"])
    w = (t_snap >= FLOW_WINDOW[0]) & (t_snap <= FLOW_WINDOW[1])
    mach = float(np.mean(np.asarray(run["mach"])[w])) if w.any() else np.nan
    return dict(series=series_of(run), N=int(run["N"]), gamma=gamma,
                spread=spread, n_fit=n, mach=mach,
                seed=str(run.get("seed", run.get("rseed", "?"))),
                tc=tc, ratio=E_B / np.maximum(E_K, 1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "ensemble_matched"))
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--out", default=str(HERE / "data" / "ensemble_matched" /
                                        "ensemble.md"))
    args = p.parse_args()

    rows = [measure(r) for r in load_runs(args.data, skip=("smoke",))]
    if not rows:
        raise SystemExit(f"no runs in {args.data}")

    groups = {}
    for r in rows:
        groups.setdefault((r["series"], r["N"]), []).append(r)

    lines = ["# Matched-forcing ensemble: realisation-averaged growth rates", "",
             "Kinematic eigenmode growth rate `Gamma t_cross` (fit of ln E_B "
             "between 1e3 x seed and E_B/E_K = 1e-4), mean +- s.d. over forcing "
             "realisations, with the per-realisation values. astronomix is driven "
             "with AthenaPK's 30-mode forcing statistics (`--forcing athenapk`).",
             "",
             "| scheme | N | realisations | `Gamma t_cross` | s.d. | s.d./mean | "
             "per realisation | Mach (2.5-5 t_cross) |",
             "|---|---|---|---|---|---|---|---|"]
    stats = {}
    for (series, N), grp in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        g = np.array([r["gamma"] for r in grp], dtype=float)
        ok = np.isfinite(g)
        g = g[ok]
        mean = g.mean() if g.size else np.nan
        sd = g.std(ddof=1) if g.size > 1 else np.nan
        stats[(series, N)] = (mean, sd, g.size)
        per = ", ".join(f"{r['gamma']:.3f}" for r in grp)
        mach = np.nanmean([r["mach"] for r in grp])
        lines.append(f"| {SERIES[series][1]} | {N} | {g.size} | **{mean:.3f}** | "
                     f"{sd:.3f} | {sd / mean:.1%} | {per} | {mach:.3f} |")

    lines += ["", "## Scheme ratios (errors: realisation scatter of both means, "
              "propagated)", "",
              "| ratio | N | value |", "|---|---|---|"]
    for N in sorted({N for _, N in stats}):
        for num, den in (("astronomix", "plm"), ("astronomix_wenoz", "plm"),
                         ("astronomix_wenoz", "astronomix")):
            if (num, N) in stats and (den, N) in stats:
                m1, s1, n1 = stats[(num, N)]
                m2, s2, n2 = stats[(den, N)]
                ratio = m1 / m2
                err = ratio * np.sqrt((s1 / np.sqrt(n1) / m1) ** 2
                                      + (s2 / np.sqrt(n2) / m2) ** 2)
                lines.append(f"| {SERIES[num][1]} / {SERIES[den][1]} | {N} | "
                             f"**{ratio:.2f} +- {err:.2f}** |")

    text = "\n".join(lines)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")

    # Every realisation's E_B/E_K against time, scheme by colour, N by panel.
    Ns = sorted({N for _, N in stats})
    fig, axes = plt.subplots(1, len(Ns), figsize=(6.0 * len(Ns), 4.6),
                             squeeze=False)
    for ax, N in zip(axes[0], Ns):
        seen = set()
        for r in rows:
            if r["N"] != N:
                continue
            color, label = SERIES[r["series"]]
            # The box starts at rest, so E_B/E_K is unbounded before the driving
            # has built the flow; start the curves at one crossing time.
            m = r["tc"] >= 1.0
            ax.semilogy(r["tc"][m], np.maximum(r["ratio"][m], 1e-300), color=color,
                        lw=1.2, alpha=0.8,
                        label=label if r["series"] not in seen else None)
            seen.add(r["series"])
        ax.set_title(f"$N = {N}$")
        ax.set_xlabel(r"$t / t_{\rm cross}$")
        ax.set_ylabel(r"$E_B / E_K$")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path(args.figures) / "dynamo_ensemble.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out} and {args.out}")


if __name__ == "__main__":
    main()
