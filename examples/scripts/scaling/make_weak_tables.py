"""
Aggregate weak-scaling NPZ results into the paper's LaTeX tables and the
weak-scaling figure.

Reads the ``weak_<setup>_pallas_<tag>_G*.npz`` files written by the weak
scaling drivers. Efficiency is T(1)/T(G) with the 1-GPU rung as baseline
(ideal = 100%); throughput is Mcell/s/GPU, millions of cell updates (cells x
time steps) per second and per GPU. Runtimes are the warm (second) execution
of the compiled program; the cold execution and the deduced one-time warm-up
are printed alongside. The LaTeX tables quote the warm time per step, since the
FD and FV ladders run different step counts (10 and 100).

    # the GH200 campaign: FD (WENO-Z) and FV (VL2) side by side, hydro and MHD,
    # plus the figure (efficiency and Mcell/s/GPU versus G)
    python examples/scripts/scaling/make_weak_tables.py --figure --copy-to-paper

    # any single tag, one table per setup (e.g. the WENO-JS ladders)
    python examples/scripts/scaling/make_weak_tables.py --tag gh200w
"""

# general
import argparse
import glob
import os
import shutil

# numerics
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "weak_scaling")
FIGURE_DIR = os.path.join(_HERE, "figures")
PAPER_FIGURE_PATH = "/e/project1/astronomix/storcks1/astronomix_paper/figures/weak_scaling.svg"

SETUPS = ["hydro", "mhd", "selfgrav"]

#: The schemes of the GH200 campaign and their labels.
SCHEMES = {
    "fd": dict(label="FD (WENO-Z)"),
    "fv": dict(label="FV (VL2)"),
}

#: Result-file tag of every campaign ladder. FD MHD runs with the staged
#: Pallas constrained-transport kernels (``pallas_ct``).
CAMPAIGN_TAGS = {
    ("hydro", "fd"): "gh200zpair",
    ("hydro", "fv"): "gh200vl2pair",
    ("mhd", "fd"): "gh200zctpair",
    ("mhd", "fv"): "gh200vl2pair",
}

#: Alternative single-GPU configurations of a campaign ladder: where faster,
#: their 1-GPU rung is the efficiency baseline (marked with a dagger). FD MHD
#: runs its multi-GPU rungs with the Pallas constrained-transport kernels,
#: which save inter-device communication but are slower on one GPU than the
#: default constrained transport (tag gh200znoct).
SINGLE_GPU_ALTERNATIVE_TAGS = {("mhd", "fd"): "gh200znoctpair"}

#: Categorical colors (validated reference palette, slots 1 and 2) for the
#: scheme, marker and line style for the physics.
SCHEME_COLORS = {"fd": "#2a78d6", "fv": "#eb6834"}
PHYSICS_STYLES = {
    "hydro": dict(marker="o", linestyle="-", label="hydro"),
    "mhd": dict(marker="s", linestyle="--", label="MHD"),
}


def _fmt_cells(n):
    exponent = int(np.floor(np.log10(n)))
    mantissa = n / 10 ** exponent
    return f"${mantissa:.1f}\\cdot10^{{{exponent}}}$"


def _fmt_step_ms(seconds_per_step):
    """Format a time per step in ms with three significant digits."""
    milliseconds = 1e3 * seconds_per_step
    return f"{milliseconds:.0f}" if milliseconds >= 100 else f"{milliseconds:.1f}"


def _fmt_grid(cells):
    nx, ny, nz = cells
    if nx == ny == nz:
        return f"${nx}^{{3}}$"
    if ny == nz:
        return f"${nx}\\times{ny}^{{2}}$"
    return f"${nx}\\times{ny}\\times{nz}$"


def load_ladder(
    setup: str,
    tag: str,
    gpus_per_node: int,
    single_gpu_alternative_tag: str = None,
) -> list:
    """
    Read one weak-scaling ladder.

    Args:
        setup: ``"hydro"``, ``"mhd"`` or ``"selfgrav"``.
        tag: The result-file tag.
        gpus_per_node: GPUs per node (for the node count).
        single_gpu_alternative_tag: Optional tag of an alternative single-GPU
            configuration; its 1-GPU rung replaces the ladder's own (flagged
            ``alternative``) where it is faster.

    Returns:
        One dict per rung, sorted by the GPU count, with the efficiency
        relative to the 1-GPU rung (NaN without one).
    """
    pattern = os.path.join(DATA_DIR, f"weak_{setup}_pallas_{tag}_G*.npz")
    rows = []
    for path in sorted(glob.glob(pattern)):
        data = np.load(path, allow_pickle=True)
        num_gpus = int(data["G"])
        runtime = float(data["runtime"])
        total_cells = int(data["total_cells"])
        iterations = int(data["iterations"])
        rows.append(dict(
            G=num_gpus,
            nodes=max(1, num_gpus // gpus_per_node),
            cells=[int(extent) for extent in data["global_cells"]],
            total_cells=total_cells,
            runtime=runtime,
            iterations=iterations,
            step_time=runtime / iterations,
            # ``runtime_cold`` / ``warmup_s`` exist for the warm-repeat
            # measurement: the first execution and its one-time cost.
            runtime_cold=float(data["runtime_cold"]) if "runtime_cold" in data else None,
            warmup=float(data["warmup_s"]) if "warmup_s" in data else None,
            # Recompute cell updates/s/GPU from the raw entries (early runs
            # recorded cells_per_s_per_gpu without the timestep factor).
            throughput=total_cells * iterations / runtime / num_gpus,
            memory_gib=float(data["total_bytes_per_dev"]) / 1024 ** 3,
        ))
    rows.sort(key=lambda row: row["G"])
    for row in rows:
        row["alternative"] = False
    if single_gpu_alternative_tag is not None:
        alternative_rows = load_ladder(setup, single_gpu_alternative_tag, gpus_per_node)
        alternative = next((row for row in alternative_rows if row["G"] == 1), None)
        own = next((row for row in rows if row["G"] == 1), None)
        if alternative is not None and (own is None or alternative["runtime"] < own["runtime"]):
            alternative["alternative"] = True
            alternative["replaced_runtime"] = own["runtime"] if own is not None else None
            rows = [alternative] + [row for row in rows if row["G"] != 1]
    baseline = next((row["runtime"] for row in rows if row["G"] == 1), None)
    for row in rows:
        row["efficiency"] = 100.0 * baseline / row["runtime"] if baseline else float("nan")
    return rows


def print_text_table(title: str, rows: list) -> None:
    """Print one ladder as a plain-text table."""
    print(f"\n===== {title} =====")
    print(f"{'GPUs':>5} {'Nodes':>5} {'Global grid':>18} {'Cells':>10} "
          f"{'Steps':>5} {'Warm[s]':>8} {'ms/step':>8} {'Cold[s]':>8} {'Warmup[s]':>9} "
          f"{'Eff.[%]':>8} {'Mcell/s/GPU':>12} {'Mem[GiB]':>9}")
    for row in rows:
        if row.get("alternative"):
            print(f"  (1-GPU rung from the alternative single-GPU configuration; "
                  f"the ladder's own 1-GPU rung took {row['replaced_runtime']:.2f} s, "
                  f"{_fmt_step_ms(row['replaced_runtime'] / row['iterations'])} ms per step)")
        cold = f"{row['runtime_cold']:>8.2f}" if row["runtime_cold"] is not None else f"{'---':>8}"
        warmup = f"{row['warmup']:>9.2f}" if row["warmup"] is not None else f"{'---':>9}"
        print(f"{row['G']:>5} {row['nodes']:>5} "
              f"{'x'.join(str(extent) for extent in row['cells']):>18} {row['total_cells']:>10.2e} "
              f"{row['iterations']:>5} {row['runtime']:>8.2f} {_fmt_step_ms(row['step_time']):>8} "
              f"{cold} {warmup} {row['efficiency']:>8.1f} "
              f"{row['throughput'] / 1e6:>12.0f} {row['memory_gib']:>9.1f}")


def single_tag_tables(tag: str, gpus_per_node: int) -> None:
    """Print the plain-text table and LaTeX rows of every setup for one tag."""
    for setup in SETUPS:
        rows = load_ladder(setup, tag, gpus_per_node)
        if not rows:
            print(f"[{setup}] no results for tag {tag}")
            continue
        print_text_table(f"{setup} ({tag})", rows)
        print("\n% LaTeX rows:")
        for row in rows:
            print(
                f"        {row['G']} & {row['nodes']} & {_fmt_grid(row['cells'])} & "
                f"{_fmt_cells(row['total_cells'])} & {_fmt_step_ms(row['step_time'])} & "
                f"{row['efficiency']:5.1f} & {row['throughput'] / 1e6:.0f} \\\\"
            )


def campaign_tables(gpus_per_node: int) -> dict:
    """
    Print the FD | FV LaTeX rows per physics (same global grid per rung) and
    the plain-text tables of the four ladders.

    Args:
        gpus_per_node: GPUs per node.

    Returns:
        ``{(physics, scheme): rows}`` of the ladders found.
    """
    ladders = {}
    for physics in ("hydro", "mhd"):
        for scheme, scheme_info in SCHEMES.items():
            rows = load_ladder(
                physics,
                CAMPAIGN_TAGS[(physics, scheme)],
                gpus_per_node,
                single_gpu_alternative_tag=SINGLE_GPU_ALTERNATIVE_TAGS.get((physics, scheme)),
            )
            if rows:
                ladders[(physics, scheme)] = rows
                print_text_table(
                    f"{physics} {scheme_info['label']} ({CAMPAIGN_TAGS[(physics, scheme)]})",
                    rows,
                )

    for physics in ("hydro", "mhd"):
        fd_rows = {row["G"]: row for row in ladders.get((physics, "fd"), [])}
        fv_rows = {row["G"]: row for row in ladders.get((physics, "fv"), [])}
        print(f"\n% LaTeX rows, {physics}: G & nodes & grid & cells & "
              f"FD (ms/step, eff, Mcell/s/GPU) & FV (ms/step, eff, Mcell/s/GPU)")
        for num_gpus in sorted(set(fd_rows) | set(fv_rows)):
            reference = fd_rows.get(num_gpus) or fv_rows.get(num_gpus)
            columns = [
                str(num_gpus),
                str(reference["nodes"]),
                _fmt_grid(reference["cells"]),
                _fmt_cells(reference["total_cells"]),
            ]
            for scheme_rows in (fd_rows, fv_rows):
                row = scheme_rows.get(num_gpus)
                if row is None:
                    columns += ["---", "---", "---"]
                else:
                    dagger = "$^{\\dagger}$" if row["alternative"] else ""
                    columns += [
                        f"{_fmt_step_ms(row['step_time'])}{dagger}",
                        f"{row['efficiency']:.1f}",
                        f"{row['throughput'] / 1e6:.0f}",
                    ]
            print("        " + " & ".join(columns) + " \\\\")
    return ladders


def plot_weak_scaling(ladders: dict, copy_to_paper: bool) -> None:
    """
    Draw efficiency and Mcell/s/GPU versus G for the four configurations.

    Args:
        ladders: ``{(physics, scheme): rows}`` from ``campaign_tables``.
        copy_to_paper: Also copy the SVG into the paper's figures folder.
    """
    # plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (efficiency_axis, throughput_axis) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    gpu_counts = sorted({row["G"] for rows in ladders.values() for row in rows})

    for (physics, scheme), rows in sorted(ladders.items()):
        style = PHYSICS_STYLES[physics]
        line_options = dict(
            color=SCHEME_COLORS[scheme],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6,
            label=f"{SCHEMES[scheme]['label']}, {style['label']}",
        )
        gpus = [row["G"] for row in rows]
        efficiency_axis.plot(gpus, [row["efficiency"] for row in rows], **line_options)
        throughput_axis.plot(gpus, [row["throughput"] / 1e6 for row in rows], **line_options)

    efficiency_axis.axhline(100.0, color="#8a8984", linestyle=":", linewidth=1.0)
    efficiency_axis.set_ylim(0, 110)
    efficiency_axis.set_ylabel(r"weak-scaling efficiency $T(1)/T(G)$ [%]")
    # FV and FD throughputs differ by an order of magnitude: logarithmic axis.
    throughput_axis.set_ylabel("Mcell/s/GPU")
    throughput_axis.set_yscale("log")
    for axis in (efficiency_axis, throughput_axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(gpu_counts)
        axis.set_xticklabels([str(count) for count in gpu_counts])
        axis.xaxis.set_minor_locator(plt.NullLocator())
        axis.set_xlabel("GPUs (4 per node)")
        axis.grid(True, which="major", alpha=0.25)
    efficiency_axis.legend(fontsize=8, loc="lower left", framealpha=0.9)

    figure.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    figure_stem = os.path.join(FIGURE_DIR, "weak_scaling")
    for extension in ("svg", "pdf", "png"):
        figure.savefig(f"{figure_stem}.{extension}", dpi=200)
    plt.close(figure)
    print(f"\nwrote {figure_stem}.{{svg,pdf,png}}")
    if copy_to_paper:
        shutil.copyfile(f"{figure_stem}.svg", PAPER_FIGURE_PATH)
        print(f"copied to {PAPER_FIGURE_PATH}")


def main():
    """Print the tables (and draw the figure) for the requested ladders."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None,
                        help="one tag, all setups (default: the FD/FV GH200 campaign)")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--figure", action="store_true",
                        help="draw the campaign figure (efficiency, Mcell/s/GPU)")
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="copy the figure SVG into the paper's figures folder")
    args = parser.parse_args()

    if args.tag is not None:
        single_tag_tables(args.tag, args.gpus_per_node)
        return
    ladders = campaign_tables(args.gpus_per_node)
    if args.figure and ladders:
        plot_weak_scaling(ladders, args.copy_to_paper)


if __name__ == "__main__":
    main()
