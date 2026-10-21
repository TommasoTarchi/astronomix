"""
Figure: time per step of astronomix's finite-volume scheme (VL2 + PLM + HLLD +
GLM, Pallas backend) against AthenaPK on one GPU, in single precision with the
double-precision curves as a reference.

Reads the results of ``benchmark.py`` (one file per precision, both codes
measured on exclusive GH200 nodes; an entry of either code can be refreshed
by a later run of that code alone) and draws the time per step and the
throughput (cell updates per second; for AthenaPK its own zone-cycles per
wall second) against the resolution N of the CP Alfvén wave (grid 2N x N x N).
Prints the table behind the figure, with the AthenaPK/astronomix ratios and the
single/double speed-ups of both codes.

Usage:
    python fv_athenapk_single_precision.py [--node gh200]
        [--results-directory results] [--figure-directory <repo>/pytests/mhd/figures/<node>]

expects ``<results-directory>/benchmark_<node>_{single,double}.json`` and writes
``fv_athenapk_single_precision.{svg,pdf,png}``.
"""

# general
import argparse
import json
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]

ASTRONOMIX_CODE = "astronomix pallas"
ATHENAPK_CODE = "AthenaPK"
LABELS = {ASTRONOMIX_CODE: "astronomix FV (VL2, Pallas)", ATHENAPK_CODE: "AthenaPK (VL2+PLM)"}
COLORS = {ASTRONOMIX_CODE: "tab:blue", ATHENAPK_CODE: "tab:red"}
MARKERS = {ASTRONOMIX_CODE: "o", ATHENAPK_CODE: "s"}


def load_benchmark(path: Path) -> dict:
    """
    The per-step times of one benchmark file.

    Args:
        path: A ``benchmark.py`` results file.

    Returns:
        A dict from code name to ``(resolutions, milliseconds_per_step,
        cell_updates_per_second)`` arrays sorted by resolution, plus ``device``.
    """
    with open(path) as file:
        entries = json.load(file)
    curves = {"device": entries[0]["device"] if entries else "unknown"}
    for code in (ASTRONOMIX_CODE, ATHENAPK_CODE):
        selected = sorted((entry for entry in entries if entry["code"] == code), key=lambda entry: entry["resolution"])
        curves[code] = (
            np.array([entry["resolution"] for entry in selected]),
            np.array([entry["milliseconds_per_step"] for entry in selected]),
            np.array([entry["zone_cycles_per_second"] for entry in selected]),
        )
    return curves


def print_table(single: dict, double: dict) -> None:
    """The numbers behind the figure, with ratios and precision speed-ups."""
    resolutions = sorted(
        set(single[ASTRONOMIX_CODE][0]) | set(single[ATHENAPK_CODE][0])
        | set(double[ASTRONOMIX_CODE][0]) | set(double[ATHENAPK_CODE][0])
    )

    def value(curves, code, resolution, column):
        resolution_values, milliseconds, throughput = curves[code]
        matches = np.nonzero(resolution_values == resolution)[0]
        if len(matches) == 0:
            return float("nan")
        return (milliseconds if column == "ms" else throughput)[matches[0]]

    print(
        f"{'N':>5} {'astronomix sp':>14} {'AthenaPK sp':>12} {'ratio sp':>9} "
        f"{'astronomix dp':>14} {'AthenaPK dp':>12} {'ratio dp':>9} {'dp/sp astronomix':>17} {'dp/sp AthenaPK':>15}"
    )
    for resolution in resolutions:
        astronomix_single = value(single, ASTRONOMIX_CODE, resolution, "ms")
        athenapk_single = value(single, ATHENAPK_CODE, resolution, "ms")
        astronomix_double = value(double, ASTRONOMIX_CODE, resolution, "ms")
        athenapk_double = value(double, ATHENAPK_CODE, resolution, "ms")
        print(
            f"{resolution:5d} {astronomix_single:11.3f} ms {athenapk_single:9.3f} ms {athenapk_single / astronomix_single:8.2f}x "
            f"{astronomix_double:11.3f} ms {athenapk_double:9.3f} ms {athenapk_double / astronomix_double:8.2f}x "
            f"{astronomix_double / astronomix_single:16.2f}x {athenapk_double / athenapk_single:14.2f}x"
        )
    print("throughput [Mcell-updates/s]:")
    for resolution in resolutions:
        print(
            f"{resolution:5d} "
            + " ".join(
                f"{label} {value(curves, code, resolution, 'throughput') / 1e6:8.1f}"
                for label, curves in (("sp", single), ("dp", double))
                for code in (ASTRONOMIX_CODE, ATHENAPK_CODE)
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", default="gh200")
    parser.add_argument("--results-directory", default=str(HERE / "results"))
    parser.add_argument("--figure-directory", default=None)
    arguments = parser.parse_args()

    results_directory = Path(arguments.results_directory)
    single = load_benchmark(results_directory / f"benchmark_{arguments.node}_single.json")
    double = load_benchmark(results_directory / f"benchmark_{arguments.node}_double.json")
    print_table(single, double)

    # -------------------------------------------------------------
    # =============== ↓ Time per step and throughput ↓ ============
    # -------------------------------------------------------------
    # Two panels rather than two y-axes: the throughput is the per-step time
    # converted to cell updates per second, shown on its own scale.
    figure, (axis_time, axis_throughput) = plt.subplots(1, 2, figsize=(12, 5))
    all_resolutions = set()
    for precision_label, curves, line_style, face in (
        ("single", single, "-", None),
        ("double", double, "--", "none"),
    ):
        for code in (ASTRONOMIX_CODE, ATHENAPK_CODE):
            resolutions, milliseconds, throughput = curves[code]
            if len(resolutions) == 0:
                continue
            all_resolutions |= set(int(value) for value in resolutions)
            style = dict(
                color=COLORS[code],
                marker=MARKERS[code],
                linestyle=line_style,
                linewidth=2,
                markersize=8,
                label=f"{LABELS[code]}, {precision_label}",
            )
            if face is not None:
                style["markerfacecolor"] = face
            axis_time.loglog(resolutions, milliseconds, **style)
            axis_throughput.loglog(resolutions, throughput / 1e6, **style)

    resolutions = sorted(all_resolutions)
    for axis in (axis_time, axis_throughput):
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(ticker.FixedLocator(resolutions))
        axis.xaxis.set_major_formatter(ticker.FixedFormatter([str(value) for value in resolutions]))
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.set_xlabel("N (grid size: 2N x N x N)", fontsize=12)
        axis.grid(True, which="major", ls="-", alpha=0.2)
    axis_time.set_ylabel("Time per step (ms)", fontsize=12)
    axis_time.set_title("Time per step vs N", fontsize=13)
    axis_throughput.set_ylabel("Cell updates per second (millions)", fontsize=12)
    axis_throughput.set_title("Throughput vs N", fontsize=13)
    axis_time.legend(loc="upper left", fontsize=9)
    device_tag = arguments.node.upper()
    figure.suptitle(f"3D CP Alfvén wave, VL2 + PLM + HLLD + GLM, one {device_tag}", fontsize=14)
    figure.tight_layout()

    figure_directory = Path(arguments.figure_directory or REPOSITORY / "pytests" / "mhd" / "figures" / arguments.node)
    figure_directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        figure.savefig(
            figure_directory / f"fv_athenapk_single_precision.{extension}",
            dpi=200 if extension == "png" else None,
        )
    print(f"wrote {figure_directory}/fv_athenapk_single_precision.{{svg,pdf,png}}")
    # -------------------------------------------------------------
    # =============== ↑ Time per step and throughput ↑ ============
    # -------------------------------------------------------------


if __name__ == "__main__":
    main()
