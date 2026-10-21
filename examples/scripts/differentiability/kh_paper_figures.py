r"""
Figures of the KH long-horizon reconstruction subsection (Sec.
``sec:multiple_shooting`` of the methods paper), from the stored campaign
results.

* ``horizon`` (CPU): (a) success rate versus horizon of single-shooting
  Gauss-Newton and the reversal-informed two-stage continuation on the 16
  reporting runs (truths 3-6 x cold starts 0-3), and the two-stage method with
  coarse data at T/2 (six further truths, one start); (b) the four tuning runs
  (truths 1-2 x starts 0-1) of every multiple-shooting formulation next to the
  two Gauss-Newton baselines on the same runs; (c) the time-reversed node
  estimates at T = 160 t_g: their velocity error at the node and the misfit
  they leave at T after forward integration, both relative to the true
  perturbation (median and range over six truths).
* ``example`` (GPU unless cached): the median-IC-error single-shooting run at
  T = 120 t_g and the two-stage run from the same truth and start: initial
  transverse velocity and terminal vorticity next to the truth.

Data: ``revision_data/kh_shooting/`` (cell files of ``kh_recon.py run`` and the
diagnostics of ``kh_node_refinement.py`` and ``kh_coarse_target.py``).

    python kh_paper_figures.py horizon --output figures/kh_horizon.svg
    python kh_paper_figures.py example --horizon 120 --output figures/kh_example_T120.svg
"""

# general
import argparse
import glob
import json
import os

# numerics
import numpy as np

DATA_DIRECTORY = "/e/project1/astronomix/storcks1/revision_data/kh_shooting"
SUCCESS_THRESHOLD = 0.1

#: Categorical slots 1-3 of the validated reference palette: single shooting,
#: the continuation, multiple shooting (all variants; told apart by marker).
SINGLE_COLOR = "#2a78d6"
CONTINUATION_COLOR = "#eb6834"
MULTIPLE_COLOR = "#1baf7a"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#d9d8d4"

#: (directory, method, tag) of every result set used.
SINGLE = ("gauss_newton", "gauss_newton", "mu0.01_it60_frozen")
SINGLE_LATE = ("reversal/reporting", "gauss_newton", "mu0.01_it60_frozen")
CONTINUATION = ("reversal/reporting", "gauss_newton", "mu0.01_it60_cnode0.5_frozen")
TUNING = "reversal/tuning"
TUNING_SETS = [
    ("single shooting (Gauss-Newton)", "gauss_newton", "mu0.01_it60_frozen", SINGLE_COLOR, "o"),
    ("two-stage continuation", "gauss_newton", "mu0.01_it60_cnode0.5_frozen", CONTINUATION_COLOR, "s"),
    ("MS, Bock, reversed nodes", "gauss_newton_multiple", "mu0.01_rho1_it60_nrev0.5_frozen", MULTIPLE_COLOR, "^"),
    ("MS, Bock, relaxed", "gauss_newton_multiple", "mu0.01_rho1_it60_nrev0.5_kappa0.1+0.1_frozen", MULTIPLE_COLOR, "v"),
    ("MS, Bock, anchored nodes", "gauss_newton_multiple", "mu0.01_rho1_it60_nrev0.5_anchor10x0.7_frozen", MULTIPLE_COLOR, "D"),
    ("MS, penalty continuation", "gauss_newton_penalty", "mu0.01_w0.01x3to10000_it60_pol60_nrev0.5_frozen", MULTIPLE_COLOR, "P"),
]
REPORTING_TRUTHS = (3, 4, 5, 6)


def load_results(directory, method, tag, truths=None):
    """
    Final IC errors and optimizer times of a result set, per horizon.

    Args:
        directory: The directory under ``DATA_DIRECTORY``.
        method: The method name of the cell files.
        tag: The configuration tag of the cell files.
        truths: Keep only these truth seeds (None: all).

    Returns:
        ``{horizon: [(truth, init, ic_error, optimizer_seconds), ...]}``.
    """
    results = {}
    for path in glob.glob(os.path.join(DATA_DIRECTORY, directory, f"{method}__{tag}__T*__t*_i*.npz")):
        with np.load(path, allow_pickle=True) as cell:
            truth = int(cell["truth_index"])
            if truths is not None and truth not in truths:
                continue
            seconds = float(cell["batch_optimizer_time"])
            if "start_optimizer_time" in cell.files:
                seconds += float(cell["start_optimizer_time"])
            results.setdefault(float(cell["horizon"]), []).append(
                (truth, int(cell["init_index"]), float(cell["final_ic_error"]), seconds)
            )
    return results


def wilson_interval(successes, total, z=1.0):
    """The Wilson score interval of a success fraction (z = 1: one sigma)."""
    fraction = successes / total
    denominator = 1 + z ** 2 / total
    centre = (fraction + z ** 2 / (2 * total)) / denominator
    half_width = z * np.sqrt(fraction * (1 - fraction) / total + z ** 2 / (4 * total ** 2)) / denominator
    return centre - half_width, centre + half_width


def success_curve(results):
    """Horizons, success fractions and Wilson intervals of a result set."""
    horizons = sorted(results)
    fractions, lower, upper = [], [], []
    for horizon in horizons:
        runs = results[horizon]
        successes = sum(error < SUCCESS_THRESHOLD for _, _, error, _ in runs)
        fractions.append(successes / len(runs))
        interval = wilson_interval(successes, len(runs))
        lower.append(fractions[-1] - interval[0])
        upper.append(interval[1] - fractions[-1])
    return np.array(horizons), np.array(fractions), np.array([lower, upper])


def style_axis(axis):
    """Recessive grid and spines."""
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(MUTED_INK)
        spine.set_linewidth(0.8)
    axis.tick_params(colors=MUTED_INK, labelcolor=INK)


def horizon_figure(output):
    """Draw the three-panel horizon figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (axis_a, axis_b, axis_c) = plt.subplots(1, 3, figsize=(13.0, 3.9), constrained_layout=True)

    # (a) the reporting runs.
    single = load_results(*SINGLE, truths=REPORTING_TRUTHS)
    single.update(load_results(*SINGLE_LATE, truths=REPORTING_TRUTHS))
    continuation = load_results(*CONTINUATION, truths=REPORTING_TRUTHS)
    for results, color, marker, label in (
        (single, SINGLE_COLOR, "o", "single shooting (Gauss-Newton)"),
        (continuation, CONTINUATION_COLOR, "s", "two-stage continuation"),
    ):
        horizons, fractions, errors = success_curve(results)
        axis_a.errorbar(horizons, fractions, yerr=errors, color=color, marker=marker, markersize=7,
                        linewidth=2.0, capsize=3, label=label)
    coarse = {}
    for path in glob.glob(os.path.join(DATA_DIRECTORY, "diagnostics", "coarse_target", "coarse_target_T*_t*.json")):
        with open(path) as handle:
            for result in json.load(handle):
                if result["target"].startswith("coarse"):
                    horizon = float(os.path.basename(path).split("_T")[1].split("_")[0])
                    coarse.setdefault(horizon, []).append((result["truth"], result["init"], result["ic_error"], 0.0))
    horizons, fractions, errors = success_curve(coarse)
    axis_a.errorbar(horizons, fractions, yerr=errors, color=CONTINUATION_COLOR, marker="s", markersize=7,
                    markerfacecolor="white", linestyle=":", linewidth=1.5, capsize=3,
                    label="two-stage, coarse data at T/2 (6 truths)")
    axis_a.set_xlabel(r"horizon $T$ [$t_g$]")
    axis_a.set_ylabel(r"success rate (IC error $< 0.1$)")
    axis_a.set_title(r"(a) reporting runs (4 truths $\times$ 4 starts)", fontsize=10, color=INK)
    axis_a.set_ylim(-0.05, 1.08)
    axis_a.set_xticks([20, 60, 80, 100, 120, 160, 200])
    axis_a.legend(fontsize=7.5, loc="lower left", framealpha=0.95)

    # (b) the tuning runs of every formulation.
    offsets = np.linspace(-10.0, 10.0, len(TUNING_SETS))
    for (label, method, tag, color, marker), offset in zip(TUNING_SETS, offsets):
        results = load_results(TUNING, method, tag)
        horizons = sorted(results)
        successes = [sum(error < SUCCESS_THRESHOLD for _, _, error, _ in results[horizon]) for horizon in horizons]
        axis_b.plot(np.array(horizons) + offset, successes, linestyle="none", marker=marker, markersize=8,
                    color=color, markeredgecolor="white", markeredgewidth=0.8, label=label)
    axis_b.set_xlabel(r"horizon $T$ [$t_g$]")
    axis_b.set_ylabel("successful runs (of 4)")
    axis_b.set_title("(b) same 4 tuning runs, all formulations", fontsize=10, color=INK)
    axis_b.set_xticks([100, 120, 160, 200])
    axis_b.set_yticks([0, 1, 2, 3, 4])
    axis_b.set_ylim(-0.3, 4.4)
    axis_b.legend(fontsize=7.5, loc="upper right", framealpha=0.95)

    # (c) what the time reversal knows at T = 160.
    errors, misfits = [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIRECTORY, "diagnostics", "node_refinement_n256",
                                              "node_refinement_T160_t*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        errors.append(record["raw_node_errors"][:-1])
        misfits.append(record["raw_terminal_misfits"][:-1])
    errors, misfits = np.array(errors), np.array(misfits)
    node_times = np.arange(errors.shape[1]) / (errors.shape[1])
    for values, linestyle, label in (
        (errors, "-", "velocity error of the reversed state"),
        (misfits, "--", "its misfit at T after forward integration"),
    ):
        axis_c.fill_between(node_times, values.min(axis=0), values.max(axis=0), color=MUTED_INK, alpha=0.15,
                            linewidth=0)
        axis_c.plot(node_times, np.median(values, axis=0), color=INK, linestyle=linestyle, marker="o",
                    markersize=5, linewidth=2.0, label=label)
    axis_c.set_yscale("log")
    axis_c.set_xlabel(r"node time $t/T$")
    axis_c.set_ylabel("relative to the true perturbation")
    axis_c.set_title(r"(c) time reversal at $T = 160\,t_g$ (6 truths)", fontsize=10, color=INK)
    axis_c.axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle=":")
    axis_c.legend(fontsize=7.5, loc="upper right", framealpha=0.95)

    for axis in (axis_a, axis_b, axis_c):
        style_axis(axis)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    figure.savefig(output)
    figure.savefig(os.path.splitext(output)[0] + ".png", dpi=200)
    print(f"-> {output}")


def example_figure(horizon, output, cache):
    """Draw the paired reconstruction example at one horizon (GPU unless cached)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single_runs = load_results(*(SINGLE if horizon <= 120 else SINGLE_LATE), truths=REPORTING_TRUTHS)[horizon]
    # The median-IC-error single-shooting run (lower median), and the
    # continuation run from the same truth and start.
    single_runs = sorted(single_runs, key=lambda run: run[2])
    truth, init, single_error, _ = single_runs[(len(single_runs) - 1) // 2]

    def cell_path(directory, method, tag):
        return os.path.join(DATA_DIRECTORY, directory, f"{method}__{tag}__T{horizon:g}__t{truth}_i{init}.npz")

    single_directory = SINGLE if horizon <= 120 else SINGLE_LATE
    paths = [cell_path(*single_directory), cell_path(*CONTINUATION)]
    if os.path.exists(cache):
        with np.load(cache) as data:
            fields = {key: data[key] for key in data.files}
    else:
        from kh_recon import select_gpu

        select_gpu()
        import jax
        import jax.numpy as jnp
        from kh_recon import KHParams, build_problem, vorticity

        problem = build_problem(KHParams(), horizon)
        terminal_prediction = jax.jit(problem.terminal_prediction)
        cell_size = problem.kh_params.box_size / problem.kh_params.num_cells
        fields = {}
        with np.load(paths[0]) as cell:
            true_control = jnp.asarray(cell["true_control"])
        fields["seed_truth"] = np.asarray(problem.seed_field(true_control)[1])
        fields["vorticity_truth"] = vorticity(np.asarray(terminal_prediction(true_control)), cell_size)
        for name, path in zip(("single", "continuation"), paths):
            with np.load(path) as cell:
                control = jnp.asarray(cell["final_control"])
                fields[f"ic_error_{name}"] = np.asarray(float(cell["final_ic_error"]))
                fields[f"relative_loss_{name}"] = np.asarray(float(cell["final_relative_loss"]))
            fields[f"seed_{name}"] = np.asarray(problem.seed_field(control)[1])
            fields[f"vorticity_{name}"] = vorticity(np.asarray(terminal_prediction(control)), cell_size)
        np.savez(cache, **fields)

    lower_y, upper_y = 0.28, 0.72
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 4.6), constrained_layout=True)
    columns = [
        ("truth", "truth"),
        ("single", f"single shooting\nIC error {float(fields['ic_error_single']):.2f}, "
                   f"J/J(0) {float(fields['relative_loss_single']):.1e}"),
        ("continuation", f"two-stage continuation\nIC error {float(fields['ic_error_continuation']):.1e}, "
                         f"J/J(0) {float(fields['relative_loss_continuation']):.1e}"),
    ]
    for row, (prefix, label) in enumerate((("seed", r"$v_y'(t=0)$"), ("vorticity", r"$\omega_z(T)$"))):
        cropped = []
        for name, _ in columns:
            field = fields[f"{prefix}_{name}"]
            num_cells = field.shape[1]
            cropped.append(field[:, int(lower_y * num_cells):int(upper_y * num_cells)])
        limit = float(np.percentile(np.abs(np.stack(cropped)), 99.5))
        for column, ((_, title), field) in enumerate(zip(columns, cropped)):
            axis = axes[row][column]
            image = axis.imshow(field.T, origin="lower", extent=[0, 1, lower_y, upper_y], cmap="RdBu_r",
                                vmin=-limit, vmax=limit, aspect="auto", rasterized=True)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=9, color=INK)
        axes[row][0].set_ylabel(label, fontsize=10)
        figure.colorbar(image, ax=axes[row][:], shrink=0.85, location="right", pad=0.01)
    figure.suptitle(rf"$T = {horizon:g}\,t_g$, truth {truth}, start {init}", fontsize=10, color=INK)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    figure.savefig(output, dpi=160)
    figure.savefig(os.path.splitext(output)[0] + ".png", dpi=160)
    print(f"-> {output} (truth {truth}, init {init}, single-shooting IC error {single_error:.3f})")


def main():
    """Draw the requested figure."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    horizon_parser = commands.add_parser("horizon")
    horizon_parser.add_argument("--output", required=True)
    example_parser = commands.add_parser("example")
    example_parser.add_argument("--horizon", type=float, default=120.0)
    example_parser.add_argument("--output", required=True)
    example_parser.add_argument("--cache", default=os.path.join(DATA_DIRECTORY, "diagnostics", "example_fields.npz"))
    args = parser.parse_args()
    if args.command == "horizon":
        horizon_figure(args.output)
    else:
        example_figure(args.horizon, args.output, args.cache)


if __name__ == "__main__":
    main()
