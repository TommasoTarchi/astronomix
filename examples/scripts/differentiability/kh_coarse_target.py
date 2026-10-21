r"""
Does coarse intermediate information make the KH reconstruction easy for the
two-stage method? (pilot of the KH shooting study, before any pre-registration)

The time-reversed state at t = T/2 is wrong mostly at large scales (|k| <= 8
holds ~80-90 % of its velocity-error energy at T = 160), in a coherent
combination the terminal observation cannot see (``kh_reversal_error.py``).
Coarse velocity data at T/2 would see exactly that part. Before multiple
shooting with coarse node data is built and pre-registered, this pilot asks
whether the same information already lets the two-stage method (Gauss-Newton
fit of the cold control to a target at T/2, then to the terminal observation)
succeed where it fails with the raw reversal: the target is the true velocity
at T/2 low-passed to |k| <= k_c (the coarse data, noiseless) plus the
reversal's |k| > k_c part. If it does, multiple shooting has to beat that, not
the raw two-stage method.

Modes (``--modes``): ``raw2`` / ``coarse2`` the two-stage method with the raw /
coarse-corrected T/2 target; ``raw3`` / ``coarse3`` three stages (fits to
T/4, T/2, T), the T/4 pin from reversing the T/2 state once more: for ``raw3``
the plain reversal (terminal-only data), for ``coarse3`` the coarse-corrected
T/2 state (its velocity replaced, its density and pressure from the reversal),
so the corrected large scales propagate into the earlier pin.

    python kh_coarse_target.py --horizon 160 --truths 1 --cutoff 8 --modes raw2 coarse2 \
        --output-directory <dir>
"""

# ==== GPU selection ====
from kh_recon import select_gpu

select_gpu()
# ruff: noqa: E402
# =======================

# general
import argparse
import json
import math
import os

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import FORWARDS, PALLAS

# KH study
from kh_recon import (
    METHOD_RUNNERS,
    KHParams,
    MethodSettings,
    build_kernels,
    build_problem,
    cold_start_control,
    reversal_node_estimates,
    truth_control,
)
from kh_reversal_error import band_filter


def multi_stage(stages, full, start_control, true_control, observation, settings):
    """
    Trust-ordered continuation: Gauss-Newton fits of the control to the pins
    of increasing time, then to the terminal observation, each stage starting
    from the previous result.

    Args:
        stages: ``[((problem, kernels), velocity target), ...]`` over
            [0, t_k] in increasing t_k (FORWARDS problems).
        full: ``(problem, kernels)`` of the full horizon.
        start_control: The cold control.
        true_control: The truth (IC-error traces only).
        observation: The terminal observation.
        settings: The Gauss-Newton settings.

    Returns:
        A dict with the IC error after every stage, the final IC error and
        relative loss and the total optimizer time.
    """
    control = start_control[None]
    stage_ic_errors = []
    optimizer_time = 0.0
    for (problem, kernels), target in stages:
        control, recorder = METHOD_RUNNERS["gauss_newton"](
            problem, kernels, target[None], control, true_control[None], settings, None
        )
        stage_ic_errors.append(float(kernels.ic_error(control, true_control[None])[0]))
        optimizer_time += recorder.optimizer_time
    problem, kernels = full
    control, recorder = METHOD_RUNNERS["gauss_newton"](
        problem, kernels, observation[None], control, true_control[None], settings, None
    )
    reference = float(kernels.feasible_loss(jnp.zeros_like(control), observation[None])[0])
    return dict(
        stage_ic_errors=stage_ic_errors,
        ic_error=float(kernels.ic_error(control, true_control[None])[0]),
        relative_loss=float(kernels.feasible_loss(control, observation[None])[0]) / reference,
        optimizer_time=optimizer_time + recorder.optimizer_time,
    )


def main():
    """Run the two- and three-stage continuations with raw and coarse-corrected pins."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--truths", type=int, nargs="+", required=True)
    parser.add_argument("--inits", type=int, nargs="+", default=[0])
    parser.add_argument("--cutoff", type=float, default=8.0,
                        help="|k| up to which the T/2 target velocity is the (coarse) data")
    parser.add_argument("--modes", nargs="+", default=["raw2", "coarse2"],
                        choices=["raw2", "coarse2", "raw3", "coarse3"])
    parser.add_argument("--num-cells", type=int, default=256)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--reversal-fraction", type=float, default=0.5)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    kh_params = KHParams(num_cells=args.num_cells, frozen_weno_weights=True)
    first_node = int(math.ceil(args.reversal_fraction * args.num_segments))
    problem = build_problem(kh_params, args.horizon, args.num_segments, PALLAS, FORWARDS)
    kernels = build_kernels(problem)
    quarter_node = first_node // 2

    def sub_horizon(node):
        sub_problem = build_problem(kh_params, args.horizon * node / args.num_segments, node, PALLAS, FORWARDS)
        return sub_problem, build_kernels(sub_problem)

    half = sub_horizon(first_node)
    quarter = sub_horizon(quarter_node) if any(mode.endswith("3") for mode in args.modes) else None
    segment_map = jax.jit(problem.segment_map)
    settings = MethodSettings()

    for truth_index in args.truths:
        true_control = truth_control(problem, truth_index)
        state = problem.initial_state(true_control)
        true_nodes = [state]
        for _ in range(args.num_segments):
            state = segment_map(state)
            true_nodes.append(state)
        observation = problem.observe(true_nodes[-1])
        reversed_nodes = reversal_node_estimates(problem, segment_map, observation)
        base = problem.observe(problem.initial_state(jnp.zeros_like(true_control)))

        def relative_error(target, node):
            true_velocity = problem.observe(true_nodes[node])
            return float(jnp.linalg.norm(target - true_velocity) / jnp.linalg.norm(true_velocity - base))

        # The T/2 pins: the reversal, and its velocity with the large scales
        # replaced by the (coarse, noiseless) data.
        raw_half = problem.observe(reversed_nodes[first_node])
        true_half = problem.observe(true_nodes[first_node])
        coarse_half = band_filter(true_half, 0, args.cutoff) + (raw_half - band_filter(raw_half, 0, args.cutoff))
        # The T/4 pins: the T/2 state reversed once more over the remaining
        # segments (for the raw state this is the plain reversal at T/4).
        corrected_state = reversed_nodes[first_node].at[1].set(coarse_half[0]).at[2].set(coarse_half[1])
        state = problem.reverse_velocity(corrected_state)
        for _ in range(first_node - quarter_node):
            state = segment_map(state)
        coarse_quarter = problem.observe(problem.reverse_velocity(state))
        raw_quarter = problem.observe(reversed_nodes[quarter_node])
        pins = dict(
            raw2=[(half, raw_half)],
            coarse2=[(half, coarse_half)],
            raw3=[(quarter, raw_quarter), (half, raw_half)],
            coarse3=[(quarter, coarse_quarter), (half, coarse_half)],
        )
        pin_errors = dict(
            raw_half=relative_error(raw_half, first_node),
            coarse_half=relative_error(coarse_half, first_node),
            raw_quarter=relative_error(raw_quarter, quarter_node),
            coarse_quarter=relative_error(coarse_quarter, quarter_node),
        )
        print(f"T={args.horizon:g} truth {truth_index}: pin errors "
              f"{json.dumps({key: round(value, 3) for key, value in pin_errors.items()})}", flush=True)

        results = []
        for init_index in args.inits:
            start_control = cold_start_control(problem, init_index)
            for mode in args.modes:
                result = multi_stage(
                    pins[mode], (problem, kernels), start_control, true_control, observation, settings
                )
                result.update(truth=truth_index, init=init_index, mode=mode, pin_errors=pin_errors)
                results.append(result)
                print(f"T={args.horizon:g} truth {truth_index} init {init_index}, {mode}: IC error after the "
                      f"stages {np.array2string(np.array(result['stage_ic_errors']), precision=3)}, final "
                      f"{result['ic_error']:.4f}, J/J(0) {result['relative_loss']:.2e} "
                      f"[{result['optimizer_time']:.0f} s]", flush=True)
        path = os.path.join(args.output_directory, f"coarse_target_T{args.horizon:g}_t{truth_index}.json")
        with open(path, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
