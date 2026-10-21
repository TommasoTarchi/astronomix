r"""
Can the late multiple-shooting chain sharpen the time-reversed node estimates?
(diagnostic of the KH shooting study, before any pre-registration)

With terminal-only data, the two-stage method of ``kh_recon.py`` fits the cold
control to the time-reversed velocity at t = T/2 and then to the observation.
It fails from T ~ 160 on because the reversed state at T/2 is too inaccurate,
not because the first stage's horizon is too long. Multiple shooting could help
there only if the lifted chain of the late nodes improves that target before
the control is fitted to it. This script tests exactly that, with the control
out of the loop:

1. The late nodes s_r .. s_{M-1} (r = M/2) start at the reversed estimates
   b_j and move in full state space to minimise the anchored penalty merit

       1/2 ||H Phi(s_{M-1}) - y||^2 + lambda/2 sum_j ||H s_j - H b_j||^2
                                   + w/2 sum_j ||Phi(s_j) - s_{j+1}||^2

   (L-BFGS on reverse-mode gradients; the anchors on the observed velocity,
   continuity on the full state). The anchor weight lambda decays over the
   stages ``--anchor-weights`` (the trust schedule): the nodes first stay near
   the reversal and become dynamically consistent with each other and the
   observation, then are released.
2. After every stage it records each late node's velocity error against the
   truth, relative to the size of the true perturbation there, next to the
   raw reversal's (the truth is used for evaluation only).
3. ``--downstream``: the two-stage method with the raw target and with the
   refined target of chosen stages, from the cold control of each init.

If no stage brings the t = T/2 node closer to the truth than the raw reversal,
the hybrid of multiple shooting and the two-stage method cannot do better than
the two-stage method, and the idea is closed.

    python kh_node_refinement.py --horizon 160 --truths 1 --inits 0 \
        --output-directory <dir> --downstream
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
import time

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import BACKWARDS, FORWARDS, PALLAS

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


# ============================================================================
#  L-BFGS
# ============================================================================

def lbfgs(value_and_gradient, start, num_iterations, memory=10, report=None):
    """
    Minimise a smooth function with L-BFGS and an Armijo backtracking search.

    Args:
        value_and_gradient: ``x -> (value, gradient)`` on flat float32 arrays.
        start: The start point.
        num_iterations: The number of accepted steps at most.
        memory: The number of stored curvature pairs.
        report: Optional ``(iteration, x, value) -> None`` called after every
            accepted step.

    Returns:
        The final point and its value.
    """
    x = start
    value, gradient = value_and_gradient(x)
    pairs = []
    for iteration in range(num_iterations):
        # Two-loop recursion for the quasi-Newton direction.
        direction = -gradient
        coefficients = []
        for step_difference, gradient_difference, inverse_curvature in reversed(pairs):
            coefficient = inverse_curvature * jnp.vdot(step_difference, direction)
            direction = direction - coefficient * gradient_difference
            coefficients.append(coefficient)
        if pairs:
            step_difference, gradient_difference, _ = pairs[-1]
            direction = direction * (
                jnp.vdot(step_difference, gradient_difference) / jnp.vdot(gradient_difference, gradient_difference)
            )
        for (step_difference, gradient_difference, inverse_curvature), coefficient in zip(
            pairs, reversed(coefficients)
        ):
            correction = inverse_curvature * jnp.vdot(gradient_difference, direction)
            direction = direction + (coefficient - correction) * step_difference
        slope = float(jnp.vdot(gradient, direction))
        if slope >= 0.0:
            # Not a descent direction (curvature lost to round-off): restart.
            pairs = []
            direction = -gradient
            slope = float(jnp.vdot(gradient, direction))
        # The first step has no curvature scale: keep it short.
        step_length = 1.0 if pairs else 1e-3 / max(float(jnp.linalg.norm(gradient)), 1e-30) * float(
            jnp.linalg.norm(x)
        )
        accepted = False
        for _ in range(20):
            trial = x + step_length * direction
            trial_value, trial_gradient = value_and_gradient(trial)
            if np.isfinite(float(trial_value)) and float(trial_value) <= float(value) + 1e-4 * step_length * slope:
                accepted = True
                break
            step_length *= 0.3
        if not accepted:
            break
        step_difference = trial - x
        gradient_difference = trial_gradient - gradient
        curvature = float(jnp.vdot(step_difference, gradient_difference))
        if curvature > 1e-12 * float(jnp.vdot(step_difference, step_difference)):
            pairs.append((step_difference, gradient_difference, 1.0 / curvature))
            pairs = pairs[-memory:]
        x, value, gradient = trial, trial_value, trial_gradient
        if report is not None:
            report(iteration, x, value)
    return x, value


# ============================================================================
#  The anchored late chain
# ============================================================================

def build_late_chain(problem, first_node):
    """
    The anchored penalty merit of the late nodes, with its gradient.

    Args:
        problem: The reconstruction problem (BACKWARDS mode).
        first_node: The index r of the first lifted node.

    Returns:
        ``merit(nodes, anchors, observation, anchor_weight, continuity_weight)
        -> (value, (data, anchor, continuity))`` and its jitted value-and-
        gradient with respect to ``nodes``; ``nodes`` stacks s_r .. s_{M-1}.
    """
    segments_in_parallel = jax.vmap(problem.segment_map)
    observe_all = jax.vmap(problem.observe)

    def merit(nodes, anchors, observation, anchor_weight, continuity_weight):
        segment_ends = segments_in_parallel(nodes)
        defects = segment_ends[:-1] - nodes[1:]
        data_residual = problem.observe(segment_ends[-1]) - observation
        anchor_residuals = observe_all(nodes) - anchors
        data_term = 0.5 * jnp.sum(data_residual ** 2)
        anchor_term = 0.5 * jnp.sum(anchor_residuals ** 2)
        continuity_term = 0.5 * jnp.sum(defects ** 2)
        value = data_term + anchor_weight * anchor_term + continuity_weight * continuity_term
        return value, (data_term, anchor_term, continuity_term)

    return merit, jax.jit(jax.value_and_grad(merit, has_aux=True))


def defect_breakdown(problem, nodes, observation):
    """
    The continuity defects of the late chain split by variable.

    Args:
        problem: The reconstruction problem.
        nodes: The late nodes s_r .. s_{M-1}.
        observation: The terminal observation (for the data residual).

    Returns:
        A dict of 1/2 sum of squared defects per variable (density, velocity,
        pressure) and the data term.
    """
    segment_ends = jax.vmap(problem.segment_map)(nodes)
    defects = segment_ends[:-1] - nodes[1:]
    squared = 0.5 * jnp.sum(defects ** 2, axis=(0, 2, 3))
    return dict(
        density=float(squared[0]),
        velocity=float(jnp.sum(squared[1:3])),
        pressure=float(jnp.sum(squared[3:])),
        data=float(0.5 * jnp.sum((problem.observe(segment_ends[-1]) - observation) ** 2)),
    )


def node_errors(problem, nodes, true_nodes, base_nodes):
    """
    The velocity error of each node relative to the true perturbation there.

    Args:
        problem: The reconstruction problem.
        nodes: The node states, (k, ...).
        true_nodes: The true states at the same times.
        base_nodes: The zero-control states at the same times.

    Returns:
        ||H s - H s*|| / ||H s* - H s0|| per node, as a numpy array.
    """
    return np.array([
        float(
            jnp.linalg.norm(problem.observe(node) - problem.observe(true_node))
            / jnp.linalg.norm(problem.observe(true_node) - problem.observe(base_node))
        )
        for node, true_node, base_node in zip(nodes, true_nodes, base_nodes)
    ])


def trajectory(problem, segment_map, start):
    """The states at the node times t_k = k T / M, k = 0..M, from ``start``."""
    states = [start]
    for _ in range(problem.num_segments):
        states.append(segment_map(states[-1]))
    return states


# ============================================================================
#  Downstream: the two-stage method with a given target
# ============================================================================

def two_stage(full, sub, start_control, true_control, observation, node_target, settings):
    """
    Two-stage single shooting: Gauss-Newton fit of the start control to the
    node target over [0, t_r], then to the observation over [0, T].

    Args:
        full: ``(problem, kernels)`` of the full horizon (FORWARDS).
        sub: ``(problem, kernels)`` of [0, t_r] (FORWARDS).
        start_control: The cold control.
        true_control: The truth (IC-error traces only).
        observation: The terminal observation.
        node_target: The velocity target at t_r.
        settings: The Gauss-Newton settings.

    Returns:
        A dict with the IC errors after each stage, the final relative loss
        and the optimizer times.
    """
    fitted, stage_one = METHOD_RUNNERS["gauss_newton"](
        sub[0], sub[1], node_target[None], start_control[None], true_control[None], settings, None
    )
    final, stage_two = METHOD_RUNNERS["gauss_newton"](
        full[0], full[1], observation[None], fitted, true_control[None], settings, None
    )
    reference = float(full[1].feasible_loss(jnp.zeros_like(final), observation[None])[0])
    return dict(
        ic_error_stage_one=float(sub[1].ic_error(fitted, true_control[None])[0]),
        ic_error=float(full[1].ic_error(final, true_control[None])[0]),
        relative_loss=float(full[1].feasible_loss(final, observation[None])[0]) / reference,
        optimizer_time=stage_one.optimizer_time + stage_two.optimizer_time,
    )


# ============================================================================
#  Main
# ============================================================================

def main():
    """Refine the late reversal nodes per truth and report node and downstream errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--truths", type=int, nargs="+", required=True)
    parser.add_argument("--inits", type=int, nargs="+", default=[0])
    parser.add_argument("--num-cells", type=int, default=256)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--reversal-fraction", type=float, default=0.5)
    parser.add_argument("--anchor-weights", type=float, nargs="+", default=[1.0, 0.1, 0.01, 0.0],
                        help="anchor weight of each refinement stage (the trust schedule)")
    parser.add_argument("--continuity-weight", type=float, default=1.0)
    parser.add_argument("--stage-iterations", type=int, default=40)
    parser.add_argument("--downstream", action="store_true",
                        help="also run the two-stage method with the raw and the refined targets")
    parser.add_argument("--downstream-stages", type=int, nargs="+", default=None,
                        help="refinement stages whose target the downstream run uses (default: all)")
    parser.add_argument("--check-gradient", action="store_true",
                        help="compare the gradient with central differences along random directions first")
    parser.add_argument("--native-adjoint", action="store_true",
                        help="reverse mode through the native tangent instead of the Pallas adjoint")
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    kh_params = KHParams(
        num_cells=args.num_cells,
        frozen_weno_weights=True,
        pallas_adjoint=not args.native_adjoint,
    )
    first_node = int(math.ceil(args.reversal_fraction * args.num_segments))
    late = slice(first_node, args.num_segments)

    # FORWARDS problems: the observation, the reversal and the Gauss-Newton
    # runs, exactly as kh_recon's run_cells builds them.
    forward_problem = build_problem(kh_params, args.horizon, args.num_segments, PALLAS, FORWARDS)
    forward_kernels = build_kernels(forward_problem)
    segment_map = jax.jit(forward_problem.segment_map)
    terminal_prediction = jax.jit(forward_problem.terminal_prediction)
    sub_problem = build_problem(
        kh_params, args.horizon * first_node / args.num_segments, first_node, PALLAS, FORWARDS
    )
    sub_kernels = build_kernels(sub_problem)
    # BACKWARDS problem for the reverse-mode gradient of the late chain.
    backward_problem = build_problem(kh_params, args.horizon, args.num_segments, PALLAS, BACKWARDS)
    merit, merit_value_and_gradient = build_late_chain(backward_problem, first_node)
    settings = MethodSettings()

    base_nodes = trajectory(forward_problem, segment_map, forward_problem.initial_state(jnp.zeros((5, 2))))
    for truth_index in args.truths:
        start_time = time.time()
        true_control = truth_control(forward_problem, truth_index)
        observation = terminal_prediction(true_control)
        true_nodes = trajectory(forward_problem, segment_map, forward_problem.initial_state(true_control))
        reversed_nodes = reversal_node_estimates(forward_problem, segment_map, observation)
        anchors = jnp.stack([forward_problem.observe(node) for node in reversed_nodes[late]])
        nodes = jnp.stack(reversed_nodes[late])
        shape = nodes.shape

        raw_errors = node_errors(forward_problem, reversed_nodes, true_nodes, base_nodes)
        record = dict(
            horizon=args.horizon, truth=truth_index, first_node=first_node,
            raw_node_errors=raw_errors.tolist(), stages=[],
        )
        print(f"truth {truth_index}: raw reversal node errors (t_k = k T/{args.num_segments}, k = 0..M): "
              f"{np.array2string(raw_errors, precision=3)}", flush=True)
        # How much of each reversed node's error the observation can see: the
        # node integrated forward to T against y, relative to the perturbation
        # at T (the same normalisation as the node errors).
        perturbation_at_end = float(jnp.linalg.norm(observation - forward_problem.observe(base_nodes[-1])))
        terminal_misfits = []
        for node_index in range(args.num_segments + 1):
            state = reversed_nodes[node_index]
            for _ in range(args.num_segments - node_index):
                state = segment_map(state)
            terminal_misfits.append(
                float(jnp.linalg.norm(forward_problem.observe(state) - observation)) / perturbation_at_end
            )
        record["raw_terminal_misfits"] = terminal_misfits
        print(f"  forward visibility: terminal misfit of each reversed node integrated to T "
              f"{np.array2string(np.array(terminal_misfits), precision=3)}", flush=True)
        breakdown = defect_breakdown(forward_problem, nodes, observation)
        record["start_defects"] = breakdown
        print(f"  start defects: {json.dumps({key: float(f'{value:.3e}') for key, value in breakdown.items()})}",
              flush=True)

        if args.check_gradient:
            # The frozen-weight linearisation is not the exact derivative of
            # the primal, so a ratio near (not at) one is expected.
            for direction_seed in range(3):
                direction = jax.random.normal(jax.random.PRNGKey(direction_seed), shape) * 1e-2
                (_, _), gradient = merit_value_and_gradient(nodes, anchors, observation, 1.0, 1.0)
                for step in (1e-1, 1e-2):
                    plus = merit(nodes + step * direction, anchors, observation, 1.0, 1.0)[0]
                    minus = merit(nodes - step * direction, anchors, observation, 1.0, 1.0)[0]
                    finite_difference = float(plus - minus) / (2 * step)
                    print(f"  [check] direction {direction_seed}, step {step:g}: gradient . d "
                          f"{float(jnp.vdot(gradient, direction)):.6e}, central difference {finite_difference:.6e}",
                          flush=True)

        refined_targets = []
        for stage, anchor_weight in enumerate(args.anchor_weights):
            def value_and_gradient(flat_nodes):
                (value, _), gradient = merit_value_and_gradient(
                    flat_nodes.reshape(shape), anchors, observation, anchor_weight, args.continuity_weight
                )
                return value, gradient.reshape(-1)

            stage_start = time.time()
            flat_nodes, _ = lbfgs(value_and_gradient, nodes.reshape(-1), args.stage_iterations)
            nodes = flat_nodes.reshape(shape)
            _, parts = merit(nodes, anchors, observation, anchor_weight, args.continuity_weight)
            errors = node_errors(forward_problem, list(nodes), true_nodes[late], base_nodes[late])
            refined_targets.append(forward_problem.observe(nodes[0]))
            breakdown = defect_breakdown(forward_problem, nodes, observation)
            print(f"    defects: {json.dumps({key: float(f'{value:.3e}') for key, value in breakdown.items()})}",
                  flush=True)
            record["stages"].append(dict(
                defects=breakdown,
                anchor_weight=anchor_weight,
                node_errors=errors.tolist(),
                data=float(parts[0]), anchor=float(parts[1]), continuity=float(parts[2]),
                seconds=time.time() - stage_start,
            ))
            print(f"  stage {stage} (lambda {anchor_weight:g}): late node errors "
                  f"{np.array2string(errors, precision=3)} (raw {np.array2string(raw_errors[late], precision=3)}); "
                  f"data {float(parts[0]):.3e}, anchor {float(parts[1]):.3e}, continuity {float(parts[2]):.3e} "
                  f"[{time.time() - stage_start:.0f} s]", flush=True)

        if args.downstream:
            record["downstream"] = []
            stages = args.downstream_stages if args.downstream_stages is not None else range(len(refined_targets))
            targets = [("raw", anchors[0])] + [(f"stage {stage}", refined_targets[stage]) for stage in stages]
            for init_index in args.inits:
                start_control = cold_start_control(forward_problem, init_index)
                for label, target in targets:
                    result = two_stage(
                        (forward_problem, forward_kernels), (sub_problem, sub_kernels),
                        start_control, true_control, observation, target, settings,
                    )
                    result.update(init=init_index, target=label)
                    record["downstream"].append(result)
                    print(f"  two-stage, init {init_index}, {label} target: IC error after stage 1 "
                          f"{result['ic_error_stage_one']:.3f}, final {result['ic_error']:.4f}, "
                          f"J/J(0) {result['relative_loss']:.2e} [{result['optimizer_time']:.0f} s]", flush=True)

        record["seconds"] = time.time() - start_time
        path = os.path.join(args.output_directory, f"node_refinement_T{args.horizon:g}_t{truth_index}.json")
        with open(path, "w") as handle:
            json.dump(record, handle, indent=1)
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
