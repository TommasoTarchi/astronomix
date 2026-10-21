r"""
Where does the time-reversal error of the KH node estimates live, and which
part of it can the terminal observation see? (diagnostic of the KH shooting
study)

For each truth the script compares the time-reversed node estimates b_k
(``kh_recon.reversal_node_estimates``) with the true states s*_k at the node
times t_k = k T / M and reports

* the velocity error per node, and the error of every variable at the first
  reversal node t_r = T/2, relative to the true perturbation there;
* the radial spectrum of the velocity error and of the true velocity
  perturbation at t_r in bands of |k| (box units);
* the forward visibility per band: the band-passed full-state error added to
  the true state at t_r, integrated to T and compared with the observation,
  relative to the perturbation at T. A band whose error stays invisible at T
  is in the forward-decaying directions the terminal data cannot constrain;
  only data at the nodes can.

The truth is used throughout (this is an analysis, not a method).

    python kh_reversal_error.py --horizon 160 --truths 1 2 7 8 9 10 --output <file.json>
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

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import FORWARDS, PALLAS

# KH study
from kh_recon import KHParams, build_problem, reversal_node_estimates, truth_control

#: Radial wavenumber bands (box units, |k| in (lower, upper]).
BANDS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 182)]
VARIABLES = ["density", "velocity_x", "velocity_y", "pressure"]


def band_filter(field, lower, upper):
    """
    Keep the Fourier components of a 2D field with lower < |k| <= upper.

    Args:
        field: The field, (..., n, n); x periodic, y treated as periodic too
            (the perturbation vanishes near the open y boundaries).
        lower: The lower band edge (exclusive; 0 keeps the mean).
        upper: The upper band edge (inclusive).

    Returns:
        The band-passed field.
    """
    num_cells = field.shape[-1]
    wavenumbers = jnp.fft.fftfreq(num_cells, d=1.0 / num_cells)
    radius = jnp.sqrt(wavenumbers[:, None] ** 2 + wavenumbers[None, :] ** 2)
    keep = (radius <= upper) & ((radius > lower) | (lower == 0))
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(field) * keep))


def band_energy(field):
    """The squared norm of a (..., n, n) field in each band of ``BANDS``."""
    return [float(jnp.sum(band_filter(field, lower, upper) ** 2)) for lower, upper in BANDS]


def main():
    """Report node errors, error spectra and per-band forward visibility per truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--truths", type=int, nargs="+", required=True)
    parser.add_argument("--num-cells", type=int, default=256)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--reversal-fraction", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kh_params = KHParams(num_cells=args.num_cells, frozen_weno_weights=True)
    problem = build_problem(kh_params, args.horizon, args.num_segments, PALLAS, FORWARDS)
    segment_map = jax.jit(problem.segment_map)
    first_node = int(math.ceil(args.reversal_fraction * args.num_segments))

    def integrate(state, num_segments):
        for _ in range(num_segments):
            state = segment_map(state)
        return state

    base_nodes = [problem.initial_state(jnp.zeros((kh_params.num_modes, 2)))]
    for _ in range(args.num_segments):
        base_nodes.append(segment_map(base_nodes[-1]))

    records = []
    for truth_index in args.truths:
        true_nodes = [problem.initial_state(truth_control(problem, truth_index))]
        for _ in range(args.num_segments):
            true_nodes.append(segment_map(true_nodes[-1]))
        observation = problem.observe(true_nodes[-1])
        reversed_nodes = reversal_node_estimates(problem, segment_map, observation)

        velocity_errors = [
            float(jnp.linalg.norm(problem.observe(b) - problem.observe(s))
                  / jnp.linalg.norm(problem.observe(s) - problem.observe(s0)))
            for b, s, s0 in zip(reversed_nodes, true_nodes, base_nodes)
        ]
        error = reversed_nodes[first_node] - true_nodes[first_node]
        perturbation = true_nodes[first_node] - base_nodes[first_node]
        variable_errors = {
            name: float(jnp.linalg.norm(error[index]) / jnp.linalg.norm(perturbation[index]))
            for index, name in enumerate(VARIABLES)
        }
        velocity_error_bands = band_energy(error[1:3])
        velocity_perturbation_bands = band_energy(perturbation[1:3])

        perturbation_at_end = float(jnp.linalg.norm(observation - problem.observe(base_nodes[-1])))
        remaining = args.num_segments - first_node

        def visibility(state_error):
            end = integrate(true_nodes[first_node] + state_error, remaining)
            return float(jnp.linalg.norm(problem.observe(end) - observation)) / perturbation_at_end

        band_visibility = [
            visibility(band_filter(error, lower, upper)) for lower, upper in BANDS
        ]
        record = dict(
            truth=truth_index,
            velocity_node_errors=velocity_errors,
            variable_errors_at_first_node=variable_errors,
            velocity_error_band_fraction=(np.array(velocity_error_bands) / sum(velocity_error_bands)).tolist(),
            velocity_perturbation_band_fraction=(
                np.array(velocity_perturbation_bands) / sum(velocity_perturbation_bands)
            ).tolist(),
            velocity_error_to_perturbation_per_band=(
                np.sqrt(np.array(velocity_error_bands) / np.maximum(velocity_perturbation_bands, 1e-30))
            ).tolist(),
            forward_visibility_total=visibility(error),
            forward_visibility_per_band=band_visibility,
            # The same error without its density and pressure parts: is the
            # missing thermodynamic state what the observation cannot see?
            forward_visibility_velocity_only=visibility(error.at[0].set(0.0).at[3].set(0.0)),
            forward_visibility_thermo_only=visibility(error.at[1].set(0.0).at[2].set(0.0)),
        )
        records.append(record)
        print(f"truth {truth_index}: node velocity errors {np.array2string(np.array(velocity_errors), precision=3)}",
              flush=True)
        print(f"  at t_r: variable errors {json.dumps({k: round(v, 3) for k, v in variable_errors.items()})}",
              flush=True)
        print(f"  bands |k| {BANDS}", flush=True)
        print(f"    error fraction        {np.array2string(np.array(record['velocity_error_band_fraction']), precision=3)}",
              flush=True)
        print(f"    perturbation fraction {np.array2string(np.array(record['velocity_perturbation_band_fraction']), precision=3)}",
              flush=True)
        print(f"    error / perturbation  {np.array2string(np.array(record['velocity_error_to_perturbation_per_band']), precision=3)}",
              flush=True)
        print(f"    visible at T          {np.array2string(np.array(band_visibility), precision=4)} "
              f"(all {record['forward_visibility_total']:.4f}, velocity part {record['forward_visibility_velocity_only']:.4f}, "
              f"density+pressure part {record['forward_visibility_thermo_only']:.4f})", flush=True)

    with open(args.output, "w") as handle:
        json.dump(dict(horizon=args.horizon, bands=BANDS, first_node=first_node, records=records), handle, indent=1)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
