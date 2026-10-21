"""
The four density cubes of the field-level-inference panel figure.

Evolves the optimised initial state (turbulent driving off, the same scheme as
the optimisation, from ``_problem.py``) to

    t = 0                 (the optimised initial state)
    t = f_pre  * t_end    (shortly before the loss time)
    t = t_end             (the loss time)
    t = f_post * t_end    (shortly after)

and saves them with the times and the target image in one npz. The loss time
comes from the optimisation's ``run_summary.json`` (``--summary``) or
``--t-end``; without either it is recovered by a correlation scan: the evolution
time whose z-projection correlates best with the logo. Correlation, not MSE,
because the logo's contrast is only ~1 %, so an MSE of the projection is
dominated by the overall amplitude and misses the narrow logo peak.
"""

# general
import argparse
import json

# gpu selection
from autocvd import autocvd

autocvd(num_gpus=1)

# jax
import jax.numpy as jnp  # noqa: E402

# numerics
import numpy as np  # noqa: E402

# field-level inference problem
from _problem import (  # noqa: E402
    finalized,
    logo_target,
    make_config,
    make_params,
    rms_velocity,
)

# astronomix functions
from astronomix import time_integration  # noqa: E402


def parse_arguments():
    """
    Parse the command line.

    Returns:
        The argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--state", type=str, default="best_state.npy")
    parser.add_argument("--summary", type=str, default="", help="run_summary.json of the optimisation")
    parser.add_argument("--out", type=str, default="panel_snaps.npz")
    parser.add_argument("--f-pre", type=float, default=0.995)
    parser.add_argument("--f-post", type=float, default=1.005)
    parser.add_argument("--t-end", type=float, default=0.0, help="loss time; > 0 skips the scan")
    parser.add_argument("--scan-lo", type=float, default=0.4)
    parser.add_argument("--scan-hi", type=float, default=1.8)
    parser.add_argument("--scan-n", type=int, default=15)
    parser.add_argument(
        "--refine-n",
        type=int,
        default=41,
        help="fine-scan samples spanning +/- one coarse step around the peak",
    )
    return parser.parse_args()


def main():
    """Evolve the optimised state to the four panel times and save the cubes."""
    args = parse_arguments()
    t_end = args.t_end
    resolution = args.resolution
    if args.summary:
        with open(args.summary) as summary_file:
            summary = json.load(summary_file)
        t_end = summary["t_end"]
        resolution = summary["resolution"]

    registered_variables, config = finalized(make_config(resolution))
    density_index = registered_variables.density_index
    best_state = jnp.asarray(np.load(args.state))
    target = logo_target(resolution, float(best_state[density_index].sum()))
    target_anomaly = target - jnp.mean(target)

    def evolve(time):
        if time <= 0.0:
            return best_state[density_index]
        final_state = time_integration(best_state, config, make_params(float(time)), registered_variables)
        return final_state[density_index]

    def logo_correlation(density):
        projection = jnp.sum(density, axis=2)
        projection_anomaly = projection - jnp.mean(projection)
        return float(
            jnp.sum(projection_anomaly * target_anomaly)
            / jnp.sqrt(jnp.sum(projection_anomaly**2) * jnp.sum(target_anomaly**2))
        )

    # -------------------------------------------------------------
    # =============== ↓ Loss time (given or scanned) ↓ ============
    # -------------------------------------------------------------
    if t_end <= 0.0:
        # The logo peak is narrow (~1-2 % of t_end), so a coarse scan brackets
        # it and a fine scan of +/- one coarse step around it refines it.
        t_center = 0.5 / max(rms_velocity(best_state, registered_variables), 1e-12)
        coarse_times = np.linspace(args.scan_lo * t_center, args.scan_hi * t_center, args.scan_n)
        coarse_correlations = [logo_correlation(evolve(time)) for time in coarse_times]
        coarse_peak = float(coarse_times[int(np.argmax(coarse_correlations))])
        coarse_step = float(coarse_times[1] - coarse_times[0])
        fine_times = np.linspace(coarse_peak - coarse_step, coarse_peak + coarse_step, args.refine_n)
        fine_correlations = [logo_correlation(evolve(time)) for time in fine_times]
        t_end = float(fine_times[int(np.argmax(fine_correlations))])
        print(f"recovered t_end = {t_end:.4f} (correlation {max(fine_correlations):.4f})", flush=True)
    # -------------------------------------------------------------
    # =============== ↑ Loss time (given or scanned) ↑ ============
    # -------------------------------------------------------------

    times = {
        "init": 0.0,
        "pre": args.f_pre * t_end,
        "target": t_end,
        "post": args.f_post * t_end,
    }
    cubes = {}
    for name, time in times.items():
        density = evolve(time)
        cubes[name] = np.asarray(density, dtype=np.float32)
        mse = float(jnp.mean((jnp.sum(density, axis=2) - target) ** 2))
        print(
            f"{name:6s}: t = {time:.4f}, logo correlation {logo_correlation(density):+.4f}, "
            f"projection MSE {mse:.4e}",
            flush=True,
        )
    np.savez(
        args.out,
        init=cubes["init"],
        pre=cubes["pre"],
        target=cubes["target"],
        post=cubes["post"],
        t_init=0.0,
        t_pre=times["pre"],
        t_target=t_end,
        t_post=times["post"],
        f_pre=args.f_pre,
        f_post=args.f_post,
        target_image=np.asarray(target, dtype=np.float32),
        N=resolution,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
