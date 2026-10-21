"""
Check the domain-decomposed finite-difference MHD jet against a single device.

Runs the jet of ``examples/scripts/forward/mhd/mhd_jet.py`` (WENO-Z,
positivity-preserving WENO, SSPRK4) for a short time on ``--gpus`` devices of
one process (or, under ``srun`` with several tasks, on one GPU per task) and
stores the final primitive state; ``--compare`` then reports
the largest difference between two stored states, overall and per variable,
and the x positions where it occurs (a faulty shard halo shows up at the shard
boundaries). ``--boundary open`` runs the jet with open instead of periodic
boundaries, ``--native`` with the native-JAX instead of the Pallas backend.
Without ``--out`` a run only reports its step count, runtime and compiled
memory, which probes whether a production size fits (``--block``, ``--lsrk4``
as in ``mhd_jet.py``)::

    python pytests/runners/mhd_jet_decomposition_check.py --res 128 --gpus 4 --t-end 0.5 --out four.npz
    python pytests/runners/mhd_jet_decomposition_check.py --compare one.npz four.npz
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np


def compare_states(reference_path, other_path):
    """
    Print the largest differences between two stored jet states.

    Args:
        reference_path: The ``.npz`` file of the reference run.
        other_path: The ``.npz`` file of the run compared with it.
    """
    reference = np.load(reference_path)
    other = np.load(other_path)
    reference_state = reference["state"].astype(np.float64)
    other_state = other["state"].astype(np.float64)
    print(f"steps: {int(reference['num_iterations'])} vs {int(other['num_iterations'])}")
    difference = np.abs(other_state - reference_state)
    for variable_index in range(reference_state.shape[0]):
        variable_scale = np.max(np.abs(reference_state[variable_index]))
        largest = difference[variable_index].max()
        x_of_largest = np.unravel_index(
            np.argmax(difference[variable_index]),
            difference[variable_index].shape,
        )[0]
        print(
            f"variable {variable_index}: max |difference| {largest:.3e} "
            f"(relative to max |value| {largest / variable_scale:.3e}) at x index {x_of_largest}"
        )
    # The largest difference per x plane shows whether the error sits at the
    # shard boundaries (a halo fault) or is spread like round-off.
    difference_per_plane = difference.max(axis=(0, 2, 3))
    worst_planes = np.argsort(difference_per_plane)[::-1][:8]
    print(
        "x planes with the largest difference: "
        + ", ".join(f"{plane} ({difference_per_plane[plane]:.2e})" for plane in worst_planes)
    )


def run_and_store(arguments):
    """
    Run the jet on ``arguments.gpus`` devices and store its final state.

    Args:
        arguments: The parsed command-line arguments.
    """
    # mhd_jet reads --gpus, --block and --lsrk4 from sys.argv and selects the
    # GPUs when it is imported, so it (and with it JAX) is imported only here,
    # after its directory is on the path; the comparison needs neither.
    jet_directory = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "forward" / "mhd"
    sys.path.insert(0, str(jet_directory))
    import jax
    import jax.numpy as jnp
    import mhd_jet
    from astronomix import (
        NATIVE_JAX,
        OPEN_BOUNDARY,
        PERIODIC_BOUNDARY,
        BackendConfig,
        SimulationParams,
        finalize_config,
        get_registered_variables,
        time_integration,
    )

    boundary_type = OPEN_BOUNDARY if arguments.boundary == "open" else PERIODIC_BOUNDARY
    config = mhd_jet.make_config(arguments.res, boundary_type=boundary_type)
    if arguments.native:
        config = config._replace(backend_config=BackendConfig(backend=NATIVE_JAX))
    registered_variables = get_registered_variables(config)
    sharding = mhd_jet.make_sharding(mhd_jet.NUM_GPUS)
    initial_state = mhd_jet.build_initial_state(
        config,
        registered_variables,
        sharding,
        arguments.res,
    )
    params = SimulationParams(
        C_cfl=mhd_jet.C_CFL,
        dt_max=0.1,
        t_end=arguments.t_end,
        gamma=mhd_jet.GAMMA,
    )
    config = finalize_config(config, initial_state.shape)
    result = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
        sharding=sharding,
    )
    final_state = result.final_state
    density = final_state[registered_variables.density_index]
    pressure = final_state[registered_variables.pressure_index]
    print(
        f"{mhd_jet.NUM_GPUS} GPU(s), {arguments.boundary} boundaries, "
        f"{'native' if arguments.native else 'Pallas'}, {arguments.res}^3: "
        f"{int(result.num_iterations)} steps in {float(result.runtime):.1f} s, "
        f"temporary memory {int(result.temporary_memory_bytes) / 1024**3:.1f} GiB per device, "
        f"min density {float(density.min()):.3e}, min pressure {float(pressure.min()):.3e}, "
        f"finite {bool(jnp.all(jnp.isfinite(final_state)))}"
    )
    # Without --out the run only measures (steps, runtime, memory), which is
    # how the size of the production run is probed.
    if arguments.out:
        state = mhd_jet.gather_to_host(final_state)
        if jax.process_index() == 0:
            np.savez(
                arguments.out,
                state=state,
                num_iterations=int(result.num_iterations),
            )
            print(f"stored -> {arguments.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--boundary", choices=("periodic", "open"), default="periodic")
    parser.add_argument("--native", action="store_true", help="native-JAX instead of Pallas")
    parser.add_argument("--out", type=str)
    # Read by mhd_jet itself (see run_and_store).
    parser.add_argument("--block", type=str)
    parser.add_argument("--lsrk4", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "OTHER"))
    arguments = parser.parse_args()
    if arguments.compare:
        compare_states(*arguments.compare)
    else:
        run_and_store(arguments)
