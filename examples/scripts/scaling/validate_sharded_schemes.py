"""
Correctness check of the sharded scaling configurations: the same global grid
integrated on one GPU and decomposed over all GPUs of the job must agree.

For every scaling configuration (FD WENO-Z and FV VL2, hydro and MHD, Pallas,
fp32) the script integrates the resolved sound wave / CP Alfvén wave on a
small global grid (2N, N, N) twice: once unsharded on each process's own GPU
(the reference) and once decomposed along x over all G devices of the job
(the production path: sharded initial condition, ``_pallas_call_sharded``
halo exchanges). The sharded final state is gathered on every process and
compared with the reference; process 0 prints the maximum absolute difference
per configuration and writes it to
``data/weak_scaling/sharded_validation_G{G:03d}.json``.

Launch under Slurm (one process per node by default, see ``sharded_layout``
in ``pytests/runners/_env.sh``), e.g. 4 GPUs on one node and 8 on two nodes::

    sbatch --nodes=1 --ntasks-per-node=4 pytests/runners/validate_sharded.sh
    sbatch --nodes=2 --ntasks-per-node=4 pytests/runners/validate_sharded.sh
"""

# general
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# distributed bootstrap and benchmark configurations (no astronomix import yet)
from _sharded_benchmarks import (  # noqa: E402
    DEFAULT_BLOCK_SHAPES,
    SCHEME_LABELS,
    benchmark_configuration,
    benchmark_settings,
    benchmark_state_builder,
    bootstrap_distributed,
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=32,
                    help="N of the global grid (2N, N, N); 2N must split over the GPUs")
parser.add_argument("--steps", type=int, default=20, help="number of time steps")
parser.add_argument("--courant", type=float, default=0.25,
                    help="fixed time step in units of the cell size (unit signal speed)")
parser.add_argument("--gpus", type=int, default=0,
                    help="SINGLE-process mode: grab this many local GPUs via autocvd")
parser.add_argument("--halo-exchange", choices=["ppermute", "pair_all_to_all"], default="pair_all_to_all",
                    help="multi-device halo exchange (BackendConfig.halo_exchange)")
args = parser.parse_args()

# jax.distributed.initialize() must run before astronomix is imported.
bootstrap_distributed(single_process_gpus=args.gpus)

# jax
import jax  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402

# numerics
import numpy as np  # noqa: E402

# astronomix containers
from astronomix.option_classes.simulation_config import (  # noqa: E402
    SnapshotSettings,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams  # noqa: E402

# astronomix functions
from astronomix.parallel.distributed import init_distributed  # noqa: E402
from astronomix.time_stepping.time_integration import time_integration  # noqa: E402
from astronomix.variable_registry.registered_variables import (  # noqa: E402
    get_registered_variables,
)

_PYTESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "pytests"
)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import _build_global_sharding  # noqa: E402

#: The standard periodic box of both waves on a (2N, N, N) grid: one
#: wavelength along every axis and a uniform cell size.
BOX_SIZE = (3.0, 1.5, 1.5)

#: Resolved, finite wave amplitudes so that the comparison sees real dynamics
#: (the MHD value is the standard nonlinear CP Alfvén wave).
AMPLITUDES = {"hydro": 1.0e-2, "mhd": 0.1}

DATA_DIR = os.path.join(_HERE, "data", "weak_scaling")


def integrate(scheme: str, physics: str, sharding) -> np.ndarray:
    """
    Integrate one configuration on the validation grid.

    Args:
        scheme: ``"fd"`` or ``"fv"``.
        physics: ``"hydro"`` or ``"mhd"``.
        sharding: The global sharding over all devices, or None for an
            unsharded run on this process's own GPU.

    Returns:
        The final primitive state, gathered to a host array.
    """
    jax.clear_caches()
    resolution = args.resolution
    global_cells = (2 * resolution, resolution, resolution)
    cell_size = BOX_SIZE[0] / global_cells[0]
    time_step = args.courant * cell_size

    config = benchmark_configuration(
        scheme,
        physics,
        DEFAULT_BLOCK_SHAPES[scheme],
        halo_exchange=args.halo_exchange,
    )
    config = config._replace(
        num_cells=StaticIntVector(*global_cells),
        fixed_timestep=True,
        num_timesteps=args.steps,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        print_elapsed_time=False,
    )
    settings = benchmark_settings(physics, BOX_SIZE, amplitude=AMPLITUDES[physics])
    state, config, params = benchmark_state_builder(physics)(
        config,
        SimulationParams(),
        sharding,
        settings,
    )
    params = params._replace(t_end=time_step * args.steps)
    registered_variables = get_registered_variables(config)

    result = time_integration(
        state,
        config,
        params,
        registered_variables,
        sharding=sharding,
    )
    final_state = result.final_state
    if sharding is None:
        return np.asarray(final_state)
    # Every process receives the whole (small) global array.
    return np.asarray(multihost_utils.process_allgather(final_state, tiled=True))


def main():
    """Compare sharded and unsharded runs of all four configurations."""
    info = init_distributed()
    num_gpus = info.global_device_count
    sharding = _build_global_sharding((1, num_gpus, 1, 1))

    records = []
    for physics in ("hydro", "mhd"):
        for scheme in ("fd", "fv"):
            # The unsharded reference runs first, before any mesh context.
            reference_state = integrate(scheme, physics, None)
            sharded_state = integrate(scheme, physics, sharding)
            difference = np.abs(sharded_state - reference_state)
            record = dict(
                scheme=scheme,
                label=SCHEME_LABELS[scheme],
                physics=physics,
                num_gpus=num_gpus,
                global_cells=[2 * args.resolution, args.resolution, args.resolution],
                steps=args.steps,
                all_finite=bool(
                    np.isfinite(reference_state).all() and np.isfinite(sharded_state).all()
                ),
                max_abs_difference=float(difference.max()),
                max_abs_difference_per_variable=[
                    float(value) for value in difference.reshape(difference.shape[0], -1).max(axis=1)
                ],
                max_abs_value=float(np.abs(reference_state).max()),
                bitwise_identical=bool(np.array_equal(sharded_state, reference_state)),
            )
            records.append(record)
            if info.process_index == 0:
                print(
                    f"[validate] {physics:5s} {SCHEME_LABELS[scheme]:20s} G={num_gpus}: "
                    f"max |sharded - single| = {record['max_abs_difference']:.3e} "
                    f"(max |state| = {record['max_abs_value']:.3e}, "
                    f"bitwise identical: {record['bitwise_identical']}, "
                    f"finite: {record['all_finite']})",
                    flush=True,
                )

    if info.process_index == 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        output_path = os.path.join(DATA_DIR, f"sharded_validation_G{num_gpus:03d}.json")
        with open(output_path, "w") as output_file:
            json.dump(records, output_file, indent=2)
        print(f"[validate] wrote {output_path}", flush=True)
    multihost_utils.sync_global_devices("sharded_validation_done")


if __name__ == "__main__":
    main()
