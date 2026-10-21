"""
Multi-node weak-scaling driver: ideal MHD, Pallas backend, finite difference
(WENO-Z, LSRK4, constrained transport) or finite volume (VL2, HLLD + GLM).

Weak scaling keeps the per-GPU sub-domain fixed while growing the global grid
with the device count, so ideal scaling is a flat wall-clock curve. The state
is built already globally sharded (each process materialises only its local
shard via ``build_cp_alfven_state_sharded``), so the full grid never lives on
one host -- including, for the finite-difference scheme, the staggered
vector-potential construction that keeps the discrete face-centered B
divergence-free.

Launch under Slurm with ``pytests/runners/weak_mhd.sh`` (one process per
node driving all of its GPUs, see ``sharded_layout`` in ``_env.sh``)::

    srun --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=none python examples/scripts/scaling/weak_scaling_mhd.py \
        --scheme fd --bx 128 --by 2048 --bz 1024 --steps 10 --pallas-ct \
        --tag gh200zct

The mesh shards the X axis only: global grid = (bx * G, by, bz). The box is
set numerically equal to the grid so the spacing stays uniform (h = 1) at
every rung; this is a timing/throughput benchmark, so wave periodicity is
irrelevant, and the wave amplitude is shrunk to 1e-6 so the (deliberately)
unresolved wavelength stays a harmless linear perturbation on the uniform
background. The finite-difference MHD state carries 11 fields (VL2: 9) versus
hydro's 5, so the production per-GPU sub-domain is half the hydro one. The
compiled program is run twice on the same allocation (cold and warm, see
``run_weak_scaling_point``); the quoted runtime is the warm run.
"""

# general
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# distributed bootstrap and benchmark configurations (no astronomix import yet)
from _sharded_benchmarks import (  # noqa: E402
    DEFAULT_TIME_STEPS,
    SCHEME_LABELS,
    benchmark_configuration,
    benchmark_settings,
    benchmark_state_builder,
    bootstrap_distributed,
    parse_block_shape,
)

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", choices=["fd", "fv"], default="fd",
                    help="fd: WENO-Z finite difference (LSRK4, CT); fv: VL2 finite volume (HLLD + GLM)")
parser.add_argument("--bx", type=int, default=128, help="per-GPU cells along sharded X")
parser.add_argument("--by", type=int, default=2048, help="cells along Y (global)")
parser.add_argument("--bz", type=int, default=1024, help="cells along Z (global)")
parser.add_argument("--steps", type=int, default=10, help="fixed number of timesteps")
parser.add_argument("--dt", type=float, default=None,
                    help="fixed timestep for h = 1 (default: the scheme's stable step)")
parser.add_argument("--block-shape", type=str, default="",
                    help="Pallas block shape bx,by,bz (default: the scheme's tuned block)")
parser.add_argument("--tag", type=str, default="gh200")
parser.add_argument("--pallas-ct", action="store_true",
                    help="FD only: route the constrained-transport helpers through "
                         "their Pallas kernels (BackendConfig.pallas_ct); under "
                         "sharding this replaces the many per-shift "
                         "collective-permutes of the native roll-based "
                         "stencils with one halo exchange per kernel")
parser.add_argument("--gpus", type=int, default=0,
                    help="SINGLE-process mode: grab this many local GPUs via "
                         "autocvd (no jax.distributed). Ignored under srun "
                         "multi-process (SLURM_NTASKS>1).")
parser.add_argument("--halo-exchange", choices=["ppermute", "pair_all_to_all"], default="pair_all_to_all",
                    help="multi-device halo exchange (BackendConfig.halo_exchange)")
args = parser.parse_args()

# jax.distributed.initialize() must run before astronomix is imported.
bootstrap_distributed(single_process_gpus=args.gpus)

# astronomix functions
from astronomix.parallel.distributed import init_distributed  # noqa: E402

_PYTESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "pytests"
)
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import run_weak_scaling_point  # noqa: E402

info = init_distributed()
num_gpus = info.global_device_count

DATA_DIR = os.path.join(_HERE, "data", "weak_scaling")

block_shape = parse_block_shape(args.block_shape, args.scheme)
time_step = args.dt if args.dt is not None else DEFAULT_TIME_STEPS[args.scheme]

# Global grid: shard X across all devices, keep Y/Z fixed per process.
global_cells = (args.bx * num_gpus, args.by, args.bz)
mesh_shape = (1, num_gpus, 1, 1)
# Box numerically equal to the grid -> uniform spacing h = 1 at every rung.
box_size = tuple(float(extent) for extent in global_cells)

base_config = benchmark_configuration(
    args.scheme,
    "mhd",
    block_shape,
    pallas_ct=args.pallas_ct,
    halo_exchange=args.halo_exchange,
)
settings = benchmark_settings("mhd", box_size)

if info.process_index == 0:
    total_cells = global_cells[0] * global_cells[1] * global_cells[2]
    print(
        f"[weak-mhd] {SCHEME_LABELS[args.scheme]} G={num_gpus} mesh={mesh_shape} "
        f"global_grid={global_cells} ({total_cells:.3e} cells, "
        f"~{total_cells / num_gpus:.3e} cells/GPU) block={block_shape} "
        f"pallas_ct={base_config.backend_config.pallas_ct} "
        f"steps={args.steps} dt={time_step}",
        flush=True,
    )

run_weak_scaling_point(
    benchmark_state_builder("mhd"),
    base_config,
    settings,
    mesh_shape=mesh_shape,
    global_cells=global_cells,
    box_size=box_size,
    cfl=1.5,
    dt=time_step,
    num_timesteps=args.steps,
    name=f"weak_mhd_pallas_{args.tag}",
    data_dir=DATA_DIR,
    setup_key="mhd_weak",
    solver_label=SCHEME_LABELS[args.scheme],
)
