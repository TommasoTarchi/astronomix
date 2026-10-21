"""
Generate all CP Alfvén-wave benchmark data of astronomix and AthenaPK on this
GPU (paper convergence figure, single-GPU runtime figure and the finite-volume
single-precision comparison) with one command:

    PYTHONPATH=<repo> python examples/scripts/forward/mhd/generate_alfven_data.py [--steps ...]

Steps (all by default, in this order; ``--steps`` selects a subset by name):

* ``athenapk_double``: AthenaPK references VL2+PLM (cfl 0.3) and RK3+PPM
  (cfl 0.4), double precision, N = 8 ... 128
  (``validation/athenapk_vl2/athenapk_alfven_reference.py``),
* ``astronomix_double``: the double-precision convergence and runtime sweep
  (``alfven_convergence.py``: FV VL2, FD WENO-Z, FD WENO-Z + PP and two
  diagnostics), which also draws the two figures,
* ``athenapk_single`` and ``astronomix_single``: the same in single precision
  (AthenaPK's ``PARTHENON_SINGLE_PRECISION`` build; astronomix FV and FD only),
* ``benchmark_double`` and ``benchmark_single``: time per step of the FV scheme
  against AthenaPK VL2 for N = 16 ... 512 (``validation/athenapk_vl2/benchmark.py``),
* ``figure_single_precision``: the FV-against-AthenaPK figure from the two
  benchmark files.

The precisions run in separate processes because ``jax_enable_x64`` is global.
The GPU is picked once (autocvd, unless a queue already set
``CUDA_VISIBLE_DEVICES``) and inherited by every child, so all series are
measured on the same device. The AthenaPK steps need ``ATHENAPK_GPU_BIN`` /
``ATHENAPK_GPU_SINGLE_BIN`` (on JUPITER run this script through
``validation/athenapk_vl2/jupiter_athenapk.sh``). Data and figures are filed
under the node tag of the live device (``pytests/mhd/{data,figures}/.../<node>/``).

Options:
    --steps NAME ...  run only these steps
    --node TAG        node tag of the benchmark files (default: gh200)
    --dry-run         print the plan without running anything
"""

# general
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
VALIDATION = REPOSITORY / "examples" / "scripts" / "validation" / "athenapk_vl2"

REFERENCE_RESOLUTIONS = ["8", "16", "32", "64", "128"]
BENCHMARK_RESOLUTIONS = ["16", "32", "64", "128", "256", "384", "512"]


def plan(node: str) -> dict:
    """The steps by name: a list of ``(title, command)`` each."""
    reference = str(VALIDATION / "athenapk_alfven_reference.py")
    convergence = str(HERE / "alfven_convergence.py")
    benchmark = str(VALIDATION / "benchmark.py")
    results = VALIDATION / "results"
    return dict(
        athenapk_double=[
            (f"AthenaPK {setup}, double", [reference, "--setup", setup, "--precision", "double",
                                           "--resolutions", *REFERENCE_RESOLUTIONS])
            for setup in ("vl2_plm", "rk3_ppm")
        ],
        astronomix_double=[("astronomix sweep, double", [convergence])],
        athenapk_single=[
            (f"AthenaPK {setup}, single", [reference, "--setup", setup, "--precision", "single",
                                           "--resolutions", *REFERENCE_RESOLUTIONS])
            for setup in ("vl2_plm", "rk3_ppm")
        ],
        astronomix_single=[("astronomix sweep, single", [convergence, "--single", "--benchmarks", "fv_vl2", "fd_wenoz"])],
        benchmark_double=[
            ("FV against AthenaPK, double", [benchmark, "--precision", "double", "--resolutions",
                                             *BENCHMARK_RESOLUTIONS, "--output",
                                             str(results / f"benchmark_{node}_double.json")])
        ],
        benchmark_single=[
            ("FV against AthenaPK, single", [benchmark, "--precision", "single", "--resolutions",
                                             *BENCHMARK_RESOLUTIONS, "--output",
                                             str(results / f"benchmark_{node}_single.json")])
        ],
        figure_single_precision=[
            ("FV against AthenaPK figure", [str(VALIDATION / "fv_athenapk_single_precision.py"), "--node", node])
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=None)
    parser.add_argument("--node", default="gh200")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    steps = plan(arguments.node)
    selected = list(steps) if arguments.steps is None else arguments.steps
    unknown = set(selected) - set(steps)
    if unknown:
        raise SystemExit(f"unknown steps {sorted(unknown)}; known: {list(steps)}")

    # One GPU selection for the whole run; the children inherit it through the
    # environment instead of each grabbing (possibly different) GPUs.
    if not arguments.dry_run and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        from autocvd import autocvd

        autocvd(num_gpus=1)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY), environment.get("PYTHONPATH", "")]))
    print(f"[generate] CUDA_VISIBLE_DEVICES={environment.get('CUDA_VISIBLE_DEVICES', '<none>')}, steps {selected}")

    start_all = time.perf_counter()
    for name in selected:
        for title, command in steps[name]:
            print(f"\n{'=' * 70}\n[generate] {name}: {title}\n   python -u {' '.join(command)}\n{'=' * 70}", flush=True)
            if arguments.dry_run:
                continue
            start = time.perf_counter()
            # check=True: a failed step must not be papered over by later steps
            # appearing to succeed; the data would silently be incomplete.
            subprocess.run([sys.executable, "-u", *command], check=True, env=environment, cwd=str(REPOSITORY))
            print(f"[generate] {title} done in {time.perf_counter() - start:.1f} s", flush=True)
    print(f"\n[generate] all done in {(time.perf_counter() - start_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
