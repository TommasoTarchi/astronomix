"""
AthenaPK reference data of the circularly polarized Alfvén wave, for the paper's
convergence and single-GPU runtime figures.

For every resolution N (grid 2N x N x N, box 3 x 1.5 x 1.5, five wave periods
to t = 5) AthenaPK's GPU build runs the wave twice:

1. a timing run with P. Grete's tuned settings (single meshblock,
   ``minimum_number_of_teams_for_boundary_kernel = 256``, ``scratch_level = 1``)
   and nothing else: no error evaluation and no output, because AthenaPK's timer
   also runs over the end-of-run work. The runtime is
   ``cycles * cells / (zone-cycles/wallsecond)``, AthenaPK's own throughput
   measured after the first two cycles, i.e. without the initialisation, as in
   the A100 and H100 references;
2. a state run, identical but with ``compute_error = true`` (AthenaPK's L1
   error in ``cpaw-errors.dat``: the conserved variables against the point values
   of the analytic wave) and the primitive state written at t = 0 and t = 5,
   from which the L1 error is also evaluated in astronomix's definition: the mean
   absolute difference over rho, v, p and B between the final state and the
   initial discrete state (the analytic solution at t = 5; astronomix compares
   against the discrete initial condition, not the point values).

Setups: ``vl2_plm`` (VL2 + PLM + HLLD, cfl 0.3) and ``rk3_ppm`` (RK3 + PPM +
HLLD, cfl 0.4), each in double or single precision (``PARTHENON_SINGLE_PRECISION``
build). Writes ``pytests/mhd/data/athenapk/<node>/<setup>/athenapk_alfven_convergence.npz``
(``N_values``, ``l1_errors``, ``runtimes``, ``iterations`` as the earlier
references, plus ``l1_errors_primitive``, ``zone_cycles_per_second``,
``milliseconds_per_cycle``, ``process_seconds`` and the per-variable errors), the
input file of the largest run and a log.

Usage (on JUPITER inside a GPU job, see jupiter_athenapk.sh):
    python athenapk_alfven_reference.py --setup vl2_plm --precision double
        [--resolutions 8 16 32 64 128] [--node gh200]
"""

# general
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# numerics
import numpy as np

# validation helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from athenapk_runner import (  # noqa: E402
    athenapk_binary,
    athenapk_environment,
    cp_alfven_input,
    performance_summary,
    read_output,
)

REPOSITORY = Path(__file__).resolve().parents[4]
REFERENCE_ROOT = REPOSITORY / "pytests" / "mhd" / "data" / "athenapk"

#: The AthenaPK configurations: integrator, reconstruction, CFL number and the
#: directory name of the double-precision reference (the names of the A100 and
#: H100 references, so the figure scripts find every node the same way).
SETUPS = {
    "vl2_plm": dict(
        integrator="vl2",
        reconstruction="plm",
        cfl=0.3,
        directory="cfl03_teams256_singleblock",
    ),
    "rk3_ppm": dict(
        integrator="rk3",
        reconstruction="ppm",
        cfl=0.4,
        directory="cfl04_teams256_singleblock_rk3_ppm",
    ),
}

#: Columns of ``cpaw-errors.dat`` after (Nx1, Nx2, Nx3, Ncycle, RMS).
ERROR_COLUMNS = ("d", "M1", "M2", "M3", "E", "B1c", "B2c", "B3c")


def gpu_node_tag() -> str:
    """A directory tag for the visible GPU (``gh200``, ``h100``, ...) from nvidia-smi."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
        capture_output=True,
        text=True,
    )
    name = query.stdout.strip().lower().replace(" ", "").replace("-", "")
    for known in ("gh200", "gb200", "b200", "h200", "h100", "a100", "v100"):
        if known in name:
            return known
    raise RuntimeError(f"Unknown GPU {query.stdout!r}; pass --node.")


def setup_directory(setup: str, precision: str) -> str:
    """The reference directory name of a setup and precision."""
    directory = SETUPS[setup]["directory"]
    return directory if precision == "double" else f"{directory}_single"


def read_cpaw_errors(run_directory: str):
    """
    AthenaPK's error file of the CP Alfvén wave.

    Args:
        run_directory: The run directory holding ``cpaw-errors.dat``.

    Returns:
        ``(per_variable_errors, cycles)``: the mean absolute errors of the eight
        conserved variables (density, momenta, energy, field) and the cycle count.
    """
    with open(os.path.join(run_directory, "cpaw-errors.dat")) as file:
        columns = [line.split() for line in file if not line.startswith("#")][-1]
    return np.array([float(value) for value in columns[5:13]]), int(columns[3])


def run_once(arguments, run_directory: str, single_precision: bool):
    """
    Run AthenaPK and time the whole process.

    Args:
        arguments: Command-line arguments after the executable.
        run_directory: The working directory.
        single_precision: Use the single-precision build.

    Returns:
        ``(standard_output, process_seconds)``; the process time includes the
        initialisation and the CUDA context creation.
    """
    start = time.perf_counter()
    result = subprocess.run(
        [athenapk_binary(gpu=True, single_precision=single_precision), *arguments],
        cwd=run_directory,
        capture_output=True,
        text=True,
        env=athenapk_environment(),
    )
    process_seconds = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"athenaPK failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.stdout, process_seconds


def primitive_l1_error(run_directory: str) -> float:
    """
    The L1 error in astronomix's definition: mean over rho, v, p, B of the mean
    absolute difference between the final and the initial primitive state.

    Args:
        run_directory: A run directory with the t = 0 and final HDF5 outputs.

    Returns:
        The L1 error.
    """
    initial_state, _, _ = read_output(os.path.join(run_directory, "parthenon.out0.00000.phdf"), "prim")
    final_state, final_time, _ = read_output(os.path.join(run_directory, "parthenon.out0.final.phdf"), "prim")
    if abs(final_time - 5.0) > 1e-6:
        raise RuntimeError(f"The final output is at t = {final_time}, not 5.")
    # AthenaPK's primitive order is (rho, v1, v2, v3, p, B1, B2, B3, psi);
    # the cleaning scalar psi is not part of the error.
    difference = np.abs(final_state[:8].astype(np.float64) - initial_state[:8].astype(np.float64))
    return float(np.mean(difference.reshape(8, -1).mean(axis=1)))


def measure_resolution(resolution: int, setup: dict, single_precision: bool) -> dict:
    """
    The timing run and the state run of one resolution.

    Args:
        resolution: N (grid 2N x N x N).
        setup: The AthenaPK configuration (an entry of ``SETUPS``).
        single_precision: Use the single-precision build.

    Returns:
        A dict with AthenaPK's L1 error and the one in astronomix's definition,
        the per-variable errors, the cycles, the runtime from the throughput,
        the throughput, the time per cycle, AthenaPK's wall time and the process
        time, and the input file of the timing run.
    """
    cells = 2 * resolution**3
    scheme = dict(
        integrator=setup["integrator"],
        reconstruction=setup["reconstruction"],
        cfl=setup["cfl"],
    )

    # --------------- ↓ Timing run ↓ ----------------
    input_text = cp_alfven_input(resolution, **scheme)
    with tempfile.TemporaryDirectory(prefix=f"athenapk_cpaw_N{resolution}_") as run_directory:
        input_file = os.path.join(run_directory, "cpaw.in")
        with open(input_file, "w") as file:
            file.write(input_text)
        standard_output, process_seconds = run_once(["-i", input_file], run_directory, single_precision)
        zone_cycles_per_second, walltime_seconds, cycles = performance_summary(standard_output)
    # --------------- ↑ Timing run ↑ ----------------

    # --------------- ↓ State run ↓ ----------------
    with tempfile.TemporaryDirectory(prefix=f"athenapk_cpaw_state_N{resolution}_") as run_directory:
        input_file = os.path.join(run_directory, "cpaw.in")
        with open(input_file, "w") as file:
            file.write(cp_alfven_input(resolution, compute_error=True, output_variables="prim", **scheme))
        run_once(["-i", input_file], run_directory, single_precision)
        per_variable_errors, state_run_cycles = read_cpaw_errors(run_directory)
        l1_error_primitive = primitive_l1_error(run_directory)
    if state_run_cycles != cycles:
        raise RuntimeError(f"The timing run took {cycles} cycles, the state run {state_run_cycles}.")
    # --------------- ↑ State run ↑ ----------------

    # As in the earlier references: the run time follows from AthenaPK's
    # throughput, which leaves out the initialisation and the first cycles.
    return dict(
        l1_errors=float(np.mean(per_variable_errors)),
        l1_errors_primitive=l1_error_primitive,
        runtimes=cells * cycles / zone_cycles_per_second,
        iterations=cycles,
        zone_cycles_per_second=zone_cycles_per_second,
        milliseconds_per_cycle=1e3 * cells / zone_cycles_per_second,
        walltime_seconds=walltime_seconds,
        process_seconds=process_seconds,
        per_variable_errors=per_variable_errors,
        input_text=input_text,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", choices=sorted(SETUPS), default="vl2_plm")
    parser.add_argument("--precision", choices=["double", "single"], default="double")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--node", default=None, help="storage tag (default: from nvidia-smi)")
    arguments = parser.parse_args()

    setup = SETUPS[arguments.setup]
    single_precision = arguments.precision == "single"
    binary = athenapk_binary(gpu=True, single_precision=single_precision)
    node = arguments.node or gpu_node_tag()
    output_directory = REFERENCE_ROOT / node / setup_directory(arguments.setup, arguments.precision)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Everything printed is also kept as the log of the reference data.
    log_lines = []

    def log(message: str):
        print(message, flush=True)
        log_lines.append(message)

    log(f"AthenaPK CP Alfvén wave reference: {arguments.setup} ({setup['integrator']} + "
        f"{setup['reconstruction']} + hlld, cfl {setup['cfl']}), {arguments.precision} precision, node {node}")
    log(f"  binary: {binary}")

    records = []
    for resolution in arguments.resolutions:
        record = measure_resolution(resolution, setup, single_precision)
        records.append(record)
        log(
            f"  N={resolution:4d}  grid={2 * resolution}x{resolution}x{resolution}  cycles={record['iterations']:5d}  "
            f"L1(AthenaPK)={record['l1_errors']:.6e}  L1(primitive)={record['l1_errors_primitive']:.6e}  "
            f"runtime={record['runtimes']:8.3f} s  ({record['milliseconds_per_cycle']:8.3f} ms/cycle, "
            f"{record['zone_cycles_per_second'] / 1e6:8.1f} Mzone-cycles/s, process {record['process_seconds']:7.2f} s)"
        )

    resolutions = np.array(arguments.resolutions)
    errors = np.array([record["l1_errors"] for record in records])
    for index in range(1, len(resolutions)):
        order = -np.log(errors[index] / errors[index - 1]) / np.log(resolutions[index] / resolutions[index - 1])
        log(f"  order N={resolutions[index - 1]} -> {resolutions[index]}: {order:.2f}")

    columns = [name for name in records[0] if name != "input_text"]
    np.savez(
        output_directory / "athenapk_alfven_convergence.npz",
        N_values=resolutions,
        **{name: np.array([record[name] for record in records]) for name in columns},
        error_columns=np.array(ERROR_COLUMNS),
        precision=arguments.precision,
        setup=arguments.setup,
        binary=binary,
    )
    with open(output_directory / "cpaw_convergence.in", "w") as file:
        file.write(records[-1]["input_text"])
    with open(output_directory / "athenapk_alfven_reference.log", "w") as file:
        file.write("\n".join(log_lines) + "\n")
    log(f"  wrote {output_directory}")


if __name__ == "__main__":
    main()
