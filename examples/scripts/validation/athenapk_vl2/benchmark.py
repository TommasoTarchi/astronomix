"""
Performance of the VL2 scheme: astronomix (native JAX and Pallas) against AthenaPK.

Both codes run the 3D circularly polarized Alfvén wave (grid 2N x N x N,
VL2 + PLM + HLLD + GLM, CFL 0.3) on the same GPU. For astronomix the time per
step is measured over a fixed number of complete steps (time-step reduction plus
both stages) inside one compiled loop; every configuration is timed several
times, interleaved with the others, and the fastest repetition is reported (a
sporadic slowdown on a shared node only ever adds time). AthenaPK's figure is its
own ``zone-cycles/wallsecond`` (single meshblock, AthenaPK's tuned GPU
settings), which also excludes initialization. The GPU clock, power draw and
throttle reasons are logged with every repetition, so that a throttled
measurement can be recognised.

Usage:
    [ATHENAPK_GPU_BIN=...] python benchmark.py --resolutions 32 64 128
        [--precisions double single] [--backends pallas native] [--output results.json]
"""

# general
import argparse
import json
import os
import re
import subprocess
import tempfile
import time

# ==== GPU selection (skipped when a queue has already assigned one) ====
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =====================================================================


# jax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    HLLD,
    NATIVE_JAX,
    PALLAS,
    VAN_LEER,
    VL2,
    StaticIntVector,
)

# astronomix containers
from astronomix import BackendConfig, SimulationConfig, SimulationParams

# astronomix functions
from astronomix import get_registered_variables
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave
from astronomix._finite_volume._timestep_estimation._timestep_estimator import _cfl_time_step
from astronomix._finite_volume._state_evolution.evolve_state import _evolve_state_fv

ATHENAPK_INPUT = """<job>
problem_id = cpaw
<problem/cpaw>
compute_error = false
b_par = 1.0
b_perp = 0.1
pres = 0.1
v_par = 0.0
dir = 1
<parthenon/mesh>
refinement = none
nghost = 2
nx1 = {nx1}
x1min = 0.0
x1max = 3.0
ix1_bc = periodic
ox1_bc = periodic
nx2 = {nx2}
x2min = 0.0
x2max = 1.5
ix2_bc = periodic
ox2_bc = periodic
nx3 = {nx3}
x3min = 0.0
x3max = 1.5
ix3_bc = periodic
ox3_bc = periodic
minimum_number_of_teams_for_boundary_kernel = 256
<parthenon/meshblock>
nx1 = {nx1}
nx2 = {nx2}
nx3 = {nx3}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 5.0
nlim = {cycles}
perf_cycle_offset = 2
<hydro>
fluid = glmmhd
eos = adiabatic
riemann = hlld
reconstruction = plm
gamma = 1.666666666666667
scratch_level = 1
"""


def gpu_status() -> str:
    """The current SM clock, power draw and throttle reasons of the visible GPU."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active",
            "--format=csv,noheader",
            "-i",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        ],
        capture_output=True,
        text=True,
    )
    return query.stdout.strip()


def astronomix_step_function(resolution: int, backend: int, dtype, steps: int):
    """A compiled function running ``steps`` complete VL2 steps, and its initial state."""
    base_config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        time_integrator=VL2,
        riemann_solver=HLLD,
        limiter=VAN_LEER,
        mhd=True,
        dimensionality=3,
        num_cells=StaticIntVector(2 * resolution, resolution, resolution),
        backend_config=BackendConfig(backend=backend),
    )
    state, config, params = setup_cp_alfven_wave(base_config, SimulationParams(C_cfl=0.3, gamma=5.0 / 3.0))
    state = state.astype(dtype)
    registered_variables = get_registered_variables(config)

    def step(_, primitive_state):
        dt = _cfl_time_step(primitive_state, config, params, registered_variables)
        return _evolve_state_fv(primitive_state, dt, params.gamma, config, params, None, registered_variables)

    run = jax.jit(lambda primitive_state: jax.lax.fori_loop(0, steps, step, primitive_state))
    memory = run.lower(state).compile().memory_analysis()
    return run, state, getattr(memory, "temp_size_in_bytes", -1)


def athenapk_milliseconds_per_cycle(resolution: int, cycles: int = 102) -> float:
    """AthenaPK's time per cycle from its zone-cycles/wallsecond (``ATHENAPK_GPU_BIN``)."""
    binary = os.environ["ATHENAPK_GPU_BIN"]
    with tempfile.TemporaryDirectory() as run_directory:
        input_file = os.path.join(run_directory, "cpaw.in")
        with open(input_file, "w") as file:
            file.write(ATHENAPK_INPUT.format(nx1=2 * resolution, nx2=resolution, nx3=resolution, cycles=cycles))
        result = subprocess.run([binary, "-i", input_file], cwd=run_directory, capture_output=True, text=True)
    match = re.search(r"zone-cycles/wallsecond\s*=\s*([\d.eE+\-]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"AthenaPK failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return 2 * resolution**3 / float(match.group(1)) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--precisions", nargs="+", default=["double", "single"])
    parser.add_argument("--backends", nargs="+", default=["pallas", "native"])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    device = jax.devices()[0].device_kind
    print(f"device: {device}", flush=True)
    results = []
    for resolution in arguments.resolutions:
        cells = 2 * resolution**3
        # enough steps for ~1 s of work per repetition, at least 10
        steps = int(max(10, min(500, 2e9 / cells / 20)))
        variants = {}
        for precision in arguments.precisions:
            dtype = jnp.float64 if precision == "double" else jnp.float32
            for backend_name in arguments.backends:
                backend = PALLAS if backend_name == "pallas" else NATIVE_JAX
                start = time.perf_counter()
                run, state, temporary_bytes = astronomix_step_function(resolution, backend, dtype, steps)
                jax.block_until_ready(run(state))
                compile_seconds = time.perf_counter() - start
                variants[(precision, backend_name)] = (run, state, temporary_bytes, compile_seconds, [])

        statuses = []
        for _ in range(arguments.repetitions):
            for key, (run, state, _, _, times) in variants.items():
                start = time.perf_counter()
                jax.block_until_ready(run(state))
                times.append(1e3 * (time.perf_counter() - start) / steps)
            statuses.append(gpu_status())

        for (precision, backend_name), (_, _, temporary_bytes, compile_seconds, times) in variants.items():
            entry = dict(
                code=f"astronomix {backend_name}",
                precision=precision,
                resolution=resolution,
                cells=cells,
                milliseconds_per_step=min(times),
                all_milliseconds_per_step=times,
                zone_cycles_per_second=cells / (min(times) * 1e-3),
                temporary_memory_mib=temporary_bytes / 2**20,
                compile_and_first_run_seconds=compile_seconds,
                gpu_status=statuses,
                device=device,
            )
            results.append(entry)
            print(
                f"N={resolution:4d} {entry['code']:18s} {precision:6s}: {entry['milliseconds_per_step']:8.2f} ms/step "
                f"({entry['zone_cycles_per_second'] / 1e6:7.1f} Mzone-cycles/s, temp {entry['temporary_memory_mib']:7.0f} MiB)",
                flush=True,
            )

        if os.environ.get("ATHENAPK_GPU_BIN"):
            repetitions = [athenapk_milliseconds_per_cycle(resolution) for _ in range(min(3, arguments.repetitions))]
            entry = dict(
                code="AthenaPK",
                precision="double",
                resolution=resolution,
                cells=cells,
                milliseconds_per_step=min(repetitions),
                all_milliseconds_per_step=repetitions,
                zone_cycles_per_second=cells / (min(repetitions) * 1e-3),
                gpu_status=[gpu_status()],
                device=device,
            )
            results.append(entry)
            print(
                f"N={resolution:4d} {'AthenaPK':18s} double: {entry['milliseconds_per_step']:8.2f} ms/step "
                f"({entry['zone_cycles_per_second'] / 1e6:7.1f} Mzone-cycles/s)",
                flush=True,
            )
        print(f"   GPU status during the repetitions: {statuses}", flush=True)

    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump(results, file, indent=1)


if __name__ == "__main__":
    main()
