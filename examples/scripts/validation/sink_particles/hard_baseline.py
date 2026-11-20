"""
Hard baseline of the sink particles on the collapse setups.

Runs the four isothermal collapse setups of ``astronomix/test_setups/self_gravity``
(Bonnor–Ebert sphere, singular isothermal sphere, rotating cloud core and
turbulent cluster) at 32³ with the formation and accretion of Federrath et al.
(2010) at the hard limit of the smoothing parameters, in float64 with the
native JAX backend and the finite-volume scheme (RK2, HLL, minmod), and

* ``reference``: records the sinks at every snapshot of a run with adaptive
  steps (number, masses, positions, formation times);
* ``gradients``: compares ``jax.grad`` with central finite differences, with
  respect to one input per setup, for two outputs: the total sink mass, and a
  fixed random combination of the mass-weighted moments of the sink positions,
  Σ m cos(2π x_d / L) and Σ m sin(2π x_d / L), which stay continuous when a
  sink crosses a periodic boundary. The comparison runs twice: shortly after
  the first formation and at the end of the run, with fixed steps, so that the
  gradient and the finite differences follow the same step sequence. The
  fixed-step loop of ``time_integration`` keeps every step for reverse mode
  (about 160 MB per step here), so a run is a scan over segments of
  ``STEPS_PER_SEGMENT`` steps, each under ``jax.checkpoint``: reverse mode
  keeps one state per segment and recomputes one segment at a time.

Each setup writes ``results/<setup>.npz``; ``results/report.md`` gathers them.

The finite-volume scheme is used because the default finite-difference scheme
breaks down in the turbulent cluster at 32³: next to the heaviest sink a nearly
empty cell reaches Mach 40–70, alternating from step to step, and the state
turns non-finite at t ≈ 0.05 (at t ≈ 0.07 with the positivity-preserving WENO
flux). The finite-volume scheme runs every setup to its end.

    python hard_baseline.py                       # every setup, both parts
    python hard_baseline.py --setups bonnor_ebert --part reference
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import math
import time
from pathlib import Path

# typing
from typing import Callable, NamedTuple

# jax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, NATIVE_JAX

# astronomix containers
from astronomix import (
    BackendConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
    SnapshotSettings,
)

# astronomix functions
from astronomix import get_registered_variables, time_integration
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
    _pad_sink_particles,
)
from astronomix.test_setups.self_gravity.bonnor_ebert_sphere3D import (
    BonnorEbertSettings,
    setup_bonnor_ebert_sphere,
)
from astronomix.test_setups.self_gravity.rotating_cloud_core3D import (
    RotatingCloudCoreSettings,
    setup_rotating_cloud_core,
)
from astronomix.test_setups.self_gravity.singular_isothermal_sphere3D import (
    SingularIsothermalSphereSettings,
    setup_singular_isothermal_sphere,
)
from astronomix.test_setups.self_gravity.turbulent_cluster3D import (
    TurbulentClusterSettings,
    setup_turbulent_cluster,
)

#: Cells along each axis.
NUM_CELLS = 32

#: Snapshots of the reference run.
NUM_SNAPSHOTS = 100

#: The hard limit of the smoothing parameters of the sink formation and
#: accretion.
HARD_LIMIT = dict(
    creation_margin=1.0,
    check_width=0.0,
    spatial_width_in_cells=0.0,
    sharing_sharpness=math.inf,
    position_update_mass=0.0,
)

#: Relative steps h of the central finite differences, f(p (1 ± h)).
RELATIVE_STEPS = (1e-1, 1e-2, 1e-3, 1e-4)

#: Fixed steps of the gradient runs per adaptive step of the reference run
#: (over the same time), so that the fixed steps stay within the CFL limit
#: although the adaptive steps shrink as the collapse proceeds.
FIXED_STEPS_PER_ADAPTIVE_STEP = 2.0

#: Fixed steps per checkpointed segment of a gradient run.
STEPS_PER_SEGMENT = 25

#: Where the comparison "shortly after the first formation" lies, as a
#: fraction of the time between the first formation and the end of the run.
SHORTLY_AFTER_FORMATION = 0.1

#: The outputs whose gradients are compared.
OUTPUT_NAMES = ("total sink mass", "position moments")

#: The weights of the random combination of the 6 position moments.
POSITION_WEIGHTS = np.random.default_rng(2026).normal(size=6)
POSITION_WEIGHTS = POSITION_WEIGHTS / np.linalg.norm(POSITION_WEIGHTS)

RESULTS = Path(__file__).resolve().parent / "results"


class Case(NamedTuple):
    """A validation setup and the input of its gradient."""

    #: The setup function, ``(config, params, settings) -> (state, config, params)``.
    setup: Callable

    #: The problem constants.
    settings: NamedTuple

    #: The field of ``settings`` the gradients are taken with respect to.
    parameter: str

    #: The name of the setup in the report.
    title: str


CASES = {
    "bonnor_ebert": Case(
        setup_bonnor_ebert_sphere,
        BonnorEbertSettings(),
        "density_contrast",
        "Bonnor–Ebert sphere",
    ),
    "singular_isothermal": Case(
        setup_singular_isothermal_sphere,
        SingularIsothermalSphereSettings(),
        "overdensity_factor",
        "Singular isothermal sphere",
    ),
    "rotating_core": Case(
        setup_rotating_cloud_core,
        RotatingCloudCoreSettings(),
        "rotation_rate",
        "Rotating cloud core",
    ),
    "turbulent_cluster": Case(
        setup_turbulent_cluster,
        TurbulentClusterSettings(),
        "mach_number",
        "Turbulent cluster",
    ),
}


def _config(**options) -> SimulationConfig:
    """The configuration shared by every run, with the given further fields."""
    return SimulationConfig(
        num_cells=NUM_CELLS,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(),
        progress_bar=False,
        sink_particle_config=SinkParticleConfig(sink_particles=True, **HARD_LIMIT),
        **options,
    )


# -------------------------------------------------------------
# ======================= ↓ Reference ↓ =======================
# -------------------------------------------------------------


def run_reference(case: Case) -> dict:
    """
    Run a setup with adaptive steps and record its sinks at every snapshot.

    Args:
        case: The setup.

    Returns:
        The reference: snapshot times, the sink fields at every snapshot, the
        gas mass, the formation time of every slot (NaN for a slot never
        occupied), the number of steps and the wall time.
    """
    config = _config(
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_total_mass=True),
    )
    state, config, params = case.setup(config, SimulationParams(), case.settings)
    registered_variables = get_registered_variables(config)

    start = time.perf_counter()
    snapshots = jax.block_until_ready(
        time_integration(state, config, params, registered_variables)
    )
    wall_time = time.perf_counter() - start

    sinks = snapshots.sink_particles
    times = np.asarray(snapshots.time_points)
    occupied = np.asarray(sinks.occupied)
    ever_occupied = occupied.any(axis=0)
    formation_time = np.where(
        ever_occupied,
        times[np.argmax(occupied, axis=0)],
        np.nan,
    )
    return dict(
        times=times,
        mass=np.asarray(sinks.mass),
        position=np.asarray(sinks.position),
        velocity=np.asarray(sinks.velocity),
        occupied=occupied,
        gas_mass=np.asarray(snapshots.total_mass),
        formation_time=formation_time,
        end_time=float(params.t_end),
        num_steps=int(snapshots.num_iterations),
        wall_time=wall_time,
    )


# -------------------------------------------------------------
# ======================= ↑ Reference ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Gradients ↓ =======================
# -------------------------------------------------------------


def _sink_outputs(sink_particles, box_length) -> jax.Array:
    """
    The compared outputs of a run: the total sink mass and the random
    combination of the mass-weighted position moments.

    Args:
        sink_particles: The sinks at the end of the run.
        box_length: The side length of the box.

    Returns:
        The two outputs, shape (2,).
    """
    mass = sink_particles.mass
    phase = 2.0 * jnp.pi * sink_particles.position / box_length
    moments = jnp.concatenate(
        [
            jnp.sum(mass[:, None] * jnp.cos(phase), axis=0),
            jnp.sum(mass[:, None] * jnp.sin(phase), axis=0),
        ]
    )
    return jnp.stack([jnp.sum(mass), jnp.dot(POSITION_WEIGHTS, moments)])


def _output_function(case: Case, end_time: float, num_segments: int) -> Callable:
    """
    The outputs of a run with fixed steps, as a function of the input of the
    setup.

    The run is a scan over ``num_segments`` segments of ``STEPS_PER_SEGMENT``
    steps, each a call of ``time_integration`` under ``jax.checkpoint``. The
    setups do not depend on time, so every segment runs from t = 0 over
    end_time / num_segments.

    Args:
        case: The setup.
        end_time: The end time of the run.
        num_segments: The number of segments.

    Returns:
        ``outputs(parameter) -> (2,)``.
    """

    def outputs(parameter):
        settings = case.settings._replace(**{case.parameter: parameter, "t_end": end_time})
        config = _config(fixed_timestep=True, num_timesteps=STEPS_PER_SEGMENT)
        state, config, params = case.setup(config, SimulationParams(), settings)
        params = params._replace(t_end=end_time / num_segments)
        registered_variables = get_registered_variables(config)

        # The scan carry needs the sinks from the start.
        state = state._replace(
            sink_particles=_pad_sink_particles(
                _empty_sink_particles(config, state.primitive_state.dtype),
                config,
            )
        )

        @jax.checkpoint
        def run_segment(state):
            return time_integration(state, config, params, registered_variables)

        final_state, _ = jax.lax.scan(
            lambda state, _: (run_segment(state), None),
            state,
            None,
            length=num_segments,
        )
        return _sink_outputs(final_state.sink_particles, settings.box_length)

    return outputs


def compare_gradients(case: Case, end_time: float, num_segments: int) -> dict:
    """
    Compare the reverse-mode gradient of the outputs with central finite
    differences at the relative steps of ``RELATIVE_STEPS``.

    Args:
        case: The setup.
        end_time: The end time of the runs.
        num_segments: The number of segments of ``STEPS_PER_SEGMENT`` steps.

    Returns:
        The outputs at the setup's input, their gradients, the finite
        differences (one row per step) and the wall times.
    """
    parameter = jnp.asarray(getattr(case.settings, case.parameter), dtype=jnp.float64)
    outputs = _output_function(case, end_time, num_segments)

    def value_and_gradients(parameter):
        value, pullback = jax.vjp(outputs, parameter)
        gradients = jnp.stack(
            [pullback(jnp.eye(len(OUTPUT_NAMES))[output])[0] for output in range(len(OUTPUT_NAMES))]
        )
        return value, gradients

    start = time.perf_counter()
    value, gradients = jax.block_until_ready(jax.jit(value_and_gradients)(parameter))
    gradient_wall_time = time.perf_counter() - start

    forward = jax.jit(outputs)
    finite_differences = []
    start = time.perf_counter()
    for relative_step in RELATIVE_STEPS:
        step = relative_step * parameter
        above = forward(parameter + step)
        below = forward(parameter - step)
        finite_differences.append(np.asarray((above - below) / (2.0 * step)))
    finite_difference_wall_time = time.perf_counter() - start

    return dict(
        end_time=end_time,
        num_steps=num_segments * STEPS_PER_SEGMENT,
        value=np.asarray(value),
        gradient=np.asarray(gradients),
        finite_differences=np.stack(finite_differences),
        gradient_wall_time=gradient_wall_time,
        finite_difference_wall_time=finite_difference_wall_time,
    )


def _end_times_and_segments(reference: dict) -> list[tuple[float, int]]:
    """
    The end times of the gradient comparisons, shortly after the first
    formation and at the end of the run, with their numbers of segments of
    ``STEPS_PER_SEGMENT`` fixed steps.

    Args:
        reference: The reference of the setup.

    Returns:
        ``[(end_time, num_segments), ...]``; empty when no sink formed.
    """
    first_formation = np.nanmin(reference["formation_time"], initial=np.inf)
    if not np.isfinite(first_formation):
        return []
    end_time = reference["end_time"]
    shortly_after = first_formation + SHORTLY_AFTER_FORMATION * (end_time - first_formation)
    steps_per_time = FIXED_STEPS_PER_ADAPTIVE_STEP * reference["num_steps"] / end_time
    return [
        (
            float(time_point),
            int(math.ceil(steps_per_time * time_point / STEPS_PER_SEGMENT)),
        )
        for time_point in (shortly_after, end_time)
    ]


# -------------------------------------------------------------
# ======================= ↑ Gradients ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================== ↓ Report ↓ =========================
# -------------------------------------------------------------


def _relative_difference(gradient, finite_difference):
    """|g − fd| / max(|g|, |fd|), and 0 where both are 0."""
    scale = np.maximum(np.abs(gradient), np.abs(finite_difference))
    return np.where(scale > 0.0, np.abs(gradient - finite_difference) / np.where(scale > 0.0, scale, 1.0), 0.0)


def write_report() -> Path:
    """Gather the results of every setup into ``results/report.md``."""
    lines = [
        "# Hard baseline of the sink particles",
        "",
        f"Generated by `hard_baseline.py`: {NUM_CELLS}³ cells, float64, native JAX backend, "
        "finite-volume scheme (RK2, HLL, minmod), formation and accretion at the hard limit "
        "of the smoothing parameters.",
        "",
    ]
    for name, case in CASES.items():
        path = RESULTS / f"{name}.npz"
        if not path.exists():
            continue
        data = dict(np.load(path))
        formation_time = data["formation_time"]
        formed = np.isfinite(formation_time)
        lines += [
            f"## {case.title} (`{name}`)",
            "",
            f"Input of the gradients: `{case.parameter}` = {getattr(case.settings, case.parameter):g}.",
            "",
            "Reference (adaptive steps):",
            "",
            f"- end time {float(data['end_time']):g}, {int(data['num_steps'])} steps, "
            f"{float(data['wall_time']):.0f} s",
            f"- sinks formed: {int(formed.sum())}; first formation at "
            + (f"t = {np.nanmin(formation_time):.4g}" if formed.any() else "never"),
            f"- total sink mass at the last snapshot (t = {data['times'][-1]:.4g}): "
            f"{data['mass'][-1].sum():.6g}, gas mass {data['gas_mass'][-1]:.6g}",
            "",
        ]
        if "comparison_end_time" not in data:
            lines += ["No gradient comparison.", ""]
            continue
        steps_header = " | ".join(f"FD h={step:g}" for step in RELATIVE_STEPS)
        lines += [
            f"| end time | steps | output | value | jax.grad | {steps_header} |",
            "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in RELATIVE_STEPS) + " |",
        ]
        for comparison in range(data["comparison_end_time"].size):
            for output, output_name in enumerate(OUTPUT_NAMES):
                gradient = data["comparison_gradient"][comparison, output]
                finite_differences = data["comparison_finite_differences"][comparison, :, output]
                errors = _relative_difference(gradient, finite_differences)
                cells = " | ".join(
                    f"{difference:.4g} ({error:.1e})"
                    for difference, error in zip(finite_differences, errors)
                )
                lines.append(
                    f"| {data['comparison_end_time'][comparison]:.4g} "
                    f"| {int(data['comparison_num_steps'][comparison])} "
                    f"| {output_name} | {data['comparison_value'][comparison, output]:.6g} "
                    f"| {gradient:.4g} | {cells} |"
                )
        lines += [
            "",
            "Each finite difference is followed by its relative difference from `jax.grad`, "
            "|g − fd| / max(|g|, |fd|).",
            "",
        ]
    report = RESULTS / "report.md"
    report.write_text("\n".join(lines))
    return report


# -------------------------------------------------------------
# ======================== ↑ Report ↑ =========================
# -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--setups", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--part", choices=["reference", "gradients", "all"], default="all")
    arguments = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    for name in arguments.setups:
        case = CASES[name]
        path = RESULTS / f"{name}.npz"
        if arguments.part in ("reference", "all"):
            print(f"{name}: reference run", flush=True)
            reference = run_reference(case)
            np.savez(path, **reference)
        reference = dict(np.load(path))
        print(
            f"{name}: {int(np.isfinite(reference['formation_time']).sum())} sinks, "
            f"first at t = {np.nanmin(reference['formation_time'], initial=np.inf):.4g}, "
            f"{reference['num_steps']} steps, {reference['wall_time']:.0f} s",
            flush=True,
        )

        if arguments.part in ("gradients", "all"):
            comparisons = []
            for end_time, num_segments in _end_times_and_segments(reference):
                print(
                    f"{name}: gradients at t = {end_time:.4g} "
                    f"({num_segments * STEPS_PER_SEGMENT} steps)",
                    flush=True,
                )
                comparisons.append(compare_gradients(case, end_time, num_segments))
            reference = {
                key: value for key, value in reference.items() if not key.startswith("comparison_")
            }
            if comparisons:
                for key in comparisons[0]:
                    reference[f"comparison_{key}"] = np.stack(
                        [np.asarray(comparison[key]) for comparison in comparisons]
                    )
            np.savez(path, **reference)

    print(f"report: {write_report()}")


if __name__ == "__main__":
    main()
