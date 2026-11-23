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

With ``--parameters smooth`` the same runs use the default (smooth)
parameters of the sink formation and accretion and write to
``results/smooth``. They take the times of the gradient comparisons from the
hard baseline, and the report gives the hard reference next to their own. A
sink is a slot with activation above ½: its formation time is the first
snapshot at which the activation of its slot exceeds ½.

The finite-volume scheme is used because the default finite-difference scheme
turns the turbulent cluster non-finite at 32³ with or without sinks, once the
collapse is no longer resolved (by t ≈ 0.046 without sinks, t ≈ 0.051 with
them). The finite-volume scheme runs every setup to its end.

    python hard_baseline.py                       # every setup, both parts
    python hard_baseline.py --setups bonnor_ebert --part reference
    python hard_baseline.py --parameters smooth    # the smooth scheme
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

#: Where the runs with the smooth parameters write their results.
SMOOTH_RESULTS = RESULTS / "smooth"


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


def _config(sink_options: dict, **options) -> SimulationConfig:
    """The configuration shared by every run, with the given
    ``SinkParticleConfig`` fields and further configuration fields."""
    return SimulationConfig(
        num_cells=NUM_CELLS,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(),
        progress_bar=False,
        sink_particle_config=SinkParticleConfig(sink_particles=True, **sink_options),
        **options,
    )


# -------------------------------------------------------------
# ======================= ↓ Reference ↓ =======================
# -------------------------------------------------------------


def run_reference(case: Case, sink_options: dict) -> dict:
    """
    Run a setup with adaptive steps and record its sinks at every snapshot.

    Args:
        case: The setup.
        sink_options: The ``SinkParticleConfig`` fields of the run.

    Returns:
        The reference: snapshot times, the sink fields at every snapshot, the
        gas mass, the formation time of every slot (the first snapshot at
        which its activation exceeds ½; NaN for a slot never a sink) and the
        number of steps.
    """
    config = _config(
        sink_options,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_total_mass=True),
    )
    state, config, params = case.setup(config, SimulationParams(), case.settings)
    registered_variables = get_registered_variables(config)

    snapshots = jax.block_until_ready(
        time_integration(state, config, params, registered_variables)
    )

    sinks = snapshots.sink_particles
    times = np.asarray(snapshots.time_points)
    activation = np.asarray(sinks.activation)
    is_sink = activation > 0.5
    formation_time = np.where(
        is_sink.any(axis=0),
        times[np.argmax(is_sink, axis=0)],
        np.nan,
    )
    return dict(
        times=times,
        mass=np.asarray(sinks.mass),
        position=np.asarray(sinks.position),
        velocity=np.asarray(sinks.velocity),
        activation=activation,
        occupied=np.asarray(sinks.occupied),
        gas_mass=np.asarray(snapshots.total_mass),
        formation_time=formation_time,
        end_time=float(params.t_end),
        num_steps=int(snapshots.num_iterations),
    )


# -------------------------------------------------------------
# ======================= ↑ Reference ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Gradients ↓ =======================
# -------------------------------------------------------------


def _sink_outputs(mass, position, box_length) -> jax.Array:
    """
    The compared outputs of a run: the total sink mass and the random
    combination of the mass-weighted position moments.

    Args:
        mass: The sink masses, shape (max_num_sinks,).
        position: The sink positions, shape (max_num_sinks, 3).
        box_length: The side length of the box.

    Returns:
        The two outputs, shape (2,).
    """
    phase = 2.0 * jnp.pi * position / box_length
    moments = jnp.concatenate(
        [
            jnp.sum(mass[:, None] * jnp.cos(phase), axis=0),
            jnp.sum(mass[:, None] * jnp.sin(phase), axis=0),
        ]
    )
    return jnp.stack([jnp.sum(mass), jnp.dot(POSITION_WEIGHTS, moments)])


def _output_function(
    case: Case,
    end_time: float,
    num_segments: int,
    sink_options: dict,
) -> Callable:
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
        sink_options: The ``SinkParticleConfig`` fields of the runs.

    Returns:
        ``outputs(parameter) -> (2,)``.
    """

    def outputs(parameter):
        settings = case.settings._replace(**{case.parameter: parameter, "t_end": end_time})
        config = _config(sink_options, fixed_timestep=True, num_timesteps=STEPS_PER_SEGMENT)
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
        return _sink_outputs(
            final_state.sink_particles.mass,
            final_state.sink_particles.position,
            settings.box_length,
        )

    return outputs


def compare_gradients(
    case: Case,
    end_time: float,
    num_segments: int,
    sink_options: dict,
) -> dict:
    """
    Compare the reverse-mode gradient of the outputs with central finite
    differences at the relative steps of ``RELATIVE_STEPS``.

    Args:
        case: The setup.
        end_time: The end time of the runs.
        num_segments: The number of segments of ``STEPS_PER_SEGMENT`` steps.
        sink_options: The ``SinkParticleConfig`` fields of the runs.

    Returns:
        The outputs at the setup's input, their gradients, the finite
        differences (one row per step) and the wall times.
    """
    parameter = jnp.asarray(getattr(case.settings, case.parameter), dtype=jnp.float64)
    outputs = _output_function(case, end_time, num_segments, sink_options)

    def value_and_gradients(parameter):
        value, pullback = jax.vjp(outputs, parameter)
        # One batched backward pass for all outputs, which recomputes each
        # checkpointed segment once.
        (gradients,) = jax.vmap(pullback)(jnp.eye(len(OUTPUT_NAMES)))
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


def _reference_at(times, outputs, occupied, time_point):
    """
    An output of the adaptive reference at ``time_point``, interpolated
    between the snapshots around it.

    Args:
        times: The snapshot times, shape (num_snapshots,).
        outputs: The output at every snapshot, shape (num_snapshots,).
        occupied: The occupied slots at every snapshot, shape
            (num_snapshots, max_num_sinks).
        time_point: The time of the comparison.

    Returns:
        ``(value, sinks_form)``: the interpolated output, and whether the
        number of sinks changes between the two snapshots around
        ``time_point``, where the output jumps and the interpolated value is
        not meaningful.
    """
    value = np.interp(time_point, times, outputs)
    after = min(int(np.searchsorted(times, time_point)), times.size - 1)
    if after == 0 or times[after] == time_point:
        return value, False
    sinks_form = occupied[after].sum() != occupied[after - 1].sum()
    return value, bool(sinks_form)


def _relative_difference(gradient, finite_difference):
    """|g − fd| / max(|g|, |fd|), and 0 where both are 0."""
    scale = np.maximum(np.abs(gradient), np.abs(finite_difference))
    return np.where(scale > 0.0, np.abs(gradient - finite_difference) / np.where(scale > 0.0, scale, 1.0), 0.0)


def write_report(smooth: bool) -> Path:
    """
    Gather the results of every setup into ``report.md`` in the results
    directory of the parameters.

    Args:
        smooth: Whether to report the runs with the smooth parameters, with
            the hard reference next to them, instead of the hard baseline.

    Returns:
        The path of the report.
    """
    results = SMOOTH_RESULTS if smooth else RESULTS
    parameters = (
        "the default (smooth) parameters" if smooth else "the hard limit of the smoothing parameters"
    )
    lines = [
        "# Sink particles on the collapse setups" if smooth else "# Hard baseline of the sink particles",
        "",
        f"Generated by `hard_baseline.py`: {NUM_CELLS}³ cells, float64, native JAX backend, "
        f"finite-volume scheme (RK2, HLL, minmod), formation and accretion with {parameters}. "
        "A sink is a slot with activation above ½.",
        "",
    ]
    for name, case in CASES.items():
        path = results / f"{name}.npz"
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
            f"- end time {float(data['end_time']):g}, {int(data['num_steps'])} steps",
            f"- sinks formed: {int(formed.sum())}; first formation at "
            + (f"t = {np.nanmin(formation_time):.4g}" if formed.any() else "never"),
            f"- total sink mass at the last snapshot (t = {data['times'][-1]:.4g}): "
            f"{data['mass'][-1].sum():.6g}, gas mass {data['gas_mass'][-1]:.6g}",
            "",
        ]
        hard_path = RESULTS / f"{name}.npz"
        if smooth and hard_path.exists():
            hard = dict(np.load(hard_path))
            hard_formation_time = hard["formation_time"]
            hard_formed = np.isfinite(hard_formation_time)
            lines += [
                "Hard reference:",
                "",
                f"- sinks formed: {int(hard_formed.sum())}; first formation at "
                + (f"t = {np.nanmin(hard_formation_time):.4g}" if hard_formed.any() else "never"),
                f"- total sink mass at the last snapshot (t = {hard['times'][-1]:.4g}): "
                f"{hard['mass'][-1].sum():.6g}",
                "",
            ]
        if "comparison_end_time" not in data:
            lines += ["No gradient comparison.", ""]
            continue
        # The outputs of the adaptive reference at every snapshot, to compare
        # with the values of the fixed-step runs.
        reference_outputs = np.asarray(
            jax.vmap(
                lambda mass, position: _sink_outputs(mass, position, case.settings.box_length)
            )(data["mass"], data["position"])
        )
        steps_header = " | ".join(f"FD h={step:g}" for step in RELATIVE_STEPS)
        lines += [
            f"| end time | steps | output | value | reference | jax.grad | {steps_header} |",
            "| --- | --- | --- | --- | --- | --- | "
            + " | ".join("---" for _ in RELATIVE_STEPS)
            + " |",
        ]
        for comparison in range(data["comparison_end_time"].size):
            end_time = data["comparison_end_time"][comparison]
            for output, output_name in enumerate(OUTPUT_NAMES):
                gradient = data["comparison_gradient"][comparison, output]
                finite_differences = data["comparison_finite_differences"][comparison, :, output]
                errors = _relative_difference(gradient, finite_differences)
                cells = " | ".join(
                    f"{difference:.4g} ({error:.1e})"
                    for difference, error in zip(finite_differences, errors)
                )
                reference_value, sinks_form = _reference_at(
                    data["times"],
                    reference_outputs[:, output],
                    data["activation"] > 0.5 if "activation" in data else data["occupied"],
                    end_time,
                )
                lines.append(
                    f"| {end_time:.4g} "
                    f"| {int(data['comparison_num_steps'][comparison])} "
                    f"| {output_name} | {data['comparison_value'][comparison, output]:.6g} "
                    f"| {reference_value:.6g}{' *' if sinks_form else ''} "
                    f"| {gradient:.4g} | {cells} |"
                )
        lines += [
            "",
            "The value is that of the fixed-step run; the reference is the same output of the "
            "adaptive run, interpolated between its snapshots. Fixed steps skip the CFL limit "
            "and the half-cell limit on sink motion, so the two must agree, except where marked "
            "with *: there sinks form between the two snapshots around the time, the output "
            "jumps, and the interpolated reference is not meaningful.",
            "",
            "Each finite difference is followed by its relative difference from `jax.grad`, "
            "|g − fd| / max(|g|, |fd|).",
            "",
        ]
    report = results / "report.md"
    report.write_text("\n".join(lines))
    return report


# -------------------------------------------------------------
# ======================== ↑ Report ↑ =========================
# -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--setups", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--part", choices=["reference", "gradients", "all"], default="all")
    parser.add_argument("--parameters", choices=["hard", "smooth"], default="hard")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="print the diagnostics of the sink update at every step of the reference runs",
    )
    arguments = parser.parse_args()
    smooth = arguments.parameters == "smooth"
    sink_options = {} if smooth else dict(HARD_LIMIT)
    results = SMOOTH_RESULTS if smooth else RESULTS
    results.mkdir(parents=True, exist_ok=True)

    for name in arguments.setups:
        case = CASES[name]
        path = results / f"{name}.npz"
        if arguments.part in ("reference", "all"):
            print(f"{name}: reference run", flush=True)
            reference = run_reference(
                case,
                dict(sink_options, print_diagnostics=arguments.diagnostics),
            )
            np.savez(path, **reference)
        reference = dict(np.load(path))
        print(
            f"{name}: {int(np.isfinite(reference['formation_time']).sum())} sinks, "
            f"first at t = {np.nanmin(reference['formation_time'], initial=np.inf):.4g}, "
            f"{reference['num_steps']} steps",
            flush=True,
        )

        if arguments.part in ("gradients", "all"):
            # The smooth runs are compared at the times of the hard baseline.
            if smooth:
                hard = dict(np.load(RESULTS / f"{name}.npz"))
                end_times_and_segments = [
                    (float(end_time), int(num_steps) // STEPS_PER_SEGMENT)
                    for end_time, num_steps in zip(
                        hard.get("comparison_end_time", []),
                        hard.get("comparison_num_steps", []),
                    )
                ]
            else:
                end_times_and_segments = _end_times_and_segments(reference)
            comparisons = []
            for end_time, num_segments in end_times_and_segments:
                print(
                    f"{name}: gradients at t = {end_time:.4g} "
                    f"({num_segments * STEPS_PER_SEGMENT} steps)",
                    flush=True,
                )
                comparisons.append(
                    compare_gradients(case, end_time, num_segments, sink_options)
                )
            reference = {
                key: value for key, value in reference.items() if not key.startswith("comparison_")
            }
            if comparisons:
                for key in comparisons[0]:
                    reference[f"comparison_{key}"] = np.stack(
                        [np.asarray(comparison[key]) for comparison in comparisons]
                    )
            np.savez(path, **reference)

    print(f"report: {write_report(smooth)}")


if __name__ == "__main__":
    main()
