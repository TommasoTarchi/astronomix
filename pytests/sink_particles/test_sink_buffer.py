"""
Sink-particle buffer round trip (stage 0 of the sink-particle plan).

Runs a 3D Kelvin-Helmholtz shear layer with a sink buffer carried alongside the
fluid and checks that the buffer comes back unchanged from ``time_integration``
and from an Orbax checkpoint restart, that the fluid is identical to the same
run without the buffer, and that the disk-checkpointing segments compile only
once. The buffer is filled with distinctive values so that a dropped,
reordered or reshaped field shows up as wrong numbers rather than zeros.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import logging

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    PERIODIC_BOUNDARY,
    FORWARDS,
    TO_DISK,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix.data_classes.simulation_state_struct import (
    SinkParticles,
    StateStruct,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
    restart_from_latest_checkpoint,
)


NUM_CELLS = 32
NUM_SINK_SLOTS = 16
T_HALF = 0.2
T_END = 0.4


def _distinctive_sinks():
    """A sink buffer whose values cannot arise by accident."""
    return SinkParticles(
        mass=jnp.arange(NUM_SINK_SLOTS) + 1.0,
        position=0.1 * jnp.arange(NUM_SINK_SLOTS * 3).reshape(NUM_SINK_SLOTS, 3),
        velocity=-0.3 * (jnp.arange(NUM_SINK_SLOTS * 3).reshape(NUM_SINK_SLOTS, 3) + 1.0),
    )


def _khi_setup(with_sinks, checkpoint_path=None):
    """A 3D periodic Kelvin-Helmholtz shear layer, optionally carrying the sink
    buffer.

    Returns ``(state, config, registered_variables)``; ``state`` is a
    ``StateStruct`` when ``with_sinks`` and a bare primitive state otherwise.
    """
    box_size = 1.0

    config = SimulationConfig(
        dimensionality = 3,
        box_size = box_size,
        num_cells = NUM_CELLS,
        differentiation_mode = FORWARDS,
        boundary_settings = BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        state_struct = with_sinks,
        sink_particles = with_sinks,
        num_sink_slots = NUM_SINK_SLOTS if with_sinks else 0,
    )
    if checkpoint_path is not None:
        config = config._replace(
            snapshot_storage_mode = TO_DISK,
            snapshot_storage_path = str(checkpoint_path),
            num_snapshots = 4,
        )

    registered_variables = get_registered_variables(config)

    grid_spacing = box_size / NUM_CELLS
    x = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, NUM_CELLS)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")

    rho = jnp.where((Y > 0.25) & (Y < 0.75), 2.0, 1.0)
    u_x = jnp.where((Y > 0.25) & (Y < 0.75), -0.5, 0.5)
    u_y = 0.01 * jnp.sin(2 * jnp.pi * X)
    u_z = 0.01 * jnp.sin(2 * jnp.pi * Z)
    p = 2.5 * jnp.ones_like(X)

    primitive_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u_x,
        velocity_y = u_y,
        velocity_z = u_z,
        gas_pressure = p,
    )

    config = finalize_config(config, primitive_state.shape)

    if with_sinks:
        state = StateStruct(primitive_state=primitive_state, sinks=_distinctive_sinks())
    else:
        state = primitive_state

    return state, config, registered_variables


def _assert_sinks_equal(sinks, expected):
    for name in SinkParticles._fields:
        actual_field = getattr(sinks, name)
        expected_field = getattr(expected, name)
        assert actual_field.shape == expected_field.shape, name
        assert actual_field.dtype == expected_field.dtype, name
        assert jnp.array_equal(actual_field, expected_field), name


def test_sink_buffer_in_memory():
    """The buffer survives an in-memory run and leaves the fluid unchanged."""
    params = SimulationParams(t_end = T_END, C_cfl = 0.4)

    state, config, registered_variables = _khi_setup(with_sinks=True)
    final = time_integration(state, config, params, registered_variables)

    reference_state, reference_config, _ = _khi_setup(with_sinks=False)
    reference_final = time_integration(
        reference_state, reference_config, params, registered_variables
    )

    _assert_sinks_equal(final.sinks, _distinctive_sinks())
    assert jnp.array_equal(final.primitive_state, reference_final)


def _run_with_restart(with_sinks, checkpoint_path):
    """Run to ``T_HALF`` writing checkpoints, then restart and run to ``T_END``.

    Returns the final state, the restart state and the number of times the
    segment runner was compiled during the first half.
    """
    state, config, registered_variables = _khi_setup(with_sinks, checkpoint_path)

    # Count the segment-runner compilations by capturing JAX's compile log.
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    jax_logger = logging.getLogger("jax")
    jax_logger.addHandler(handler)
    try:
        with jax.log_compiles():
            time_integration(
                state, config, SimulationParams(t_end = T_HALF, C_cfl = 0.4),
                registered_variables,
            )
    finally:
        jax_logger.removeHandler(handler)
    num_segment_compiles = sum(
        "Compiling" in record.getMessage() and "_run_segment" in record.getMessage()
        for record in records
    )

    params = SimulationParams(t_end = T_END, C_cfl = 0.4)
    restored_primitive_state, params, restart = restart_from_latest_checkpoint(
        checkpoint_path, params
    )
    if with_sinks:
        restored_state = StateStruct(primitive_state=restored_primitive_state)
    else:
        restored_state = restored_primitive_state
    final = time_integration(
        restored_state, config, params, registered_variables, restart_state=restart
    )

    return final, restart, num_segment_compiles


def test_sink_buffer_checkpoint_restart(tmp_path):
    """The buffer survives a checkpoint restart, the fluid is unchanged and the
    segments compile once."""
    final, restart, num_segment_compiles = _run_with_restart(
        True, tmp_path / "with_sinks"
    )
    reference_final, reference_restart, _ = _run_with_restart(
        False, tmp_path / "without_sinks"
    )

    _assert_sinks_equal(restart.sinks, _distinctive_sinks())
    _assert_sinks_equal(final.sinks, _distinctive_sinks())
    assert reference_restart.sinks is None
    assert jnp.array_equal(final.primitive_state, reference_final)
    assert num_segment_compiles == 1
