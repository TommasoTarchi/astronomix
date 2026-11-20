"""
The sink particle slots: completing sinks given by mass, position and velocity
alone, filling free slots wherever they are, and restoring a checkpoint that
holds only the mass, position and velocity of the sinks.
"""

# numerics
import numpy as np

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import TO_DISK

# astronomix containers
from astronomix import SinkParticles

# astronomix functions
from astronomix import (
    finalize_state,
    restart_from_latest_checkpoint,
    time_integration,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
    _pad_sink_particles,
)
from astronomix._snapshotting._orbax_storage import (
    loop_checkpointer,
    save_loop_checkpoint,
)

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    CLUMP_CENTER,
    _form_sinks_once,
    _setup,
    _with_sink,
)


def test_sinks_given_by_mass_alone_are_completed():
    """Sinks given by mass, position and velocity alone get, at the start of a
    run, occupied slots with activation 1 where the mass is positive, empty
    slots elsewhere, and empty slots appended up to max_num_sinks."""
    _, config, _, _ = _setup()
    max_num_sinks = config.sink_particle_config.max_num_sinks
    sink_particles = SinkParticles(
        mass=jnp.array([0.5, 0.0, 0.3]),
        position=jnp.full((3, 3), 0.1),
        velocity=jnp.zeros((3, 3)),
    )

    completed = _pad_sink_particles(sink_particles, config)

    expected_occupied = np.zeros(max_num_sinks, dtype=bool)
    expected_occupied[[0, 2]] = True
    assert completed.occupied.dtype == bool, (
        f"the occupied flags are {completed.occupied.dtype}, not boolean"
    )
    np.testing.assert_array_equal(
        completed.occupied,
        expected_occupied,
        err_msg="the occupied slots are not those with a positive mass",
    )
    np.testing.assert_array_equal(
        completed.activation,
        expected_occupied.astype(float),
        err_msg="the activation is not 1 on the slots with mass and 0 elsewhere",
    )
    for name, field in completed._asdict().items():
        assert field.shape[0] == max_num_sinks, (
            f"{name} has {field.shape[0]} slots, not max_num_sinks = {max_num_sinks}"
        )


def test_given_occupied_flags_and_activation_are_kept():
    """Occupied flags and activations passed in are kept, even for a massless
    candidate, which the masses alone would call an empty slot."""
    _, config, _, _ = _setup()
    sink_particles = SinkParticles(
        mass=jnp.array([0.0, 0.0]),
        position=jnp.full((2, 3), 0.1),
        velocity=jnp.zeros((2, 3)),
        activation=jnp.array([0.3, 0.0]),
        occupied=jnp.array([True, False]),
    )

    completed = _pad_sink_particles(sink_particles, config)

    assert bool(completed.occupied[0]) and not bool(completed.occupied[1]), (
        f"the given occupied flags were replaced: {completed.occupied[:2]}"
    )
    assert completed.activation[0] == sink_particles.activation[0], (
        f"the given activation {sink_particles.activation[0]} became {completed.activation[0]}"
    )


def test_new_sink_takes_first_free_slot():
    """A new sink takes the first free slot, even when it lies before an
    occupied one: occupied slots need not be contiguous at the front."""
    state, config, params, registered_variables = _setup()
    dtype = state.primitive_state.dtype
    far_sink = _with_sink(_empty_sink_particles(config, dtype), 1, 1.0, (0.1, 0.1, 0.1))

    sink_particles = _form_sinks_once(state, config, params, registered_variables, far_sink)

    occupied = np.asarray(sink_particles.occupied)
    assert occupied[0] and occupied[1] and not occupied[2:].any(), (
        f"the occupied slots are {np.flatnonzero(occupied)}, not the new sink in slot 0 and "
        "the far sink in slot 1"
    )
    assert float(sink_particles.activation[0]) == 1.0, (
        f"the new sink has activation {float(sink_particles.activation[0])}, not 1"
    )
    assert jnp.all(jnp.abs(sink_particles.position[0] - CLUMP_CENTER) < CELL_SIZE), (
        f"the new sink is at {sink_particles.position[0]}, not at the clump centre"
    )
    np.testing.assert_array_equal(
        sink_particles.position[1],
        far_sink.position[1],
        err_msg="the far sink in slot 1 was overwritten",
    )


def test_checkpoint_without_occupied_and_activation_restores(tmp_path):
    """A checkpoint that holds only the mass, position and velocity of the
    sinks restores, and the run completes its sinks from the masses: the sink
    at the clump centre is occupied with activation 1 and accretes, and the
    other slots stay empty.

    Args:
        tmp_path: pytest's temporary directory, for the checkpoints.
    """
    pytest.importorskip("orbax.checkpoint")
    state, config, params, registered_variables = _setup()
    dtype = state.primitive_state.dtype
    sink_mass = 0.5
    saved_sinks = SinkParticles(
        mass=jnp.array([sink_mass, 0.0], dtype=dtype),
        position=jnp.array([[CLUMP_CENTER] * 3, [0.0] * 3], dtype=dtype),
        velocity=jnp.zeros((2, 3), dtype=dtype),
    )
    checkpoint_path = tmp_path / "mass_position_velocity"
    with loop_checkpointer(checkpoint_path) as writer:
        save_loop_checkpoint(
            writer,
            0,
            time=jnp.asarray(0.0, dtype=dtype),
            primitive_state=state.primitive_state,
            driving_state=None,
            num_iterations=jnp.asarray(0),
            sink_particles=saved_sinks,
        )

    primitive_state, restart_params, restart_state = restart_from_latest_checkpoint(
        str(checkpoint_path),
        params,
    )
    restored_sinks = restart_state.sink_particles
    assert restored_sinks.occupied is None and restored_sinks.activation is None, (
        "the checkpoint holds fields beyond mass, position and velocity"
    )

    config = config._replace(
        fixed_timestep=False,
        snapshot_storage_mode=TO_DISK,
        snapshot_storage_path=str(tmp_path / "restarted"),
        num_snapshots=1,
    )
    final_state = time_integration(
        finalize_state(config, primitive_state),
        config,
        restart_params,
        registered_variables,
        restart_state=restart_state,
    )

    sink_particles = final_state.sink_particles
    max_num_sinks = config.sink_particle_config.max_num_sinks
    expected_occupied = np.zeros(max_num_sinks, dtype=bool)
    expected_occupied[0] = True
    np.testing.assert_array_equal(
        sink_particles.occupied,
        expected_occupied,
        err_msg="after the restart the occupied slots are not the restored sink's alone",
    )
    np.testing.assert_array_equal(
        sink_particles.activation,
        expected_occupied.astype(float),
        err_msg="after the restart the activation is not 1 on the restored sink and 0 elsewhere",
    )
    assert float(sink_particles.mass[0]) > sink_mass, (
        f"the restored sink did not accrete: its mass is {float(sink_particles.mass[0])}"
    )
