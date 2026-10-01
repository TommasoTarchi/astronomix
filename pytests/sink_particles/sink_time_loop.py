"""
Sink particles in the time loop pytest.

Checks full runs of the time loop with sinks forming, through the in-memory
snapshots and through the disk snapshots and restart, and the configuration
requirements enforced by ``finalize_config``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import tempfile
from pathlib import Path

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import TO_DISK
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import (
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import StaticIntVector
from astronomix._modules._sink_particles._sink_particle_options import SinkParticleConfig

# astronomix functions
from astronomix import (
    get_registered_variables,
    restart_from_latest_checkpoint,
    time_integration,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    setup_sink_formation,
)

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    NUM_CELLS,
    SETTINGS,
    _num_sinks,
    _setup,
)


@pytest.mark.parametrize(
    "self_gravity_version",
    [SIMPLE_SOURCE, SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE],
    ids=["simple_source", "second_order_conservative", "fourth_order_conservative"],
)
def test_clump_collapse(self_gravity_version):
    """A sink forms during a run of the time loop and total mass is conserved,
    for each finite-difference self-gravity treatment. The run goes through
    the in-memory snapshots, which record the sinks at every snapshot."""
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
        ),
        return_snapshots=True,
        num_snapshots=5,
        snapshot_settings=SnapshotSettings(return_states=True),
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    # About 30 adaptive steps, a few free-fall times of the clump peak.
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SETTINGS._replace(t_end=0.06),
    )
    registered_variables = get_registered_variables(config)

    snapshots = time_integration(state, config, params, registered_variables)

    # No sink at t = 0 (recorded before the first formation call); one sink
    # from the next snapshot on.
    num_sinks = jnp.sum(snapshots.sink_particles.mass > 0.0, axis=1)
    assert num_sinks[0] == 0
    assert jnp.all(num_sinks[1:] == 1)

    # Gas plus sink mass at every snapshot equals the initial gas mass.
    cell_volume = CELL_SIZE**3
    initial_mass = jnp.sum(state.primitive_state[registered_variables.density_index])
    initial_mass = initial_mass * cell_volume
    gas_mass = jnp.sum(
        snapshots.states[:, registered_variables.density_index],
        axis=(1, 2, 3),
    ) * cell_volume
    sink_mass = jnp.sum(snapshots.sink_particles.mass, axis=1)
    assert jnp.allclose(gas_mass + sink_mass, initial_mass, rtol=1e-4)


def test_disk_snapshots_with_sinks(tmp_path):
    """Disk snapshots carry the sinks: a run restarted from a checkpoint ends
    with the same gas state and the same sinks as the uninterrupted run."""
    pytest.importorskip("orbax.checkpoint")
    state, config, params, registered_variables = _setup()

    # Two segments of adaptive steps; each ends exactly on its snapshot time
    # and writes a checkpoint. The sink forms in the first step.
    uninterrupted_path = str(tmp_path / "uninterrupted")
    config = config._replace(
        fixed_timestep=False,
        snapshot_storage_mode=TO_DISK,
        snapshot_storage_path=uninterrupted_path,
        num_snapshots=2,
    )
    final_state = time_integration(state, config, params, registered_variables)
    assert _num_sinks(final_state.sink_particles) == 1

    # Restart from the checkpoint of the first segment and run the second
    # segment again, writing to a separate directory.
    primitive_state, restart_params, restart_state = restart_from_latest_checkpoint(
        uninterrupted_path,
        params,
        step=1,
    )
    assert _num_sinks(restart_state.sink_particles) == 1
    restart_config = config._replace(
        snapshot_storage_path=str(tmp_path / "restarted"),
        num_snapshots=1,
    )
    restarted_state = time_integration(
        StateStruct(primitive_state=primitive_state),
        restart_config,
        restart_params,
        registered_variables,
        restart_state=restart_state,
    )

    assert jnp.array_equal(restarted_state.primitive_state, final_state.primitive_state)
    for restarted_field, final_field in zip(
        restarted_state.sink_particles,
        final_state.sink_particles,
    ):
        assert jnp.array_equal(restarted_field, final_field)


@pytest.mark.parametrize(
    "unsupported_options, state_shape",
    [
        (dict(dimensionality=2), (4, 16, 16)),
        (dict(gravity_config=GravityConfig(self_gravity=False)), (5, 16, 16, 16)),
        (dict(state_struct=False), (5, 16, 16, 16)),
    ],
    ids=["2d", "no_self_gravity", "no_state_struct"],
)
def test_sink_particle_config_requirements(unsupported_options, state_shape):
    """``finalize_config`` must reject each configuration that sink particle
    formation does not support.

    Args:
        unsupported_options: The config fields that make the configuration
            unsupported, on top of an otherwise valid one.
        state_shape: The primitive state shape matching the dimensionality.
    """
    supported_options = dict(
        dimensionality=3,
        gravity_config=GravityConfig(self_gravity=True),
        state_struct=True,
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    config = SimulationConfig(**{**supported_options, **unsupported_options})

    with pytest.raises(ValueError):
        finalize_config(config, state_shape)


if __name__ == "__main__":
    test_clump_collapse(SIMPLE_SOURCE)
    test_clump_collapse(SECOND_ORDER_CONSERVATIVE)
    test_clump_collapse(FOURTH_ORDER_CONSERVATIVE)
    test_disk_snapshots_with_sinks(Path(tempfile.mkdtemp()))
    test_sink_particle_config_requirements(dict(dimensionality=2), (4, 16, 16))
    test_sink_particle_config_requirements(
        dict(gravity_config=GravityConfig(self_gravity=False)),
        (5, 16, 16, 16),
    )
    test_sink_particle_config_requirements(dict(state_struct=False), (5, 16, 16, 16))
