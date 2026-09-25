"""
Sink formation with hard criteria (stage 1a of the sink-particle plan).

Uses a small periodic box with self-gravity holding a cold, infalling Gaussian
overdensity, centred on a cell, whose centre exceeds the Truelove threshold
tenfold. Checks a single formation call for conservation and for the density
left behind, checks that a shock-compressed sheet forms no sink unless the
transfer checks are switched off, and checks that a full run forms a single
sink and survives a checkpoint restart.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    PERIODIC_BOUNDARY,
    FORWARDS,
    TO_DISK,
)

# astronomix containers
from astronomix import (
    GravityConfig,
    PositivityConfig,
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
from astronomix._modules._sink_particles._sink_formation import (
    TRUELOVE_JEANS_NUMBER,
    _sink_formation,
)


NUM_CELLS = 32
NUM_SINK_SLOTS = 8
GAMMA = 5 / 3
SOUND_SPEED_SQUARED = 0.01
T_HALF = 0.05
T_END = 0.1

DX = 1.0 / NUM_CELLS
VOLUME = DX**3


def _empty_sinks():
    return SinkParticles(
        mass=jnp.zeros(NUM_SINK_SLOTS),
        position=jnp.zeros((NUM_SINK_SLOTS, 3)),
        velocity=jnp.zeros((NUM_SINK_SLOTS, 3)),
    )


def _setup(density, velocity, config_overrides=None):
    """A periodic, self-gravitating 3D box with sinks, holding the given gas at
    ``SOUND_SPEED_SQUARED``. Returns ``(primitive_state, config, registered_variables)``."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        dimensionality = 3,
        box_size = 1.0,
        num_cells = NUM_CELLS,
        differentiation_mode = FORWARDS,
        boundary_settings = BoundarySettings(periodic, periodic, periodic),
        gravity_config = GravityConfig(self_gravity = True),
        positivity_config = PositivityConfig(preserving_flux = True),
        state_struct = True,
        sink_particles = True,
        num_sink_slots = NUM_SINK_SLOTS,
        **(config_overrides or {}),
    )
    registered_variables = get_registered_variables(config)

    primitive_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = density,
        velocity_x = velocity[0],
        velocity_y = velocity[1],
        velocity_z = velocity[2],
        gas_pressure = SOUND_SPEED_SQUARED * density / GAMMA,
    )
    config = finalize_config(config, primitive_state.shape)
    return primitive_state, config, registered_variables


def _cell_offsets(centre):
    x = (jnp.arange(NUM_CELLS) + 0.5) * DX
    return jnp.stack(jnp.meshgrid(x, x, x, indexing="ij")) - centre


def _collapsing_blob(config_overrides=None):
    """Infalling Gaussian overdensity centred on cell (16, 16, 16)."""
    offsets = _cell_offsets(16.5 * DX)
    profile = jnp.exp(-jnp.sum(offsets**2, axis=0) / (2 * 0.08**2))
    density = 1.0 + 20.0 * profile
    velocity = -2.0 * offsets * profile
    return _setup(density, velocity, config_overrides)


def _shock_sheet(config_overrides=None):
    """Two uniform streams colliding along x into a dense sheet, with no
    convergence along y and z and the same density everywhere in the sheet."""
    offsets = _cell_offsets(16.5 * DX)
    density = jnp.where(jnp.abs(offsets[0]) < 2 * DX, 20.0, 1.0)
    velocity = jnp.stack([
        -jnp.sign(offsets[0]),
        jnp.zeros_like(density),
        jnp.zeros_like(density),
    ])
    return _setup(density, velocity, config_overrides)


def _gas_totals(primitive_state, registered_variables):
    """Total gas mass, momentum and energy."""
    rho = primitive_state[registered_variables.density_index]
    u = primitive_state[1:4]
    p = primitive_state[registered_variables.pressure_index]
    mass = jnp.sum(rho) * VOLUME
    momentum = jnp.sum(rho * u, axis=(1, 2, 3)) * VOLUME
    energy = jnp.sum(0.5 * rho * jnp.sum(u**2, axis=0) + p / (GAMMA - 1)) * VOLUME
    return mass, momentum, energy


def test_single_transfer():
    """One formation call opens one slot, conserves mass and momentum, keeps
    the energy books straight and removes no more than the 25% cap."""
    state, config, registered_variables = _collapsing_blob()
    params = SimulationParams(gamma = GAMMA)

    new_state, sinks = _sink_formation(
        state, _empty_sinks(), config, params, registered_variables
    )

    assert jnp.sum(sinks.mass > 0) == 1
    slot = jnp.argmax(sinks.mass)
    assert jnp.allclose(sinks.position[slot], 16.5 * DX)

    mass_0, momentum_0, energy_0 = _gas_totals(state, registered_variables)
    mass_1, momentum_1, energy_1 = _gas_totals(new_state, registered_variables)
    assert jnp.isclose(mass_1 + jnp.sum(sinks.mass), mass_0, rtol=1e-13)
    assert jnp.allclose(
        momentum_1 + jnp.sum(sinks.mass[:, None] * sinks.velocity, axis=0),
        momentum_0,
        rtol=0,
        atol=1e-13,
    )

    # The gas loses its energy in proportion to the mass it gives; the sink
    # keeps only the bulk kinetic energy of what it took.
    rho_0 = state[registered_variables.density_index]
    rho_1 = new_state[registered_variables.density_index]
    removed_fraction = 1.0 - rho_1 / rho_0
    cell_energy = (
        0.5 * rho_0 * jnp.sum(state[1:4] ** 2, axis=0)
        + state[registered_variables.pressure_index] / (GAMMA - 1)
    ) * VOLUME
    assert jnp.isclose(energy_0 - energy_1, jnp.sum(removed_fraction * cell_energy), rtol=1e-10)
    sink_kinetic = jnp.sum(0.5 * sinks.mass * jnp.sum(sinks.velocity**2, axis=1))
    assert energy_1 + sink_kinetic <= energy_0

    # No cell gives more than the cap. With the hard criteria, a cell that
    # gives mass is left at exactly max(rho_thr, 0.75 rho); this does not
    # hold once the criteria are smooth.
    assert jnp.all(rho_1 >= 0.75 * rho_0 * (1 - 1e-12))
    rho_thr = (
        TRUELOVE_JEANS_NUMBER**2 * jnp.pi * SOUND_SPEED_SQUARED
        / (params.gravitational_constant * DX**2)
    )
    gave = rho_1 < rho_0
    assert jnp.sum(gave) > 1
    expected = jnp.maximum(rho_thr, 0.75 * rho_0)
    assert jnp.allclose(rho_1[gave], expected[gave], rtol=1e-12)


def test_shock_forms_no_sink():
    """A shock-compressed sheet above the threshold forms no sink with the
    transfer checks on, but does with all of them off."""
    params = SimulationParams(gamma = GAMMA)

    state, config, registered_variables = _shock_sheet()
    _, sinks = _sink_formation(state, _empty_sinks(), config, params, registered_variables)
    assert jnp.sum(sinks.mass > 0) == 0

    state, config, registered_variables = _shock_sheet(dict(
        sink_converging_flow_check = False,
        sink_jeans_check = False,
        sink_bound_check = False,
    ))
    _, sinks = _sink_formation(state, _empty_sinks(), config, params, registered_variables)
    assert jnp.sum(sinks.mass > 0) > 0


def _run(checkpoint_path, restart):
    """Run the collapsing blob to ``T_END`` with disk checkpoints, optionally
    stopping at ``T_HALF`` and restarting from the latest checkpoint."""
    state, config, registered_variables = _collapsing_blob()
    config = config._replace(
        snapshot_storage_mode = TO_DISK,
        snapshot_storage_path = str(checkpoint_path),
        num_snapshots = 2,
    )
    initial = StateStruct(primitive_state=state, sinks=_empty_sinks())

    if not restart:
        params = SimulationParams(t_end = T_END, C_cfl = 0.4, gamma = GAMMA)
        return time_integration(initial, config, params, registered_variables), state

    params = SimulationParams(t_end = T_HALF, C_cfl = 0.4, gamma = GAMMA)
    time_integration(initial, config._replace(num_snapshots = 1), params, registered_variables)

    params = SimulationParams(t_end = T_END, C_cfl = 0.4, gamma = GAMMA)
    restored_state, params, restart_state = restart_from_latest_checkpoint(
        checkpoint_path, params
    )
    assert jnp.sum(restart_state.sinks.mass > 0) == 1
    final = time_integration(
        StateStruct(primitive_state=restored_state),
        config._replace(num_snapshots = 1),
        params,
        registered_variables,
        restart_state=restart_state,
    )
    return final, state


def test_single_sink_and_restart(tmp_path):
    """A collapsing blob forms a single sink, the total mass is conserved, and a
    run restarted after formation continues with the same sinks."""
    final, initial_state = _run(tmp_path / "uninterrupted", restart=False)
    restarted, _ = _run(tmp_path / "restarted", restart=True)

    assert jnp.sum(final.sinks.mass > 0) == 1

    initial_mass = jnp.sum(initial_state[0]) * VOLUME
    for result in (final, restarted):
        total_mass = jnp.sum(result.primitive_state[0]) * VOLUME + jnp.sum(result.sinks.mass)
        assert jnp.isclose(total_mass, initial_mass, rtol=1e-12)

    for name in SinkParticles._fields:
        assert jnp.array_equal(getattr(restarted.sinks, name), getattr(final.sinks, name)), name
