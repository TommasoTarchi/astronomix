"""
Sink particle dynamics operators pytest (float64).

Checks the operators of ``_sink_particle_dynamics.py`` one by one: the sink
density of the triangular-shaped-cloud (TSC) mass assignment (total mass, ghost
cells, empty slots, and weights whose derivative is continuous across a cell
face), the zero self-force and the equal and opposite forces of the
particle-mesh gravity, the force of an external potential read back with the
6th-order (finite difference) and the 2nd-order (classic finite volume, VL2)
stencil, and the kick and the drift. Three runs of the time loop with
different step sizes check that the kick-drift-kick integration of two sinks
on a circular orbit converges at second order.

Run on the CPU with::

    JAX_PLATFORMS=cpu python -m pytest pytests/sink_particles/test_sink_dynamics_operators.py
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") == "cpu":
    # XLA:CPU compiles the unrolled step in seconds at optimisation level 0
    # (minutes and tens of GB otherwise); the numerics differ at round-off only.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true",
    )
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
    SinkParticles,
)
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._sink_particles._sink_particle_dynamics import (
    _drift,
    _kick,
    _sink_acceleration,
    _sink_density,
    _tsc_weights,
)
from astronomix._modules._sink_particles._sink_particle_formation import (
    _empty_sink_particles,
)
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    SinkFormationSettings,
    setup_sink_formation,
)
from astronomix.time_stepping.time_integration import _prepare_padded_state


NUM_CELLS = 16
CELL_SIZE = 1.0 / NUM_CELLS
MAX_NUM_SINKS = 4

PERIODIC_AXIS = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
OPEN_AXIS = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
PERIODIC = BoundarySettings(x=PERIODIC_AXIS, y=PERIODIC_AXIS, z=PERIODIC_AXIS)
OPEN = BoundarySettings(x=OPEN_AXIS, y=OPEN_AXIS, z=OPEN_AXIS)


@pytest.fixture(autouse=True, scope="module")
def float64():
    """Run every test of this module in float64, without switching float64 on
    for the float32 tests collected in the same session."""
    with jax.enable_x64(True):
        yield


# -------------------------------------------------------------
# ======================= ↓ Helpers ↓ =========================
# -------------------------------------------------------------


def _setup(boundary_settings, gravity_config=GravityConfig(self_gravity=True), **config_options):
    """A 16³ unit box of uniform gas at rest, with sink particles on.

    Args:
        boundary_settings: The boundaries of the box.
        gravity_config: The gravity configuration; self-gravity is always on.
        **config_options: Further ``SimulationConfig`` fields, e.g. the solver.

    Returns:
        ``(primitive_state, config, params, registered_variables)``, with the
        primitive state padded and its ghost cells filled, as in the time loop.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        gravity_config=gravity_config,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
        **config_options,
    )
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SinkFormationSettings(
            peak_overdensity=0.0,
            boundary_settings=boundary_settings,
        ),
    )
    registered_variables = get_registered_variables(config)
    primitive_state = _prepare_padded_state(
        state.primitive_state,
        config,
        params,
        registered_variables,
    )
    return primitive_state, config, params, registered_variables


def _sinks(masses, positions, config, velocities=None):
    """Sink particles with the given sinks in the first slots and the other
    slots empty.

    Args:
        masses: The masses of the sinks, shape (N,).
        positions: Their positions, shape (N, 3).
        config: The simulation configuration.
        velocities: Their velocities, shape (N, 3); zero by default.

    Returns:
        Sink particles with ``MAX_NUM_SINKS`` slots.
    """
    num_sinks = len(masses)
    if velocities is None:
        velocities = np.zeros((num_sinks, 3))
    sink_particles = _empty_sink_particles(config, jnp.float64)
    return SinkParticles(
        mass=sink_particles.mass.at[:num_sinks].set(jnp.asarray(masses)),
        position=sink_particles.position.at[:num_sinks].set(jnp.asarray(positions)),
        velocity=sink_particles.velocity.at[:num_sinks].set(jnp.asarray(velocities)),
    )


def _in_cells(*cell_coordinates):
    """A position given in units of the cell size."""
    return np.array(cell_coordinates) * CELL_SIZE


def _weight_derivatives_by_cell(position, grid_shape, config):
    """The derivative of every TSC weight of one sink with respect to its
    position, keyed by the cell the weight belongs to.

    The weights are stored by their offset from the cell that contains the
    sink, and that cell changes when the sink crosses a face, so derivatives
    on the two sides of a face can only be compared cell by cell.

    Args:
        position: The position of the sink, shape (3,).
        grid_shape: The shape of a padded field.
        config: The simulation configuration.

    Returns:
        A dict from the padded cell index (i, j, k) to the derivative of that
        cell's weight with respect to the position, shape (3,).
    """

    def weights(sink_position):
        _, cell_weights = _tsc_weights(sink_position[None, :], grid_shape, config)
        return cell_weights[0]

    cell_indices, _ = _tsc_weights(position[None, :], grid_shape, config)
    cells = zip(*(np.asarray(index[0]) for index in cell_indices))
    derivatives = np.asarray(jax.jacfwd(weights)(position))
    return {tuple(int(i) for i in cell): derivative for cell, derivative in zip(cells, derivatives)}


def _orbit_setup(num_timesteps):
    """Two unit sinks 8 cells apart on their Newtonian circular orbit in a 16³
    open box of thin hot gas, with fixed steps over a quarter orbit.

    Args:
        num_timesteps: The number of fixed steps.

    Returns:
        ``(state, config, params, registered_variables)``.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        fixed_timestep=True,
        num_timesteps=num_timesteps,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
    )
    # Thin hot gas: its mass does not change the orbit, and its pressure keeps
    # it from collapsing onto the sinks.
    separation = 8 * CELL_SIZE
    speed = np.sqrt(1.0 / (2.0 * separation))
    period = 2.0 * np.pi * (0.5 * separation) / speed
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SinkFormationSettings(
            background_density=1e-9,
            peak_overdensity=0.0,
            sound_speed_squared=11.0,
            t_end=0.25 * period,
            boundary_settings=OPEN,
        ),
    )
    # The centre of mass sits off the cell corners and faces, so that no
    # symmetry of the grid hides an error.
    center = 0.5 + _in_cells(0.3, 0.2, 0.1)
    offset = np.array([0.5 * separation, 0.0, 0.0])
    sink_particles = _sinks(
        [1.0, 1.0],
        [center - offset, center + offset],
        config,
        velocities=[[0.0, -speed, 0.0], [0.0, speed, 0.0]],
    )
    state = state._replace(sink_particles=sink_particles)
    return state, config, params, get_registered_variables(config)


# -------------------------------------------------------------
# ======================= ↑ Helpers ↑ =========================
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary_settings",
    [PERIODIC, OPEN],
    ids=["periodic", "open"],
)
def test_sink_density(boundary_settings):
    """The sink density holds the sink masses, leaves the ghost cells empty and
    ignores empty slots, for sinks at a cell centre, on a cell face and, in the
    periodic box, across the boundary; the derivative of every TSC weight is
    continuous across a cell face.

    Args:
        boundary_settings: The boundaries of the box.
    """
    primitive_state, config, _, _ = _setup(boundary_settings)
    grid_shape = primitive_state.shape[1:]
    is_periodic = boundary_settings == PERIODIC

    # In the open box every sink stays at least 2 cells from the edge, so that
    # all of its mass lands inside the box.
    if is_periodic:
        third_position = _in_cells(0.2, 15.9, 3.6)
    else:
        third_position = _in_cells(2.2, 13.6, 3.4)
    masses = [0.7, 1.3, 0.4]
    positions = [_in_cells(7.5, 8.5, 6.5), _in_cells(9.0, 5.3, 10.7), third_position]
    sink_particles = _sinks(masses, positions, config)
    sink_density = _sink_density(sink_particles, grid_shape, config)

    total_mass = float(jnp.sum(sink_density)) * CELL_SIZE**3
    assert abs(total_mass - sum(masses)) <= 1e-12 * sum(masses), (
        f"the sink density holds a mass of {total_mass!r}, but the sinks have {sum(masses)!r}"
    )

    num_ghost_cells = config.num_ghost_cells
    if num_ghost_cells > 0:
        interior = np.zeros(grid_shape, dtype=bool)
        interior[(slice(num_ghost_cells, -num_ghost_cells),) * 3] = True
        ghost_cell_density = np.asarray(sink_density)[~interior]
        assert np.all(ghost_cell_density == 0.0), (
            f"the sink density is non-zero in {np.count_nonzero(ghost_cell_density)} ghost cells"
        )

    # An empty slot placed anywhere, and moving, adds nothing.
    moved_empty_slot = sink_particles._replace(
        position=sink_particles.position.at[-1].set(jnp.asarray(_in_cells(5.5, 5.5, 5.5))),
        velocity=sink_particles.velocity.at[-1].set(1.0),
    )
    assert jnp.array_equal(
        _sink_density(moved_empty_slot, grid_shape, config),
        sink_density,
    ), "moving an empty slot changed the sink density"

    # Across the face between two cells, along each axis, the derivative of
    # each cell's weight changes only by the second derivative of the weights
    # (at most 2 / Δx²) times the distance between the two points.
    for axis in range(3):
        face_position = _in_cells(7.3, 8.6, 6.4)
        face_position[axis] = 8.0 * CELL_SIZE
        derivatives = {}
        for side in (-1.0, 1.0):
            position = face_position.copy()
            position[axis] += side * 1e-4 * CELL_SIZE
            derivatives[side] = _weight_derivatives_by_cell(
                jnp.asarray(position),
                grid_shape,
                config,
            )
        for cell in derivatives[-1.0].keys() | derivatives[1.0].keys():
            left = derivatives[-1.0].get(cell, np.zeros(3))
            right = derivatives[1.0].get(cell, np.zeros(3))
            jump = np.max(np.abs(left - right))
            assert jump <= 1e-3 / CELL_SIZE, (
                f"the derivative of the TSC weight of cell {cell} jumps by {jump:.3e} across "
                f"the cell face along axis {axis}, more than 1e-3 / Δx = {1e-3 / CELL_SIZE:.3e}"
            )


@pytest.mark.parametrize(
    "boundary_settings",
    [PERIODIC, OPEN],
    ids=["periodic", "open"],
)
def test_force_balance(boundary_settings):
    """Without gas, a lone sink feels no force from itself, and the forces
    between three sinks of different masses add up to zero.

    Args:
        boundary_settings: The boundaries of the box.
    """
    primitive_state, config, params, registered_variables = _setup(boundary_settings)
    primitive_state = primitive_state.at[registered_variables.density_index].set(0.0)
    gravitational_constant = params.gravitational_constant

    lone_sink = _sinks([1.0], [_in_cells(6.3, 9.6, 7.2)], config)
    acceleration = _sink_acceleration(
        primitive_state,
        lone_sink,
        config,
        params,
        registered_variables,
    )
    force_scale = gravitational_constant * 1.0 / (10.0 * CELL_SIZE) ** 2
    self_acceleration = float(jnp.linalg.norm(acceleration[0]))
    assert self_acceleration <= 1e-10 * force_scale, (
        f"a lone sink feels an acceleration of {self_acceleration:.3e} from itself, more than "
        f"1e-10 of G M / (10 Δx)² = {float(force_scale):.3e}"
    )

    masses = jnp.array([1.0, 0.4, 2.3])
    three_sinks = _sinks(
        masses,
        [_in_cells(5.1, 8.3, 7.5), _in_cells(9.4, 5.8, 9.7), _in_cells(7.9, 10.7, 6.2)],
        config,
    )
    acceleration = _sink_acceleration(
        primitive_state,
        three_sinks,
        config,
        params,
        registered_variables,
    )
    forces = masses[:, None] * acceleration[:3]
    largest_force = float(jnp.max(jnp.linalg.norm(forces, axis=-1)))
    net_force = float(jnp.linalg.norm(jnp.sum(forces, axis=0)))
    assert net_force <= 1e-10 * largest_force, (
        f"the forces between the three sinks add up to {net_force:.3e}, more than 1e-10 of "
        f"the largest force {largest_force:.3e}"
    )


@pytest.mark.parametrize(
    "solver_options",
    [
        dict(),
        dict(solver_mode=FINITE_VOLUME),
        dict(solver_mode=FINITE_VOLUME, time_integrator=VL2),
    ],
    ids=["finite_difference", "classic_finite_volume", "vl2"],
)
def test_external_potential_on_sink(solver_options):
    """In uniform gas, a sink feels exactly the force of a quadratic external
    potential, with the 6th-order read-back of the finite-difference path and
    the 2nd-order one of classic finite volume and VL2: a centred difference of
    either order is exact for a quadratic, and the TSC reading of a linear
    field is exact.

    Args:
        solver_options: The ``SimulationConfig`` fields that select the solver.
    """
    primitive_state, config, params, registered_variables = _setup(
        PERIODIC,
        gravity_config=GravityConfig(self_gravity=True, external_potential=True),
        **solver_options,
    )
    center = jnp.full((3,), 0.5)
    omega_squared = 3.0
    cell_centers = get_helper_data(config).geometric_centers
    params = params._replace(
        gravitational_potential=0.5
        * omega_squared
        * jnp.sum((cell_centers - center) ** 2, axis=-1)
    )

    # At least 5 cells from the edges: the 6th-order stencil plus the TSC
    # weights reach 4 cells, and the quadratic is not periodic.
    position = jnp.array([0.37, 0.58, 0.63])
    sink_particles = _sinks([0.1], [position], config)
    acceleration = _sink_acceleration(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )[0]

    expected_acceleration = -omega_squared * (position - center)
    error = float(jnp.linalg.norm(acceleration - expected_acceleration))
    assert error <= 1e-10 * float(jnp.linalg.norm(expected_acceleration)), (
        f"the sink acceleration {np.asarray(acceleration)} differs from −ω² (x − c) = "
        f"{np.asarray(expected_acceleration)} by {error:.3e}, more than 1e-10 relative"
    )


def test_kick_and_drift_skip_empty_slots():
    """A kick and a drift move the filled slots only, even with an
    acceleration in every slot. A sink crossing the periodic boundary along x
    is wrapped back into the box; a sink leaving through the open boundary
    along y is not."""
    mixed_boundaries = BoundarySettings(x=PERIODIC_AXIS, y=OPEN_AXIS, z=OPEN_AXIS)
    _, config, _, _ = _setup(mixed_boundaries)
    time_step = 0.05

    masses = [0.5, 0.8]
    positions = np.array([[0.95, 0.5, 0.5], [0.4, 0.97, 0.3]])
    velocities = np.array([[2.0, 0.3, -0.1], [0.1, 1.5, 0.2]])
    sink_particles = _sinks(masses, positions, config, velocities=velocities)
    acceleration = 1.0 + jnp.arange(3.0 * MAX_NUM_SINKS).reshape(MAX_NUM_SINKS, 3)

    kicked = _kick(sink_particles, acceleration, time_step)
    np.testing.assert_allclose(
        kicked.velocity[:2],
        velocities + np.asarray(acceleration[:2]) * time_step,
        rtol=1e-14,
        err_msg="the kicked velocities of the filled slots are not v + a Δt",
    )
    assert jnp.array_equal(kicked.position, sink_particles.position), "the kick moved the sinks"
    assert jnp.array_equal(kicked.mass, sink_particles.mass), "the kick changed the sink masses"
    assert jnp.all(kicked.velocity[2:] == 0.0), (
        f"the kick changed the velocities of the empty slots: {kicked.velocity[2:]}"
    )

    drifted = _drift(sink_particles, time_step, config)
    expected_positions = positions + velocities * time_step
    # x = 1.05 lies past the periodic boundary and is wrapped to 0.05; y =
    # 1.045 lies past the open boundary and stays there.
    expected_positions[0, 0] -= 1.0
    np.testing.assert_allclose(
        drifted.position[:2],
        expected_positions,
        rtol=1e-14,
        atol=1e-15,
        err_msg=(
            "the drifted positions of the filled slots are not x + v Δt, wrapped along the "
            "periodic x axis and not along the open y axis"
        ),
    )
    assert jnp.array_equal(drifted.velocity, sink_particles.velocity), (
        "the drift changed the sink velocities"
    )
    assert jnp.array_equal(drifted.mass, sink_particles.mass), "the drift changed the sink masses"
    assert jnp.all(drifted.position[2:] == 0.0), (
        f"the drift moved the empty slots: {drifted.position[2:]}"
    )


def test_kdk_second_order():
    """Two sinks on a circular orbit, integrated over a quarter orbit with
    fixed steps Δt, Δt/2 and Δt/4: the differences between the final
    positions shrink as Δt², so their ratio is 5 (a first-order scheme gives
    3)."""
    final_positions = []
    for num_timesteps in (128, 256, 512):
        state, config, params, registered_variables = _orbit_setup(num_timesteps)
        final_state = time_integration(state, config, params, registered_variables)
        final_positions.append(np.asarray(final_state.sink_particles.position[:2]))

    coarse, medium, fine = final_positions
    ratio = np.linalg.norm(coarse - fine) / np.linalg.norm(medium - fine)
    assert 4.0 <= ratio <= 6.0, (
        f"the convergence ratio is {ratio:.3f}, outside [4, 6] (second order gives 5, "
        "first order 3)"
    )
