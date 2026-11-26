"""
Motion of sink particles and their gravity, following Gong & Ostriker (2013),
ApJS 204, 8.

Every gravitational interaction that involves a sink goes through the grid
(particle-mesh): each sink's mass is spread over the 3×3×3 cells around it with
triangular-shaped-cloud (TSC) weights (Gong & Ostriker 2013, Eq. 16), and the
force on the sink is read back from the same cells with the same weights. The
kicks solve the Poisson equation for gas and sink density together; the gas
feels the sinks through the potential of the sink density alone, solved once
per step and added to its own, which by linearity is the same field.
Depositing and reading back with the same weights gives a sink no force from
itself and makes the forces between any two masses equal and opposite. The
TSC weights and their first derivative are continuous in the sink position,
so derivatives with respect to the positions have no jumps.

The sinks move with a kick-drift-kick leapfrog (Gong & Ostriker 2013, Section
2.4), once per hydro step, and adaptive steps keep a sink from moving more than
half a cell per step (Federrath et al. 2010, ApJ 713, 269, Eq. 22).

A forming seed, of activation 0 < a < 1, moves in proportion to its
activation: it drifts by a v Δt, and after the accretion of each step its
velocity is tied to that of the gas of its TSC cells, with weight 1 − a. As
a → 0 the seed stops moving and its velocity tends to that of its gas; at
a = 1 both are the plain sink leapfrog.

The TSC cells follow the gas boundaries: they wrap around periodic axes and end
at the edge of the box on non-periodic ones. Where the Poisson solve treats the
box differently, a sink within reach of the edge (4 cells, 2 with VL2) feels
a force from itself, and the forces between masses are not exactly equal and
opposite. This happens with periodic gas and ``poisson_manual_open_boundaries``
and with mixed boundaries (solved as fully open).
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    FINITE_VOLUME,
    STATE_TYPE,
    VL2,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._gravity._gravity import (
    _compute_total_potential,
    _gravitational_acceleration,
    _second_order_gravitational_acceleration,
)
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _interior_cell_indices,
    _periodic_box,
)


#: The largest distance, in cells, that a sink may move in one adaptive step.
#: Federrath et al. (2010), Eq. 22, keep a sink from crossing more than half a
#: cell per step, so that it stays almost fixed relative to the grid while the
#: gas sees it at its mid-step position.
MAXIMUM_SINK_DISPLACEMENT_IN_CELLS = 0.5


def _tsc_offsets() -> np.ndarray:
    """
    Integer offsets (i, j, k) of the 27 cells around the cell that contains a
    sink, each in {−1, 0, +1}. They only depend on the stencil, so they are
    computed with NumPy at trace time.

    Returns:
        The offsets, shape (27, 3).
    """
    offset_range = np.arange(-1, 2)
    return np.stack(
        np.meshgrid(offset_range, offset_range, offset_range, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def _tsc_weights(
    position: jax.Array,
    grid_shape: tuple,
    config: SimulationConfig,
) -> tuple[tuple, jax.Array]:
    """
    The cells and the triangular-shaped-cloud (TSC) weights of the 27 cells
    around each sink (Gong & Ostriker 2013, Eq. 16).

    Along each axis, with Δx the cell size and d the offset of the sink from
    the centre of the cell that contains it,

        d = x / Δx − (⌊x / Δx⌋ + ½),
        W₋₁ = ½ (½ − d)²,   W₀ = ¾ − d²,   W₊₁ = ½ (½ + d)²,

    and the weight of a cell is the product of its three 1D weights. The
    containing cell comes from ``jnp.floor``, whose derivative is zero, so the
    derivative of a weight is that of d, 1 / Δx, times the polynomial's slope.
    Cells past a non-periodic edge of the box get weight zero.

    Args:
        position: The sink positions, shape (num_sinks, 3).
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        ``(cell_indices, weights)``: a tuple of three integer index arrays into
        the padded field, and the weights, each of shape (num_sinks, 27).
    """
    grid_spacing = config.grid_spacing
    offsets = _tsc_offsets()

    scaled_position = position / grid_spacing
    containing_cell = jnp.floor(scaled_position)
    offset_from_cell_center = scaled_position - (containing_cell + 0.5)

    # The 1D weights of the neighbours at −1, 0 and +1 along each axis, shape
    # (num_sinks, 3 axes, 3 neighbours).
    weights_along_axes = jnp.stack(
        [
            0.5 * (0.5 - offset_from_cell_center) ** 2,
            0.75 - offset_from_cell_center**2,
            0.5 * (0.5 + offset_from_cell_center) ** 2,
        ],
        axis=-1,
    )
    # For every one of the 27 cells, pick the 1D weight of its offset along
    # each axis and multiply them, shape (num_sinks, 27).
    weights = jnp.prod(
        weights_along_axes[:, np.arange(3), offsets + 1],
        axis=-1,
    )

    # The cell indices are counted from the first interior cell; along periodic
    # axes they wrap around the interior, and cells past a non-periodic edge
    # are flagged, since there is no gas there.
    cell_indices, _, is_real_cell = _interior_cell_indices(
        containing_cell.astype(jnp.int32)[:, None, :] + offsets[None, :, :],
        grid_shape,
        config,
    )
    weights = jnp.where(is_real_cell, weights, 0.0)

    return cell_indices, weights


def _sink_density(
    sink_particles: SinkParticles,
    grid_shape: tuple,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    The mass density of the sinks on the (padded) grid: each sink adds
    M W / ΔV to each of its 27 TSC cells.

    The ghost cells stay empty, so each sink's mass is counted once. Along a
    periodic axis the TSC cells wrap around the interior instead, reaching the
    real cells across the boundary. Empty slots have zero mass and add
    nothing.

    Args:
        sink_particles: The sink particles.
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        The sink mass density, with shape ``grid_shape``.
    """
    cell_indices, weights = _tsc_weights(
        sink_particles.position,
        grid_shape,
        config,
    )
    cell_volume = config.grid_spacing**3
    density_contributions = sink_particles.mass[:, None] * weights / cell_volume
    sink_density = jnp.zeros(grid_shape, dtype=sink_particles.mass.dtype)
    return sink_density.at[cell_indices].add(density_contributions)


def _sink_potential(
    sink_particles: SinkParticles,
    grid_shape: tuple,
    config: SimulationConfig,
    params: SimulationParams,
) -> FIELD_TYPE:
    """
    The gravitational potential of the sinks alone on the (padded) grid: the
    Poisson solve of ``_sink_density``, the same solve the gas density gets.
    The solve is linear, so adding this potential to that of the gas gives the
    potential of gas and sinks together.

    Args:
        sink_particles: The sink particles.
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        The potential of the sinks, with shape ``grid_shape``.
    """
    return _compute_gravitational_potential(
        _sink_density(sink_particles, grid_shape, config),
        config.grid_spacing,
        config,
        params.gravitational_constant,
    )


def _acceleration_field(
    gravitational_potential: FIELD_TYPE,
    config: SimulationConfig,
) -> jax.Array:
    """
    The gravitational acceleration −∇Φ at the cell centres, with the centred
    difference that the gas coupling of the active solver applies to the gas
    momentum.

    With the same difference on both sides, the force of a sink on the gas and
    the force of the gas on the sink are equal and opposite. Every scheme
    that takes gravity from the explicit sources (finite difference,
    ``MUSCL_HANCOCK`` and ``RK2_SSP``) uses the 6th-order difference
    (``_gravitational_acceleration``); VL2 uses the 2nd-order one
    (``_second_order_gravitational_acceleration``), −(Φᵢ₊₁ − Φᵢ₋₁) / (2Δx),
    which is the average ½(g₋½ + g₊½) of its face accelerations. Any centred
    difference is antisymmetric, so a sink feels no force from itself.

    Args:
        gravitational_potential: The total potential at the cell centres.
        config: The simulation configuration.

    Returns:
        The acceleration, shape (3, nx, ny, nz).
    """
    uses_vl2 = config.solver_mode == FINITE_VOLUME and config.solver_config.time_integrator == VL2
    if uses_vl2:
        acceleration_along_axis = _second_order_gravitational_acceleration
    else:
        acceleration_along_axis = _gravitational_acceleration
    return jnp.stack(
        [
            acceleration_along_axis(
                gravitational_potential,
                spatial_axis,
                config.grid_spacing,
            )
            for spatial_axis in range(3)
        ]
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _sink_acceleration(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> jax.Array:
    """
    The gravitational acceleration of every sink slot, from the potential of
    the gas and the sinks together, plus the external potential when it is on.

    One Poisson solve of the gas density plus the sink density gives the
    potential, the acceleration is taken at the cell centres with the stencil
    of the gas coupling (``_acceleration_field``), and each sink reads it back
    from its 27 TSC cells with the weights it was deposited with. Empty slots
    get the finite acceleration at their position, which the kick ignores.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The acceleration of every slot, shape (max_num_sinks, 3).
    """
    gas_density = primitive_state[registered_variables.density_index]
    grid_shape = gas_density.shape

    gravitational_potential = _compute_total_potential(
        gas_density + _sink_density(sink_particles, grid_shape, config),
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )
    acceleration_field = _acceleration_field(gravitational_potential, config)

    # The acceleration in the 27 cells of every sink, shape (3, num_sinks, 27),
    # summed with the TSC weights. An elementwise product and a sum, rather
    # than a matrix product, keep the full precision of the field: on GPUs a
    # float32 matrix product may run in reduced precision.
    cell_indices, weights = _tsc_weights(
        sink_particles.position,
        grid_shape,
        config,
    )
    acceleration_in_cells = acceleration_field[(slice(None),) + cell_indices]
    return jnp.sum(weights[None, :, :] * acceleration_in_cells, axis=-1).T


def _kick(
    sink_particles: SinkParticles,
    acceleration: jax.Array,
    time_step,
) -> SinkParticles:
    """
    Kick the sinks, v ← v + a Δt, in the active slots. Empty slots keep their
    zero velocity, whatever acceleration they are given.

    Args:
        sink_particles: The sink particles.
        acceleration: The acceleration of every slot, shape (max_num_sinks, 3).
        time_step: The time interval of the kick.

    Returns:
        The sink particles with the kicked velocities.
    """
    is_active = sink_particles.activation[:, None] > 0.0
    kicked_velocity = sink_particles.velocity + acceleration * time_step
    return sink_particles._replace(
        velocity=jnp.where(is_active, kicked_velocity, sink_particles.velocity),
    )


def _warn_sinks_left_box(has_left_box: np.ndarray, position: np.ndarray) -> None:
    """
    Host callback: print a warning for every sink that has just left the box
    through a non-periodic boundary.

    Args:
        has_left_box: Whether each slot has just left the box, shape
            (max_num_sinks,).
        position: The positions after the drift, shape (max_num_sinks, 3).
    """
    position = np.asarray(position)
    for slot in np.flatnonzero(np.asarray(has_left_box)):
        print(
            f"WARNING: sink particle {slot} left the box through a non-periodic "
            f"boundary, at position {position[slot]}. It keeps its slot; one "
            "cell past the edge its gravity no longer acts, and it moves in a "
            "straight line."
        )


def _drift(
    sink_particles: SinkParticles,
    time_step,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Drift the sinks, x ← x + a v Δt, in the active slots, and wrap them back
    into the box along periodic axes: a forming seed moves in proportion to
    its activation a, a sink (a = 1) by v Δt. Empty slots stay where they
    are.

    Along a non-periodic axis a sink may leave the box. It keeps its slot, and
    a warning is printed for every active slot that goes from inside to
    outside the box in this drift. The warning is printed from inside the
    step, so it is printed again when a step is retried and when the
    checkpointed loop recomputes a step in reverse mode.

    Args:
        sink_particles: The sink particles.
        time_step: The time interval of the drift.
        config: The simulation configuration.

    Returns:
        The sink particles with the drifted positions.
    """
    position = sink_particles.position
    is_periodic, box_length = _periodic_box(config)
    box_length = jnp.asarray(box_length, dtype=position.dtype)
    is_active = sink_particles.activation > 0.0

    drifted_position = (
        position
        + sink_particles.activation[:, None] * sink_particles.velocity * time_step
    )
    wrapped_position = jnp.mod(drifted_position, box_length)
    # A position just below zero can round to exactly L, the same point as 0.
    wrapped_position = jnp.where(
        wrapped_position >= box_length,
        wrapped_position - box_length,
        wrapped_position,
    )
    drifted_position = jnp.where(is_periodic, wrapped_position, drifted_position)
    new_position = jnp.where(is_active[:, None], drifted_position, position)

    def is_inside_box(slot_position):
        """Whether each slot lies inside the box along every non-periodic axis."""
        return jnp.all(
            is_periodic | ((slot_position >= 0.0) & (slot_position < box_length)),
            axis=-1,
        )

    has_left_box = is_active & is_inside_box(position) & ~is_inside_box(new_position)
    jax.lax.cond(
        jnp.any(has_left_box),
        lambda: jax.debug.callback(
            _warn_sinks_left_box,
            has_left_box,
            new_position,
        ),
        lambda: None,
    )

    return sink_particles._replace(position=new_position)


def _couple_to_gas(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles]:
    """
    Tie the velocity of every forming seed to the gas of its TSC cells,

        u ← u + (1 − a) (u_cm − u),
        u_cm = (m u + Σ W ρ v ΔV) / (m + Σ W ρ ΔV),

    with a the activation, m the mass and u the velocity of the seed, and W
    its TSC weights: u_cm is the centre-of-mass velocity of the seed and the
    gas of its cells, so a seed that holds more mass than that gas barely
    drags it. The momentum the seed gains is taken from the same cells in
    proportion to W ρ, which lowers the velocity of each by the same amount
    times W; momentum is conserved, and the kinetic energy the coupling
    removes is not given back to the gas.

    A sink (a = 1) and its cells are left exactly as they are, and so are
    the empty slots. As a → 0 the seed's mass goes to 0 with it, and its
    velocity tends to the mean velocity of its gas.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles)`` after the coupling.
    """
    cell_volume = config.grid_spacing**3
    density = primitive_state[registered_variables.density_index]
    velocity_rows = (
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
    )
    velocity = sink_particles.velocity
    activation = sink_particles.activation

    cell_indices, weights = _tsc_weights(sink_particles.position, density.shape, config)
    gas_mass = weights * density[cell_indices] * cell_volume
    gas_velocity = jnp.stack(
        [primitive_state[row][cell_indices] for row in velocity_rows],
        axis=-1,
    )
    total_gas_mass = jnp.sum(gas_mass, axis=-1)
    total_mass = sink_particles.mass + total_gas_mass

    # u_cm − u = Σ W ρ ΔV (v − u) / (m + Σ W ρ ΔV). The inner wheres keep the
    # divisions away from zero, so that neither gives a non-finite value, in
    # derivatives either.
    has_mass = total_mass > 0.0
    relative_center_of_mass_velocity = jnp.sum(
        gas_mass[..., None] * (gas_velocity - velocity[:, None, :]),
        axis=1,
    ) / jnp.where(has_mass, total_mass, 1.0)[:, None]
    velocity_change = jnp.where(
        ((activation > 0.0) & has_mass)[:, None],
        (1.0 - activation)[:, None] * relative_center_of_mass_velocity,
        0.0,
    )

    has_gas = total_gas_mass > 0.0
    gas_velocity_change = jnp.where(
        has_gas[:, None, None],
        -(sink_particles.mass[:, None] * velocity_change)[:, None, :]
        * weights[..., None]
        / jnp.where(has_gas, total_gas_mass, 1.0)[:, None, None],
        0.0,
    )
    gas_velocity_change = jnp.zeros(
        density.shape + (3,),
        dtype=density.dtype,
    ).at[cell_indices].add(gas_velocity_change)
    for axis, row in enumerate(velocity_rows):
        primitive_state = primitive_state.at[row].add(gas_velocity_change[..., axis])

    return primitive_state, sink_particles._replace(velocity=velocity + velocity_change)


def _sink_time_step(
    sink_particles: SinkParticles,
    config: SimulationConfig,
) -> jax.Array:
    """
    The largest time step in which no active slot moves more than
    ``MAXIMUM_SINK_DISPLACEMENT_IN_CELLS`` cells, at the largest speed of the
    active slots at the start of the step. The kicks change the speed within
    the step, so a sink can move a further ½ |a| Δt².

    Only adaptive runs apply this limit; fixed-step runs take the steps they
    are given.

    Args:
        sink_particles: The sink particles at the start of the step.
        config: The simulation configuration.

    Returns:
        The time step, or +∞ when no active slot moves.
    """
    is_active = sink_particles.activation > 0.0
    speed_squared = jnp.where(
        is_active,
        jnp.sum(sink_particles.velocity**2, axis=-1),
        0.0,
    )
    largest_speed_squared = jnp.max(speed_squared)
    is_moving = largest_speed_squared > 0.0

    # The inner where keeps the square root and the division away from zero,
    # so that neither gives a non-finite value, in derivatives either.
    largest_speed = jnp.sqrt(jnp.where(is_moving, largest_speed_squared, 1.0))
    return jnp.where(
        is_moving,
        MAXIMUM_SINK_DISPLACEMENT_IN_CELLS * config.grid_spacing / largest_speed,
        jnp.inf,
    )
