"""
Sink particle formation following Federrath et al. (2010), ApJ 713, 269.

Once per time step, the gas is checked for regions that should turn into sink
particles, using the creation checks of Section 2.2 of the paper. Only the
creation of sinks is handled here: the sinks do not move, do not accrete and
do not remove gas from the grid.

The formation runs in two stages. First, cheap checks are evaluated on the
whole grid, giving a mask of candidate cells. Then, for a fixed-size list of
candidates, the gas in the control volume around each candidate (all cells
within the accretion radius r_acc) is gathered and the remaining checks are
evaluated on it. Candidates that pass every check become new sinks.

The refinement check of Section 2.2.2 is not applied: astronomix uses a
uniform grid, so every cell is already on the highest level of refinement.
"""

# general
from functools import partial

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    IDEAL_GAS,
    PERIODIC_BOUNDARY,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._gravity._gravity import _compute_total_potential
from astronomix._stencil_operations._stencil_operations import _shift


def _empty_sink_particles(
    config: SimulationConfig,
    dtype,
) -> SinkParticles:
    """
    Create sink particle arrays with every slot empty (zero mass).

    Args:
        config: The simulation configuration; supplies the number of slots.
        dtype: The floating-point type of the arrays.

    Returns:
        Sink particles with ``max_num_sinks`` empty slots.
    """
    max_num_sinks = config.sink_particle_config.max_num_sinks
    return SinkParticles(
        mass=jnp.zeros((max_num_sinks,), dtype=dtype),
        position=jnp.zeros((max_num_sinks, 3), dtype=dtype),
        velocity=jnp.zeros((max_num_sinks, 3), dtype=dtype),
    )


def _control_volume_offsets(config: SimulationConfig) -> np.ndarray:
    """
    Integer cell offsets (i, j, k) of the control volume around a cell.

    The control volume holds every cell whose centre lies within the accretion
    radius of the central cell, i.e. i² + j² + k² ≤ (r_acc / Δx)² (Federrath et
    al. 2010, Eq. 3). The offsets are computed with NumPy at trace time, since
    they only depend on the static configuration.

    Args:
        config: The simulation configuration.

    Returns:
        The offsets, shape (num_offsets, 3), with (0, 0, 0) among them.
    """
    radius_in_cells = config.sink_particle_config.accretion_radius_in_cells
    max_offset = int(np.floor(radius_in_cells))
    offset_range = np.arange(-max_offset, max_offset + 1)
    offsets = np.stack(
        np.meshgrid(offset_range, offset_range, offset_range, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    inside_control_volume = np.sum(offsets**2, axis=-1) <= radius_in_cells**2
    return offsets[inside_control_volume]


def _sound_speed_squared(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """
    The squared sound speed in every cell.

    Args:
        primitive_state: The primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        c_s² = γP/ρ for an ideal gas, or the constant isothermal c_s².
    """
    density = primitive_state[registered_variables.density_index]
    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        return params.gamma * pressure / density
    return jnp.full_like(density, params.isothermal_sound_speed**2)


def _interior_mask(
    density: FIELD_TYPE,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Boolean mask that is True on the physical cells and False on ghost cells.

    Args:
        density: Any (padded) field, used for its shape.
        config: The simulation configuration.

    Returns:
        The interior mask, with the shape of ``density``.
    """
    num_ghost_cells = config.num_ghost_cells
    interior = jnp.zeros(density.shape, dtype=bool)
    interior_slices = tuple(
        slice(num_ghost_cells, size - num_ghost_cells) for size in density.shape
    )
    return interior.at[interior_slices].set(True)


def _converging_flow_mask(velocity: jax.Array) -> FIELD_TYPE:
    """
    Boolean mask of the cells toward which the flow converges along every axis.

    Along each axis d, the neighbour at +1 must not move away from the cell,
    v_d(+1) − v_d(0) ≤ 0, and the neighbour at −1 must not move away either,
    v_d(−1) − v_d(0) ≥ 0 (Federrath et al. 2010, Section 2.2.3). The
    inequalities are not strict, so that gas at rest or in uniform motion
    passes, as in Federrath's FLASH code; only flow away from the cell fails.

    Args:
        velocity: The (padded) velocity field, with the three components on
            the last axis.

    Returns:
        The converging-flow mask, with the shape of one field.
    """
    converging = jnp.ones(velocity.shape[:-1], dtype=bool)
    for axis in range(3):
        velocity_component = velocity[..., axis]
        # _shift(field, -1, axis) holds the value of the neighbour at +1.
        velocity_at_plus_one = _shift(velocity_component, -1, axis)
        velocity_at_minus_one = _shift(velocity_component, 1, axis)
        converging = (
            converging
            & (velocity_at_plus_one - velocity_component <= 0.0)
            & (velocity_at_minus_one - velocity_component >= 0.0)
        )
    return converging


def _potential_minimum_mask(
    gravitational_potential: FIELD_TYPE,
    offsets: np.ndarray,
) -> FIELD_TYPE:
    """
    Boolean mask of the cells where the potential is lowest in their control
    volume.

    A cell passes if its potential is not above the potential of any cell in
    its control volume, φ(0) ≤ min φ (Federrath et al. 2010, Eq. 4).

    Args:
        gravitational_potential: The (padded) gravitational potential.
        offsets: The control-volume offsets, shape (num_offsets, 3).

    Returns:
        The potential-minimum mask, with the shape of the potential.
    """
    minimum_in_volume = gravitational_potential
    for offset in offsets:
        # Shifting by −offset brings the value at cell + offset to each cell.
        potential_at_offset = gravitational_potential
        for axis in range(3):
            if offset[axis] != 0:
                potential_at_offset = _shift(
                    potential_at_offset,
                    -int(offset[axis]),
                    axis,
                )
        minimum_in_volume = jnp.minimum(minimum_in_volume, potential_at_offset)
    return gravitational_potential <= minimum_in_volume


def _periodic_box(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Which axes are periodic, and the box length along each axis.

    Both only depend on the static configuration, so they are computed with
    NumPy at trace time.

    Args:
        config: The simulation configuration.

    Returns:
        ``(is_periodic, box_length)``, two arrays of shape (3,).
    """
    boundary_settings = config.boundary_settings
    is_periodic = np.array(
        [
            axis_settings.left_boundary == PERIODIC_BOUNDARY
            and axis_settings.right_boundary == PERIODIC_BOUNDARY
            for axis_settings in (
                boundary_settings.x,
                boundary_settings.y,
                boundary_settings.z,
            )
        ]
    )
    box_length = np.array(
        [config.box_size.x, config.box_size.y, config.box_size.z]
    )
    return is_periodic, box_length


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _form_sink_particles(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> SinkParticles:
    """
    Create new sink particles where the gas passes all creation checks.

    A new sink gets the mass above the density threshold in its control volume,
    Σ (ρ − ρ_res) ΔV over the cells with ρ > ρ_res, and the centre of mass and
    centre-of-mass velocity (Eq. 12) of all gas in the control volume. The gas
    itself is left on the grid.

    Args:
        primitive_state: The (padded) primitive state after the hydro update.
        sink_particles: The current sink particles.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The sink particles, with any newly created sinks appended.
    """

    sink_particle_config = config.sink_particle_config
    grid_spacing = config.grid_spacing
    cell_volume = grid_spacing**3
    accretion_radius = sink_particle_config.accretion_radius_in_cells * grid_spacing

    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [
            primitive_state[registered_variables.velocity_index.x],
            primitive_state[registered_variables.velocity_index.y],
            primitive_state[registered_variables.velocity_index.z],
        ],
        axis=-1,
    )

    # The density threshold is the density at which the Jeans length equals
    # 2 r_acc, the smallest Jeans length the grid resolves (Federrath et al.
    # 2010, Eq. 32). It depends on the local sound speed, so it is evaluated
    # in every cell.
    sound_speed_squared = _sound_speed_squared(
        primitive_state,
        config,
        params,
        registered_variables,
    )
    density_threshold = (
        jnp.pi
        * sound_speed_squared
        / (4.0 * params.gravitational_constant * accretion_radius**2)
    )

    # The gravitational potential of the gas (plus any external potential),
    # from a Poisson solve on the updated density. The hydro update computes it
    # internally but does not return it, and for intermediate states only.
    gravitational_potential = _compute_total_potential(
        density,
        grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )
    offsets = _control_volume_offsets(config)

    # -------------------------------------------------------------
    # =============== ↓ Stage 1: grid-wide checks ↓ ===============
    # -------------------------------------------------------------

    # Density threshold check (Section 2.2.1), converging flow check
    # (Section 2.2.3) and gravitational potential minimum check (Section
    # 2.2.4). Ghost cells are excluded, as they only mirror physical cells or
    # hold boundary values.
    candidate_mask = (
        (density > density_threshold)
        & _converging_flow_mask(velocity)
        & _potential_minimum_mask(gravitational_potential, offsets)
        & _interior_mask(density, config)
    )

    # -------------------------------------------------------------
    # =============== ↑ Stage 1: grid-wide checks ↑ ===============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =========== ↓ Stage 2: control-volume checks ↓ ==============
    # -------------------------------------------------------------

    # JIT needs fixed array sizes, so the candidate cells are collected into a
    # list of fixed length; unused entries are marked as invalid.
    max_num_candidates = sink_particle_config.max_num_candidates
    num_candidates_found = jnp.sum(candidate_mask)
    candidate_indices = jnp.stack(
        jnp.nonzero(candidate_mask, size=max_num_candidates, fill_value=0),
        axis=-1,
    )
    candidate_is_valid = jnp.arange(max_num_candidates) < num_candidates_found

    # Cell indices of every candidate's control volume. The indices wrap
    # around the grid, which is the right neighbourhood for periodic
    # boundaries and never happens for ghost-cell boundaries, where the ghost
    # layer is wider than the control volume.
    grid_shape = np.array(density.shape)
    control_volume_indices = tuple(
        (candidate_indices[:, None, axis] + offsets[None, :, axis]) % grid_shape[axis]
        for axis in range(3)
    )

    # Fields over the control volumes, shape (num_candidates, num_offsets),
    # with a trailing vector axis for the velocity.
    density_in_volume = density[control_volume_indices]
    density_threshold_in_volume = density_threshold[control_volume_indices]
    velocity_in_volume = velocity[control_volume_indices]
    cell_mass_in_volume = density_in_volume * cell_volume
    gas_mass_in_volume = jnp.sum(cell_mass_in_volume, axis=-1)

    # Centre of mass, measured from the candidate cell with the (unwrapped)
    # offsets so that a control volume crossing a periodic boundary stays in
    # one piece, and the centre-of-mass velocity (Eq. 12).
    candidate_positions = (candidate_indices - config.num_ghost_cells + 0.5) * grid_spacing
    offset_positions = offsets * grid_spacing
    center_of_mass = candidate_positions + jnp.einsum(
        "ck,kd->cd",
        cell_mass_in_volume,
        offset_positions,
    ) / gas_mass_in_volume[:, None]
    center_of_mass_velocity = jnp.einsum(
        "ck,ckd->cd",
        cell_mass_in_volume,
        velocity_in_volume,
    ) / gas_mass_in_volume[:, None]

    # Next to a periodic boundary, the centre of mass can come out just past
    # the edge of the box; it is wrapped back into the box along periodic axes.
    is_periodic, box_length = _periodic_box(config)
    center_of_mass = jnp.where(
        is_periodic,
        jnp.mod(center_of_mass, box_length),
        center_of_mass,
    )

    # Mass of a new sink: the gas above the density threshold in its control
    # volume, i.e. the mass the paper's accretion step would transfer.
    mass_above_threshold = jnp.sum(
        jnp.maximum(density_in_volume - density_threshold_in_volume, 0.0) * cell_volume,
        axis=-1,
    )

    # Proximity check (Section 2.2.7): no new sink within r_acc of an existing
    # sink. Along periodic axes the separation is taken to the nearest
    # periodic copy of the sink (minimum-image convention), as in Federrath's
    # FLASH code. Only filled slots are compared: empty slots have zero mass
    # and sit at the origin, where they would otherwise block candidates.
    separation = candidate_positions[:, None, :] - sink_particles.position[None, :, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    distance_to_sink = jnp.linalg.norm(separation, axis=-1)
    sink_is_filled = sink_particles.mass > 0.0
    sink_within_accretion_radius = jnp.logical_and(
        distance_to_sink <= accretion_radius,
        sink_is_filled[None, :],
    )
    far_from_existing_sinks = jnp.logical_not(
        jnp.any(sink_within_accretion_radius, axis=-1)
    )

    candidate_passes = jnp.logical_and(
        candidate_is_valid,
        far_from_existing_sinks,
    )

    # -------------------------------------------------------------
    # =========== ↑ Stage 2: control-volume checks ↑ ==============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Appending new sinks ↓ ===================
    # -------------------------------------------------------------

    # Sinks are only appended, so the filled slots are contiguous at the
    # front. Each passing candidate gets the next free slot; failing
    # candidates get the out-of-range slot max_num_sinks. With mode="drop",
    # writes to an out-of-range slot are discarded, which also discards new
    # sinks once every slot is taken.
    max_num_sinks = sink_particle_config.max_num_sinks
    num_existing_sinks = jnp.sum(sink_particles.mass > 0.0)
    slot = jnp.where(
        candidate_passes,
        num_existing_sinks + jnp.cumsum(candidate_passes) - 1,
        max_num_sinks,
    )

    sink_particles = SinkParticles(
        mass=sink_particles.mass.at[slot].set(mass_above_threshold, mode="drop"),
        position=sink_particles.position.at[slot].set(center_of_mass, mode="drop"),
        velocity=sink_particles.velocity.at[slot].set(
            center_of_mass_velocity,
            mode="drop",
        ),
    )

    # Warn when a limit set by the configuration cut something off.
    num_new_sinks = jnp.sum(candidate_passes)
    jax.lax.cond(
        num_candidates_found > max_num_candidates,
        lambda: jax.debug.print(
            "WARNING: {} sink particle candidates found, only the first {} "
            "are checked this step (increase max_num_candidates).",
            num_candidates_found,
            max_num_candidates,
        ),
        lambda: None,
    )
    jax.lax.cond(
        num_existing_sinks + num_new_sinks > max_num_sinks,
        lambda: jax.debug.print(
            "WARNING: {} new sink particles discarded, all {} slots are taken "
            "(increase max_num_sinks).",
            num_existing_sinks + num_new_sinks - max_num_sinks,
            max_num_sinks,
        ),
        lambda: None,
    )

    # -------------------------------------------------------------
    # ================= ↑ Appending new sinks ↑ ===================
    # -------------------------------------------------------------

    return sink_particles
