"""
Sink particle formation following Federrath et al. (2010), ApJ 713, 269, with
smooth checks (``_sink_particle_smoothing.py``).

Once per time step, two things happen:

1. Cells that pass relaxed creation checks (Section 2.2), with the density,
   Jeans and bound thresholds lowered by the creation margin μ and the
   converging-flow and potential checks widened by ω / μ, become candidates:
   occupied slots with activation 0 and no mass, at the centre of their cell.
   This step is hard, but it changes neither the gas nor the dynamics.
2. Every occupied slot gets a creation weight w in [0, 1], the product of one
   smooth factor per creation check, evaluated around its current position,
   and its activation grows to max(a, w). A candidate with activation 1 is a
   sink.

With every smoothing parameter at its hard limit, a cell that passes the hard
checks becomes a candidate whose weight is 1 at once, so a sink forms where
the hard checks pass.

The creation runs in two stages. First, cheap checks are evaluated on the
whole grid, giving a mask of candidate cells. Then, for a fixed-size list of
candidates, the gas in the control volume around each candidate (all cells
within the accretion radius r_acc) is gathered and the remaining checks are
evaluated on it.

Only gas in interior cells counts, as in the accretion step. A control volume
wraps around periodic axes and ends at the edge of the box on non-periodic
axes. Reflective walls are treated like the other edges: mirrored gas is not
counted.

The refinement check of Section 2.2.2 is not applied: astronomix uses a
uniform grid, so every cell is already on the highest level of refinement.
"""

# typing
from types import NoneType
from typing import NamedTuple, Union

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    GHOST_CELLS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._geometry.boundaries import _axis_slice, _periodic_boundaries
from astronomix._modules._gravity._gravity import _compute_total_potential
from astronomix._modules._sink_particles._sink_particle_dynamics import _sink_density
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _accretion_offsets,
    _collect_candidates,
    _control_volume_energies,
    _control_volume_offsets,
    _density_threshold,
    _interior_cell_indices,
    _interior_mask,
    _periodic_box,
    _raise_sink_slots_full,
    _select_separated_candidates,
    _sound_speed_squared,
)
from astronomix._modules._sink_particles._sink_particle_smoothing import (
    _cell_weights,
    _radial_weight,
    _safe_divide,
    _smooth_step,
    _soft_maximum,
    _soft_minimum,
)
from astronomix._stencil_operations._stencil_operations import _shift


# -------------------------------------------------------------
# ===================== ↓ Cell quantities ↓ ===================
# -------------------------------------------------------------


class CellQuantities(NamedTuple):
    """The cell quantities that the creation and accretion checks read, on
    the (padded) grid."""

    #: The gas density ρ.
    density: FIELD_TYPE

    #: The gas velocity, with the three components on the last axis.
    velocity: jax.Array

    #: The squared sound speed c_s².
    sound_speed_squared: FIELD_TYPE

    #: The density threshold ρ_thr of Eq. 32.
    density_threshold: FIELD_TYPE

    #: The potential of the gas, of the occupied slots (their masses spread
    #: with the TSC weights of their motion) and the external potential.
    potential: FIELD_TYPE

    #: The six velocity differences u_k of the converging-flow check, in units
    #: of c_s and positive where the flow converges, shape (6, ...).
    converging_velocity: jax.Array

    #: |B|², or ``None`` without a magnetic field.
    magnetic_field_squared: Union[FIELD_TYPE, NoneType]


def _cell_quantities(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> CellQuantities:
    """
    The cell quantities of the creation and accretion checks, with one
    Poisson solve of the gas density plus the slots' masses.

    Along each axis d, the converging-flow check compares the cell with its
    neighbours at ±1: u = (v_d(0) − v_d(+1)) / c_s and (v_d(−1) − v_d(0)) / c_s,
    both ≥ 0 where neither neighbour moves away from the cell (Federrath et
    al. 2010, Section 2.2.3). It reads the velocity of the ghost cells, since
    these ±1 neighbours are what the hydro solver sees at the boundary.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles, whose masses enter the potential.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The cell quantities.
    """
    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [
            primitive_state[registered_variables.velocity_index.x],
            primitive_state[registered_variables.velocity_index.y],
            primitive_state[registered_variables.velocity_index.z],
        ],
        axis=-1,
    )
    sound_speed_squared = _sound_speed_squared(
        primitive_state,
        config,
        params,
        registered_variables,
    )
    potential = _compute_total_potential(
        density + _sink_density(sink_particles, density.shape, config),
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )

    sound_speed = jnp.sqrt(sound_speed_squared)
    converging_velocity = []
    for axis in range(3):
        velocity_component = velocity[..., axis]
        # _shift(field, -1, axis) holds the value of the neighbour at +1.
        velocity_at_plus_one = _shift(velocity_component, -1, axis)
        velocity_at_minus_one = _shift(velocity_component, 1, axis)
        converging_velocity.append(
            (velocity_component - velocity_at_plus_one) / sound_speed
        )
        converging_velocity.append(
            (velocity_at_minus_one - velocity_component) / sound_speed
        )

    magnetic_field_squared = None
    if config.mhd:
        magnetic_field_squared = (
            primitive_state[registered_variables.magnetic_index.x] ** 2
            + primitive_state[registered_variables.magnetic_index.y] ** 2
            + primitive_state[registered_variables.magnetic_index.z] ** 2
        )

    return CellQuantities(
        density=density,
        velocity=velocity,
        sound_speed_squared=sound_speed_squared,
        density_threshold=_density_threshold(sound_speed_squared, config, params),
        potential=potential,
        converging_velocity=jnp.stack(converging_velocity),
        magnetic_field_squared=magnetic_field_squared,
    )


# -------------------------------------------------------------
# ===================== ↑ Cell quantities ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Cells around a slot ↓ =================
# -------------------------------------------------------------


class SlotNeighbourhood(NamedTuple):
    """The cells around each of a set of positions: those of the accretion
    offsets around the cell that contains it, which hold every cell within
    r_acc. Arrays have the shape (num_positions, num_offsets)."""

    #: A tuple of three integer index arrays into the padded field.
    cell_indices: tuple

    #: Whether each cell is a real cell; cells past a non-periodic edge are
    #: not, and every weight is 0 on them.
    is_real_cell: jax.Array

    #: The separation from the position to each cell centre, to the nearest
    #: periodic copy along periodic axes, with a trailing vector axis.
    separation: jax.Array

    #: The squared length of the separation.
    distance_squared: jax.Array

    #: The radial weight K of each cell.
    radial_weight: jax.Array

    #: The cell weight h of each cell.
    cell_weight: jax.Array


def _slot_neighbourhood(
    position: jax.Array,
    grid_shape: tuple,
    config: SimulationConfig,
) -> SlotNeighbourhood:
    """
    The cells around each position, with their separations and weights.

    Args:
        position: The positions, shape (num_positions, 3).
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        The neighbourhood of each position.
    """
    sink_particle_config = config.sink_particle_config
    grid_spacing = config.grid_spacing
    accretion_radius = sink_particle_config.accretion_radius_in_cells * grid_spacing
    spatial_width_in_cells = sink_particle_config.spatial_width_in_cells
    offsets = _accretion_offsets(config)
    is_periodic, box_length = _periodic_box(config)

    containing_cell = jnp.floor(position / grid_spacing).astype(jnp.int32)
    cell_indices, interior_index, is_real_cell = _interior_cell_indices(
        containing_cell[:, None, :] + offsets[None, :, :],
        grid_shape,
        config,
    )

    cell_centers = (interior_index + 0.5) * grid_spacing
    separation = cell_centers - position[:, None, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    distance_squared = jnp.sum(separation**2, axis=-1)

    radial_weight = jnp.where(
        is_real_cell,
        _radial_weight(
            distance_squared,
            accretion_radius,
            spatial_width_in_cells * grid_spacing,
        ),
        0.0,
    )
    _, cell_weight = _cell_weights(
        position,
        grid_shape,
        config,
        spatial_width_in_cells,
        offsets,
    )

    return SlotNeighbourhood(
        cell_indices=cell_indices,
        is_real_cell=is_real_cell,
        separation=separation,
        distance_squared=distance_squared,
        radial_weight=radial_weight,
        cell_weight=cell_weight,
    )


def _near_slots_mask(
    neighbourhood: SlotNeighbourhood,
    occupied: jax.Array,
    grid_shape: tuple,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Boolean mask of the real cells within r_acc of an occupied slot.

    Args:
        neighbourhood: The neighbourhood of every slot.
        occupied: Whether each slot is occupied, shape (num_slots,).
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        The mask, with shape ``grid_shape``.
    """
    accretion_radius = (
        config.sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    )
    is_near = (
        occupied[:, None]
        & neighbourhood.is_real_cell
        & (neighbourhood.distance_squared <= accretion_radius**2)
    )
    return jnp.zeros(grid_shape, dtype=bool).at[neighbourhood.cell_indices].max(is_near)


def _slot_separation_squared(
    position: jax.Array,
    config: SimulationConfig,
) -> jax.Array:
    """
    The squared distances between slots, to the nearest periodic copy along
    periodic axes.

    Args:
        position: The slot positions, shape (num_slots, 3).
        config: The simulation configuration.

    Returns:
        The squared distances, shape (num_slots, num_slots).
    """
    is_periodic, box_length = _periodic_box(config)
    separation = position[:, None, :] - position[None, :, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    return jnp.sum(separation**2, axis=-1)


# -------------------------------------------------------------
# =================== ↑ Cells around a slot ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Checks at a position ↓ ================
# -------------------------------------------------------------


class SlotChecks(NamedTuple):
    """The quantities of the creation checks at a set of positions, each of
    shape (num_positions,) unless stated otherwise."""

    #: Whether any cell weight h is above 0; a position past a non-periodic
    #: edge has no cell to read its values from.
    has_cells: jax.Array

    #: The density threshold ρ_thr read at the position.
    density_threshold: jax.Array

    #: ρ / ρ_thr, with the slot's own mass added to its density.
    density_ratio: jax.Array

    #: The six converging-flow velocities u_k, shape (num_positions, 6).
    converging_velocity: jax.Array

    #: (φ − min_T φ) / c_s², the height above the potential minimum of the
    #: control volume.
    potential_height: jax.Array

    #: |E_grav| / (2 E_th + E_mag).
    jeans_ratio: jax.Array

    #: |E_grav| / (E_th + E_kin + E_mag).
    bound_ratio: jax.Array

    #: The centre-of-mass velocity of the control volume, shape
    #: (num_positions, 3).
    center_of_mass_velocity: jax.Array


def _slot_checks(
    neighbourhood: SlotNeighbourhood,
    mass: jax.Array,
    cells: CellQuantities,
    config: SimulationConfig,
) -> SlotChecks:
    """
    The quantities of the creation checks at a set of positions.

    A cell quantity q is read at a position as Σ h q / Σ h. The control
    volume holds the gas within r_acc, each cell with the mass M = K ρ ΔV;
    its gravitational energy is measured from the soft maximum of the
    potential, and the soft minimum and maximum have the temperature ω c_s²,
    with c_s² read at the position.

    Args:
        neighbourhood: The cells around each position.
        mass: The mass of the slot at each position, added to its density,
            shape (num_positions,).
        cells: The cell quantities.
        config: The simulation configuration.

    Returns:
        The quantities of the checks.
    """
    check_width = config.sink_particle_config.check_width
    cell_volume = config.grid_spacing**3
    cell_indices = neighbourhood.cell_indices
    cell_weight = neighbourhood.cell_weight
    radial_weight = neighbourhood.radial_weight

    total_cell_weight = jnp.sum(cell_weight, axis=-1)

    def read(field):
        """The field read at each position with the cell weights."""
        return _safe_divide(
            jnp.sum(cell_weight * field[cell_indices], axis=-1),
            total_cell_weight,
        )

    density = read(cells.density) + mass / cell_volume
    density_threshold = read(cells.density_threshold)
    sound_speed_squared = read(cells.sound_speed_squared)
    potential = read(cells.potential)
    converging_velocity = jnp.stack(
        [read(component) for component in cells.converging_velocity],
        axis=-1,
    )

    potential_in_reach = cells.potential[cell_indices]
    minimum_potential = _soft_minimum(
        potential_in_reach,
        radial_weight,
        check_width,
        sound_speed_squared,
    )
    maximum_potential = _soft_maximum(
        potential_in_reach,
        radial_weight,
        check_width,
        sound_speed_squared,
    )

    magnetic_field_squared = None
    if cells.magnetic_field_squared is not None:
        magnetic_field_squared = cells.magnetic_field_squared[cell_indices]
    energies = _control_volume_energies(
        radial_weight * cells.density[cell_indices] * cell_volume,
        radial_weight * cell_volume,
        cells.velocity[cell_indices],
        cells.sound_speed_squared[cell_indices],
        magnetic_field_squared,
        maximum_potential[:, None] - potential_in_reach,
    )

    return SlotChecks(
        has_cells=total_cell_weight > 0.0,
        density_threshold=density_threshold,
        density_ratio=_safe_divide(density, density_threshold),
        converging_velocity=converging_velocity,
        potential_height=_safe_divide(
            potential - minimum_potential,
            sound_speed_squared,
        ),
        jeans_ratio=_safe_divide(
            energies.gravitational_energy_magnitude,
            2.0 * energies.thermal_energy + energies.magnetic_energy,
        ),
        bound_ratio=_safe_divide(
            energies.gravitational_energy_magnitude,
            energies.thermal_energy + energies.kinetic_energy + energies.magnetic_energy,
        ),
        center_of_mass_velocity=energies.center_of_mass_velocity,
    )


def _passes_relaxed_checks(
    checks: SlotChecks,
    config: SimulationConfig,
) -> jax.Array:
    """
    Whether each position passes the relaxed creation checks (c) to (g): the
    density, Jeans and bound ratios above μ, every u_k ≥ −ω / μ and the
    potential height ≤ ω / μ.

    Args:
        checks: The quantities of the checks.
        config: The simulation configuration.

    Returns:
        The result, shape (num_positions,).
    """
    sink_particle_config = config.sink_particle_config
    creation_margin = sink_particle_config.creation_margin
    relaxed_width = sink_particle_config.check_width / creation_margin
    return (
        checks.has_cells
        & (checks.density_ratio > creation_margin)
        & jnp.all(checks.converging_velocity >= -relaxed_width, axis=-1)
        & (checks.potential_height <= relaxed_width)
        & (checks.jeans_ratio > creation_margin)
        & (checks.bound_ratio > creation_margin)
    )


# -------------------------------------------------------------
# =================== ↑ Checks at a position ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Creation of candidates ↓ ==============
# -------------------------------------------------------------


def _fill_potential_ghost_cells(
    gravitational_potential: FIELD_TYPE,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Overwrite the ghost cells of the potential for the potential-minimum check,
    so that a control volume only compares real cells.

    Along a periodic axis the ghost cells get periodic copies of the interior
    potential. Along a non-periodic axis they get +∞, which marks them as
    cells past the edge; this includes the ghost cells that also lie beyond a
    periodic axis.

    Args:
        gravitational_potential: The (padded) gravitational potential.
        config: The simulation configuration.

    Returns:
        The potential with its ghost cells overwritten.
    """
    if config.boundary_handling != GHOST_CELLS:
        return gravitational_potential

    num_ghost_cells = config.num_ghost_cells
    num_axes = gravitational_potential.ndim
    is_periodic, _ = _periodic_box(config)
    for axis in range(num_axes):
        if is_periodic[axis]:
            gravitational_potential = _periodic_boundaries(
                gravitational_potential,
                num_ghost_cells,
                axis=axis,
            )
        else:
            left_ghost_cells = _axis_slice(axis, 0, num_ghost_cells, num_axes)
            right_ghost_cells = _axis_slice(axis, -num_ghost_cells, None, num_axes)
            gravitational_potential = (
                gravitational_potential.at[left_ghost_cells]
                .set(jnp.inf)
                .at[right_ghost_cells]
                .set(jnp.inf)
            )
    return gravitational_potential


def _potential_height_on_grid(
    potential: FIELD_TYPE,
    sound_speed_squared: FIELD_TYPE,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    (φ − min_T φ) / c_s² in every cell, with the soft minimum over the cell's
    control volume weighted by K, at the temperature ω c_s² of the cell: the
    potential height of ``_slot_checks`` for a position at a cell centre.

    Cells past a non-periodic edge are left out. The values are meaningful in
    interior cells only.

    Args:
        potential: The (padded) potential.
        sound_speed_squared: The (padded) c_s².
        config: The simulation configuration.

    Returns:
        The potential height, with the shape of the potential.
    """
    sink_particle_config = config.sink_particle_config
    check_width = sink_particle_config.check_width
    accretion_radius_in_cells = sink_particle_config.accretion_radius_in_cells
    spatial_width_in_cells = sink_particle_config.spatial_width_in_cells

    # The control volume, with the radial weight K of each offset, computed
    # with NumPy at trace time as in ``_radial_weight``; offsets of weight 0
    # are dropped.
    offsets = _control_volume_offsets(config)
    if spatial_width_in_cells == 0.0:
        radial_weights = np.ones(offsets.shape[0])
    else:
        distance_in_cells = np.sqrt(np.sum(offsets**2, axis=-1))
        scaled = np.clip(
            (accretion_radius_in_cells - distance_in_cells) / spatial_width_in_cells,
            0.0,
            1.0,
        )
        radial_weights = scaled * scaled * (3.0 - 2.0 * scaled)
    offsets = offsets[radial_weights > 0.0]
    radial_weights = radial_weights[radial_weights > 0.0]

    filled_potential = _fill_potential_ghost_cells(potential, config)

    def at_offset(offset):
        """The filled potential of the cell at ``offset`` from each cell."""
        shifted = filled_potential
        for axis in range(3):
            if offset[axis] != 0:
                # Shifting by −offset brings the value at cell + offset.
                shifted = _shift(shifted, -int(offset[axis]), axis)
        return shifted

    minimum = filled_potential
    for offset in offsets:
        minimum = jnp.minimum(minimum, at_offset(offset))
    minimum = jnp.where(jnp.isfinite(minimum), minimum, 0.0)

    if check_width > 0.0:
        temperature = check_width * sound_speed_squared
        safe_temperature = jnp.where(temperature > 0.0, temperature, 1.0)
        weighted_sum = jnp.zeros_like(potential)
        normalization = jnp.zeros_like(potential)
        for offset, radial_weight in zip(offsets, radial_weights):
            shifted = at_offset(offset)
            is_real = jnp.isfinite(shifted)
            value = jnp.where(is_real, shifted, minimum)
            weight = jnp.where(
                is_real,
                radial_weight * jnp.exp(-(value - minimum) / safe_temperature),
                0.0,
            )
            weighted_sum = weighted_sum + weight * value
            normalization = normalization + weight
        minimum = jnp.where(
            temperature > 0.0,
            _safe_divide(weighted_sum, normalization),
            minimum,
        )

    return _safe_divide(potential - minimum, sound_speed_squared)


def _create_candidates(
    sink_particles: SinkParticles,
    cells: CellQuantities,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Step 1: create candidates where the gas passes the relaxed creation
    checks.

    On every cell: (a) it is not a ghost cell, (b) it lies farther than r_acc
    from every occupied slot, (c) ρ / ρ_thr > μ, (d) every u_k ≥ −ω / μ and
    (e) its potential height is ≤ ω / μ. The first ``max_num_candidates``
    such cells are listed, and on each listed control volume (f) the Jeans
    ratio and (g) the bound ratio exceed μ. Among the cells that pass, only
    those at least r_acc apart are kept, the deepest potential first.

    A new candidate takes a free slot, the k-th in the k-th free slot, with
    activation 0 and no mass, at the centre of its cell and with the
    centre-of-mass velocity (Eq. 12) of the gas in its control volume. When no
    slot is free the run stops with an error, since the gas would otherwise
    collapse without a candidate.

    Args:
        sink_particles: The current sink particles.
        cells: The cell quantities.
        config: The simulation configuration.

    Returns:
        The sink particles with the new candidates.
    """
    sink_particle_config = config.sink_particle_config
    creation_margin = sink_particle_config.creation_margin
    relaxed_width = sink_particle_config.check_width / creation_margin
    grid_spacing = config.grid_spacing
    accretion_radius = sink_particle_config.accretion_radius_in_cells * grid_spacing
    grid_shape = cells.density.shape

    # -------------------------------------------------------------
    # =============== ↓ Stage 1: grid-wide checks ↓ ===============
    # -------------------------------------------------------------

    # The checks only decide where candidates are created.
    grid_cells = jax.lax.stop_gradient(cells)
    is_near_slot = _near_slots_mask(
        _slot_neighbourhood(sink_particles.position, grid_shape, config),
        sink_particles.occupied,
        grid_shape,
        config,
    )
    candidate_mask = (
        _interior_mask(grid_cells.density, config)
        & ~is_near_slot
        & (grid_cells.density > creation_margin * grid_cells.density_threshold)
        & jnp.all(grid_cells.converging_velocity >= -relaxed_width, axis=0)
        & (
            _potential_height_on_grid(
                grid_cells.potential,
                grid_cells.sound_speed_squared,
                config,
            )
            <= relaxed_width
        )
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
    candidate_indices, num_candidates_found = _collect_candidates(
        candidate_mask,
        max_num_candidates,
    )
    candidate_is_valid = jnp.arange(max_num_candidates) < num_candidates_found

    # Centre of the candidate cells. Unused entries of the list point at the
    # corner of the padded grid; they never pass, since they are invalid.
    candidate_positions = (
        candidate_indices - config.num_ghost_cells + 0.5
    ) * grid_spacing
    checks = _slot_checks(
        _slot_neighbourhood(candidate_positions, grid_shape, config),
        jnp.zeros(max_num_candidates, dtype=candidate_positions.dtype),
        cells,
        config,
    )
    candidate_passes = (
        candidate_is_valid
        & (jax.lax.stop_gradient(checks.jeans_ratio) > creation_margin)
        & (jax.lax.stop_gradient(checks.bound_ratio) > creation_margin)
    )

    # Proximity among the new candidates (Section 2.2.7): several nearby cells
    # can pass every check in the same call, e.g. when the potential minimum
    # is shared by equivalent cells. Only candidates at least r_acc apart are
    # kept, the deepest potential first.
    is_periodic, box_length = _periodic_box(config)
    candidate_passes = _select_separated_candidates(
        candidate_passes,
        candidate_positions,
        grid_cells.potential[tuple(candidate_indices.T)],
        accretion_radius,
        is_periodic,
        box_length,
    )

    # -------------------------------------------------------------
    # =========== ↑ Stage 2: control-volume checks ↑ ==============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Filling free slots ↓ ====================
    # -------------------------------------------------------------

    # The k-th passing candidate takes the k-th free slot; failing candidates,
    # and passing ones beyond the free slots, get the out-of-range slot
    # max_num_sinks, whose writes mode="drop" discards.
    max_num_sinks = sink_particle_config.max_num_sinks
    free_slots = jnp.nonzero(
        ~sink_particles.occupied,
        size=max_num_sinks,
        fill_value=max_num_sinks,
    )[0]
    num_free_slots = max_num_sinks - jnp.sum(sink_particles.occupied)
    new_candidate_rank = jnp.cumsum(candidate_passes) - 1
    slot = jnp.where(
        candidate_passes,
        free_slots.at[new_candidate_rank].get(mode="fill", fill_value=max_num_sinks),
        max_num_sinks,
    )

    sink_particles = sink_particles._replace(
        mass=sink_particles.mass.at[slot].set(0.0, mode="drop"),
        position=sink_particles.position.at[slot].set(
            candidate_positions,
            mode="drop",
        ),
        velocity=sink_particles.velocity.at[slot].set(
            checks.center_of_mass_velocity,
            mode="drop",
        ),
        activation=sink_particles.activation.at[slot].set(0.0, mode="drop"),
        occupied=sink_particles.occupied.at[slot].set(True, mode="drop"),
    )
    num_new_candidates = jnp.sum(candidate_passes)

    # Candidates beyond max_num_candidates are only skipped for this step:
    # they still pass the grid-wide checks at the next step and are checked
    # then, so a warning is enough. A candidate without a free slot would
    # leave its gas to collapse without one, so the run stops with an error
    # instead (a host callback that raises aborts the jitted computation).
    jax.lax.cond(
        num_candidates_found > max_num_candidates,
        lambda: jax.debug.print(
            "WARNING: {} sink particle candidate cells found, only the first {} "
            "are checked this step (increase max_num_candidates).",
            num_candidates_found,
            max_num_candidates,
        ),
        lambda: None,
    )
    jax.lax.cond(
        num_new_candidates > num_free_slots,
        lambda: jax.debug.callback(
            _raise_sink_slots_full,
            num_new_candidates - num_free_slots,
            max_num_sinks,
        ),
        lambda: None,
    )

    # -------------------------------------------------------------
    # ================= ↑ Filling free slots ↑ ====================
    # -------------------------------------------------------------

    return sink_particles


# -------------------------------------------------------------
# =================== ↑ Creation of candidates ↑ ==============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Activation ↓ ======================
# -------------------------------------------------------------


class CreationFactors(NamedTuple):
    """The factors of the creation weight of each slot, each of shape
    (num_slots,), in [0, 1]."""

    #: step_ω(ρ / ρ_thr − 1).
    density: jax.Array

    #: Π_k [1 − step_ω(−u_k)].
    converging_flow: jax.Array

    #: 1 − step_ω(potential height).
    potential_minimum: jax.Array

    #: step_ω(Jeans ratio − 1).
    jeans_instability: jax.Array

    #: step_ω(bound ratio − 1).
    bound_state: jax.Array

    #: Π_{j ≠ i} [1 − a_j K(d_ij − ℓ Δx)] over the other occupied slots.
    proximity: jax.Array


def _creation_factors(
    checks: SlotChecks,
    sink_particles: SinkParticles,
    slot_separation_squared: jax.Array,
    config: SimulationConfig,
) -> CreationFactors:
    """
    The factors of the creation weight of every slot.

    Each factor is 0 wherever its hard check fails and 1 beyond a band of
    width ω past its threshold. The proximity factor is 0 within r_acc of a
    sink with activation 1, and its band lies between r_acc and r_acc + ℓ Δx,
    outside the radius of the creation check (b).

    Args:
        checks: The quantities of the checks at every slot.
        sink_particles: The sink particles, with the activations of the
            previous step.
        slot_separation_squared: The squared distances between slots, shape
            (num_slots, num_slots).
        config: The simulation configuration.

    Returns:
        The factors.
    """
    sink_particle_config = config.sink_particle_config
    check_width = sink_particle_config.check_width
    spatial_width = sink_particle_config.spatial_width_in_cells * config.grid_spacing
    accretion_radius = (
        sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    )

    num_slots = sink_particles.mass.shape[0]
    is_other_occupied_slot = sink_particles.occupied[None, :] & ~jnp.eye(
        num_slots,
        dtype=bool,
    )
    proximity_weight = _radial_weight(
        slot_separation_squared,
        accretion_radius + spatial_width,
        spatial_width,
    )

    return CreationFactors(
        density=_smooth_step(checks.density_ratio - 1.0, check_width),
        converging_flow=jnp.prod(
            1.0 - _smooth_step(-checks.converging_velocity, check_width),
            axis=-1,
        ),
        potential_minimum=1.0 - _smooth_step(checks.potential_height, check_width),
        jeans_instability=_smooth_step(checks.jeans_ratio - 1.0, check_width),
        bound_state=_smooth_step(checks.bound_ratio - 1.0, check_width),
        proximity=jnp.prod(
            jnp.where(
                is_other_occupied_slot,
                1.0 - sink_particles.activation[None, :] * proximity_weight,
                1.0,
            ),
            axis=-1,
        ),
    )


def _activate(
    sink_particles: SinkParticles,
    checks: SlotChecks,
    slot_separation_squared: jax.Array,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Step 2: raise the activation of every occupied slot to its creation
    weight w, the product of the creation factors, a ← max(a, w). All
    activations are updated together, with the previous ones in the
    proximity factor. A slot without a cell to read its values from gets
    w = 0.

    Args:
        sink_particles: The sink particles.
        checks: The quantities of the checks at every slot.
        slot_separation_squared: The squared distances between slots.
        config: The simulation configuration.

    Returns:
        The sink particles with the new activations.
    """
    factors = _creation_factors(
        checks,
        sink_particles,
        slot_separation_squared,
        config,
    )
    weight = jnp.where(
        checks.has_cells,
        factors.density
        * factors.converging_flow
        * factors.potential_minimum
        * factors.jeans_instability
        * factors.bound_state
        * factors.proximity,
        0.0,
    )
    return sink_particles._replace(
        activation=jnp.where(
            sink_particles.occupied,
            jnp.maximum(sink_particles.activation, weight),
            0.0,
        )
    )


# -------------------------------------------------------------
# ======================= ↑ Activation ↑ ======================
# -------------------------------------------------------------
