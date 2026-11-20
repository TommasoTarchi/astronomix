"""
Sink particle formation following Federrath et al. (2010), ApJ 713, 269.

Once per time step, the gas is checked for regions that should turn into sink
particles, using the creation checks of Section 2.2 of the paper. A new sink is
created with zero mass at the centre of its cell and gets its mass from the
accretion step that follows in the same time step
(``_sink_particle_accretion.py``), as in Federrath's FLASH code.

The formation runs in two stages. First, cheap checks are evaluated on the
whole grid, giving a mask of candidate cells. Then, for a fixed-size list of
candidates, the gas in the control volume around each candidate (all cells
within the accretion radius r_acc) is gathered and the remaining checks are
evaluated on it. Candidates that pass every check become new sinks.

Only gas in interior cells counts, as in the accretion step. A control volume
wraps around periodic axes and ends at the edge of the box on non-periodic
axes. Reflective walls are treated like the other edges: mirrored gas is not
counted.

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
from astronomix._stencil_operations._stencil_operations import _shift


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


def _fill_potential_ghost_cells(
    gravitational_potential: FIELD_TYPE,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Overwrite the ghost cells of the potential for the potential-minimum check,
    so that a control volume only compares real cells.

    Along a periodic axis the ghost cells get periodic copies of the interior
    potential. Along a non-periodic axis they get +∞, which can never be the
    minimum; this includes the ghost cells that also lie beyond a periodic axis.

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

    A new sink takes a free slot, with activation 1 and zero mass, at the
    centre of its cell and with the centre-of-mass velocity (Eq. 12) of the
    gas in its control volume. It gets its mass from the accretion step that
    follows. The starting velocity only sets the frame of the first accretion
    checks; FLASH starts from zero velocity, which would judge a clump moving
    as a whole in a different frame than the creation checks below.

    Args:
        primitive_state: The (padded) primitive state after the hydro update.
        sink_particles: The current sink particles.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The sink particles with the new sinks in free slots, the k-th new sink
        in the k-th free slot.
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

    # The density threshold (Federrath et al. 2010, Eq. 32) depends on the
    # local sound speed, so it is evaluated in every cell.
    sound_speed_squared = _sound_speed_squared(
        primitive_state,
        config,
        params,
        registered_variables,
    )
    density_threshold = _density_threshold(
        sound_speed_squared,
        config,
        params,
    )

    # The gravitational potential of the gas and the sinks (plus any external
    # potential), from one Poisson solve of the updated gas density plus the
    # sinks' masses spread with the TSC weights of their motion, so that the
    # checks feel the sinks that already exist.
    gravitational_potential = _compute_total_potential(
        density + _sink_density(sink_particles, density.shape, config),
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
    # hold boundary values. The converging flow check reads their velocity,
    # since these ±1 neighbours are what the hydro solver sees at the boundary;
    # the potential check does not count them as gas.
    candidate_mask = (
        (density > density_threshold)
        & _converging_flow_mask(velocity)
        & _potential_minimum_mask(
            _fill_potential_ghost_cells(gravitational_potential, config),
            offsets,
        )
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
    candidate_indices, num_candidates_found = _collect_candidates(
        candidate_mask,
        max_num_candidates,
    )
    candidate_is_valid = jnp.arange(max_num_candidates) < num_candidates_found

    # Cell indices of every candidate's control volume. Cells past a
    # non-periodic boundary get zero volume, and so zero mass, which removes
    # them from every sum below.
    control_volume_indices, _, is_real_cell = _interior_cell_indices(
        candidate_indices[:, None, :] - config.num_ghost_cells + offsets[None, :, :],
        density.shape,
        config,
    )
    cell_volume_in_volume = jnp.where(is_real_cell, cell_volume, 0.0)

    # Fields over the control volumes, shape (num_candidates, num_offsets),
    # with a trailing vector axis for the velocity.
    density_in_volume = density[control_volume_indices]
    velocity_in_volume = velocity[control_volume_indices]
    cell_mass_in_volume = density_in_volume * cell_volume_in_volume

    # Centre of the candidate cells. Unused entries of the list point at the
    # corner of the padded grid; they never pass, since they are invalid.
    candidate_positions = (candidate_indices - config.num_ghost_cells + 0.5) * grid_spacing

    # Proximity check (Section 2.2.7): no new sink within r_acc of an occupied
    # slot. Along periodic axes the separation is taken to the nearest
    # periodic copy of the sink (minimum-image convention), as in Federrath's
    # FLASH code. Empty slots sit at the origin, where they would otherwise
    # block candidates.
    is_periodic, box_length = _periodic_box(config)
    separation = candidate_positions[:, None, :] - sink_particles.position[None, :, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    distance_to_sink = jnp.linalg.norm(separation, axis=-1)
    sink_within_accretion_radius = jnp.logical_and(
        distance_to_sink <= accretion_radius,
        sink_particles.occupied[None, :],
    )
    far_from_existing_sinks = jnp.logical_not(
        jnp.any(sink_within_accretion_radius, axis=-1)
    )

    # The energies of the gas in each control volume, with the gravitational
    # energy measured from the largest potential in the volume, as in
    # Federrath's FLASH code.
    potential_in_volume = gravitational_potential[control_volume_indices]
    maximum_potential_in_volume = jnp.max(
        jnp.where(is_real_cell, potential_in_volume, -jnp.inf),
        axis=-1,
        keepdims=True,
    )
    potential_below_maximum = jnp.where(
        is_real_cell,
        maximum_potential_in_volume - potential_in_volume,
        0.0,
    )
    magnetic_field_squared_in_volume = None
    if config.mhd:
        magnetic_field_squared = (
            primitive_state[registered_variables.magnetic_index.x] ** 2
            + primitive_state[registered_variables.magnetic_index.y] ** 2
            + primitive_state[registered_variables.magnetic_index.z] ** 2
        )
        magnetic_field_squared_in_volume = magnetic_field_squared[control_volume_indices]
    energies = _control_volume_energies(
        cell_mass_in_volume,
        cell_volume_in_volume,
        velocity_in_volume,
        sound_speed_squared[control_volume_indices],
        magnetic_field_squared_in_volume,
        potential_below_maximum,
    )

    # Jeans instability check (Section 2.2.5): the gravitational energy of the
    # gas in the control volume must exceed twice its thermal energy plus its
    # magnetic energy, |E_grav| > 2 E_th + E_mag.
    jeans_unstable = (
        energies.gravitational_energy_magnitude
        > 2.0 * energies.thermal_energy + energies.magnetic_energy
    )

    # Bound state check (Section 2.2.6): the total energy of the gas in the
    # control volume must be negative, E_grav + E_th + E_kin + E_mag < 0.
    bound = (
        -energies.gravitational_energy_magnitude
        + energies.thermal_energy
        + energies.kinetic_energy
        + energies.magnetic_energy
        < 0.0
    )

    candidate_cell_indices = tuple(candidate_indices.T)
    candidate_passes = (
        candidate_is_valid
        & far_from_existing_sinks
        & jeans_unstable
        & bound
    )

    # Proximity among the new sinks (Section 2.2.7): several nearby cells can
    # pass every check in the same call, e.g. when the potential minimum is
    # shared by equivalent cells (the potential check is not strict). Only
    # candidates at least r_acc apart are kept, the deepest potential first.
    candidate_passes = _select_separated_candidates(
        candidate_passes,
        candidate_positions,
        gravitational_potential[candidate_cell_indices],
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
    new_sink_rank = jnp.cumsum(candidate_passes) - 1
    slot = jnp.where(
        candidate_passes,
        free_slots.at[new_sink_rank].get(mode="fill", fill_value=max_num_sinks),
        max_num_sinks,
    )

    sink_particles = sink_particles._replace(
        mass=sink_particles.mass.at[slot].set(0.0, mode="drop"),
        position=sink_particles.position.at[slot].set(
            candidate_positions,
            mode="drop",
        ),
        velocity=sink_particles.velocity.at[slot].set(
            energies.center_of_mass_velocity,
            mode="drop",
        ),
        activation=sink_particles.activation.at[slot].set(1.0, mode="drop"),
        occupied=sink_particles.occupied.at[slot].set(True, mode="drop"),
    )
    num_new_sinks = jnp.sum(candidate_passes)

    # Candidates beyond max_num_candidates are only skipped for this step:
    # they are still above the threshold at the next step and are checked
    # then, so a warning is enough. A new sink without a free slot would be
    # lost, leaving its gas above the threshold on the grid, so the run stops
    # with an error instead (a host callback that raises aborts the jitted
    # computation).
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
        num_new_sinks > num_free_slots,
        lambda: jax.debug.callback(
            _raise_sink_slots_full,
            num_new_sinks - num_free_slots,
            max_num_sinks,
        ),
        lambda: None,
    )

    # -------------------------------------------------------------
    # ================= ↑ Filling free slots ↑ ====================
    # -------------------------------------------------------------

    return sink_particles
