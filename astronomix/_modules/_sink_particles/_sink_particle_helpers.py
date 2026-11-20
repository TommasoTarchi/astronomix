"""
Helpers shared by the formation, the accretion and the motion of sink
particles: the sink particle slots, the stencils around a cell or a sink, the
density threshold, the geometry of the box, the search for candidate cells and
the energies of the gas in a control volume.
"""

# general
import inspect

# typing
from typing import NamedTuple

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

try:  # the stable alias (newer jax)
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map as _shard_map

# The keyword that turns off shard_map's check that an output is the same on
# every device, which it cannot infer through the collectives used below.
_NO_REPLICATION_CHECK = (
    {"check_vma": False}
    if "check_vma" in inspect.signature(_shard_map).parameters
    else {"check_rep": False}
)

# astronomix helpers
from astronomix._pallas_helpers import _current_pallas_mesh

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


# -------------------------------------------------------------
# ==================== ↓ Sink particle slots ↓ =================
# -------------------------------------------------------------


def _raise_sink_slots_full(num_discarded, max_num_sinks):
    """Stop the run when new sinks do not fit in the sink particle slots.

    Called from inside the jitted formation step through
    ``jax.debug.callback``; the raised error aborts the jitted computation.

    Args:
        num_discarded: The number of new sinks without a free slot.
        max_num_sinks: The number of sink particle slots.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(
        f"{int(num_discarded)} new sink particles cannot be stored: all "
        f"{int(max_num_sinks)} sink particle slots are taken. Increase "
        "max_num_sinks in SinkParticleConfig."
    )


def _empty_sink_particles(
    config: SimulationConfig,
    dtype,
) -> SinkParticles:
    """
    Create sink particle arrays with every slot empty: not occupied, and zero
    in every other field.

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
        activation=jnp.zeros((max_num_sinks,), dtype=dtype),
        occupied=jnp.zeros((max_num_sinks,), dtype=bool),
    )


def _pad_sink_particles(
    sink_particles: SinkParticles,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Complete the sink particles a run starts with, and append empty slots to
    sink particles with fewer slots than ``max_num_sinks``, e.g. sinks
    restored from a checkpoint of a run with fewer slots.

    Sinks given by mass, position and velocity alone, such as sinks placed by
    hand or restored from a checkpoint that holds only these fields, are
    completed from their masses: a slot with a positive mass is occupied by a
    sink of activation 1, and the other slots are empty, with activation 0.

    Args:
        sink_particles: The sink particles passed into the run.
        config: The simulation configuration; supplies the number of slots.

    Returns:
        Sink particles with every field set and ``max_num_sinks`` slots.

    Raises:
        ValueError: If the sink particles have more slots than
            ``max_num_sinks``; dropping slots could drop occupied ones.
    """
    max_num_sinks = config.sink_particle_config.max_num_sinks
    num_slots = sink_particles.mass.shape[0]
    if num_slots > max_num_sinks:
        raise ValueError(
            f"{num_slots} sink particle slots passed in, but max_num_sinks is "
            f"{max_num_sinks}. Increase max_num_sinks in SinkParticleConfig."
        )

    holds_mass = sink_particles.mass > 0.0
    if sink_particles.occupied is None:
        sink_particles = sink_particles._replace(occupied=holds_mass)
    if sink_particles.activation is None:
        sink_particles = sink_particles._replace(
            activation=holds_mass.astype(sink_particles.mass.dtype)
        )

    num_missing_slots = max_num_sinks - num_slots
    return jax.tree.map(
        lambda field: jnp.pad(
            field,
            [(0, num_missing_slots)] + [(0, 0)] * (field.ndim - 1),
        ),
        sink_particles,
    )


# -------------------------------------------------------------
# ==================== ↑ Sink particle slots ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ========================= ↓ Stencils ↓ =======================
# -------------------------------------------------------------


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


def _accretion_offsets(config: SimulationConfig) -> np.ndarray:
    """
    Integer cell offsets (i, j, k), from the cell containing a sink, of every
    cell that can lie within the accretion radius of that sink.

    A sink can sit anywhere inside its cell, at most √3/2 cells from the cell
    centre, so the offsets are those with i² + j² + k² ≤ (r_acc/Δx + √3/2)².
    Whether a cell is actually within r_acc is checked per sink. The offsets
    are computed with NumPy at trace time, since they only depend on the
    static configuration.

    Args:
        config: The simulation configuration.

    Returns:
        The offsets, shape (num_offsets, 3), with (0, 0, 0) among them.
    """
    reach_in_cells = (
        config.sink_particle_config.accretion_radius_in_cells + 0.5 * np.sqrt(3.0)
    )
    max_offset = int(np.floor(reach_in_cells))
    offset_range = np.arange(-max_offset, max_offset + 1)
    offsets = np.stack(
        np.meshgrid(offset_range, offset_range, offset_range, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    within_reach = np.sum(offsets**2, axis=-1) <= reach_in_cells**2
    return offsets[within_reach]


# -------------------------------------------------------------
# ========================= ↑ Stencils ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ Sound speed and threshold ↓ ===============
# -------------------------------------------------------------


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


def _density_threshold(
    sound_speed_squared: FIELD_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
) -> FIELD_TYPE:
    """
    The density threshold ρ_res in every cell.

    ρ_res is the density at which the Jeans length λ_J = (π c_s² / (G ρ))^½
    equals 2 r_acc, the smallest Jeans length the grid resolves (Federrath et
    al. 2010, Eq. 32). It uses the local c_s².

    Args:
        sound_speed_squared: The squared sound speed in every cell.
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        The density threshold, with the shape of ``sound_speed_squared``.
    """
    accretion_radius = (
        config.sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    )
    return (
        jnp.pi
        * sound_speed_squared
        / (4.0 * params.gravitational_constant * accretion_radius**2)
    )


# -------------------------------------------------------------
# ================ ↑ Sound speed and threshold ↑ ===============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ==================== ↓ Geometry of the box ↓ =================
# -------------------------------------------------------------


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


def _interior_cell_indices(
    interior_index: jax.Array,
    grid_shape: tuple,
    config: SimulationConfig,
) -> tuple[tuple, jax.Array, jax.Array]:
    """
    Map cell indices counted from the first interior cell, which may lie past
    the edge of the box, to real (interior) cells.

    Along a periodic axis the index wraps around the interior, which reaches
    the real cell across the boundary whether or not ghost cells are used
    (the ghost layer only holds copies). Along a non-periodic axis there is no
    gas beyond the boundary, so indices leaving the interior are invalid. They
    are clipped into the interior, so that they never read a ghost cell.

    Args:
        interior_index: The cell indices, shape (..., 3).
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        ``(cell_indices, interior_index, is_real_cell)``: a tuple of three
        index arrays into the padded field, the wrapped and clipped interior
        indices, and whether each index is valid, shape (...).
    """
    num_ghost_cells = config.num_ghost_cells
    num_interior_cells = np.array(grid_shape) - 2 * num_ghost_cells
    is_periodic, _ = _periodic_box(config)

    interior_index = jnp.where(
        is_periodic,
        interior_index % num_interior_cells,
        interior_index,
    )
    is_real_cell = jnp.all(
        (interior_index >= 0) & (interior_index < num_interior_cells),
        axis=-1,
    )
    interior_index = jnp.clip(interior_index, 0, num_interior_cells - 1)
    cell_indices = tuple(
        interior_index[..., axis] + num_ghost_cells for axis in range(3)
    )
    return cell_indices, interior_index, is_real_cell


# -------------------------------------------------------------
# ==================== ↑ Geometry of the box ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Search for candidates ↓ ================
# -------------------------------------------------------------


def _select_separated_candidates(
    candidate_passes: jax.Array,
    candidate_positions: jax.Array,
    potential_at_candidates: jax.Array,
    accretion_radius: float,
    is_periodic: np.ndarray,
    box_length: np.ndarray,
) -> jax.Array:
    """
    Keep only passing candidates that lie farther than r_acc from every other
    kept candidate, so that new sinks formed in the same call respect the
    proximity rule (Section 2.2.7) among themselves.

    The candidates are taken in order of increasing potential, the deepest
    first, with exact ties going to the lower candidate index. Each one is kept
    if it passes and no candidate kept before it lies within r_acc. The pass is
    sequential: a single parallel rule ("drop a candidate if a higher-priority
    passing candidate is within r_acc") would also drop candidates that are
    only near a candidate dropped itself.

    Args:
        candidate_passes: Whether each candidate passes the creation checks,
            shape (num_candidates,).
        candidate_positions: The candidate cell centres, shape
            (num_candidates, 3).
        potential_at_candidates: The gravitational potential at each candidate
            cell, shape (num_candidates,).
        accretion_radius: The accretion radius r_acc.
        is_periodic: Which axes are periodic, shape (3,).
        box_length: The box length along each axis, shape (3,).

    Returns:
        Whether each candidate becomes a sink, shape (num_candidates,).
    """
    # Distances between candidates, to the nearest periodic copy along
    # periodic axes, as in the proximity check to existing sinks.
    separation = candidate_positions[:, None, :] - candidate_positions[None, :, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    within_accretion_radius = (
        jnp.linalg.norm(separation, axis=-1) <= accretion_radius
    )

    # Failing candidates go last; the stable sort keeps exact ties in
    # candidate order.
    order = jnp.argsort(
        jnp.where(candidate_passes, potential_at_candidates, jnp.inf),
        stable=True,
    )

    def keep_if_separated(rank, kept):
        candidate = order[rank]
        near_kept = jnp.any(kept & within_accretion_radius[candidate])
        return kept.at[candidate].set(candidate_passes[candidate] & ~near_kept)

    return jax.lax.fori_loop(
        0,
        candidate_passes.shape[0],
        keep_if_separated,
        jnp.zeros_like(candidate_passes),
    )


def _collect_candidates(
    candidate_mask: jax.Array,
    max_num_candidates: int,
) -> tuple[jax.Array, jax.Array]:
    """
    The first ``max_num_candidates`` candidate cells, in the grid's row-major
    order, and the number of candidate cells.

    On one device this is ``jnp.nonzero`` with a fixed size. When the grid is
    split across devices (a mesh set by ``pallas_mesh_context``, as
    ``time_integration`` does for a sharded run), ``jnp.nonzero`` on the
    global mask would make every device gather the whole mask. Instead, each
    device lists the first ``max_num_candidates`` candidates of its own part
    of the grid, in global indices, and only these short lists are gathered.
    Sorting them in row-major order gives the same list as on one device: a
    cell among the first ``max_num_candidates`` of the grid is also among the
    first ``max_num_candidates`` of its device.

    Args:
        candidate_mask: Boolean mask of the candidate cells, shape of the
            (padded) grid.
        max_num_candidates: The length of the list.

    Returns:
        ``(candidate_indices, num_candidates_found)``: the cell indices, shape
        (max_num_candidates, 3), with unused entries set to 0, and the total
        number of candidate cells.
    """
    grid_shape = candidate_mask.shape

    # The distributed list needs a 4-axis state mesh (variables, x, y, z) that
    # splits the grid evenly; otherwise the mask is searched as a whole.
    mesh = _current_pallas_mesh()
    split = False
    if mesh is not None and len(mesh.axis_names) == 4:
        devices_per_axis = [mesh.shape[name] for name in mesh.axis_names[1:]]
        split = any(num > 1 for num in devices_per_axis) and all(
            size % num == 0 for size, num in zip(grid_shape, devices_per_axis)
        )
    if not split:
        candidate_indices = jnp.stack(
            jnp.nonzero(candidate_mask, size=max_num_candidates, fill_value=0),
            axis=-1,
        )
        return candidate_indices, jnp.sum(candidate_mask)

    # The state mesh may use integer axis names, which JAX's named collectives
    # would read as array axes. Alias the same devices (same order, so no data
    # movement) under string names, as the distributed Poisson solve does.
    axis_names = ("x", "y", "z")
    named_mesh = jax.sharding.Mesh(mesh.devices, ("vars",) + axis_names)

    def _local_list(local_mask):
        # The first candidates of this device's part of the grid, in global
        # indices: local index + device index × local size along each axis.
        local_indices = jnp.stack(
            jnp.nonzero(local_mask, size=max_num_candidates, fill_value=0),
            axis=-1,
        )
        offsets = jnp.stack(
            [
                jax.lax.axis_index(name) * local_mask.shape[axis]
                for axis, name in enumerate(axis_names)
            ]
        )
        local_count = jnp.sum(local_mask)
        local_is_valid = jnp.arange(max_num_candidates) < local_count

        # Gather the short lists of all devices, shape
        # (num_devices × max_num_candidates, ...), and the total count.
        indices = jax.lax.all_gather(
            local_indices + offsets, axis_names, tiled=True
        )
        is_valid = jax.lax.all_gather(local_is_valid, axis_names, tiled=True)
        num_candidates_found = jax.lax.psum(local_count, axis_names)

        # Valid entries first, in row-major order (x, then y, then z). A
        # lexicographic sort avoids a flat index, which overflows int32 on
        # large grids.
        order = jnp.lexsort(
            (indices[:, 2], indices[:, 1], indices[:, 0], ~is_valid)
        )[:max_num_candidates]
        candidate_indices = jnp.where(
            is_valid[order, None],
            indices[order],
            0,
        )
        return candidate_indices, num_candidates_found

    return _shard_map(
        _local_list,
        mesh=named_mesh,
        in_specs=(PartitionSpec(*axis_names),),
        out_specs=(PartitionSpec(), PartitionSpec()),
        **_NO_REPLICATION_CHECK,
    )(candidate_mask)


# -------------------------------------------------------------
# =================== ↑ Search for candidates ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============== ↓ Energies of a control volume ↓ ==============
# -------------------------------------------------------------


class ControlVolumeEnergies(NamedTuple):
    """The centre-of-mass velocity and the energies of the gas in control
    volumes, each with the leading shape of the volumes."""

    #: The centre-of-mass velocity (Eq. 12), with a trailing vector axis.
    center_of_mass_velocity: jax.Array

    #: The thermal energy E_th = ½ Σ M c_s² (Eq. 5).
    thermal_energy: jax.Array

    #: The kinetic energy relative to the centre-of-mass motion,
    #: E_kin = ½ Σ M |v − v_cm|² (Eq. 11).
    kinetic_energy: jax.Array

    #: The magnetic energy E_mag = ½ Σ |B|² ΔV (Eq. 9, with the magnetic energy
    #: density ½ B² of astronomix's field units instead of the paper's B² / 8π).
    magnetic_energy: jax.Array

    #: The magnitude of the gravitational energy, |E_grav| = Σ M (φ_ref − φ).
    gravitational_energy_magnitude: jax.Array


def _control_volume_energies(
    cell_mass: jax.Array,
    cell_volume: jax.Array,
    velocity: jax.Array,
    sound_speed_squared: jax.Array,
    magnetic_field_squared,
    potential_depth: jax.Array,
) -> ControlVolumeEnergies:
    """
    The centre-of-mass velocity and the energies of the gas in control
    volumes, for the Jeans instability and bound state checks (Federrath et
    al. 2010, Sections 2.2.5 and 2.2.6).

    Each cell enters with a mass and a volume, so that a cell outside a volume
    (mass and volume 0) or one counted with a weight drops out of every sum.
    The gravitational energy is measured from a reference potential φ_ref, the
    largest potential in the volume in Federrath's FLASH code: measuring from
    it removes the part of the potential that is constant across the volume,
    which the gas outside the volume contributes.

    Args:
        cell_mass: The gas mass M of each cell, shape (..., num_cells).
        cell_volume: The volume of each cell, shape (..., num_cells), for the
            magnetic energy.
        velocity: The gas velocity of each cell, shape (..., num_cells, 3).
        sound_speed_squared: c_s² of each cell, shape (..., num_cells).
        magnetic_field_squared: |B|² of each cell, shape (..., num_cells), or
            ``None`` without a magnetic field.
        potential_depth: φ_ref − φ of each cell, shape (..., num_cells).

    Returns:
        The centre-of-mass velocity and the energies of each volume. A volume
        without gas has the centre-of-mass velocity 0.
    """
    gas_mass = jnp.sum(cell_mass, axis=-1)
    safe_gas_mass = jnp.where(gas_mass > 0.0, gas_mass, 1.0)
    center_of_mass_velocity = (
        jnp.einsum("...k,...kd->...d", cell_mass, velocity) / safe_gas_mass[..., None]
    )

    thermal_energy = 0.5 * jnp.sum(cell_mass * sound_speed_squared, axis=-1)

    velocity_relative_to_center_of_mass = (
        velocity - center_of_mass_velocity[..., None, :]
    )
    kinetic_energy = 0.5 * jnp.sum(
        cell_mass * jnp.sum(velocity_relative_to_center_of_mass**2, axis=-1),
        axis=-1,
    )

    if magnetic_field_squared is None:
        magnetic_energy = jnp.zeros_like(thermal_energy)
    else:
        magnetic_energy = 0.5 * jnp.sum(
            magnetic_field_squared * cell_volume,
            axis=-1,
        )

    gravitational_energy_magnitude = jnp.sum(cell_mass * potential_depth, axis=-1)

    return ControlVolumeEnergies(
        center_of_mass_velocity=center_of_mass_velocity,
        thermal_energy=thermal_energy,
        kinetic_energy=kinetic_energy,
        magnetic_energy=magnetic_energy,
        gravitational_energy_magnitude=gravitational_energy_magnitude,
    )


# -------------------------------------------------------------
# ============== ↑ Energies of a control volume ↑ ==============
# -------------------------------------------------------------
