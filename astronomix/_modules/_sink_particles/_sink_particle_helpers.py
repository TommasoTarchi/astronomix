"""
Helpers shared by the sink particle modules: the slot arrays (empty slots,
padding of the sinks passed into a run, the error when the slots run out),
the control volume of the creation checks, the squared sound speed and the
density threshold (Federrath et al. 2010, ApJ 713, 269, Eq. 32), the mapping
of cell indices across the box boundaries, and the fixed-length list of
candidate cells.
"""

# general
import inspect

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
    GHOST_CELLS,
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
from astronomix._fluid_equations._equation_of_state import sound_speed_squared
from astronomix._geometry.boundaries import _axis_slice, _periodic_boundaries


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
    Create sink particle arrays with every slot empty: every field is zero.

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
        home_cell=jnp.zeros((max_num_sinks, 3), dtype=jnp.int32),
        generation=jnp.zeros((max_num_sinks,), dtype=jnp.int32),
    )


def _pad_sink_particles(
    sink_particles: SinkParticles,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Bring the sink particles passed into a run to the layout the run uses:
    fill in the fields left as ``None`` and append empty slots up to
    ``max_num_sinks``.

    Sinks placed by hand, or restored from a checkpoint that holds only mass,
    position and velocity, have no ``activation``, ``home_cell`` or
    ``generation``. A slot with mass gets activation 1, the cell containing
    its position as home cell and generation 0; a slot with zero mass is
    empty. Slots are appended e.g. to sinks restored from a checkpoint of a
    run with fewer slots; the given slots stay at the front.

    Args:
        sink_particles: The sink particles passed into the run.
        config: The simulation configuration; supplies the number of slots
            and the grid spacing.

    Returns:
        Sink particles with every field set and ``max_num_sinks`` slots.

    Raises:
        ValueError: If the sink particles have more slots than
            ``max_num_sinks``; dropping slots could drop active ones.
    """
    max_num_sinks = config.sink_particle_config.max_num_sinks
    num_slots = sink_particles.mass.shape[0]
    if num_slots > max_num_sinks:
        raise ValueError(
            f"{num_slots} sink particle slots passed in, but max_num_sinks is "
            f"{max_num_sinks}. Increase max_num_sinks in SinkParticleConfig."
        )

    mass = jnp.asarray(sink_particles.mass)
    activation = sink_particles.activation
    if activation is None:
        activation = (mass > 0.0).astype(mass.dtype)
    is_active = jnp.asarray(activation) > 0.0
    home_cell = sink_particles.home_cell
    if home_cell is None:
        containing_cell = jnp.floor(
            jnp.asarray(sink_particles.position) / config.grid_spacing
        ).astype(jnp.int32)
        home_cell = jnp.where(is_active[:, None], containing_cell, 0)
    generation = sink_particles.generation
    if generation is None:
        generation = jnp.zeros((num_slots,), dtype=jnp.int32)
    sink_particles = sink_particles._replace(
        activation=activation,
        home_cell=home_cell,
        generation=generation,
    )

    num_missing_slots = max_num_sinks - num_slots
    return jax.tree.map(
        lambda field: jnp.pad(
            field,
            [(0, num_missing_slots)] + [(0, 0)] * (field.ndim - 1),
        ),
        sink_particles,
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
        c_s² = γP/ρ for an ideal gas, the constant isothermal c_s², or
        Γ K ρ^(Γ−1) for a polytropic gas, with the shape of the density.
    """
    density = primitive_state[registered_variables.density_index]
    # A barotropic state has no pressure row.
    pressure = None
    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
    return jnp.broadcast_to(
        jnp.asarray(
            sound_speed_squared(density, pressure, params, config),
            dtype=density.dtype,
        ),
        density.shape,
    )


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


def _polytropic_resolved_density(
    config: SimulationConfig,
    params: SimulationParams,
):
    """
    The density ρ* at which a polytropic gas is at its own density threshold,
    ρ* = ρ_res(ρ*). With c_s² = Γ K ρ^(Γ−1),

        ρ* = (π Γ K / (4 G r_acc²))^(1 / (2 − Γ)),   Γ < 2.

    Gas denser than ρ* is above its threshold, and gas at ρ* has the Jeans
    length 2 r_acc.

    Args:
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        ρ*, a scalar.
    """
    accretion_radius = (
        config.sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    )
    exponent = params.polytropic_exponent
    return (
        jnp.pi
        * exponent
        * params.polytropic_constant
        / (4.0 * params.gravitational_constant * accretion_radius**2)
    ) ** (1.0 / (2.0 - exponent))


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
