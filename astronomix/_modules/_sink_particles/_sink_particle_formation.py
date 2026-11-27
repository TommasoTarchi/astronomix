"""
Sink particle formation with fixed seeds, following the creation checks of
Federrath et al. (2010), ApJ 713, 269, Section 2.2, made smooth
(``_sink_particle_smoothing.py``).

Every cell holds an inactive seed from the start of the run, and a seed turns
into a sink gradually. Once per time step every seed gets a creation weight
w in [0, 1], the product of one factor per creation check, and its activation
grows to max(a, w). A seed of activation 0 has no effect on anything; one of
activation 1 is a sink. A seed is stored in a slot of ``SinkParticles`` from
the step its activation first exceeds 0, with its cell as home cell; the
inactive seed of the next generation then takes its place in that cell.

The factors of seed s, at position y_s, are

    f_ρ    = step_ω(ρ_s / ρ_res − 1), the seed's mass added to its density,
    f_conv = Π_k [1 − step_ω(−δ_k)], the six converging-flow velocities,
    f_φ    = 1 − step_ω((φ_s − min φ) / c_s²), with min φ the lowest
             potential within r_acc of y_s,
    f_J    = step_ω(|E_grav| / (2 E_th + E_mag) − 1),
    f_b    = step_ω(|E_grav| / (E_th + E_kin + E_mag) − 1),
    f_prox = Π_{s' ≠ s} [1 − a_s' K(|y_s' − y_s| − ℓ Δx)],
    f_home = Π_{s'} [1 − c_s' K(|y_s' − x_c(s)| − ℓ Δx)], over the seeds s'
             that activated before s in its home cell c(s), with c_s' the
             claim of s' on its home cell,

each exactly 0 where its hard check fails. A cell quantity is read at y_s
with the cell weights h, and the control volume holds the gas within r_acc
of y_s with the radial weights K, and its gravitational energy is measured
from the soft maximum of the potential at the temperature ω c_s². Only the
deepest cell of a control volume passes the potential-minimum check fully;
f_φ falls to 0 over ω c_s² above it. The potential φ is that of the gas, of
the seeds and of
any external field. All activations are updated together, with those of the
previous step in f_prox, so seeds that pass in the same step do not hold
each other back.

The weights of the inactive seeds are computed only where they can be above
0. f_ρ, f_conv, f_prox and f_home need no control volume, and f_φ is also
computed on the whole grid, with a slack for rounding: the interior cells
where all five can be above 0 are listed, in a list of fixed length, and only
the listed cells get the factors that need a control volume, f_φ again among
them. A cell leaving or entering the list has w = 0, so the list changes no
result unless it is full.

After the activations and the new seeds, every seed merges into the more
active seeds within its reach (``_sink_particle_merging.py``), giving up its
claim on its home cell with what it gives.

With every width at its hard limit (ω = ℓ = δ = 0), each factor is a hard
check of Federrath et al. (2010), a seed becomes a full sink in the first
step its cell passes them all, and no seed merges.

Only gas in interior cells counts. A control volume wraps around periodic
axes and ends at the edge of the box on non-periodic axes. Reflective walls
are treated like the other edges: mirrored gas is not counted.

The refinement check of Section 2.2.2 is not applied: astronomix uses a
uniform grid, so every cell is already on the highest level of refinement.
"""

# general
from functools import partial

# numerics
import numpy as np

# typing
from types import NoneType
from typing import NamedTuple, Union

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
    _sound_speed_squared,
)
from astronomix._modules._sink_particles._sink_particle_merging import _merge_seeds
from astronomix._modules._sink_particles._sink_particle_smoothing import (
    _cell_weights,
    _radial_weight,
    _safe_divide,
    _smooth_step,
    _soft_maximum,
)
from astronomix._stencil_operations._stencil_operations import _shift


# -------------------------------------------------------------
# ===================== ↓ Cell quantities ↓ ===================
# -------------------------------------------------------------


class CellQuantities(NamedTuple):
    """The cell quantities that the creation and accretion checks read, on
    the (padded) grid. Accretion reads only the first four, and its
    quantities leave the others ``None``."""

    #: The gas density ρ.
    density: FIELD_TYPE

    #: The gas velocity, with the three components on the last axis.
    velocity: jax.Array

    #: The squared sound speed c_s².
    sound_speed_squared: FIELD_TYPE

    #: The density threshold ρ_res of Eq. 32.
    density_threshold: FIELD_TYPE

    #: The potential of the gas, of the seeds (their masses spread with the
    #: TSC weights of their motion) and the external potential.
    potential: Union[FIELD_TYPE, NoneType] = None

    #: The six velocity differences δ_k of the converging-flow check, in
    #: units of c_s and positive where the flow converges, shape (6, ...).
    converging_velocity: Union[jax.Array, NoneType] = None

    #: |B|², or ``None`` without a magnetic field.
    magnetic_field_squared: Union[FIELD_TYPE, NoneType] = None


def _gas_quantities(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> CellQuantities:
    """
    The cell quantities that accretion reads: ρ, v, c_s² and ρ_res.

    Args:
        primitive_state: The (padded) primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The cell quantities, with the potential, the converging-flow
        velocities and |B|² left ``None``.
    """
    sound_speed_squared = _sound_speed_squared(
        primitive_state,
        config,
        params,
        registered_variables,
    )
    return CellQuantities(
        density=primitive_state[registered_variables.density_index],
        velocity=jnp.stack(
            [
                primitive_state[registered_variables.velocity_index.x],
                primitive_state[registered_variables.velocity_index.y],
                primitive_state[registered_variables.velocity_index.z],
            ],
            axis=-1,
        ),
        sound_speed_squared=sound_speed_squared,
        density_threshold=_density_threshold(sound_speed_squared, config, params),
    )


def _cell_quantities(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> CellQuantities:
    """
    The cell quantities of the creation checks, with one Poisson solve of the
    gas density plus the seeds' masses.

    Along each axis d, the converging-flow check compares the cell with its
    neighbours at ±1: δ = (v_d(0) − v_d(+1)) / c_s and (v_d(−1) − v_d(0)) / c_s,
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
    cells = _gas_quantities(primitive_state, config, params, registered_variables)
    density = cells.density
    potential = _compute_total_potential(
        density + _sink_density(sink_particles, density.shape, config),
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )

    sound_speed = jnp.sqrt(cells.sound_speed_squared)
    converging_velocity = []
    for axis in range(3):
        velocity_component = cells.velocity[..., axis]
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

    return cells._replace(
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


def _minimum_image(
    separation: jax.Array,
    config: SimulationConfig,
) -> jax.Array:
    """
    Separations taken to the nearest periodic copy along periodic axes.

    Args:
        separation: The separations, with the three components on the last
            axis.
        config: The simulation configuration.

    Returns:
        The separations, with the shape of ``separation``.
    """
    is_periodic, box_length = _periodic_box(config)
    return jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )


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

    containing_cell = jnp.floor(position / grid_spacing).astype(jnp.int32)
    cell_indices, interior_index, is_real_cell = _interior_cell_indices(
        containing_cell[:, None, :] + offsets[None, :, :],
        grid_shape,
        config,
    )

    cell_centers = (interior_index + 0.5) * grid_spacing
    separation = _minimum_image(cell_centers - position[:, None, :], config)
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


def _separation_squared(
    position: jax.Array,
    other_position: jax.Array,
    config: SimulationConfig,
) -> jax.Array:
    """
    The squared distances between two sets of points, to the nearest
    periodic copy along periodic axes.

    Args:
        position: The first points, shape (num_positions, 3).
        other_position: The second points, shape (num_other_positions, 3).
        config: The simulation configuration.

    Returns:
        The squared distances, shape (num_positions, num_other_positions).
    """
    separation = _minimum_image(
        position[:, None, :] - other_position[None, :, :],
        config,
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

    #: ρ / ρ_res, with the seed's own mass added to its density.
    density_ratio: jax.Array

    #: The six converging-flow velocities δ_k, shape (num_positions, 6).
    converging_velocity: jax.Array

    #: (φ − min φ) / c_s², the height above the lowest potential of the
    #: control volume.
    potential_height: jax.Array

    #: |E_grav| / (2 E_th + E_mag).
    jeans_ratio: jax.Array

    #: |E_grav| / (E_th + E_kin + E_mag).
    bound_ratio: jax.Array

    #: The centre-of-mass velocity of the control volume, shape
    #: (num_positions, 3).
    center_of_mass_velocity: jax.Array


def _read_at_positions(
    field: FIELD_TYPE,
    neighbourhood: SlotNeighbourhood,
) -> jax.Array:
    """
    A cell quantity q read at each position with the cell weights,
    Σ h q / Σ h.

    Args:
        field: The (padded) field of q.
        neighbourhood: The cells around each position.

    Returns:
        q at each position, shape (num_positions,); 0 at a position with no
        cell weight above 0.
    """
    cell_weight = neighbourhood.cell_weight
    return _safe_divide(
        jnp.sum(cell_weight * field[neighbourhood.cell_indices], axis=-1),
        jnp.sum(cell_weight, axis=-1),
    )


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
    potential, at the temperature ω c_s² with c_s² read at the position, and
    the potential height from the lowest potential of the cells with K > 0.

    Args:
        neighbourhood: The cells around each position.
        mass: The mass of the seed at each position, added to its density,
            shape (num_positions,).
        cells: The cell quantities.
        config: The simulation configuration.

    Returns:
        The quantities of the checks.
    """
    check_width = config.sink_particle_config.check_width
    cell_volume = config.grid_spacing**3
    cell_indices = neighbourhood.cell_indices
    radial_weight = neighbourhood.radial_weight

    def read(field):
        return _read_at_positions(field, neighbourhood)

    density = read(cells.density) + mass / cell_volume
    density_threshold = read(cells.density_threshold)
    sound_speed_squared = read(cells.sound_speed_squared)
    potential = read(cells.potential)
    converging_velocity = jnp.stack(
        [read(component) for component in cells.converging_velocity],
        axis=-1,
    )

    potential_in_reach = cells.potential[cell_indices]
    minimum_potential = jnp.min(
        jnp.where(radial_weight > 0.0, potential_in_reach, jnp.inf),
        axis=-1,
    )
    minimum_potential = jnp.where(jnp.isfinite(minimum_potential), minimum_potential, 0.0)
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
        has_cells=jnp.sum(neighbourhood.cell_weight, axis=-1) > 0.0,
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


# -------------------------------------------------------------
# =================== ↑ Checks at a position ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Creation weight ↓ ===================
# -------------------------------------------------------------


class CreationFactors(NamedTuple):
    """The factors of the creation weight of each seed, each of shape
    (num_positions,), in [0, 1]."""

    #: f_ρ = step_ω(ρ / ρ_res − 1).
    density: jax.Array

    #: f_conv = Π_k [1 − step_ω(−δ_k)].
    converging_flow: jax.Array

    #: f_φ = 1 − step_ω(potential height).
    potential_minimum: jax.Array

    #: f_J = step_ω(Jeans ratio − 1).
    jeans_instability: jax.Array

    #: f_b = step_ω(bound ratio − 1).
    bound_state: jax.Array

    #: f_prox, from the active seeds around.
    proximity: jax.Array

    #: f_home, from the earlier seeds of the home cell.
    home: jax.Array


def _exclusion_factor(
    distance_squared: jax.Array,
    strength: jax.Array,
    counts: jax.Array,
    config: SimulationConfig,
) -> jax.Array:
    """
    Π [1 − s K(d − ℓ Δx)] over the seeds that count, with s the strength of
    each seed: 0 where a seed of strength 1 lies within r_acc, 1 where every
    seed that counts lies farther than r_acc + ℓ Δx. With the activations as
    strengths this is f_prox, with the claims on the home cell as strengths
    and the distances from the home cell it is f_home.

    Args:
        distance_squared: The squared distance d² to each seed, shape
            (num_positions, num_slots).
        strength: The strength s of each seed, shape (num_slots,).
        counts: Whether each seed counts, shape (num_positions, num_slots).
        config: The simulation configuration.

    Returns:
        The factor, shape (num_positions,).
    """
    sink_particle_config = config.sink_particle_config
    spatial_width = sink_particle_config.spatial_width_in_cells * config.grid_spacing
    accretion_radius = sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    shifted_radial_weight = _radial_weight(
        distance_squared,
        accretion_radius + spatial_width,
        spatial_width,
    )
    return jnp.prod(
        jnp.where(counts, 1.0 - strength[None, :] * shifted_radial_weight, 1.0),
        axis=-1,
    )


def _creation_factors(
    checks: SlotChecks,
    proximity: jax.Array,
    home: jax.Array,
    config: SimulationConfig,
) -> CreationFactors:
    """
    The factors of the creation weight of a set of seeds.

    Args:
        checks: The quantities of the checks at every seed.
        proximity: f_prox of every seed.
        home: f_home of every seed.
        config: The simulation configuration.

    Returns:
        The factors.
    """
    check_width = config.sink_particle_config.check_width
    return CreationFactors(
        density=_smooth_step(checks.density_ratio - 1.0, check_width),
        converging_flow=jnp.prod(
            1.0 - _smooth_step(-checks.converging_velocity, check_width),
            axis=-1,
        ),
        potential_minimum=1.0 - _smooth_step(checks.potential_height, check_width),
        jeans_instability=_smooth_step(checks.jeans_ratio - 1.0, check_width),
        bound_state=_smooth_step(checks.bound_ratio - 1.0, check_width),
        proximity=proximity,
        home=home,
    )


def _creation_weight(
    checks: SlotChecks,
    factors: CreationFactors,
) -> jax.Array:
    """
    The creation weight w, the product of the factors; 0 for a seed without
    a cell to read its values from.

    Args:
        checks: The quantities of the checks at every seed.
        factors: The factors of every seed.

    Returns:
        w, shape (num_positions,).
    """
    return jnp.where(
        checks.has_cells,
        factors.density
        * factors.converging_flow
        * factors.potential_minimum
        * factors.jeans_instability
        * factors.bound_state
        * factors.proximity
        * factors.home,
        0.0,
    )


# -------------------------------------------------------------
# ===================== ↑ Creation weight ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Potential minimum on the grid ↓ ===========
# -------------------------------------------------------------

#: How much looser the grid-wide potential-minimum check of the cell
#: selection is than f_φ > 0, which holds for a potential height h < ω (h ≤ 0
#: at ω = 0), as a share of |φ| / c_s². The grid and ``_slot_checks`` take
#: the same minimum, but ``_slot_checks`` reads φ and c_s² at the cell centre
#: with the cell weights, so their heights can differ by rounding. A listed
#: cell with h ≥ ω has f_φ = 0 and so w = 0.
POTENTIAL_HEIGHT_SLACK_IN_POTENTIALS = 1e-4


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
    (φ − min φ) / c_s² in every cell, with the minimum over the cells of
    its control volume with K > 0: the potential height of ``_slot_checks``
    for a seed at a cell centre.

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
    accretion_radius_in_cells = sink_particle_config.accretion_radius_in_cells

    # The control volume; with an edge of width ℓ > 0, K is 0 at r_acc itself
    # (``_radial_weight``), so offsets at that distance are dropped.
    offsets = _control_volume_offsets(config)
    if sink_particle_config.spatial_width_in_cells > 0.0:
        distance_in_cells = np.sqrt(np.sum(offsets**2, axis=-1))
        offsets = offsets[distance_in_cells < accretion_radius_in_cells]

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

    return _safe_divide(potential - minimum, sound_speed_squared)


def _may_pass_potential_minimum(
    cells: CellQuantities,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Boolean mask of the cells whose inactive seed can have f_φ > 0, from the
    potential height on the grid, with the slack of
    ``POTENTIAL_HEIGHT_SLACK_IN_POTENTIALS``.

    Args:
        cells: The cell quantities.
        config: The simulation configuration.

    Returns:
        The mask, with the shape of the potential; meaningful in interior
        cells only.
    """
    check_width = config.sink_particle_config.check_width
    potential_height = _potential_height_on_grid(
        cells.potential,
        cells.sound_speed_squared,
        config,
    )
    slack = POTENTIAL_HEIGHT_SLACK_IN_POTENTIALS * _safe_divide(
        jnp.abs(cells.potential),
        cells.sound_speed_squared,
    )
    return potential_height <= check_width + slack


# -------------------------------------------------------------
# =============== ↑ Potential minimum on the grid ↑ ===========
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ Cells of the inactive seeds ↓ ============
# -------------------------------------------------------------


def _cells_without_weight(
    sink_particles: SinkParticles,
    grid_shape: tuple,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Boolean mask of the cells whose inactive seed has f_prox = 0 or
    f_home = 0: the cells within r_acc of a seed of activation 1, and the
    home cells of the seeds with their whole claim on it that lie within
    r_acc of it.

    Args:
        sink_particles: The sink particles.
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        The mask, with shape ``grid_shape``.
    """
    sink_particle_config = config.sink_particle_config
    spatial_width = sink_particle_config.spatial_width_in_cells * config.grid_spacing
    accretion_radius = sink_particle_config.accretion_radius_in_cells * config.grid_spacing

    def is_within_accretion_radius(distance_squared):
        """Whether K(d − ℓ Δx) = 1, the same test as ``_exclusion_factor``."""
        return (
            _radial_weight(
                distance_squared,
                accretion_radius + spatial_width,
                spatial_width,
            )
            >= 1.0
        )

    is_full_sink = sink_particles.activation >= 1.0

    neighbourhood = _slot_neighbourhood(sink_particles.position, grid_shape, config)
    near_full_sink = (
        is_full_sink[:, None]
        & neighbourhood.is_real_cell
        & is_within_accretion_radius(neighbourhood.distance_squared)
    )
    mask = jnp.zeros(grid_shape, dtype=bool).at[neighbourhood.cell_indices].max(
        near_full_sink
    )

    home_indices, home_interior_index, home_is_real_cell = _interior_cell_indices(
        sink_particles.home_cell,
        grid_shape,
        config,
    )
    home_centers = (home_interior_index + 0.5) * config.grid_spacing
    near_home = (
        (sink_particles.home_claim >= 1.0)
        & home_is_real_cell
        & is_within_accretion_radius(
            jnp.sum(
                _minimum_image(home_centers - sink_particles.position, config) ** 2,
                axis=-1,
            )
        )
    )
    return mask.at[home_indices].max(near_home)


# -------------------------------------------------------------
# ================ ↑ Cells of the inactive seeds ↑ ============
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _form_sink_particles(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> SinkParticles:
    """
    Raise the activation of every active seed to its creation weight, store
    the inactive seeds whose weight is above 0 in free slots, and merge every
    seed into the more active seeds within its reach.

    An active seed gets w at its position, with its own mass added to its
    density, and a ← max(a, w). An inactive seed sits at the centre of its
    cell with no mass. When its weight is above 0 it takes the next free
    slot, with activation w, its cell as home cell, the number of active
    seeds of that home cell as generation, its whole claim on that cell, and
    the centre-of-mass velocity of its control volume (Eq. 12). It gets its
    mass from the merging and from the accretion step that follows. All
    weights use the activations the seeds had before this call.

    Args:
        primitive_state: The (padded) primitive state after the hydro update.
        sink_particles: The current sink particles.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The sink particles with the new activations, the new seeds and the
        merged seeds.
    """
    sink_particle_config = config.sink_particle_config
    max_num_sinks = sink_particle_config.max_num_sinks
    max_num_candidates = sink_particle_config.max_num_candidates
    check_width = sink_particle_config.check_width
    grid_spacing = config.grid_spacing

    cells = _cell_quantities(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    grid_shape = cells.density.shape
    activation = sink_particles.activation
    is_active = activation > 0.0

    # -------------------------------------------------------------
    # ============== ↓ Weights of the active seeds ↓ ==============
    # -------------------------------------------------------------

    slot_checks = _slot_checks(
        _slot_neighbourhood(sink_particles.position, grid_shape, config),
        sink_particles.mass,
        cells,
        config,
    )
    is_other_active_seed = is_active[None, :] & ~jnp.eye(max_num_sinks, dtype=bool)
    slot_proximity = _exclusion_factor(
        _separation_squared(sink_particles.position, sink_particles.position, config),
        activation,
        is_other_active_seed,
        config,
    )

    # The earlier seeds of each seed's home cell: active, with the same home
    # cell and a lower generation.
    home_cell = sink_particles.home_cell
    generation = sink_particles.generation
    is_earlier_seed_of_home = (
        is_active[None, :]
        & jnp.all(home_cell[None, :, :] == home_cell[:, None, :], axis=-1)
        & (generation[None, :] < generation[:, None])
    )
    slot_home = _exclusion_factor(
        _separation_squared(
            (home_cell + 0.5) * grid_spacing,
            sink_particles.position,
            config,
        ),
        sink_particles.home_claim,
        is_earlier_seed_of_home,
        config,
    )

    slot_weight = _creation_weight(
        slot_checks,
        _creation_factors(slot_checks, slot_proximity, slot_home, config),
    )
    new_activation = jnp.where(is_active, jnp.maximum(activation, slot_weight), 0.0)

    # -------------------------------------------------------------
    # ============== ↑ Weights of the active seeds ↑ ==============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ========= ↓ Stage 1: cells where w can be above 0 ↓ =========
    # -------------------------------------------------------------

    # The listing only decides where weights are computed, and every cell left
    # out has w = 0, so it carries no derivative.
    grid_cells = jax.lax.stop_gradient(cells)
    density_factor = _smooth_step(
        _safe_divide(grid_cells.density, grid_cells.density_threshold) - 1.0,
        check_width,
    )
    converging_flow_factor = jnp.prod(
        1.0 - _smooth_step(-grid_cells.converging_velocity, check_width),
        axis=0,
    )
    candidate_mask = (
        _interior_mask(grid_cells.density, config)
        & (density_factor > 0.0)
        & (converging_flow_factor > 0.0)
        & _may_pass_potential_minimum(grid_cells, config)
        & ~_cells_without_weight(
            jax.lax.stop_gradient(sink_particles),
            grid_shape,
            config,
        )
    )

    # JIT needs fixed array sizes, so the cells are collected into a list of
    # fixed length; unused entries are marked as invalid.
    candidate_indices, num_candidates_found = _collect_candidates(
        candidate_mask,
        max_num_candidates,
    )
    candidate_is_valid = jnp.arange(max_num_candidates) < num_candidates_found

    # -------------------------------------------------------------
    # ========= ↑ Stage 1: cells where w can be above 0 ↑ =========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ========== ↓ Stage 2: weights of the listed seeds ↓ =========
    # -------------------------------------------------------------

    # The inactive seed of each listed cell sits at the cell centre. Unused
    # entries of the list point at the corner of the padded grid; their
    # weight is set to 0.
    candidate_cell = candidate_indices - config.num_ghost_cells
    candidate_positions = (candidate_cell + 0.5) * grid_spacing
    candidate_checks = _slot_checks(
        _slot_neighbourhood(candidate_positions, grid_shape, config),
        jnp.zeros(max_num_candidates, dtype=candidate_positions.dtype),
        cells,
        config,
    )
    distance_squared_to_seeds = _separation_squared(
        candidate_positions,
        sink_particles.position,
        config,
    )
    candidate_proximity = _exclusion_factor(
        distance_squared_to_seeds,
        activation,
        jnp.broadcast_to(is_active[None, :], distance_squared_to_seeds.shape),
        config,
    )
    # Every active seed of a listed cell activated before its inactive seed.
    is_seed_of_cell = is_active[None, :] & jnp.all(
        home_cell[None, :, :] == candidate_cell[:, None, :],
        axis=-1,
    )
    candidate_home = _exclusion_factor(
        distance_squared_to_seeds,
        sink_particles.home_claim,
        is_seed_of_cell,
        config,
    )
    candidate_weight = jnp.where(
        candidate_is_valid,
        _creation_weight(
            candidate_checks,
            _creation_factors(
                candidate_checks,
                candidate_proximity,
                candidate_home,
                config,
            ),
        ),
        0.0,
    )
    is_new_seed = candidate_weight > 0.0

    # -------------------------------------------------------------
    # ========== ↑ Stage 2: weights of the listed seeds ↑ =========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Storing new seeds ↓ =====================
    # -------------------------------------------------------------

    # Each new seed gets the next free slot; the other entries get the
    # out-of-range slot max_num_sinks. With mode="drop", writes to an
    # out-of-range slot are discarded, which also discards new seeds once
    # every slot is taken.
    free_slots = jnp.nonzero(~is_active, size=max_num_sinks, fill_value=max_num_sinks)[0]
    new_seed_rank = jnp.cumsum(is_new_seed) - 1
    slot = jnp.where(
        is_new_seed & (new_seed_rank < max_num_sinks),
        free_slots[jnp.clip(new_seed_rank, 0, max_num_sinks - 1)],
        max_num_sinks,
    )
    sink_particles = SinkParticles(
        mass=sink_particles.mass.at[slot].set(0.0, mode="drop"),
        position=sink_particles.position.at[slot].set(
            candidate_positions,
            mode="drop",
        ),
        velocity=sink_particles.velocity.at[slot].set(
            candidate_checks.center_of_mass_velocity,
            mode="drop",
        ),
        activation=new_activation.at[slot].set(candidate_weight, mode="drop"),
        home_cell=home_cell.at[slot].set(
            candidate_cell.astype(home_cell.dtype),
            mode="drop",
        ),
        generation=generation.at[slot].set(
            jnp.sum(is_seed_of_cell, axis=-1).astype(generation.dtype),
            mode="drop",
        ),
        home_claim=sink_particles.home_claim.at[slot].set(1.0, mode="drop"),
    )
    num_new_seeds = jnp.sum(is_new_seed)
    num_free_slots = jnp.sum(~is_active)

    # Cells beyond max_num_candidates are only skipped for this step, a
    # jump in their weights from w to 0, so a warning is printed. A new seed
    # without a free slot would be lost, leaving its gas to collapse without
    # one, so the run stops with an error instead (a host callback that
    # raises aborts the jitted computation).
    jax.lax.cond(
        num_candidates_found > max_num_candidates,
        lambda: jax.debug.print(
            "WARNING: {} cells may form sink particles, only the first {} are "
            "checked this step (increase max_num_candidates).",
            num_candidates_found,
            max_num_candidates,
        ),
        lambda: None,
    )
    jax.lax.cond(
        num_new_seeds > num_free_slots,
        lambda: jax.debug.callback(
            _raise_sink_slots_full,
            num_new_seeds - num_free_slots,
            max_num_sinks,
        ),
        lambda: None,
    )

    # -------------------------------------------------------------
    # ================= ↑ Storing new seeds ↑ =====================
    # -------------------------------------------------------------

    position = sink_particles.position
    return _merge_seeds(
        sink_particles,
        _minimum_image(position[None, :, :] - position[:, None, :], config),
        _read_at_positions(
            cells.density_threshold,
            _slot_neighbourhood(position, grid_shape, config),
        ),
        config,
    )
