"""
Motion of sink particles and their gravity, following Gong & Ostriker (2013),
ApJS 204, 8.

Every gravitational interaction that involves a sink goes through the grid
(particle-mesh): each sink's mass is spread over the 3×3×3 cells around it with
triangular-shaped-cloud (TSC) weights (Gong & Ostriker 2013, Eq. 16) and added
to the gas density of the Poisson solve, and the force on the sink is read
back from the same cells with the same weights. Depositing and reading back
with the same weights gives a sink no force from itself and makes the forces
between any two masses equal and opposite. The TSC weights and their first
derivative are continuous in the sink position, so derivatives with respect to
the positions have no jumps.

The sinks move with a kick-drift-kick leapfrog (Gong & Ostriker 2013, Section
2.4), once per hydro step, and adaptive steps keep a sink from moving more than
half a cell per step (Federrath et al. 2010, ApJ 713, 269, Eq. 22).

The TSC cells follow the gas boundaries: they wrap around periodic axes and end
at the edge of the box on non-periodic ones. Where the Poisson solve treats the
box differently, a sink within reach of the edge (2 cells, 4 on the
finite-difference path) feels a force from itself, and the forces between
masses are not exactly equal and opposite. This happens with periodic gas and
``poisson_manual_open_boundaries``, with mixed boundaries (solved as fully
open), and with classic finite volume on a periodic box, whose ghost cells lie
inside the periodic Poisson domain.
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
    FINITE_DIFFERENCE,
    STATE_TYPE,
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
)
from astronomix._modules._sink_particles._sink_particle_formation import (
    _interior_cell_indices,
)
from astronomix._stencil_operations._stencil_operations import _stencil_add


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


def _acceleration_field(
    gravitational_potential: FIELD_TYPE,
    config: SimulationConfig,
) -> jax.Array:
    """
    The gravitational acceleration −∇Φ at the cell centres, with the centred
    difference that the gas coupling of the active solver applies to the gas
    momentum.

    With the same difference on both sides, the force of a sink on the gas and
    the force of the gas on the sink are equal and opposite. All three
    finite-difference couplings use the 6th-order difference
    (``_gravitational_acceleration``); classic finite volume and VL2 use the
    2nd-order one, −(Φᵢ₊₁ − Φᵢ₋₁) / (2Δx), which is also the average
    ½(g₋½ + g₊½) of the face accelerations of VL2. Any centred difference is
    antisymmetric, so a sink feels no force from itself.

    Args:
        gravitational_potential: The total potential at the cell centres.
        config: The simulation configuration.

    Returns:
        The acceleration, shape (3, nx, ny, nz).
    """
    grid_spacing = config.grid_spacing
    if config.solver_mode == FINITE_DIFFERENCE:
        return jnp.stack(
            [
                _gravitational_acceleration(
                    gravitational_potential,
                    spatial_axis,
                    grid_spacing,
                )
                for spatial_axis in range(3)
            ]
        )
    return jnp.stack(
        [
            -_stencil_add(
                gravitational_potential,
                indices=(1, -1),
                factors=(1.0, -1.0),
                axis=spatial_axis,
            )
            / (2.0 * grid_spacing)
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

    The sink density is added to the gas density of one Poisson solve, the
    acceleration is taken at the cell centres with the stencil of the gas
    coupling (``_acceleration_field``), and each sink reads it back from its
    27 TSC cells with the weights it was deposited with. Empty slots get the
    finite acceleration at their position, which the kick ignores.

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
        gas_density,
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
        sink_density=_sink_density(sink_particles, grid_shape, config),
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
