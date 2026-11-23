"""
Building blocks of the smooth sink formation and accretion: a smooth step for
the checks, a soft minimum and maximum, the radial weight of the accretion
radius, the cell weights that read a cell quantity at a sink, and the
point-mass potential of the bound check.

Each block has a width or a sharpness, a static configuration value. Its
formula divides by the width, so the hard case (width 0, sharpness ∞) is a
separate branch, chosen at trace time. The hard branch is the limit of the
formula, except where a docstring names a difference: there it follows the
hard checks of Federrath et al. (2010) as the rest of the sink code applies
them.

Distances enter squared, so that the derivatives stay finite where a distance
is 0.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_dynamics import _tsc_offsets
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _interior_cell_indices,
)


def _smooth_step(x: jax.Array, width: float) -> jax.Array:
    """
    The smooth step of a strict check x > 0: 0 for x ≤ 0, 1 for x ≥ ω, and
    3 t² − 2 t³ with t = x / ω in between.

    It is exactly 0 or 1 outside the band of width ω, and its derivative is
    continuous, 0 at both ends of the band. A non-strict check x ≥ 0 is
    1 − step(−x).

    Args:
        x: The checked quantity.
        width: The width ω of the band, static. At 0, the step is 1 for x > 0
            and 0 otherwise.

    Returns:
        The smooth step, with the shape of ``x``.
    """
    if width == 0.0:
        return (x > 0.0).astype(jnp.result_type(x))
    scaled = jnp.clip(x / width, 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _safe_distance(distance_squared: jax.Array) -> jax.Array:
    """
    The distance from its square, with a derivative of 0 instead of NaN where
    the distance is 0.

    Args:
        distance_squared: The squared distance.

    Returns:
        The distance.
    """
    is_positive = distance_squared > 0.0
    return jnp.where(
        is_positive,
        jnp.sqrt(jnp.where(is_positive, distance_squared, 1.0)),
        0.0,
    )


def _soft_minimum(
    values: jax.Array,
    weights: jax.Array,
    check_width: float,
    temperature_scale: jax.Array,
) -> jax.Array:
    """
    The soft minimum over the last axis,

        min_T[f] = Σ k f e^(−f/T) / Σ k e^(−f/T),

    with weights k and the temperature T = ω × ``temperature_scale``.

    Being a weighted average of the values, it never lies below their
    minimum. The minimum is subtracted from the exponent (with its derivative
    stopped), so the largest exponential is 1. Cells are left out by a weight
    of 0, and their values are not read, even when infinite.

    Args:
        values: The values f, shape (..., n).
        weights: The weights k ≥ 0, shape (..., n).
        check_width: The width ω, static. At 0, the result is the minimum
            over the cells with k > 0.
        temperature_scale: The scale of the temperature, shape (...), e.g.
            c_s². Where the temperature is 0, the result is the minimum over
            the cells with k > 0, the limit T → 0.

    Returns:
        The soft minimum, shape (...); 0 where every weight is 0.
    """
    is_included = weights > 0.0
    any_included = jnp.any(is_included, axis=-1)
    minimum = jnp.where(
        any_included,
        jnp.min(jnp.where(is_included, values, jnp.inf), axis=-1),
        0.0,
    )
    if check_width == 0.0:
        return minimum

    reference = jax.lax.stop_gradient(minimum)
    included_values = jnp.where(is_included, values, reference[..., None])
    temperature = check_width * temperature_scale
    has_temperature = temperature > 0.0
    safe_temperature = jnp.where(has_temperature, temperature, 1.0)
    boltzmann_weights = weights * jnp.exp(
        -(included_values - reference[..., None]) / safe_temperature[..., None]
    )
    normalization = jnp.sum(boltzmann_weights, axis=-1)
    safe_normalization = jnp.where(normalization > 0.0, normalization, 1.0)
    soft_minimum = (
        jnp.sum(boltzmann_weights * included_values, axis=-1) / safe_normalization
    )
    soft_minimum = jnp.where(normalization > 0.0, soft_minimum, 0.0)
    return jnp.where(has_temperature, soft_minimum, minimum)


def _soft_maximum(
    values: jax.Array,
    weights: jax.Array,
    check_width: float,
    temperature_scale: jax.Array,
) -> jax.Array:
    """
    The soft maximum over the last axis, max_T[f] = −min_T[−f]: the soft
    minimum (``_soft_minimum``) at the temperature −T.

    Args:
        values: The values f, shape (..., n).
        weights: The weights k ≥ 0, shape (..., n).
        check_width: The width ω, static. At 0, the result is the maximum
            over the cells with k > 0.
        temperature_scale: The scale of the temperature, shape (...).

    Returns:
        The soft maximum, shape (...); 0 where every weight is 0.
    """
    return -_soft_minimum(-values, weights, check_width, temperature_scale)


def _radial_weight(
    distance_squared: jax.Array,
    accretion_radius: float,
    width: float,
) -> jax.Array:
    """
    The radial weight K(r) = step_w(r_acc − r): 1 within r_acc − w of a sink,
    0 beyond r_acc, smooth in between.

    Args:
        distance_squared: The squared distance r² from the sink.
        accretion_radius: The accretion radius r_acc.
        width: The width w = ℓ Δx of the band, static. At 0, the weight is 1
            for r ≤ r_acc, as in the hard checks; the limit of the formula
            would give 0 at r = r_acc.

    Returns:
        The weight, with the shape of ``distance_squared``.
    """
    distance = _safe_distance(distance_squared)
    if width == 0.0:
        return (distance <= accretion_radius).astype(jnp.result_type(distance))
    return _smooth_step(accretion_radius - distance, width)


def _cell_weights(
    position: jax.Array,
    grid_shape: tuple,
    config: SimulationConfig,
    spatial_width_in_cells: float,
) -> tuple[tuple, jax.Array]:
    """
    The cell weights h of the 27 cells around each sink: 1 on the cell that
    contains the sink, passing smoothly to a neighbouring cell within ½ ℓ cells
    of a face.

    Along each axis, with d the offset of the sink from a cell centre in
    cells, the weight is step_ℓ(½ + ½ ℓ − |d|), and the weight of a cell is
    the product over the axes. For ℓ ≤ 1 the weights of the cells around a
    sink sum to 1, except next to a non-periodic edge: cells past it get
    weight 0.

    Args:
        position: The sink positions, shape (num_sinks, 3).
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.
        spatial_width_in_cells: The width ℓ, in [0, 1], static. At 0, the
            weight is 1 on the cell that contains the sink, found with
            ``floor`` as in the motion and the accretion of the sinks, and 0
            elsewhere; on a face the sink belongs to the upper cell, where the
            limit of the formula would give 0 to both.

    Returns:
        ``(cell_indices, weights)``: a tuple of three integer index arrays into
        the padded field, and the weights, each of shape (num_sinks, 27).
    """
    offsets = _tsc_offsets()

    scaled_position = position / config.grid_spacing
    containing_cell = jnp.floor(scaled_position)
    offset_from_cell_center = scaled_position - (containing_cell + 0.5)

    # The 1D weights of the neighbours at −1, 0 and +1 along each axis, shape
    # (num_sinks, 3 axes, 3 neighbours).
    if spatial_width_in_cells == 0.0:
        weights_along_axes = jnp.broadcast_to(
            jnp.array([0.0, 1.0, 0.0], dtype=position.dtype),
            position.shape + (3,),
        )
    else:
        distance_to_centers = jnp.abs(
            offset_from_cell_center[..., None]
            - jnp.arange(-1, 2, dtype=position.dtype)
        )
        weights_along_axes = _smooth_step(
            0.5 + 0.5 * spatial_width_in_cells - distance_to_centers,
            spatial_width_in_cells,
        )

    # For every one of the 27 cells, pick the 1D weight of its offset along
    # each axis and multiply them, shape (num_sinks, 27).
    weights = jnp.prod(
        weights_along_axes[:, jnp.arange(3), offsets + 1],
        axis=-1,
    )

    cell_indices, _, is_real_cell = _interior_cell_indices(
        containing_cell.astype(jnp.int32)[:, None, :] + offsets[None, :, :],
        grid_shape,
        config,
    )
    return cell_indices, jnp.where(is_real_cell, weights, 0.0)


def _point_mass_potential(
    distance_squared: jax.Array,
    mass: jax.Array,
    accretion_radius: float,
    softening_length: float,
    gravitational_constant,
) -> jax.Array:
    """
    The potential of a sink as a point mass, softened at r = 0 and 0 at r_acc,

        ψ(r) = −G m (1 / √(r² + λ²) − 1 / √(r_acc² + λ²)),

    the point-mass term of the bound check of accretion.

    Args:
        distance_squared: The squared distance r² from the sink.
        mass: The sink mass m, broadcastable to ``distance_squared``.
        accretion_radius: The accretion radius r_acc.
        softening_length: The softening λ = ℓ Δx, static. At 0, r = 0 is
            treated as r = r_acc, where ψ is 0, as in the hard bound check;
            the limit of the formula would diverge there.
        gravitational_constant: The gravitational constant G.

    Returns:
        The potential, with the shape of ``distance_squared``.
    """
    if softening_length == 0.0:
        is_positive = distance_squared > 0.0
        safe_distance_squared = jnp.where(
            is_positive,
            distance_squared,
            accretion_radius**2,
        )
        inverse_distance = 1.0 / jnp.sqrt(safe_distance_squared)
        inverse_accretion_radius = 1.0 / accretion_radius
    else:
        softening_squared = softening_length**2
        inverse_distance = 1.0 / jnp.sqrt(distance_squared + softening_squared)
        inverse_accretion_radius = 1.0 / jnp.sqrt(
            accretion_radius**2 + softening_squared
        )
    return -gravitational_constant * mass * (inverse_distance - inverse_accretion_radius)
