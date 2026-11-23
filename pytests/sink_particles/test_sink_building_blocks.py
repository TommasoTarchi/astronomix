"""
The building blocks of the smooth sink formation and accretion
(``_sink_particle_smoothing.py``): each is exact outside its band, reduces to
its hard branch as its width goes to 0, and has finite derivatives where a
distance or a volume is 0.
"""

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import BoundarySettings

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_dynamics import _tsc_offsets
from astronomix._modules._sink_particles._sink_particle_smoothing import (
    _cell_weights,
    _point_mass_potential,
    _radial_weight,
    _smooth_step,
    _soft_maximum,
    _soft_minimum,
    _soft_minimum_weights,
)
from astronomix.time_stepping.time_integration import _prepare_padded_state

# sink particle test helpers
from _sink_helpers import CELL_SIZE, SETTINGS, _setup

pytestmark = pytest.mark.x64

#: A width small enough that a smooth block equals its hard branch at every
#: test point farther than 1e-3 from the threshold.
SMALL_WIDTH = 1e-6

#: The index of the containing cell, offset (0, 0, 0), among the 27 cells.
CONTAINING_CELL = int(np.flatnonzero(np.all(_tsc_offsets() == 0, axis=-1))[0])


def _padded_grid(settings=SETTINGS):
    """The configuration of the clump setup and the shape of a padded field.

    Args:
        settings: The problem constants of the setup.

    Returns:
        ``(config, grid_shape)``.
    """
    state, config, params, registered_variables = _setup(settings=settings)
    padded_state = _prepare_padded_state(
        state.primitive_state,
        config,
        params,
        registered_variables,
    )
    return config, padded_state.shape[1:]


# -------------------------------------------------------------
# ======================= ↓ Smooth step ↓ =====================
# -------------------------------------------------------------


def test_smooth_step_is_exact_outside_its_band():
    """The smooth step is exactly 0 for x ≤ 0 and exactly 1 for x ≥ ω, and
    rises strictly in between, through ½ at the middle of the band."""
    width = 0.3
    outside = jnp.array([-1.0, -1e-12, 0.0, 0.3, 0.31, 2.0])
    np.testing.assert_array_equal(
        _smooth_step(outside, width),
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        err_msg="the smooth step is not exactly 0 below its band and 1 above it",
    )

    inside = _smooth_step(jnp.array([0.075, 0.15, 0.225]), width)
    assert jnp.all(jnp.diff(inside) > 0.0) and 0.0 < inside[0] and inside[-1] < 1.0, (
        f"the smooth step does not rise strictly inside its band: {inside}"
    )
    np.testing.assert_allclose(
        inside[1], 0.5, rtol=1e-14, err_msg="the smooth step is not ½ at the middle of its band"
    )


def test_smooth_step_derivative_vanishes_at_band_ends():
    """The derivative of the smooth step is continuous: 0 at both ends of the
    band and 3 / (2ω) at its middle."""
    width = 0.3
    derivative = jax.vmap(jax.grad(lambda x: _smooth_step(x, width)))
    slopes = derivative(jnp.array([1e-9, width - 1e-9, 0.5 * width]))
    assert abs(slopes[0]) < 1e-6 and abs(slopes[1]) < 1e-6, (
        f"the derivative at the ends of the band is {slopes[:2]}, not 0"
    )
    np.testing.assert_allclose(
        slopes[2], 1.5 / width, rtol=1e-12, err_msg="the derivative at the middle is not 3 / (2ω)"
    )


def test_smooth_step_reduces_to_hard_step():
    """At width 0 the step is 1 for x > 0 and 0 otherwise, also at x = 0; a
    small width gives the same values away from 0."""
    x = jnp.concatenate([jnp.linspace(-1.0, -1e-3, 20), jnp.linspace(1e-3, 1.0, 20)])
    hard = _smooth_step(x, 0.0)
    np.testing.assert_array_equal(hard, x > 0.0, err_msg="the hard step is not x > 0")
    assert _smooth_step(jnp.array(0.0), 0.0) == 0.0, "the hard step is not 0 at x = 0"
    np.testing.assert_array_equal(
        _smooth_step(x, SMALL_WIDTH),
        hard,
        err_msg="a narrow smooth step differs from the hard step away from 0",
    )


# -------------------------------------------------------------
# ======================= ↑ Smooth step ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ Soft minimum and maximum ↓ ===============
# -------------------------------------------------------------


def _random_volumes():
    """Values and weights of 5 volumes of 20 cells, every fourth cell left
    out with weight 0 and an infinite value."""
    rng = np.random.default_rng(0)
    values = jnp.asarray(rng.normal(size=(5, 20)))
    weights = jnp.asarray(rng.uniform(0.1, 1.0, size=(5, 20)))
    weights = weights.at[:, ::4].set(0.0)
    values = values.at[:, ::4].set(jnp.inf)
    return values, weights


def test_soft_minimum_hard_branch_is_minimum_over_weighted_cells():
    """At width 0 the soft minimum is the minimum over the cells with a
    positive weight; cells left out with weight 0 are not read, even when
    their value is infinite."""
    values, weights = _random_volumes()
    expected = jnp.min(jnp.where(weights > 0.0, values, jnp.inf), axis=-1)
    hard = _soft_minimum(values, weights, 0.0, jnp.ones(5))
    np.testing.assert_array_equal(
        hard, expected, err_msg="the hard soft minimum is not the minimum over weighted cells"
    )


def test_soft_minimum_never_lies_below_minimum():
    """The soft minimum is a weighted average of the values, so it never lies
    below their minimum, at any temperature; it is finite although the cells
    left out hold infinite values."""
    values, weights = _random_volumes()
    minimum = _soft_minimum(values, weights, 0.0, jnp.ones(5))
    for width in (10.0, 1.0, 0.1):
        soft = _soft_minimum(values, weights, width, jnp.ones(5))
        assert jnp.all(jnp.isfinite(soft)), f"the soft minimum at width {width} is not finite"
        assert jnp.all(soft >= minimum), (
            f"the soft minimum at width {width} lies below the minimum by "
            f"{float(jnp.max(minimum - soft)):.3e}"
        )


def test_soft_minimum_reduces_to_minimum():
    """At a small width the soft minimum equals the minimum."""
    values, weights = _random_volumes()
    np.testing.assert_allclose(
        _soft_minimum(values, weights, SMALL_WIDTH, jnp.ones(5)),
        _soft_minimum(values, weights, 0.0, jnp.ones(5)),
        rtol=0.0,
        atol=1e-12,
        err_msg="a narrow soft minimum differs from the minimum",
    )


def test_soft_maximum_is_mirrored_soft_minimum():
    """The soft maximum is −min_T[−f]: at width 0 the maximum over the
    weighted cells, and never above that maximum."""
    rng = np.random.default_rng(1)
    values = jnp.asarray(rng.normal(size=(5, 20)))
    weights = jnp.asarray(rng.uniform(0.1, 1.0, size=(5, 20))).at[:, ::4].set(0.0)
    maximum = jnp.max(jnp.where(weights > 0.0, values, -jnp.inf), axis=-1)
    np.testing.assert_array_equal(
        _soft_maximum(values, weights, 0.0, jnp.ones(5)),
        maximum,
        err_msg="the hard soft maximum is not the maximum over weighted cells",
    )
    soft = _soft_maximum(values, weights, 1.0, jnp.ones(5))
    np.testing.assert_allclose(
        soft,
        -_soft_minimum(-values, weights, 1.0, jnp.ones(5)),
        rtol=1e-14,
        err_msg="the soft maximum is not −min_T[−f]",
    )
    assert jnp.all(soft <= maximum), "the soft maximum lies above the maximum"


@pytest.mark.parametrize("width", [0.0, 0.1], ids=["hard", "smooth"])
def test_soft_minimum_of_empty_volume_is_zero_with_finite_gradient(width):
    """A volume whose weights are all 0 gives 0, and a volume with a
    temperature scale of 0 gives the minimum over its weighted cells, the
    limit T → 0, both with finite derivatives.

    Args:
        width: The width ω of the soft minimum.
    """
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    weights = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    scale = jnp.array([1.0, 0.0])

    def total(values, weights):
        return jnp.sum(_soft_minimum(values, weights, width, scale))

    result = _soft_minimum(values, weights, width, scale)
    assert result[0] == 0.0, f"the soft minimum of an empty volume is {result[0]}, not 0"
    assert result[1] == 3.0, (
        f"the soft minimum at a temperature scale of 0 is {result[1]}, not the minimum 3"
    )
    gradients = jax.grad(total, argnums=(0, 1))(values, weights)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients), (
        f"the derivatives of the soft minimum are not finite: {gradients}"
    )


def test_soft_minimum_weights_average_to_soft_minimum():
    """The weights of the soft minimum sum to 1, are 0 on the cells left out
    with weight 0, and average the values to the soft minimum; at width 0
    they are 1 on the cell of the minimum and 0 elsewhere, and a small width
    gives the same weights."""
    values, weights = _random_volumes()
    width = 0.5
    soft = _soft_minimum_weights(values, weights, width, jnp.ones(5))
    np.testing.assert_allclose(
        jnp.sum(soft, axis=-1), 1.0, rtol=1e-14, err_msg="the weights do not sum to 1"
    )
    np.testing.assert_array_equal(
        soft[:, ::4], 0.0, err_msg="the cells left out have a weight above 0"
    )
    np.testing.assert_allclose(
        jnp.sum(soft * jnp.where(weights > 0.0, values, 0.0), axis=-1),
        _soft_minimum(values, weights, width, jnp.ones(5)),
        rtol=1e-13,
        err_msg="the weights do not average the values to the soft minimum",
    )

    hard = _soft_minimum_weights(values, weights, 0.0, jnp.ones(5))
    is_minimum = jnp.arange(20) == jnp.argmin(
        jnp.where(weights > 0.0, values, jnp.inf), axis=-1
    )[:, None]
    np.testing.assert_array_equal(
        hard, is_minimum, err_msg="the hard weights are not 1 on the minimum alone"
    )
    np.testing.assert_allclose(
        _soft_minimum_weights(values, weights, SMALL_WIDTH, jnp.ones(5)),
        hard,
        rtol=0.0,
        atol=1e-12,
        err_msg="the weights at a small width differ from the hard weights",
    )


# -------------------------------------------------------------
# ================ ↑ Soft minimum and maximum ↑ ===============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ Radial weight ↓ ====================
# -------------------------------------------------------------


def test_radial_weight_hard_branch_includes_accretion_radius():
    """At width 0 the radial weight is 1 for r ≤ r_acc, r = r_acc included, as
    in the hard checks, and 0 beyond."""
    accretion_radius = 2.5
    distance = jnp.array([0.0, 1.0, 2.5, 2.5 + 1e-9, 3.0])
    np.testing.assert_array_equal(
        _radial_weight(distance**2, accretion_radius, 0.0),
        [1.0, 1.0, 1.0, 0.0, 0.0],
        err_msg="the hard radial weight is not r ≤ r_acc",
    )


def test_radial_weight_is_exact_outside_its_band():
    """The radial weight is exactly 1 within r_acc − w and exactly 0 beyond
    r_acc, and lies strictly between in the band."""
    accretion_radius, width = 2.5, 0.5
    distance = jnp.array([0.0, 1.9, 2.0, 2.25, 2.5, 3.0])
    weight = _radial_weight(distance**2, accretion_radius, width)
    np.testing.assert_array_equal(
        weight[jnp.array([0, 1, 2, 4, 5])],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        err_msg="the radial weight is not exactly 1 inside r_acc − w and 0 beyond r_acc",
    )
    assert 0.0 < weight[3] < 1.0, f"the radial weight in its band is {weight[3]}"


def test_radial_weight_reduces_to_hard_branch():
    """A narrow radial weight equals the hard one away from r_acc."""
    accretion_radius = 2.5
    distance = jnp.concatenate(
        [jnp.linspace(0.0, 2.499, 20), jnp.linspace(2.501, 4.0, 20)]
    )
    np.testing.assert_array_equal(
        _radial_weight(distance**2, accretion_radius, SMALL_WIDTH),
        _radial_weight(distance**2, accretion_radius, 0.0),
        err_msg="a narrow radial weight differs from the hard one away from r_acc",
    )


@pytest.mark.parametrize("width", [0.0, 0.5], ids=["hard", "smooth"])
def test_radial_weight_has_finite_gradient_at_zero_distance(width):
    """The derivative of the radial weight with respect to the separation is
    finite (0) where the separation is 0.

    Args:
        width: The width w of the radial weight.
    """
    gradient = jax.grad(lambda separation: _radial_weight(jnp.sum(separation**2), 2.5, width))(
        jnp.zeros(3)
    )
    np.testing.assert_array_equal(
        gradient, 0.0, err_msg=f"the derivative at zero separation is {gradient}, not 0"
    )


# -------------------------------------------------------------
# ====================== ↑ Radial weight ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ Cell weights ↓ =====================
# -------------------------------------------------------------


def _positions_in_cells(*cell_coordinates):
    """Positions given in units of the cell size, shape (n, 3)."""
    return jnp.array(cell_coordinates) * CELL_SIZE


@pytest.mark.parametrize("spatial_width", [0.0, 0.4, 1.0], ids=["hard", "narrow", "widest"])
def test_cell_weights_sum_to_one(spatial_width):
    """The cell weights around a sink sum to 1 for ℓ ≤ 1, anywhere in a
    periodic box: inside a cell, on a face, on an edge, on a corner and
    across the periodic boundary.

    Args:
        spatial_width: The width ℓ in cells.
    """
    config, grid_shape = _padded_grid()
    positions = _positions_in_cells(
        (16.3, 16.5, 16.5),
        (16.0, 16.5, 16.5),
        (16.0, 16.0, 16.5),
        (16.0, 16.0, 16.0),
        (16.95, 7.02, 3.5),
        (0.01, 31.99, 0.5),
    )
    _, weights = _cell_weights(positions, grid_shape, config, spatial_width)
    np.testing.assert_allclose(
        jnp.sum(weights, axis=-1),
        1.0,
        rtol=1e-14,
        err_msg=f"the cell weights at ℓ = {spatial_width} do not sum to 1",
    )


def test_cell_weights_hard_branch_picks_containing_cell():
    """At ℓ = 0 the weight is 1 on the cell found with ``floor`` and 0
    elsewhere; a sink on a face belongs to the upper cell."""
    config, grid_shape = _padded_grid()
    positions = _positions_in_cells((16.3, 9.7, 4.5), (16.0, 9.5, 4.5))
    cell_indices, weights = _cell_weights(positions, grid_shape, config, 0.0)

    expected = np.zeros(weights.shape)
    expected[:, CONTAINING_CELL] = 1.0
    np.testing.assert_array_equal(weights, expected, err_msg="the hard weights are not one-hot")
    containing_x = int(cell_indices[0][1, CONTAINING_CELL]) - config.num_ghost_cells
    assert containing_x == 16, (
        f"a sink on the face x = 16 Δx belongs to cell {containing_x} along x, not the upper cell 16"
    )


def test_cell_weights_are_one_on_containing_cell_away_from_faces():
    """For ℓ = 0.4 the weight is exactly 1 on the containing cell farther than
    ½ ℓ cells from every face, and shared between two cells closer to a face."""
    config, grid_shape = _padded_grid()
    spatial_width = 0.4
    away_from_faces = _positions_in_cells((16.7, 16.5, 16.3))
    _, weights = _cell_weights(away_from_faces, grid_shape, config, spatial_width)
    expected = np.zeros(weights.shape)
    expected[:, CONTAINING_CELL] = 1.0
    np.testing.assert_array_equal(
        weights, expected, err_msg="away from the faces the weight is not 1 on the containing cell only"
    )

    near_face = _positions_in_cells((16.95, 16.5, 16.5))
    _, weights = _cell_weights(near_face, grid_shape, config, spatial_width)
    num_cells_with_weight = int(jnp.sum(weights > 0.0))
    assert num_cells_with_weight == 2, (
        f"near a face {num_cells_with_weight} cells have weight, not the 2 on either side of it"
    )


def test_cell_weights_reduce_to_hard_branch():
    """A narrow ℓ gives the hard weights away from the faces."""
    config, grid_shape = _padded_grid()
    positions = _positions_in_cells((16.3, 9.7, 4.5), (0.2, 31.8, 12.6))
    np.testing.assert_array_equal(
        _cell_weights(positions, grid_shape, config, SMALL_WIDTH)[1],
        _cell_weights(positions, grid_shape, config, 0.0)[1],
        err_msg="narrow cell weights differ from the hard ones away from the faces",
    )


def test_cell_weights_vanish_past_non_periodic_edge():
    """Next to an open edge the cells past it get weight 0, so the weights of
    a sink within ½ ℓ cells of the edge sum to less than 1."""
    config, grid_shape = _padded_grid(SETTINGS._replace(boundary_settings=BoundarySettings()))
    position = _positions_in_cells((0.05, 16.5, 16.5))
    _, weights = _cell_weights(position, grid_shape, config, 0.4)
    past_edge = _tsc_offsets()[:, 0] == -1
    np.testing.assert_array_equal(
        weights[0, past_edge], 0.0, err_msg="the cells past the open edge have weight"
    )
    total = float(jnp.sum(weights))
    assert 0.0 < total < 1.0, f"next to the open edge the weights sum to {total}, not below 1"


# -------------------------------------------------------------
# ====================== ↑ Cell weights ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================== ↓ Point-mass potential ↓ =================
# -------------------------------------------------------------


@pytest.mark.parametrize("softening", [0.0, 0.5], ids=["hard", "softened"])
def test_point_mass_potential_is_zero_at_accretion_radius(softening):
    """The point-mass potential is 0 at r = r_acc, with or without softening.

    Args:
        softening: The softening length λ.
    """
    potential = _point_mass_potential(jnp.array(2.5**2), 3.0, 2.5, softening, 1.0)
    assert abs(float(potential)) < 1e-15, f"ψ(r_acc) = {float(potential)}, not 0"


def test_point_mass_potential_hard_branch_treats_zero_distance_as_accretion_radius():
    """Without softening, r = 0 is treated as r = r_acc, where ψ is 0, and
    elsewhere ψ = −G m (1/r − 1/r_acc)."""
    distance = jnp.array([0.0, 1.0, 2.0])
    potential = _point_mass_potential(distance**2, 3.0, 2.5, 0.0, 1.0)
    np.testing.assert_allclose(
        potential,
        [0.0, -3.0 * (1.0 - 0.4), -3.0 * (0.5 - 0.4)],
        rtol=1e-14,
        atol=1e-15,
        err_msg="the hard point-mass potential is not −G m (1/r − 1/r_acc), with 0 at r = 0",
    )


def test_point_mass_potential_reduces_to_hard_branch():
    """A small softening gives the hard potential for r > 0."""
    distance = jnp.linspace(0.5, 3.0, 20)
    np.testing.assert_allclose(
        _point_mass_potential(distance**2, 3.0, 2.5, SMALL_WIDTH, 1.0),
        _point_mass_potential(distance**2, 3.0, 2.5, 0.0, 1.0),
        rtol=1e-10,
        atol=1e-12,
        err_msg="a slightly softened potential differs from the hard one",
    )


@pytest.mark.parametrize("softening", [0.0, 0.5], ids=["hard", "softened"])
def test_point_mass_potential_has_finite_gradient_at_zero_distance(softening):
    """The derivative of the point-mass potential with respect to the
    separation is finite where the separation is 0.

    Args:
        softening: The softening length λ.
    """
    gradient = jax.grad(
        lambda separation: _point_mass_potential(jnp.sum(separation**2), 3.0, 2.5, softening, 1.0)
    )(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(gradient)), f"the derivative at zero separation is {gradient}"


# -------------------------------------------------------------
# ================== ↑ Point-mass potential ↑ =================
# -------------------------------------------------------------
