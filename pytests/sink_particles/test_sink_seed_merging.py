"""
A seed merges into a more active seed within its reach: it hands over its
mass and momentum and leaves its slot, and the fraction it gives is a
continuous function of the distance and of the activations. Two seeds are
placed by hand along x and merged once with the default widths.
"""

# general
from functools import cache

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_formation import _minimum_image
from astronomix._modules._sink_particles._sink_particle_helpers import _empty_sink_particles
from astronomix._modules._sink_particles._sink_particle_merging import _merge_seeds

# sink particle test helpers
from _sink_helpers import CELL_SIZE, CLUMP_CENTER, _density_threshold, _setup

pytestmark = pytest.mark.x64

#: The more active seed (slot 0) and the less active one (slot 1): activation,
#: mass and velocity.
RECEIVER = (0.9, 0.3, (1.0, 0.0, 0.0))
DONOR = (0.05, 0.1, (0.0, 2.0, -1.0))

#: The distance of the less active seed from the more active one along x, in
#: cells, well within the accretion radius of 2.5 cells.
SEPARATION_IN_CELLS = 0.5

#: The relative tolerance of mass and momentum conservation.
CONSERVATION_TOLERANCE = 1e-12


@cache
def _config():
    """The configuration with the default widths, and ρ_res of the setup."""
    _, config, params, _ = _setup()
    return config, float(_density_threshold(config, params))


def _two_seeds(separation_in_cells, donor_activation, dtype=jnp.float64):
    """Sink particles with the receiver at the clump centre and the donor
    ``separation_in_cells`` cells along x from it, each with its whole claim
    on the cell it lies in."""
    config, _ = _config()
    empty = _empty_sink_particles(config, dtype)
    positions = jnp.stack(
        [
            jnp.full((3,), CLUMP_CENTER, dtype=dtype),
            jnp.full((3,), CLUMP_CENTER, dtype=dtype).at[0].add(separation_in_cells * CELL_SIZE),
        ]
    )
    return empty._replace(
        mass=empty.mass.at[:2].set(jnp.asarray([RECEIVER[1], DONOR[1]], dtype=dtype)),
        position=empty.position.at[:2].set(positions),
        velocity=empty.velocity.at[:2].set(jnp.asarray([RECEIVER[2], DONOR[2]], dtype=dtype)),
        activation=empty.activation.at[0].set(RECEIVER[0]).at[1].set(donor_activation),
        home_cell=empty.home_cell.at[:2].set(jnp.floor(positions / CELL_SIZE).astype(jnp.int32)),
        home_claim=empty.home_claim.at[:2].set(1.0),
    )


def _merge(sink_particles):
    """One merging step, as at the end of the formation."""
    config, density_threshold = _config()
    position = sink_particles.position
    return _merge_seeds(
        sink_particles,
        _minimum_image(position[None, :, :] - position[:, None, :], config),
        jnp.full(position.shape[0], density_threshold, dtype=position.dtype),
        config,
    )


def _given_fraction(separation_in_cells, donor_activation):
    """The fraction of its mass the donor gives in one step."""
    sinks = _two_seeds(separation_in_cells, donor_activation)
    merged = _merge(sinks)
    return 1.0 - merged.mass[1] / sinks.mass[1]


def test_weak_seed_merges_into_stronger_one():
    """A seed of activation 0.05, half a cell from one of activation 0.9,
    gives it all its mass and momentum in one step and leaves its slot; the
    receiver moves to the centre of mass, with ε ρ_res ΔV added to its mass,
    and keeps its activation and its claim on its home cell."""
    config, density_threshold = _config()
    sinks = _two_seeds(SEPARATION_IN_CELLS, DONOR[0])
    merged = _merge(sinks)

    np.testing.assert_array_equal(
        np.asarray(merged.activation[1:]),
        0.0,
        err_msg="the donor's slot, or a slot that was empty, is not empty after the merge",
    )
    for field in ("mass", "home_claim"):
        assert float(getattr(merged, field)[1]) == 0.0, (
            f"the donor keeps {field} = {float(getattr(merged, field)[1])} after giving everything"
        )
    assert float(merged.activation[0]) == RECEIVER[0], (
        f"the receiver's activation changed from {RECEIVER[0]} to {float(merged.activation[0])}"
    )
    assert float(merged.home_claim[0]) == 1.0, "the receiver gave up part of its home claim"

    total_mass = RECEIVER[1] + DONOR[1]
    np.testing.assert_allclose(
        float(merged.mass[0]),
        total_mass,
        rtol=CONSERVATION_TOLERANCE,
        err_msg="the merge does not conserve mass",
    )
    momentum = RECEIVER[1] * np.asarray(RECEIVER[2]) + DONOR[1] * np.asarray(DONOR[2])
    np.testing.assert_allclose(
        np.asarray(jnp.sum(merged.mass[:, None] * merged.velocity, axis=0)),
        momentum,
        rtol=CONSERVATION_TOLERANCE,
        err_msg="the merge does not conserve momentum",
    )

    position_update_mass = (
        config.sink_particle_config.position_update_mass * density_threshold * CELL_SIZE**3
    )
    expected_shift = DONOR[1] * SEPARATION_IN_CELLS * CELL_SIZE / (total_mass + position_update_mass)
    np.testing.assert_allclose(
        np.asarray(merged.position[0] - sinks.position[0]),
        [expected_shift, 0.0, 0.0],
        rtol=1e-12,
        atol=1e-15,
        err_msg="the receiver did not move to the regularized centre of mass",
    )


def _check_band(name, values, band, given_fraction):
    """The given fraction over a scan that crosses a band: 1 on one side, 0
    on the other, continuous and monotonic, and inside the band its
    derivative agrees with a central finite difference."""
    fractions = np.array([float(given_fraction(value)) for value in values])
    assert fractions[0] == 1.0 and fractions[-1] == 0.0, (
        f"{name}: the given fraction goes from {fractions[0]} to {fractions[-1]}, "
        "not from 1 to 0 across the band"
    )
    assert np.all(np.diff(fractions) <= 0.0), (
        f"{name}: the given fraction is not monotonic: {fractions}"
    )
    # The smooth step 3t² − 2t³ has slope at most 1.5 / width.
    largest_step = 1.5 * (values[1] - values[0]) / (band[1] - band[0])
    assert np.max(np.abs(np.diff(fractions))) <= largest_step * (1.0 + 1e-9), (
        f"{name}: the given fraction jumps by {np.max(np.abs(np.diff(fractions)))} "
        f"between neighbouring values, more than the {largest_step} of a smooth step"
    )

    gradient = jax.grad(given_fraction)
    for value in np.linspace(*band, 5)[1:-1]:
        step = 1e-6 * (band[1] - band[0])
        finite_difference = (
            float(given_fraction(value + step)) - float(given_fraction(value - step))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            float(gradient(value)),
            finite_difference,
            rtol=1e-6,
            err_msg=f"{name}: jax.grad disagrees with a finite difference at {value}",
        )


def test_merging_is_continuous():
    """The fraction a seed of activation 0.05 gives one of activation 0.9
    falls from 1 to 0 as their distance crosses the band from r_acc to
    r_acc + ℓ Δx, and the fraction a seed gives one of activation 0.9 falls
    from 1 to 0 as their activation ratio crosses the band from 1 − δ to 1,
    so that seeds of equal activation do not merge."""
    config, _ = _config()
    sink_particle_config = config.sink_particle_config
    accretion_radius = sink_particle_config.accretion_radius_in_cells
    spatial_width = sink_particle_config.spatial_width_in_cells
    merging_width = sink_particle_config.merging_width

    distance_band = (accretion_radius, accretion_radius + spatial_width)
    _check_band(
        "distance",
        np.linspace(distance_band[0] - 0.2, distance_band[1] + 0.2, 37),
        distance_band,
        lambda separation: _given_fraction(separation, DONOR[0]),
    )

    ratio_band = (1.0 - merging_width, 1.0)
    _check_band(
        "activation ratio",
        np.linspace(ratio_band[0] - 0.05, ratio_band[1], 31),
        ratio_band,
        lambda ratio: _given_fraction(SEPARATION_IN_CELLS, ratio * RECEIVER[0]),
    )
