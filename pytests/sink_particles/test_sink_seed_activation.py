"""
The activation of a seed is a continuous function of the gas, at the default
widths: a Gaussian clump at rest whose peak overdensity is scanned, one call
of ``_form_sink_particles`` each.

In this clump the Jeans check decides whether the centre forms a sink: the
density check already passes at a peak density of ρ_res (overdensity ≈ 128),
the Jeans check only from about twice that. Below the Jeans threshold no seed
is active and every activation is exactly 0; above it, one seed at the
centre has an activation that rises continuously from 0 to 1 across the band
of the Jeans factor, and inside the band ``jax.grad`` of that activation
agrees with a finite difference.
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
from astronomix._modules._sink_particles._sink_particle_formation import (
    _cell_quantities,
    _form_sink_particles,
    _slot_checks,
    _slot_neighbourhood,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
)

# sink particle test helpers
from _sink_helpers import CELL_SIZE, NUM_CELLS, SETTINGS, _density_threshold, _setup

pytestmark = pytest.mark.x64

#: The scanned peak overdensities: one at 0.9 ρ_res, where no cell passes the
#: density check, and 21 across the Jeans threshold (overdensity ≈ 265) and
#: the band of the Jeans factor (up to ≈ 286).
LOWEST_PEAK_OVERDENSITY_IN_DENSITY_THRESHOLDS = 0.9
JEANS_SCAN = tuple(np.linspace(250.0, 300.0, 21))

#: The cell at the clump centre.
CENTER_CELL = (NUM_CELLS // 2,) * 3

#: The largest change of the activation between neighbouring scan points
#: (2.5 apart in overdensity): the Jeans factor rises by at most about 0.2
#: there, while a hard check would jump from 0 to 1.
LARGEST_ACTIVATION_STEP = 0.3

#: The step of the central finite difference, in overdensity.
FINITE_DIFFERENCE_STEP = 1e-3


@cache
def _clump():
    """The clump setup, and the primitive state as a function of the peak
    overdensity, at a uniform c_s².

    Returns:
        ``(state_at, config, params, registered_variables)``.
    """
    state, config, params, registered_variables = _setup()
    primitive_state = state.primitive_state
    profile = (
        primitive_state[registered_variables.density_index] - SETTINGS.background_density
    ) / SETTINGS.peak_overdensity

    def state_at(peak_overdensity):
        density = SETTINGS.background_density + peak_overdensity * profile
        return (
            primitive_state.at[registered_variables.density_index]
            .set(density)
            .at[registered_variables.pressure_index]
            .set(density * SETTINGS.sound_speed_squared / SETTINGS.gamma)
        )

    return state_at, config, params, registered_variables


def _form_at(peak_overdensity):
    """The sink particles after one call of ``_form_sink_particles`` on the
    clump of the given peak overdensity, starting from no seeds."""
    state_at, config, params, registered_variables = _clump()
    primitive_state = state_at(peak_overdensity)
    sink_particles = _form_sink_particles(
        primitive_state,
        _empty_sink_particles(config, primitive_state.dtype),
        config,
        params,
        registered_variables,
    )
    return sink_particles


def _jeans_ratio_at_center(peak_overdensity):
    """|E_grav| / (2 E_th + E_mag) of the control volume of the centre cell."""
    state_at, config, params, registered_variables = _clump()
    primitive_state = state_at(peak_overdensity)
    empty = _empty_sink_particles(config, primitive_state.dtype)
    center = (jnp.asarray(CENTER_CELL, dtype=primitive_state.dtype) + 0.5) * CELL_SIZE
    checks = _slot_checks(
        _slot_neighbourhood(center[None, :], primitive_state.shape[1:], config),
        jnp.zeros(1, dtype=primitive_state.dtype),
        _cell_quantities(primitive_state, empty, config, params, registered_variables),
        config,
    )
    return float(checks.jeans_ratio[0])


@cache
def _scan():
    """The scan: for each peak overdensity, the Jeans ratio at the centre and
    the sink particles after formation."""
    _, config, params, _ = _clump()
    lowest_peak = (
        LOWEST_PEAK_OVERDENSITY_IN_DENSITY_THRESHOLDS * float(_density_threshold(config, params))
        - SETTINGS.background_density
    )
    peaks = (lowest_peak,) + JEANS_SCAN
    return [(peak, _jeans_ratio_at_center(peak), _form_at(peak)) for peak in peaks]


def test_no_seed_below_the_jeans_threshold():
    """Where the control volume of the centre is not Jeans unstable, no slot
    is active and every activation is exactly 0, whether or not the density
    check passes."""
    below = [(peak, sinks) for peak, jeans_ratio, sinks in _scan() if jeans_ratio <= 1.0]
    assert len(below) >= 3, f"only {len(below)} scan points lie below the Jeans threshold"
    for peak, sinks in below:
        assert jnp.all(sinks.activation == 0.0), (
            f"at peak overdensity {peak:.1f}, below the Jeans threshold, activations "
            f"{sinks.activation[sinks.activation != 0.0]} are not 0"
        )


def test_activation_rises_continuously_above_the_jeans_threshold():
    """Above the Jeans threshold one seed is active, at the centre cell and
    with that cell as home cell. Its activation never drops along the scan,
    takes values strictly between 0 and 1 in the band of the Jeans factor,
    reaches 1, and changes by less than ``LARGEST_ACTIVATION_STEP`` between
    neighbouring scan points."""
    above = [(peak, sinks) for peak, jeans_ratio, sinks in _scan() if jeans_ratio > 1.0]
    assert len(above) >= 3, f"only {len(above)} scan points lie above the Jeans threshold"

    activations = []
    for peak, sinks in above:
        num_active = int(jnp.sum(sinks.activation > 0.0))
        assert num_active == 1, (
            f"at peak overdensity {peak:.1f}, {num_active} slots are active, not 1"
        )
        cell = tuple(int(index) for index in jnp.floor(sinks.position[0] / CELL_SIZE))
        assert cell == CENTER_CELL and tuple(sinks.home_cell[0].tolist()) == CENTER_CELL, (
            f"at peak overdensity {peak:.1f} the seed sits in cell {cell} with home cell "
            f"{sinks.home_cell[0].tolist()}, not the centre cell {CENTER_CELL}"
        )
        activations.append(float(sinks.activation[0]))

    activations = np.asarray(activations)
    steps = np.diff(np.concatenate([[0.0], activations]))
    assert np.all(steps >= 0.0), f"the activation drops along the scan: {activations}"
    assert np.sum((activations > 0.0) & (activations < 1.0)) >= 3, (
        f"fewer than 3 scan points lie inside the band: {activations}"
    )
    assert activations[-1] == 1.0, f"the activation does not reach 1: {activations}"
    assert np.max(steps) < LARGEST_ACTIVATION_STEP, (
        f"the activation jumps by {np.max(steps):.3f} between neighbouring scan points: "
        f"{activations}"
    )


def test_activation_gradient_matches_finite_difference():
    """Inside the band of the Jeans factor, the derivative of the seed's
    activation with respect to the peak overdensity from ``jax.grad`` agrees
    with a central finite difference."""

    def activation(peak_overdensity):
        return _form_at(peak_overdensity).activation[0]

    in_band = [
        peak
        for peak, _, sinks in _scan()
        if 0.0 < float(sinks.activation[0]) < 1.0
    ]
    assert in_band, "no scan point lies inside the band"
    gradient = jax.grad(activation)
    for peak in in_band:
        derivative = float(gradient(peak))
        finite_difference = float(
            (activation(peak + FINITE_DIFFERENCE_STEP) - activation(peak - FINITE_DIFFERENCE_STEP))
            / (2.0 * FINITE_DIFFERENCE_STEP)
        )
        assert derivative > 0.0, (
            f"at peak overdensity {peak:.1f} the derivative of the activation is {derivative}"
        )
        np.testing.assert_allclose(
            derivative,
            finite_difference,
            rtol=1e-5,
            err_msg=(
                f"at peak overdensity {peak:.1f}, jax.grad gives {derivative} and the "
                f"finite difference {finite_difference}"
            ),
        )
