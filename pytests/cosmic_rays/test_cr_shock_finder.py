"""
Unit tests for the shock finder and the DSA injection step (CPU, seconds).

They check that

* the Mach estimate is exact for Rankine-Hugoniot jumps, with and without
  CRs, so the M > 1.3 gate of the shock criterion rejects weak compressions;
* the WENO smoothness sensor is non-negative (both of its terms are squares);
* one injection step conserves energy exactly (gas loss = CR gain), removes
  exactly ``zeta * e_diss * v_2 * dt`` from the gas for a resolved shock, keeps
  ``(1 - escape_fraction)`` of it in the CRs, respects the thermal-energy cap,
  does nothing (instead of NaN) for an invalid zone, and is AD-safe in
  CR-free cells (``P_cr ** (3/4)`` has an infinite derivative at 0);
* ``OUTERMOST_SHOCK`` selects the forward shock when a stronger inner shock
  exists;
* a shock near the origin of a spherical grid receives its CRs.
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# The Mach inversion and the energy bookkeeping are asserted to rel = 1e-10 to
# 1e-12, which needs double precision.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    SPHERICAL,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    OUTERMOST_SHOCK,
    STRONGEST_SHOCK,
)

# astronomix containers
from astronomix import SimulationConfig
from astronomix.option_classes.simulation_config import BackendConfig
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    CosmicRayConfig,
    CosmicRayParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_n_from_pressure,
    cosmic_ray_pressure_from_n,
)
from astronomix._modules._cosmic_rays.cr_injection import inject_crs_at_strongest_shock
from astronomix.shock_finder.shock_finder import (
    find_shock_zone,
    mach_number_squared,
    shock_criteria,
    shock_sensor,
)
from astronomix.test_setups.reference_solutions.cr_shock_tube import (
    cosmic_ray_shock_tube_regions,
)

GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
NUM_CELLS = 128


# -------------------------------------------------------------
# ===================== ↓ Helpers ↓ ===========================
# -------------------------------------------------------------


def _rankine_hugoniot(mach, upstream_density, upstream_pressure, gamma=GAMMA):
    """
    Exact jump of a right-moving shock into gas at rest.

    Args:
        mach: The upstream Mach number.
        upstream_density: The upstream density.
        upstream_pressure: The upstream pressure.
        gamma: The adiabatic index.

    Returns:
        The downstream density, the downstream pressure, the lab-frame
        downstream velocity and the shock speed.
    """
    pressure_ratio = (2 * gamma * mach**2 - (gamma - 1)) / (gamma + 1)
    compression = (gamma + 1) * mach**2 / ((gamma - 1) * mach**2 + 2)
    shock_speed = mach * np.sqrt(gamma * upstream_pressure / upstream_density)
    return (
        upstream_density * compression,
        upstream_pressure * pressure_ratio,
        shock_speed * (1 - 1 / compression),
        shock_speed,
    )


def _setup(shock_selection=STRONGEST_SHOCK, num_cells=NUM_CELLS):
    """
    Cartesian finite-volume configuration with cosmic rays and DSA.

    Args:
        shock_selection: The shock-selection tag of the injection.
        num_cells: The number of cells.

    Returns:
        The (not yet finalized) configuration, the helper data and the
        registered variables.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        num_cells=num_cells,
        box_size=1.0,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=True,
            shock_selection=shock_selection,
        ),
        riemann_solver=HLL,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    return config, helper_data, registered_variables


def _smooth_step(num_cells, centre, width):
    """1 left of ``centre``, 0 right of it, tanh-smoothed over ``width`` cells."""
    cell_index = np.arange(num_cells)
    return 0.5 * (1.0 - np.tanh((cell_index - centre) / width))


def _single_shock_state(mach=10.0, width=0.6, upstream_cosmic_ray_pressure=0.0, centre=64.3):
    """
    A resolved right-moving gas shock (upstream rho = p = 1 at rest).

    Args:
        mach: The upstream Mach number.
        width: The tanh width of the shock in cells.
        upstream_cosmic_ray_pressure: The uniform cosmic-ray pressure.
        centre: The shock position in cells.

    Returns:
        The primitive state, the finalized configuration, the helper data, the
        registered variables and the exact jump ``(rho2, p2, u2, v_shock)``.
    """
    config, helper_data, registered_variables = _setup()
    downstream_density, downstream_pressure, downstream_velocity, shock_speed = (
        _rankine_hugoniot(mach, 1.0, 1.0)
    )
    step_profile = _smooth_step(NUM_CELLS, centre, width)
    density = 1.0 + (downstream_density - 1.0) * step_profile
    pressure = 1.0 + (downstream_pressure - 1.0) * step_profile
    velocity = downstream_velocity * step_profile
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.asarray(density),
        velocity_x=jnp.asarray(velocity),
        gas_pressure=jnp.asarray(pressure),
        cosmic_ray_pressure=jnp.full(NUM_CELLS, upstream_cosmic_ray_pressure),
    )
    config = finalize_config(config, state.shape)
    exact_jump = (downstream_density, downstream_pressure, downstream_velocity, shock_speed)
    return state, config, helper_data, registered_variables, exact_jump


def _energies(state, registered_variables, helper_data):
    """
    Volume-integrated thermal and cosmic-ray energies of a 1D primitive state.

    Args:
        state: The primitive state.
        registered_variables: The registered variables.
        helper_data: The helper data (cell volumes).

    Returns:
        The thermal gas energy and the cosmic-ray energy.
    """
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        state[registered_variables.cosmic_ray_n_index]
    )
    gas_pressure = state[registered_variables.pressure_index] - cosmic_ray_pressure
    cell_volumes = helper_data.cell_volumes
    return (
        jnp.sum(gas_pressure / (GAMMA - 1) * cell_volumes),
        jnp.sum(cosmic_ray_pressure / (GAMMA_CR - 1) * cell_volumes),
    )


# -------------------------------------------------------------
# ===================== ↑ Helpers ↑ ===========================
# -------------------------------------------------------------


@pytest.mark.parametrize("mach", [1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 10.0, 100.0])
def test_mach_estimate_exact_for_gas_shocks(mach):
    """The Mach inversion recovers a gamma = 5/3 Rankine-Hugoniot jump exactly."""
    _, downstream_pressure, _, _ = _rankine_hugoniot(mach, 1.0, 1.0)
    mach_squared = float(
        mach_number_squared(1.0, 0.0, downstream_pressure, 0.0, GAMMA, GAMMA_CR)
    )
    assert np.sqrt(mach_squared) == pytest.approx(mach, rel=1e-10)


@pytest.mark.parametrize("zeta", [0.0, 0.5])
def test_mach_estimate_exact_for_cr_shocks(zeta):
    """
    The Mach inversion recovers the composite gas + CR jump of the Pfrommer
    et al. (2017) tube (M = 10.00 / 9.56), with and without injection (energy
    is conserved across the jump either way).
    """
    regions = cosmic_ray_shock_tube_regions(injection_efficiency=zeta)
    upstream_pressure, upstream_cosmic_ray_pressure, upstream_density = 0.1, 0.05, 0.125
    downstream_pressure = regions["P_th2"] + regions["P_cr2"]
    upstream_effective_gamma = (
        GAMMA_CR * upstream_cosmic_ray_pressure
        + GAMMA * (upstream_pressure - upstream_cosmic_ray_pressure)
    ) / upstream_pressure
    mach_exact = regions["vs"] / np.sqrt(
        upstream_effective_gamma * upstream_pressure / upstream_density
    )
    mach_squared = float(
        mach_number_squared(
            upstream_pressure,
            upstream_cosmic_ray_pressure,
            downstream_pressure,
            regions["P_cr2"],
            GAMMA,
            GAMMA_CR,
        )
    )
    assert np.sqrt(mach_squared) == pytest.approx(mach_exact, rel=1e-6)
    assert mach_exact == pytest.approx(10.00 if zeta == 0 else 9.559, abs=2e-3)


def test_mach_estimate_rejects_reverse_jumps():
    """A jump with the upstream on the left (P2 < P1) is never a right-moving shock."""
    for pressure_ratio in [0.9, 0.5, 0.1, 1e-3]:
        assert float(mach_number_squared(1.0, 0.0, pressure_ratio, 0.0)) < 1.0


def test_shock_sensor_non_negative():
    """The WENO-JS indicator is a sum of squares, also at a concave maximum."""
    concave_peak = jnp.array([1.0, 1.0, 2.0, 1.0, 1.0])
    assert float(shock_sensor(concave_peak)[2]) == pytest.approx(13 / 12 * 4.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pressure = jnp.asarray(rng.uniform(0.1, 10.0, 64))
        assert float(jnp.min(shock_sensor(pressure))) >= 0.0


def test_shock_criteria_flags_only_real_shocks():
    """The M > 1.3 gate: a resolved M = 1.2 compression is not flagged, M = 2 is."""
    for mach, expected in [(1.2, False), (2.0, True), (10.0, True)]:
        state, config, helper_data, registered_variables, _ = _single_shock_state(
            mach=mach,
            width=0.3,
        )
        flagged = bool(
            jnp.any(shock_criteria(state, config, registered_variables, helper_data))
        )
        assert flagged == expected, mach


@pytest.mark.parametrize("escape_fraction", [0.0, 0.5])
def test_injection_energy_bookkeeping(escape_fraction):
    """One step: gas loses zeta * e_diss * v2 * dt; CRs keep (1 - f_esc) of it."""
    zeta, dt = 0.5, 1e-4
    state, config, helper_data, registered_variables, exact_jump = _single_shock_state()
    downstream_density, downstream_pressure, _, shock_speed = exact_jump
    cosmic_ray_params = CosmicRayParams(
        diffusive_shock_acceleration_efficiency=zeta,
        escape_fraction=escape_fraction,
    )
    new_state = inject_crs_at_strongest_shock(
        state,
        GAMMA,
        helper_data,
        cosmic_ray_params,
        config,
        registered_variables,
        dt,
    )
    gas_energy_before, cr_energy_before = _energies(state, registered_variables, helper_data)
    gas_energy_after, cr_energy_after = _energies(new_state, registered_variables, helper_data)
    gas_loss = float(gas_energy_before - gas_energy_after)
    cr_gain = float(cr_energy_after - cr_energy_before)

    # Exact dissipated energy flux of this Rankine-Hugoniot shock.
    compression = downstream_density / 1.0
    dissipated_energy_density = (
        downstream_pressure / (GAMMA - 1) - 1.0 / (GAMMA - 1) * compression**GAMMA
    )
    expected = zeta * dissipated_energy_density * (shock_speed / compression) * dt
    assert gas_loss == pytest.approx(expected, rel=0.02)
    assert cr_gain == pytest.approx((1 - escape_fraction) * gas_loss, rel=1e-12)

    # Only a few cells change, and none of them loses more than it has.
    pressure_index = registered_variables.pressure_index
    changed = np.flatnonzero(np.asarray(new_state[pressure_index] != state[pressure_index]))
    assert 1 <= changed.size <= 6
    new_gas_pressure = new_state[pressure_index] - cosmic_ray_pressure_from_n(
        new_state[registered_variables.cosmic_ray_n_index]
    )
    assert float(jnp.min(new_gas_pressure)) > 0.0


def test_injection_linear_in_efficiency_and_capped():
    """Linear in zeta below the cap; the cap bounds the per-cell thermal removal."""
    state, config, helper_data, registered_variables, _ = _single_shock_state()
    dt = 1e-4
    cr_energies = []
    for zeta in (0.1, 0.2):
        new_state = inject_crs_at_strongest_shock(
            state,
            GAMMA,
            helper_data,
            CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta),
            config,
            registered_variables,
            dt,
        )
        cr_energies.append(float(_energies(new_state, registered_variables, helper_data)[1]))
    assert cr_energies[1] == pytest.approx(2 * cr_energies[0], rel=1e-12)

    cap = 1e-4
    new_state = inject_crs_at_strongest_shock(
        state,
        GAMMA,
        helper_data,
        CosmicRayParams(
            diffusive_shock_acceleration_efficiency=0.5,
            max_thermal_fraction_per_step=cap,
        ),
        config,
        registered_variables,
        1e-2,
    )
    old_thermal_energy = state[registered_variables.pressure_index] / (GAMMA - 1)
    new_cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        new_state[registered_variables.cosmic_ray_n_index]
    )
    new_thermal_energy = (
        new_state[registered_variables.pressure_index] - new_cosmic_ray_pressure
    ) / (GAMMA - 1)
    removed = old_thermal_energy - new_thermal_energy
    assert float(jnp.max(removed / old_thermal_energy)) <= cap * (1 + 1e-9)
    assert float(jnp.max(removed)) > 0.0


def test_injection_invalid_zone_is_a_no_op():
    """No shock (a pure pressure pulse at rest / an expansion): nothing, and no NaN."""
    config, helper_data, registered_variables = _setup()
    cell_index = np.arange(NUM_CELLS)
    pressure_pulse = 1.0 + 10.0 * (cell_index == 64)
    pressure_ramp = 1.0 + 0.5 * np.tanh((cell_index - 64) / 2.0)
    for pressure in (pressure_pulse, pressure_ramp):
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=jnp.ones(NUM_CELLS),
            velocity_x=jnp.zeros(NUM_CELLS),
            gas_pressure=jnp.asarray(pressure),
            cosmic_ray_pressure=jnp.zeros(NUM_CELLS),
        )
        finalized_config = finalize_config(config, state.shape)
        new_state = inject_crs_at_strongest_shock(
            state,
            GAMMA,
            helper_data,
            CosmicRayParams(
                diffusive_shock_acceleration_start_time=0.0,
                diffusive_shock_acceleration_efficiency=0.5,
            ),
            finalized_config,
            registered_variables,
            1e-3,
        )
        assert bool(jnp.all(jnp.isfinite(new_state)))
        np.testing.assert_array_equal(np.asarray(new_state), np.asarray(state))


def test_injection_is_ad_safe_in_cr_free_cells():
    """
    Tangents through an injection step are finite where P_cr = 0 (every
    CR-free cell), and d E_cr / d zeta = E_cr / zeta.
    """
    state, config, helper_data, registered_variables, _ = _single_shock_state(
        upstream_cosmic_ray_pressure=0.0,
    )
    dt = 1e-4

    def injection_step(primitive_state, zeta):
        return inject_crs_at_strongest_shock(
            primitive_state,
            GAMMA,
            helper_data,
            CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta),
            config,
            registered_variables,
            dt,
        )

    def injection_step_at_fixed_efficiency(primitive_state):
        return injection_step(primitive_state, 0.3)

    tangent = jnp.asarray(np.random.default_rng(1).normal(size=state.shape)) * 1e-3
    _, state_tangent = jax.jvp(injection_step_at_fixed_efficiency, (state,), (tangent,))
    assert bool(jnp.all(jnp.isfinite(state_tangent)))

    def cr_energy(zeta):
        return _energies(injection_step(state, zeta), registered_variables, helper_data)[1]

    value, gradient = jax.value_and_grad(cr_energy)(0.3)
    assert np.isfinite(float(gradient))
    assert float(gradient) == pytest.approx(float(value) / 0.3, rel=1e-10)

    # The conversion helper itself: zero tangent at P_cr = 0, exact elsewhere.
    derivative_at_zero = jax.grad(cosmic_ray_n_from_pressure)(0.0)
    derivative_at_two = jax.grad(cosmic_ray_n_from_pressure)(2.0)
    assert float(derivative_at_zero) == 0.0
    assert float(derivative_at_two) == pytest.approx(0.75 * 2.0 ** (-0.25), rel=1e-12)


def test_outermost_selection_picks_the_forward_shock():
    """Strong inner shock + weaker outer one: STRONGEST -> inner, OUTERMOST -> outer."""
    # An outer M = 3 shock runs into (1, 1, 0); an inner M = 10 shock runs into
    # the outer shock's downstream state.
    outer_density, outer_pressure, outer_velocity, _ = _rankine_hugoniot(3.0, 1.0, 1.0)
    inner_density, inner_pressure, inner_velocity_jump, _ = _rankine_hugoniot(
        10.0,
        outer_density,
        outer_pressure,
    )
    outer_step = _smooth_step(NUM_CELLS, 90.3, 0.6)
    inner_step = _smooth_step(NUM_CELLS, 40.3, 0.6)
    density = (
        1.0
        + (outer_density - 1.0) * outer_step
        + (inner_density - outer_density) * inner_step
    )
    pressure = (
        1.0
        + (outer_pressure - 1.0) * outer_step
        + (inner_pressure - outer_pressure) * inner_step
    )
    velocity = outer_velocity * outer_step + inner_velocity_jump * inner_step
    picked_index = {}
    for selection in (STRONGEST_SHOCK, OUTERMOST_SHOCK):
        config, helper_data, registered_variables = _setup(shock_selection=selection)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=jnp.asarray(density),
            velocity_x=jnp.asarray(velocity),
            gas_pressure=jnp.asarray(pressure),
            cosmic_ray_pressure=jnp.zeros(NUM_CELLS),
        )
        config = finalize_config(config, state.shape)
        shock_index, _, _ = find_shock_zone(
            state,
            config,
            registered_variables,
            helper_data,
            shock_selection=selection,
        )
        picked_index[selection] = int(shock_index)
    assert abs(picked_index[STRONGEST_SHOCK] - 40) <= 2
    assert abs(picked_index[OUTERMOST_SHOCK] - 90) <= 2


def test_spherical_injection_near_the_origin():
    """
    A flagged M = 2 shock a few cells from r = 0 in spherical geometry gets
    its CRs, conservatively and with no negative per-cell share.

    Near the origin the upstream reference cell is larger than the zone cells
    (V_ref > V_i), so the weights must compare energy densities,
    ``(e_i - e_ref) V_i``; the volume-integrated form ``e_i V_i - e_ref V_ref``
    is negative in every zone cell, and the clipped step would inject nothing.
    """
    num_cells, zeta, dt = 64, 0.5, 1e-3
    config = SimulationConfig(
        geometry=SPHERICAL,
        num_cells=num_cells,
        box_size=1.0,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=True,
        ),
        riemann_solver=HLL,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    downstream_density, downstream_pressure, downstream_velocity, _ = _rankine_hugoniot(
        2.0,
        1.0,
        1.0,
    )
    step_profile = _smooth_step(num_cells, 4.3, 2.0)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.asarray(1.0 + (downstream_density - 1.0) * step_profile),
        velocity_x=jnp.asarray(downstream_velocity * step_profile),
        gas_pressure=jnp.asarray(1.0 + (downstream_pressure - 1.0) * step_profile),
        cosmic_ray_pressure=jnp.zeros(num_cells),
    )
    config = finalize_config(config, state.shape)

    # The shock criteria flag this state, so the solver would call the injection.
    assert bool(jnp.any(shock_criteria(state, config, registered_variables, helper_data)))

    new_state = inject_crs_at_strongest_shock(
        state,
        GAMMA,
        helper_data,
        CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta),
        config,
        registered_variables,
        dt,
    )
    gas_energy_before, cr_energy_before = _energies(state, registered_variables, helper_data)
    gas_energy_after, cr_energy_after = _energies(new_state, registered_variables, helper_data)
    gas_loss = float(gas_energy_before - gas_energy_after)
    cr_gain = float(cr_energy_after - cr_energy_before)
    assert cr_gain > 0.0
    assert cr_gain == pytest.approx(gas_loss, rel=1e-12)
    new_cosmic_ray_pressure = np.asarray(
        cosmic_ray_pressure_from_n(new_state[registered_variables.cosmic_ray_n_index])
    )
    assert float(np.min(new_cosmic_ray_pressure)) >= 0.0
    new_gas_pressure = (
        np.asarray(new_state[registered_variables.pressure_index]) - new_cosmic_ray_pressure
    )
    old_pressure = np.asarray(state[registered_variables.pressure_index])
    assert float(np.min(old_pressure - new_gas_pressure)) >= 0.0
