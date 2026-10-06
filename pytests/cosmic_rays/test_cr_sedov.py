"""
1D spherical Sedov-Taylor blast with cosmic-ray shock acceleration (CPU, ~30 s).

E = 1 into rho = 1 (P_0 = 1e-4) at t = 0.1, 2001 radial cells, finite-volume
HLL, float64 -- the setup of Pfrommer et al. (2017, Sec. 4.2) in 1D. The
reference is the exact self-similar two-fluid solution for a constant
injection efficiency (``cr_sedov_two_fluid``): zeta = 0.5 gives R = 0.42113
(between the gamma = 7/5 Sedov radius 0.41116 that Pfrommer et al. compare
with and the gamma = 5/3 radius 0.45849), a post-shock
P_cr/P_th = zeta/[2(1 - zeta)] = 0.5 and a CR share of the blast energy of
0.529.

Measured errors (zeta = 0.5, DSA from t = 0): shock radius -0.39 / -0.21 /
-0.02 % at 1001 / 2001 / 4001 cells, E_cr/E 0.562 / 0.539 / 0.525; zeta = 0.1
at 2001 cells: +0.27 % (the zeta = 0 run's own bias is +0.15 %), E_cr/E 0.136
vs 0.143. Injecting from t = 0 exercises the injection guards: the first
flagged "shock" is a one-cell pressure pulse whose weights sum to ~0.
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

# Total-energy conservation is asserted at rel = 1e-10, which needs double
# precision.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    SPHERICAL,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)
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
    time_integration,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)
from astronomix.test_setups.reference_solutions.cr_sedov_two_fluid import (
    sedov_radius,
    sedov_two_fluid,
)

NUM_CELLS = 2001
T_END = 0.1
NUM_INJECTION_CELLS = 5
AMBIENT_PRESSURE = 1e-4
GAMMA_CR = 4.0 / 3.0


def run_cr_sedov(zeta, escape_fraction=0.0, num_cells=NUM_CELLS, start_time=0.0):
    """
    Run the spherical blast with cosmic-ray injection.

    Args:
        zeta: The injection efficiency (0 switches the injection off).
        escape_fraction: The fraction of the injected CR energy that escapes.
        num_cells: The number of radial cells.
        start_time: The start time of the diffusive shock acceleration.

    Returns:
        A dict with ``finite`` (all final values finite), ``r_sh`` (shock
        radius), ``e0`` (initial energy), ``e_tot`` (final total energy),
        ``e_cr`` (final CR energy), ``x_behind`` (mean P_cr / P_th just behind
        the shock) and ``min_p_th`` (minimum final thermal pressure).
    """

    # --------------- ↓ Setup ↓ ----------------
    config = SimulationConfig(
        geometry=SPHERICAL,
        first_order_fallback=True,
        progress_bar=False,
        num_cells=num_cells,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=zeta > 0,
        ),
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        riemann_solver=HLL,
    )
    params = SimulationParams(
        t_end=T_END,
        dt_max=1e-5,
        cosmic_ray_params=CosmicRayParams(
            diffusive_shock_acceleration_start_time=start_time,
            diffusive_shock_acceleration_efficiency=zeta,
            escape_fraction=escape_fraction,
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    radii = helper_data.geometric_centers

    # The unit blast energy is deposited as thermal energy in the innermost
    # cells.
    injection_radius = helper_data.outer_cell_boundaries[NUM_INJECTION_CELLS]
    injection_pressure = (params.gamma - 1) / (4 / 3 * np.pi * injection_radius**3)
    gas_pressure = AMBIENT_PRESSURE * jnp.ones_like(radii)
    gas_pressure = gas_pressure.at[: NUM_INJECTION_CELLS + 1].set(injection_pressure)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.ones_like(radii),
        velocity_x=jnp.zeros_like(radii),
        gas_pressure=gas_pressure,
        cosmic_ray_pressure=jnp.zeros_like(radii),
    )
    config = finalize_config(config, state.shape)
    cell_volumes = np.asarray(helper_data.cell_volumes)
    initial_energy = float(
        np.sum(
            np.asarray(state[registered_variables.pressure_index])
            / (params.gamma - 1)
            * cell_volumes
        )
    )
    # --------------- ↑ Setup ↑ ----------------

    final_state = np.asarray(time_integration(state, config, params, registered_variables))

    # --------------- ↓ Diagnostics ↓ ----------------
    radii = np.asarray(radii)
    density = final_state[registered_variables.density_index]
    velocity = final_state[registered_variables.velocity_index]
    cosmic_ray_pressure = np.asarray(
        cosmic_ray_pressure_from_n(final_state[registered_variables.cosmic_ray_n_index])
    )
    thermal_pressure = final_state[registered_variables.pressure_index] - cosmic_ray_pressure

    # The shock radius is the outer crossing of (rho_max + rho_0) / 2, linearly
    # interpolated between the two bracketing cells.
    peak_index = int(np.argmax(density))
    half_density = 0.5 * (density[peak_index] + 1.0)
    shock_index = peak_index + int(np.argmax(density[peak_index:] < half_density))
    inner_radius = radii[shock_index - 1]
    outer_radius = radii[shock_index]
    inner_density = density[shock_index - 1]
    outer_density = density[shock_index]
    shock_radius = inner_radius + (half_density - inner_density) * (
        outer_radius - inner_radius
    ) / (outer_density - inner_density)

    cosmic_ray_energy = float(np.sum(cosmic_ray_pressure / (GAMMA_CR - 1) * cell_volumes))
    thermal_energy = float(np.sum(thermal_pressure / (params.gamma - 1) * cell_volumes))
    kinetic_energy = float(np.sum(0.5 * density * velocity**2 * cell_volumes))
    behind_shock = (radii > 0.96 * shock_radius) & (radii < 0.99 * shock_radius)
    # --------------- ↑ Diagnostics ↑ ----------------

    return dict(
        finite=bool(np.all(np.isfinite(final_state))),
        r_sh=shock_radius,
        e0=initial_energy,
        e_tot=cosmic_ray_energy + thermal_energy + kinetic_energy,
        e_cr=cosmic_ray_energy,
        x_behind=float(
            np.mean(cosmic_ray_pressure[behind_shock] / thermal_pressure[behind_shock])
        ),
        min_p_th=float(np.min(thermal_pressure)),
    )


def test_sedov_without_injection_matches_gamma_five_thirds():
    """Without injection the blast follows the gamma = 5/3 Sedov radius and conserves energy."""
    result = run_cr_sedov(zeta=0.0)
    assert result["finite"]
    exact_radius = sedov_radius(sedov_two_fluid(0.0)["alpha"], time=T_END)
    assert exact_radius == pytest.approx(0.45849, abs=1e-5)
    assert result["r_sh"] == pytest.approx(exact_radius, rel=5e-3)
    assert result["e_cr"] == 0.0
    assert result["e_tot"] == pytest.approx(result["e0"], rel=1e-10)


def test_sedov_with_injection_matches_two_fluid_self_similar_solution():
    """
    With zeta = 0.5 from t = 0 the blast follows the self-similar two-fluid
    solution: shock radius, CR energy share, post-shock P_cr / P_th, and an
    exactly conserved total energy.
    """
    zeta = 0.5
    reference = sedov_two_fluid(zeta)
    exact_radius = sedov_radius(reference["alpha"], time=T_END)
    assert exact_radius == pytest.approx(0.42113, abs=1e-5)
    assert reference["X0"] == pytest.approx(0.5)

    result = run_cr_sedov(zeta=zeta, start_time=0.0)
    assert result["finite"]
    assert result["min_p_th"] > 0.0
    assert result["r_sh"] == pytest.approx(exact_radius, rel=1e-2)

    # The two-fluid blast is clearly softer than the pure gas blast.
    gas_only_radius = sedov_radius(sedov_two_fluid(0.0)["alpha"], time=T_END)
    assert result["r_sh"] < 0.97 * gas_only_radius
    assert result["e_cr"] / result["e_tot"] == pytest.approx(
        reference["E_cr_fraction"],
        abs=0.04,
    )

    # P_cr / P_th just behind the shock is ~ zeta / [2 (1 - zeta)]; it rises
    # further inwards.
    assert result["x_behind"] == pytest.approx(reference["X0"], rel=0.15)

    # The injection moves energy between the fluids; the total is conserved.
    assert result["e_tot"] == pytest.approx(result["e0"], rel=1e-10)


def test_sedov_escape_removes_energy():
    """
    ``escape_fraction`` removes a fraction of the injected CR energy: the
    blast loses energy, holds less CR energy and is smaller than without
    escape.
    """
    without_escape = run_cr_sedov(zeta=0.5, num_cells=1001)
    with_escape = run_cr_sedov(zeta=0.5, escape_fraction=0.5, num_cells=1001)
    assert with_escape["finite"]
    assert with_escape["e_tot"] < 0.9 * with_escape["e0"]
    assert with_escape["e_cr"] < without_escape["e_cr"]
    assert with_escape["r_sh"] < without_escape["r_sh"]
