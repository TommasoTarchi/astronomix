"""
Positivity-preserving WENO pytest.

Two stress problems where the bare finite-difference scheme fails and the
``weno_admissible_face_state`` / ``weno_positivity_preserving`` options must
hold, plus a smooth-flow check that the options do not cost order of accuracy:

* a cold dense slab rammed at Mach ~800 into tenuous gas (the default interface
  sound speed goes negative there and the run blows up within a few steps);
* a strong double rarefaction close to vacuum (positivity of density and
  pressure);
* an advected entropy wave (fifth-order convergence with the options on).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import numpy as np

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DOUBLE_PRECISION,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import BoundarySettings1D

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.reference_solutions.riemann_solver import (
    _exact_riemann_ideal_gas,
)

jax.config.update("jax_enable_x64", True)


def _run_riemann(left, right, gamma, t_end, num_cells, **options):
    """Run a 1D Riemann problem (rho, u, p per side) on [0, 1], diaphragm at 0.5."""
    config = SimulationConfig(
        dimensionality=1,
        num_cells=num_cells,
        numerical_precision=DOUBLE_PRECISION,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        **options,
    )
    params = SimulationParams(t_end=t_end, gamma=gamma, C_cfl=0.4)
    registered_variables = get_registered_variables(config)
    x = get_helper_data(config).geometric_centers
    is_left = x < 0.5
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.where(is_left, left[0], right[0]),
        velocity_x=jnp.where(is_left, left[1], right[1]),
        gas_pressure=jnp.where(is_left, left[2], right[2]),
    )
    config = finalize_config(config, state.shape)
    final = np.asarray(time_integration(state, config, params, registered_variables))
    exact_density, _, _ = _exact_riemann_ideal_gas(*left, *right, gamma, x, t_end, 0.5)
    return final, registered_variables, np.asarray(exact_density)


def test_cold_dense_ram():
    """Mach ~800 cold slab into tenuous gas: finite, positive, near exact."""
    left, right = (100.0, 1.0, 1e-4), (1.0, 0.0, 1e-4)
    for options in (dict(weno_admissible_face_state=True), dict(weno_positivity_preserving=True)):
        final, registered_variables, exact = _run_riemann(left, right, 5.0 / 3.0, 0.3, 400, **options)
        density = final[registered_variables.density_index]
        pressure = final[registered_variables.pressure_index]
        assert np.all(np.isfinite(final)), options
        assert density.min() > 0.0 and pressure.min() > 0.0, options
        # the error is dominated by the smeared density-400 contact
        assert np.mean(np.abs(density - exact)) < 3.0, options


def test_near_vacuum_double_rarefaction():
    """Double rarefaction with u = -+3.5 (vacuum at u = -+3.74): positive."""
    left, right = (1.0, -3.5, 0.4), (1.0, 3.5, 0.4)
    final, registered_variables, exact = _run_riemann(
        left, right, 1.4, 0.1, 400, weno_positivity_preserving=True
    )
    density = final[registered_variables.density_index]
    pressure = final[registered_variables.pressure_index]
    assert density.min() > 0.0 and pressure.min() > 0.0
    assert np.mean(np.abs(density - exact)) < 1e-2


def test_smooth_fifth_order():
    """An advected entropy wave stays fifth order with the options on."""
    errors = []
    for num_cells in (32, 64):
        config = SimulationConfig(
            dimensionality=1,
            num_cells=num_cells,
            numerical_precision=DOUBLE_PRECISION,
            boundary_handling=PERIODIC_ROLL,
            num_ghost_cells=0,
            boundary_settings=BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            weno_positivity_preserving=True,
        )
        params = SimulationParams(t_end=1.0, gamma=5.0 / 3.0, C_cfl=0.4)
        registered_variables = get_registered_variables(config)
        x = get_helper_data(config).geometric_centers
        density = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=jnp.ones_like(x),
            gas_pressure=jnp.ones_like(x),
        )
        # 1D finalize keeps ghost cells; the convergence test needs the roll
        config = finalize_config(config, state.shape)._replace(
            boundary_handling=PERIODIC_ROLL, num_ghost_cells=0
        )
        final = np.asarray(time_integration(state, config, params, registered_variables))
        errors.append(np.mean(np.abs(final[registered_variables.density_index] - np.asarray(density))))
    order = np.log2(errors[0] / errors[1])
    assert order > 4.5, f"observed order {order:.2f}"
