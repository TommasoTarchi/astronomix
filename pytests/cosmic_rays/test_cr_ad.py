"""
Forward-mode AD through a full CR time integration with DSA (CPU, ~30 s).

The Pfrommer+17 tube with CR-FREE initial conditions (P_cr = 0 everywhere,
every CR is injected at the shock), 201 cells, t = 0.2, finite-volume HLL,
float64. Guards:

* tangents are finite. Before 2026-09-25 they were NaN in every direction:
  ``P_cr ** (3/4)`` has an infinite derivative at P_cr = 0 (inf * 0 = NaN in
  every CR-free cell), measured d(sum P_cr)/d zeta = NaN;
* with a FIXED time step the JVP equals the central finite difference to
  round-off (measured 16.48068 vs 16.48068 in zeta, 8.65235 vs 8.65235 in the
  left-state pressure scale), i.e. the tangent of the injection operator is
  exact for frozen shock-zone indices.

Documented, NOT asserted: with the adaptive CFL step (``dt`` is
``stop_gradient``-ed) the state-direction JVP differs from the converged FD by
27 % (8.58 vs 6.78), while it agrees to 0.2 % without injection and to 0.6 % in
the zeta direction. The argmax-selected shock zone makes the response a
staircase (FD with h = 1e-4 gives 5.5-6.4 against the local 8.65 at fixed dt),
and the injection samples the sub-cell shock position every step, so the
dt dependence dropped by the stop-gradient is O(1) here. Gradient-based fits
through DSA need the smooth, local injection design of ``cr_code.md`` 4b.
"""

# ==== device ====
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 1D, 201 cells; no GPU needed
# ruff: noqa: E402
# =================

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    BackendConfig,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    CosmicRayConfig,
    CosmicRayParams,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)

NUM_CELLS = 201
NUM_STEPS = 400
T_END = 0.2


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Run this module in float64 and restore the previous setting afterwards."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _objective():
    config = SimulationConfig(
        geometry=CARTESIAN,
        num_cells=NUM_CELLS,
        box_size=10.0,
        cosmic_ray_config=CosmicRayConfig(cosmic_rays=True, diffusive_shock_acceleration=True),
        riemann_solver=HLL,
        progress_bar=False,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        fixed_timestep=True,
        num_timesteps=NUM_STEPS,
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    x = helper_data.geometric_centers
    left = x < 5.0

    def initial_state(scale):
        return construct_primitive_state(
            config=config, registered_variables=rv,
            density=jnp.where(left, 1.0, 0.125), velocity_x=jnp.zeros_like(x),
            gas_pressure=jnp.where(left, 17.172 * scale, 0.05),
            cosmic_ray_pressure=jnp.zeros_like(x),
        )

    config = finalize_config(config, initial_state(1.0).shape)

    def total_cr_pressure(zeta, scale):
        params = SimulationParams(
            t_end=T_END,
            cosmic_ray_params=CosmicRayParams(diffusive_shock_acceleration_efficiency=zeta),
        )
        out = time_integration(initial_state(scale), config, params, rv)
        return jnp.sum(cosmic_ray_pressure_from_n(out[rv.cosmic_ray_n_index]))

    return total_cr_pressure


@pytest.mark.parametrize("direction", ["zeta", "state"])
def test_cr_jvp_matches_fd_at_fixed_dt(direction):
    f = _objective()
    if direction == "zeta":
        g = lambda a: f(0.3 + a, 1.0)  # noqa: E731
    else:
        g = lambda a: f(0.3, 1.0 + a)  # noqa: E731
    value, tangent = jax.jvp(g, (0.0,), (1.0,))
    assert np.isfinite(float(value)) and float(value) > 0.0
    assert np.isfinite(float(tangent))
    h = 1e-6
    fd = (float(g(h)) - float(g(-h))) / (2 * h)
    assert float(tangent) == pytest.approx(fd, rel=1e-5)
