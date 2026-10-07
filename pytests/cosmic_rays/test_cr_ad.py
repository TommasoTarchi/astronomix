"""
Forward-mode AD through a full CR time integration with DSA (CPU, ~10 s).

The Pfrommer et al. (2017) shock tube with CR-free initial conditions
(P_cr = 0 everywhere, every CR is injected at the shock), 201 cells, t = 0.2,
finite-volume HLL, float64. Guards:

* tangents are finite. ``P_cr ** (3/4)`` has an infinite derivative at
  P_cr = 0, so a naive conversion gives inf * 0 = NaN in every CR-free cell
  and d(sum P_cr)/d zeta = NaN;
* with a fixed time step the JVP equals the central finite difference to
  round-off (16.48068 in zeta, 8.65235 in the left-state pressure scale),
  i.e. the tangent of the injection operator is exact for frozen shock-zone
  indices.

Adaptive time steps are not tested: ``dt`` is ``stop_gradient``-ed, and the
argmax-selected shock zone makes the response a staircase that samples the
sub-cell shock position every step, so the dropped dt dependence is O(1) for
the state direction (JVP 8.58 vs a converged finite difference of 6.78).
Gradient-based fits through DSA need a smooth, local injection model.
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

# The JVP is compared with a central finite difference at rel = 1e-5, which
# needs double precision.
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

NUM_CELLS = 201
NUM_STEPS = 400
T_END = 0.2

#: Left-state thermal pressure of the Pfrommer et al. (2017) shock tube.
THERMAL_PRESSURE_LEFT = 17.172


def _build_total_cosmic_ray_pressure():
    """
    Set up the CR-free tube and return its scalar objective.

    Returns:
        A function ``total_cosmic_ray_pressure(zeta, scale)`` that runs the
        fixed-step integration with injection efficiency ``zeta`` and the
        left-state pressure multiplied by ``scale``, and returns the summed
        final cosmic-ray pressure.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        num_cells=NUM_CELLS,
        box_size=10.0,
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=True,
        ),
        riemann_solver=HLL,
        progress_bar=False,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        fixed_timestep=True,
        num_timesteps=NUM_STEPS,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    x = helper_data.geometric_centers
    left = x < 5.0

    def initial_state(scale):
        return construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=jnp.where(left, 1.0, 0.125),
            velocity_x=jnp.zeros_like(x),
            gas_pressure=jnp.where(left, THERMAL_PRESSURE_LEFT * scale, 0.05),
            cosmic_ray_pressure=jnp.zeros_like(x),
        )

    config = finalize_config(config, initial_state(1.0).shape)

    def total_cosmic_ray_pressure(zeta, scale):
        params = SimulationParams(
            t_end=T_END,
            cosmic_ray_params=CosmicRayParams(
                diffusive_shock_acceleration_efficiency=zeta,
            ),
        )
        final_state = time_integration(
            initial_state(scale),
            config,
            params,
            registered_variables,
        )
        return jnp.sum(
            cosmic_ray_pressure_from_n(final_state[registered_variables.cosmic_ray_n_index])
        )

    return total_cosmic_ray_pressure


@pytest.mark.parametrize("direction", ["zeta", "state"])
def test_cr_jvp_matches_fd_at_fixed_dt(direction):
    """
    At a fixed time step the JVP in the injection efficiency or in the
    left-state pressure is finite and equals the central finite difference.
    """
    total_cosmic_ray_pressure = _build_total_cosmic_ray_pressure()

    def perturbed_objective(perturbation):
        if direction == "zeta":
            return total_cosmic_ray_pressure(0.3 + perturbation, 1.0)
        return total_cosmic_ray_pressure(0.3, 1.0 + perturbation)

    value, tangent = jax.jvp(perturbed_objective, (0.0,), (1.0,))
    assert np.isfinite(float(value))
    assert float(value) > 0.0
    assert np.isfinite(float(tangent))

    step = 1e-6
    finite_difference = (
        float(perturbed_objective(step)) - float(perturbed_objective(-step))
    ) / (2 * step)
    assert float(tangent) == pytest.approx(finite_difference, rel=1e-5)
