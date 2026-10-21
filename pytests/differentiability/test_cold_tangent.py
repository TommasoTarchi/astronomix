"""
The tangent-only cold-face linearisation (``ad_tangent_llf_cold_factor``).

In cold, kinetic-dominated gas far below the temperature floor the linearised
WENO flux advances a tangent with a linear high-order scheme that is not
stable at sharp density edges; the option takes the derivative of the flux on
faces with a cold side through the monotone Lax-Friedrichs flux instead and
leaves the primal untouched.

What is checked, on a periodic 1D flow with a cold, fast, structured density
band next to warm gas (positivity-preserving WENO, dual energy, cold-crush
blend):

* the primal with the option is that without it, bit for bit;
* the tangent with the option is finite and differs from the plain one;
* it agrees with a central finite difference no worse than the plain tangent
  does (the warm gas sits above the cold-face threshold and drives a strong
  blast into the cold band, so neither linearisation is close to the
  difference; the option changes the linearisation by design);
* ``finalize_config`` refuses the option without the cold-crush blend.

Run on the CPU::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_cold_tangent.py
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix import NATIVE_JAX, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import DOUBLE_PRECISION, FINITE_DIFFERENCE

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

jax.config.update("jax_enable_x64", True)

NUM_CELLS = 64
NUM_STEPS = 8
#: the floor specific pressure; the cold band sits three orders below it
FLOOR_SPECIFIC_PRESSURE = 1e-2


def make_config(cold_factor):
    """A 1D periodic PP-WENO configuration with the cold-crush blend on."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=1,
        num_cells=NUM_CELLS,
        box_size=1.0,
        numerical_precision=DOUBLE_PRECISION,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        boundary_settings=periodic,
        weno_positivity_preserving=True,
        positivity_config=PositivityConfig(coldcrush_blend=True, coldcrush_blend_factor=8.0),
        dual_energy=True,
        weno_ad_frozen_weights=True,
        ad_tangent_llf_cold_factor=cold_factor,
        fixed_timestep=True,
        num_timesteps=NUM_STEPS,
    )


def initial_density(log_density_amplitude):
    """A smooth density band with a sharp-edged knot, scaled by the control."""
    x = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
    band = 1.0 + 0.5 * np.sin(2.0 * np.pi * x)
    knot = np.where(np.abs(x - 0.3) < 0.05, 4.0, 0.0)
    return jnp.asarray(band + knot) * jnp.exp(log_density_amplitude * jnp.asarray(np.cos(4.0 * np.pi * x)))


def evolve(cold_factor):
    """Return ``control -> final density`` for the given cold factor."""
    config = make_config(cold_factor)
    registered_variables = get_registered_variables(config)
    x = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
    # cold where x < 0.5 (p / rho ~ 1e-5, far below the floor), and warm elsewhere
    # ABOVE the cold-face threshold of the factor 1000 (p / rho > 1000 x 1e-2), so
    # the per-face selection is exercised
    pressure = jnp.asarray(np.where(x < 0.5, 1e-5, 100.0))
    velocity = jnp.full(NUM_CELLS, 3.0)
    params = SimulationParams(
        C_cfl=0.3,
        dt_max=2e-3,
        t_end=NUM_STEPS * 2e-3,
        minimum_density=1e-8,
        minimum_pressure=1e-10,
        minimum_specific_pressure=FLOOR_SPECIFIC_PRESSURE,
    )

    def run(control):
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=initial_density(control),
            velocity_x=velocity,
            gas_pressure=pressure,
        )
        final_config = finalize_config(config, state.shape)
        final_state = time_integration(state, final_config, params, registered_variables)
        return final_state[registered_variables.density_index]

    return run


def test_primal_is_unchanged_bit_for_bit():
    control = jnp.asarray(0.05)
    plain = np.asarray(jax.jit(evolve(0.0))(control))
    cold = np.asarray(jax.jit(evolve(1000.0))(control))
    assert np.all(np.isfinite(plain))
    assert np.array_equal(plain, cold)


def test_tangent_is_finite_changed_and_no_worse_than_the_plain_one():
    control = jnp.asarray(0.05)
    tangents = {}
    for factor in (0.0, 1000.0):
        _, tangents[factor] = jax.jvp(evolve(factor), (control,), (jnp.asarray(1.0),))
        assert np.all(np.isfinite(np.asarray(tangents[factor])))
    assert not np.allclose(np.asarray(tangents[0.0]), np.asarray(tangents[1000.0]))

    step = 1e-5
    run = jax.jit(evolve(0.0))
    finite_difference = (np.asarray(run(control + step)) - np.asarray(run(control - step))) / (2.0 * step)
    scale = np.linalg.norm(finite_difference)
    error_plain = np.linalg.norm(np.asarray(tangents[0.0]) - finite_difference) / scale
    error_cold = np.linalg.norm(np.asarray(tangents[1000.0]) - finite_difference) / scale
    print(f"\n[cold tangent] |JVP - FD| / |FD|: plain {error_plain:.3e}, cold LLF {error_cold:.3e}")
    assert error_cold <= error_plain


def test_option_requires_the_cold_crush_blend():
    config = make_config(1000.0)._replace(positivity_config=PositivityConfig(coldcrush_blend=False))
    with pytest.raises(ValueError, match="coldcrush_blend"):
        finalize_config(config, (4, NUM_CELLS))
