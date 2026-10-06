"""
The cold-crush flux blend in 1D and 2D finite-difference MHD.

Finite-difference MHD carries all three momentum components in every
dimensionality, so the first-order flux the blend mixes in and its face
temperature must use all three; this test runs a few blended steps of a cold,
compressive 1D and 2D MHD flow and checks that the result is finite and that
mass is conserved.

    JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_coldcrush_blend_mhd.py
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# testing
import pytest

# jax
import jax

# The mass check compares sums to 1e-12.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

NUM_CELLS = 32


def _run_blended_mhd(dimensionality):
    """A few steps of a cold, converging MHD flow with the cold-crush blend on.

    Args:
        dimensionality: 1 or 2.

    Returns:
        The initial and the final primitive state and the registered variables.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    boundary_settings = periodic if dimensionality == 1 else BoundarySettings(periodic, periodic)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=dimensionality,
        mhd=True,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=boundary_settings,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        positivity_config=PositivityConfig(coldcrush_blend=True),
        progress_bar=False,
    )
    registered_variables = get_registered_variables(config)

    coordinates = (jnp.arange(NUM_CELLS) + 0.5) / NUM_CELLS
    if dimensionality == 1:
        x = coordinates
        y = jnp.zeros_like(coordinates)
    else:
        x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    zeros = jnp.zeros_like(x)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=1.0 + 0.5 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y),
        # converging flow towards x = 0.5, with an out-of-plane component
        velocity_x=-0.5 * jnp.sin(2 * jnp.pi * (x - 0.5)),
        velocity_y=0.3 * jnp.sin(2 * jnp.pi * x),
        velocity_z=0.2 * jnp.cos(2 * jnp.pi * x),
        gas_pressure=0.05 + zeros,
        magnetic_field_x=0.2 + zeros,
        magnetic_field_y=0.1 + zeros,
        magnetic_field_z=zeros,
    )
    config = finalize_config(config, state.shape)

    # A temperature floor just below the gas temperature makes the blend act.
    params = SimulationParams(C_cfl=0.8, t_end=0.02, minimum_specific_pressure=0.04)
    final_state = time_integration(state, config, params, registered_variables)
    return np.asarray(state), np.asarray(final_state), registered_variables


@pytest.mark.parametrize("dimensionality", [1, 2])
def test_coldcrush_blend_mhd_low_dimensional(dimensionality):
    """Blended 1D and 2D MHD steps stay finite and conserve mass."""
    initial_state, final_state, registered_variables = _run_blended_mhd(dimensionality)
    assert np.all(np.isfinite(final_state))
    density_index = registered_variables.density_index
    np.testing.assert_allclose(
        final_state[density_index].sum(),
        initial_state[density_index].sum(),
        rtol=1e-12,
    )


if __name__ == "__main__":
    for dimensionality in (1, 2):
        test_coldcrush_blend_mhd_low_dimensional(dimensionality)
        print(f"{dimensionality}D: ok")
