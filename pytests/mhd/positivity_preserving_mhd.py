"""
Positivity-preserving WENO pytest for ideal MHD.

For ideal MHD a single Lax-Friedrichs split state q +- F/alpha is often not
admissible at the fast speed when beta is low (Wu 2018); the scalings act on
the weighted pairs of each cell's update instead. Two checks:

* the low-beta blast wave (Balsara & Spicer 1999; ambient beta 2.5e-4): the
  scheme without positivity preservation fails within half the run, with it
  density and pressure stay positive to the end;
* a circularly polarised Alfven wave at beta = 0.02: the limiter stays out of
  the way in smooth flow (an unpaired limiter falls back to first order here).
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
from astronomix import NATIVE_JAX, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import DOUBLE_PRECISION, IDEAL_GAS, StaticIntVector

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)
from astronomix.test_setups.mhd.alfven_wave3D import (
    CPAlfvenWave3DSettings,
    cp_alfven_wave_solution,
    setup_cp_alfven_wave,
)

jax.config.update("jax_enable_x64", True)


def test_low_beta_blast():
    """Blast wave into beta = 2.5e-4 plasma: positive density and pressure to t = 0.01."""
    num_cells = 50
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        equation_of_state=IDEAL_GAS,
        dimensionality=2,
        num_cells=num_cells,
        box_size=1.0,
        mhd=True,
        numerical_precision=DOUBLE_PRECISION,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        positivity_config=PositivityConfig(clamp_in_estimates=False),
        weno_positivity_preserving=True,
    )
    params = SimulationParams(
        C_cfl=0.75, gamma=1.4, t_end=0.01, minimum_density=1e-30, minimum_pressure=1e-30
    )
    registered_variables = get_registered_variables(config)
    centres = (jnp.arange(num_cells) + 0.5) / num_cells - 0.5
    x, y = jnp.meshgrid(centres, centres, indexing="ij")
    zero = jnp.zeros_like(x)
    field_x = jnp.full_like(x, 100.0 / jnp.sqrt(4.0 * jnp.pi))
    bx_face, by_face, bz_face = initialize_interface_fields(field_x, zero, zero, dimensionality=2)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.ones_like(x),
        velocity_x=zero,
        velocity_y=zero,
        gas_pressure=jnp.where(jnp.sqrt(x**2 + y**2) < 0.1, 1000.0, 0.1),
        magnetic_field_x=field_x,
        magnetic_field_y=zero,
        magnetic_field_z=zero,
        interface_magnetic_field_x=bx_face,
        interface_magnetic_field_y=by_face,
        interface_magnetic_field_z=bz_face,
    )
    config = finalize_config(config, state.shape)
    final = np.asarray(time_integration(state, config, params, registered_variables))
    assert np.all(np.isfinite(final))
    assert final[registered_variables.density_index].min() > 0.0
    assert final[registered_variables.pressure_index].min() > 0.0


def test_low_beta_alfven_wave_order():
    """CP Alfven wave at beta = 0.02 converges at high order with the limiter on."""
    settings = CPAlfvenWave3DSettings(p_0=0.01, t_end=1.0)
    errors = []
    for n in (8, 16):
        config = SimulationConfig(
            num_cells=StaticIntVector(2 * n, n, n),
            numerical_precision=DOUBLE_PRECISION,
            backend_config=BackendConfig(backend=NATIVE_JAX),
            positivity_config=PositivityConfig(clamp_in_estimates=False),
            return_snapshots=True,
            snapshot_settings=SnapshotSettings(return_states=False, return_final_state=True),
            num_snapshots=2,
            weno_positivity_preserving=True,
        )
        params = SimulationParams(C_cfl=0.75, minimum_density=1e-30, minimum_pressure=1e-30)
        state, config, params = setup_cp_alfven_wave(config, params, settings)
        registered_variables = get_registered_variables(config)
        final = time_integration(state, config, params, registered_variables).final_state
        exact = cp_alfven_wave_solution(
            config, registered_variables, params, get_helper_data(config), settings
        )
        indices = (
            registered_variables.density_index,
            *registered_variables.velocity_index,
            registered_variables.pressure_index,
            *registered_variables.magnetic_index,
        )
        errors.append(float(np.mean([jnp.mean(jnp.abs(final[i] - exact[i])) for i in indices])))
    order = np.log2(errors[0] / errors[1])
    # measured: 6.2e-3 -> 3.6e-4 (order 4.1); the unpaired limiter gave ~0.2
    assert errors[1] < 1e-3, errors
    assert order > 3.5, (errors, order)


if __name__ == "__main__":
    test_low_beta_blast()
    test_low_beta_alfven_wave_order()
