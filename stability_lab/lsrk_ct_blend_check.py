"""Does flux blending act under MHD + RK4_LSRK? (It was silently skipped.)

Runs a few steps of rough isothermal MHD at 16^3 with and without the deep-void
LLF blend (factor large enough to blend everywhere below rho ~ 1) and reports
the difference. Zero difference = blending ignored.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
import jax
import jax.numpy as jnp
import numpy as np
from astronomix import (BackendConfig, BoundarySettings, BoundarySettings1D, NATIVE_JAX, PERIODIC_BOUNDARY,
                        PositivityConfig, SimulationConfig, SimulationParams, construct_primitive_state,
                        finalize_config, get_registered_variables, initialize_interface_fields, time_integration)
from astronomix.option_classes.simulation_config import ISOTHERMAL, RK4_LSRK, RK4_SSP

n = 16
periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
rho = jnp.exp(0.5 * jax.random.normal(k1, (n, n, n)))
v = 0.3 * jax.random.normal(k2, (3, n, n, n))
bz = jnp.full((n, n, n), 0.5)
zero = jnp.zeros_like(rho)
bxf, byf, bzf = initialize_interface_fields(zero, zero, bz)
for integrator in (RK4_SSP, RK4_LSRK):
    finals = []
    for blend in (False, True):
        config = SimulationConfig(
            equation_of_state=ISOTHERMAL, dimensionality=3, num_cells=n, mhd=True,
            time_integrator=integrator, backend_config=BackendConfig(backend=NATIVE_JAX),
            boundary_settings=BoundarySettings(periodic, periodic, periodic),
            positivity_config=PositivityConfig(deepvoid_blend=blend, deepvoid_blend_factor=1e10),
        )
        rv = get_registered_variables(config)
        state = construct_primitive_state(
            config=config, registered_variables=rv, density=rho,
            velocity_x=v[0], velocity_y=v[1], velocity_z=v[2],
            magnetic_field_x=zero, magnetic_field_y=zero, magnetic_field_z=bz,
            interface_magnetic_field_x=bxf, interface_magnetic_field_y=byf, interface_magnetic_field_z=bzf,
        )
        config = finalize_config(config, state.shape)
        params = SimulationParams(t_end=0.02, isothermal_sound_speed=1.0, minimum_density=1e-10, C_cfl=0.5)
        finals.append(np.asarray(time_integration(state, config, params, rv)))
    name = "RK4_SSP " if integrator == RK4_SSP else "RK4_LSRK"
    print(f"{name}: max |blend - no blend| in rho = {np.max(np.abs(finals[1][0] - finals[0][0])):.3e}", flush=True)
