"""
3D self-gravitating slab advection correctness pytest (fast).

Advects a self-gravitating density slab across a periodic cubic box at a single
low resolution and checks the final state against the analytic solution for the
three finite-difference self-gravity treatments (simple source, flux-based
source, corrected flux-based source). This is the fast correctness check; the
resolution-sweep convergence figure lives in
``examples/scripts/forward/self_gravity/slab_convergence.py``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# Self-gravity source terms are compared against the analytic slab solution, so
# run in double precision to keep round-off well below the tolerance.
jax.config.update("jax_enable_x64", True)

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import (
    GravityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)
from astronomix._modules._sink_particles._sink_particle_options import SinkParticleConfig
from astronomix.data_classes.simulation_state_struct import StateStruct

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.self_gravity.slab_advection import (
    setup_slab_advection,
    slab_advection_solution,
)


# Cubic box of side length 3 pi (one wavelength along each axis).
BOX_LENGTH = float(3.0 * jnp.pi)
BOX = StaticFloatVector(BOX_LENGTH, BOX_LENGTH, BOX_LENGTH)

# One base configuration per self-gravity treatment, with sink particles enabled.
CONFIG_LIST = [
    SimulationConfig(
        box_size=BOX,
        mhd=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=version,
        ),
        dimensionality=3,
        progress_bar=False,
        state_struct=True,
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    for version in (SIMPLE_SOURCE, SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE)
]


def test_slab_advection_with_sinks(N=16, tol=5e-2):
    """Advect the self-gravitating slab at ``N``^3 with sink particle formation
    on and check every scheme.

    Each self-gravity treatment is integrated at a single low resolution. The mean
    L1 error of the final state against the analytic slab solution, over the five
    primitive variables, must stay below ``tol``. The gas stays far below the sink
    density threshold throughout, so no sink particle may form.

    Args:
        N: The per-dimension resolution of the cubic grid.
        tol: The maximum allowed mean L1 error per scheme.
    """
    for base_config in CONFIG_LIST:
        config = base_config._replace(num_cells=StaticIntVector(N, N, N))

        initial_state, config, params = setup_slab_advection(
            config,
            SimulationParams(C_cfl=1.5),
        )
        initial_state = StateStruct(primitive_state=initial_state)

        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        final_state = time_integration(
            initial_state, config, params, registered_variables
        )
        true_final_state = slab_advection_solution(
            config, registered_variables, params, helper_data
        )

        indices = (
            registered_variables.density_index,
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
            registered_variables.pressure_index,
        )
        l1 = jnp.mean(
            jnp.stack([
                jnp.mean(jnp.abs(final_state.primitive_state[i] - true_final_state[i]))
                for i in indices
            ])
        )
        version = base_config.gravity_config.self_gravity_version
        assert l1 < tol, f"slab advection (gravity v{version}) L1 {l1:.3e} exceeds {tol}"

        num_sinks = int(jnp.sum(final_state.sink_particles.mass > 0.0))
        assert num_sinks == 0, (
            f"slab advection (gravity v{version}): {num_sinks} spurious sink particles formed"
        )


if __name__ == "__main__":
    test_slab_advection_with_sinks()
