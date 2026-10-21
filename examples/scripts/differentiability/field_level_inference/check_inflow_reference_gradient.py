"""
Check that the reverse-mode gradient of the field-level inference loss is the
same on the Pallas and the native backend (3D ideal MHD with the
positivity-preserving WENO, whose Pallas inflow reference must sit behind
``diffable_pallas_call``).

Usage::

    python check_inflow_reference_gradient.py [--resolution 64] [--time-fraction 0.05]
"""

# general
import argparse
import os

# jax
import jax
import jax.numpy as jnp

# astronomix functions
from astronomix import NATIVE_JAX, PALLAS

# local
from _problem import (
    driven_turbulence,
    finalized,
    initial_velocity,
    logo_target,
    make_config,
    make_loss,
    rms_velocity,
)

DATA_DIRECTORY = "/e/project1/astronomix/storcks1/revision_data/field_level_inference"


def main():
    """Compare the Pallas and native gradients of the inference loss."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--time-fraction", type=float, default=0.05)
    args = parser.parse_args()

    turbulence, _ = driven_turbulence(
        args.resolution, os.path.join(DATA_DIRECTORY, f"turbulence_N{args.resolution}.npy")
    )
    registered_variables, _ = finalized(make_config(args.resolution))
    t_loss = args.time_fraction * 0.5 / rms_velocity(turbulence, registered_variables)
    target = logo_target(args.resolution, float(turbulence[registered_variables.density_index].sum()))
    velocity = initial_velocity(turbulence, registered_variables)

    results = {}
    for name, backend in (("pallas", PALLAS), ("native", NATIVE_JAX)):
        _, config = finalized(make_config(args.resolution, backward=True, backend=backend))
        loss = make_loss(turbulence, config, registered_variables, t_loss, target)
        results[name] = jax.block_until_ready(jax.jit(jax.value_and_grad(loss))(velocity))
        value, gradient = results[name]
        print(f"{name}: loss {float(value):.10e}, |grad| {float(jnp.linalg.norm(gradient)):.10e}", flush=True)

    gradient_pallas, gradient_native = results["pallas"][1], results["native"][1]
    relative = jnp.linalg.norm(gradient_pallas - gradient_native) / jnp.linalg.norm(gradient_native)
    print(f"relative gradient difference pallas vs native: {float(relative):.3e}", flush=True)


if __name__ == "__main__":
    main()
