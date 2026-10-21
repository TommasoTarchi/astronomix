"""
Check the Pallas adjoint of the ideal-MHD WENO flux on the field-level
inference problem: the reverse-mode gradient of the projected-density loss with
respect to the initial velocity, over a fraction of the loss time, through the
Pallas adjoint (``BackendConfig.pallas_adjoint``) and through the transposed
native-JAX tangent, and a central finite-difference check of the directional
derivative along a smooth random velocity perturbation.

Prints the relative difference and the cosine of the two gradients, and for
every finite-difference step the relative error of both directional
derivatives; with ``--double`` everything runs in float64.

Usage::

    python check_flux_adjoint_gradient.py --resolution 64 --time-fraction 0.03 [--double]
"""

# general
import argparse
import os
import sys

# The precision has to be fixed before jax is imported anywhere.
if "--double" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"

# gpu selection
from autocvd import autocvd

autocvd(num_gpus=1)

# jax
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# numerics
import numpy as np  # noqa: E402

# field-level inference problem
from _problem import (  # noqa: E402
    driven_turbulence,
    finalized,
    initial_velocity,
    logo_target,
    make_config,
    make_loss,
    rms_velocity,
)

DATA_DIRECTORY = "/e/project1/astronomix/storcks1/revision_data/field_level_inference"


def parse_arguments():
    """
    Parse the command line.

    Returns:
        The argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=0.03,
        help="fraction of the loss time t_end = 0.5 l_0 / v_rms the loss integrates over",
    )
    parser.add_argument("--double", action="store_true", help="run in float64")
    parser.add_argument(
        "--steps",
        type=str,
        default="1e-1,3e-2,1e-2,3e-3",
        help="finite-difference steps, relative to the rms of the velocity perturbation",
    )
    return parser.parse_args()


def smooth_perturbation(shape, seed=0):
    """
    A smooth random velocity perturbation: white noise low-pass filtered to
    the lowest eight wavenumbers per axis, normalised to unit rms.

    Args:
        shape: The shape ``(3, N, N, N)``.
        seed: The random seed.

    Returns:
        The perturbation (numpy, float64).
    """
    noise = np.random.default_rng(seed).standard_normal(shape)
    spectrum = np.fft.fftn(noise, axes=(1, 2, 3))
    wavenumbers = np.fft.fftfreq(shape[1]) * shape[1]
    kx, ky, kz = np.meshgrid(wavenumbers, wavenumbers, wavenumbers, indexing="ij")
    spectrum *= (np.abs(kx) <= 8) & (np.abs(ky) <= 8) & (np.abs(kz) <= 8)
    perturbation = np.real(np.fft.ifftn(spectrum, axes=(1, 2, 3)))
    return perturbation / np.sqrt(np.mean(perturbation**2))


def main():
    """Compare the two reverse modes and check them by finite differences."""
    args = parse_arguments()
    dtype = jnp.float64 if args.double else jnp.float32
    turbulence, _ = driven_turbulence(
        args.resolution,
        os.path.join(DATA_DIRECTORY, f"turbulence_N{args.resolution}.npy"),
    )
    turbulence = turbulence.astype(dtype)
    registered_variables, _ = finalized(make_config(args.resolution))
    velocity_rms = rms_velocity(turbulence, registered_variables)
    loss_time = args.time_fraction * 0.5 / velocity_rms
    target = logo_target(args.resolution, float(turbulence[registered_variables.density_index].sum())).astype(dtype)
    velocity = initial_velocity(turbulence, registered_variables)
    print(
        f"N = {args.resolution}, {'float64' if args.double else 'float32'}, "
        f"loss time {loss_time:.4f} t_0 ({args.time_fraction} t_end)",
        flush=True,
    )

    # -------------------------------------------------------------
    # =========== ↓ Native tangent vs Pallas adjoint ↓ ============
    # -------------------------------------------------------------
    losses = {}
    gradients = {}
    for flux_adjoint in ("native", "pallas"):
        _, config = finalized(make_config(
            args.resolution,
            backward=True,
            pallas_adjoint=flux_adjoint == "pallas",
        ))
        losses[flux_adjoint] = make_loss(turbulence, config, registered_variables, loss_time, target)
        value, gradient = jax.block_until_ready(jax.jit(jax.value_and_grad(losses[flux_adjoint]))(velocity))
        gradients[flux_adjoint] = gradient
        print(
            f"{flux_adjoint:6s}: loss {float(value):.10e}, |grad| {float(jnp.linalg.norm(gradient)):.10e}",
            flush=True,
        )
    native, pallas = gradients["native"], gradients["pallas"]
    relative = jnp.linalg.norm(pallas - native) / jnp.linalg.norm(native)
    cosine = jnp.sum(pallas * native) / (jnp.linalg.norm(pallas) * jnp.linalg.norm(native))
    print(
        f"Pallas adjoint vs native tangent: relative difference {float(relative):.3e}, "
        f"cosine {float(cosine):.12f}",
        flush=True,
    )
    # -------------------------------------------------------------
    # =========== ↑ Native tangent vs Pallas adjoint ↑ ============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Finite-difference directional check ↓ ========
    # -------------------------------------------------------------
    direction = jnp.asarray(smooth_perturbation(velocity.shape), dtype=dtype)
    forward_loss = jax.jit(losses["native"])
    directional = {name: float(jnp.sum(gradient * direction)) for name, gradient in gradients.items()}
    for relative_step in (float(step) for step in args.steps.split(",")):
        step = relative_step * velocity_rms
        finite_difference = (
            float(forward_loss(velocity + step * direction))
            - float(forward_loss(velocity - step * direction))
        ) / (2.0 * step)
        errors = ", ".join(
            f"{name} {abs(value - finite_difference) / abs(finite_difference):.3e}"
            for name, value in directional.items()
        )
        print(
            f"step {relative_step:.0e} v_rms: finite difference {finite_difference:.8e}, "
            f"relative error of <grad, d>: {errors}",
            flush=True,
        )
    # -------------------------------------------------------------
    # ============ ↑ Finite-difference directional check ↑ ========
    # -------------------------------------------------------------


if __name__ == "__main__":
    main()
