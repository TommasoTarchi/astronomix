"""
Convenience functions for operations that combine multiple elements
of an array based on some stencil, e.g. b_i <- a_{i + 1} + a_{i - 1}.
Allows for code "closer to the math".
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp

def custom_roll(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """Periodic roll of ``input_array`` by ``shift`` along ``axis``.

    Equivalent to ``jnp.roll`` but expressed via two static slices and a
    concatenate, which keeps ``shift`` / ``axis`` compile-time constants so the
    stencil helpers built on top of it fuse cleanly.

    Inside a multi-device ``pallas_mesh_context`` (a sharded
    ``time_integration``), a roll along the split axis is a shard_map halo
    exchange of ``|shift|`` planes (``sharded_roll``) instead of GSPMD's
    all-to-all reshard of the whole block; values identical.

    Args:
        input_array: The array to roll.
        shift: The (signed) number of positions to roll by.
        axis: The axis along which to roll.

    Returns:
        The rolled array.
    """
    from astronomix._pallas_helpers import sharded_roll
    rolled = sharded_roll(input_array, shift, axis)
    if rolled is not None:
        return rolled
    return _custom_roll_local(input_array, shift, axis)


@partial(jax.jit, static_argnames=["shift", "axis"])
def _custom_roll_local(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """The single-device roll (two static slices + a concatenate)."""
    i = (-shift) % input_array.shape[axis]
    return jax.lax.concatenate(
        [
            jax.lax.slice_in_dim(input_array, i, input_array.shape[axis], axis=axis),
            jax.lax.slice_in_dim(input_array, 0, i, axis=axis),
        ],
        dimension=axis,
    )


def _shift(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """Shift ``input_array`` by ``shift`` along ``axis``.

    A thin indirection over :func:`custom_roll`: the shift is currently periodic,
    but routing every stencil through this single entry point leaves room to
    support other boundary conditions later without touching call sites.
    """
    return custom_roll(input_array, shift, axis)

# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["indices", "axis"])
def _stencil_add(
    input_array: jnp.ndarray,
    indices: Tuple[int, ...],
    factors: Tuple[Union[float, Float[Array, ""]], ...],
    axis: int,
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.

    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """

    output = sum(
        factor * custom_roll(input_array, -index, axis=axis)
        for factor, index in zip(factors, indices)
    )

    return output
