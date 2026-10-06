"""Shared Pallas-backend utilities used across the FD and FV paths.

This module is the single place that:
- imports Pallas / Triton (and exposes ``pl is None`` if unavailable),
- normalises ``config.backend_config.pallas_block_shape`` to a 3-tuple,
- builds Triton ``CompilerParams`` from config knobs,
- exposes the ``backend == PALLAS`` predicate,
- provides the ``_pallas_call_sharded`` multi-GPU wrapper that turns an
  opaque ``pl.pallas_call`` into a ``shard_map`` + ppermute halo-exchange
  body when the user runs on a multi-device mesh.

Every Pallas kernel module under ``astronomix`` should import from here so
new knobs / fallbacks only need to be added once.
"""

# general
import contextvars
import os
from contextlib import contextmanager

# jax
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

# astronomix containers
from astronomix.option_classes.simulation_config import PALLAS, SimulationConfig

# Pallas / Triton are optional: a CPU-only or older JAX install may lack one or
# both. We import them defensively so the rest of the module loads (callers gate
# on ``pl is None`` / ``pltriton is None``) rather than failing at import time.
try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - Pallas optional
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - Triton GPU backend optional
    pltriton = None


def _backend_is_pallas(config: SimulationConfig) -> bool:
    """Return whether the configured backend is the Pallas/Triton GPU backend."""
    return config.backend_config.backend == PALLAS


def _default_pallas_block_shape(ndim: int) -> tuple[int, int, int]:
    """Return the default Pallas block shape ``(bx, by, bz)`` for ``ndim`` spatial
    dimensions (inactive dimensions forced to 1).

    The 3D default (2, 2, 32) keeps one element per thread at the default
    ``num_warps=4`` (128-cell blocks) with a 256-byte-contiguous fast axis:
    on an A100 this ran the dp MHD WENO kernel ~10% faster end-to-end than
    the previous (4, 4, 8) at identical results (2026-07 Alfvén-wave sweep;
    128-cell blocks with longer z-rows all tie within noise, larger or
    smaller blocks register-spill or idle threads)."""
    if ndim == 1:
        return (128, 1, 1)
    if ndim == 2:
        return (16, 16, 1)
    return (2, 2, 32)


def _as_3tuple_block_shape(block_shape, ndim: int, spatial_shape=None) -> tuple[int, int, int]:
    """Normalise whatever the user supplied (None / str / tuple) to
    ``(bx, by, bz)`` with the inactive dims forced to 1.  Pallas grid
    construction depends on this tuple being canonical.

    When ``spatial_shape`` is given, each active block dimension is clamped
    to its grid extent, so a default tuned for production grids (e.g. bz=32)
    stays a valid tiling on small grids (bz -> nz) instead of tripping the
    grid-divisibility support predicates and silently dropping the whole run
    to the native backend."""
    if block_shape is None:
        parts = _default_pallas_block_shape(ndim)
    elif isinstance(block_shape, str):
        parts = tuple(int(p.strip()) for p in block_shape.split(",") if p.strip())
    else:
        parts = tuple(int(x) for x in block_shape)
    if len(parts) == 1:
        parts = (parts[0], 1, 1)
    elif len(parts) == 2:
        parts = (parts[0], parts[1], 1)
    elif len(parts) >= 3:
        parts = parts[:3]
    else:
        parts = _default_pallas_block_shape(ndim)
    if ndim == 1:
        parts = (parts[0], 1, 1)
    elif ndim == 2:
        parts = (parts[0], parts[1], 1)
    if spatial_shape is not None:
        parts = tuple(
            min(int(b), int(n)) for b, n in zip(parts, tuple(spatial_shape) + (1, 1))
        )[:3]
        parts = tuple(parts) + (1,) * (3 - len(parts))
    return parts


def _pallas_compiler_params(config: SimulationConfig):
    """Return Triton ``CompilerParams`` (or None if the Triton backend is
    not available / the user opted out via ``pallas_use_triton=False``)."""
    use_triton = config.backend_config.pallas_use_triton
    if use_triton and pltriton is not None:
        return pltriton.CompilerParams(
            num_warps=config.backend_config.pallas_num_warps,
        )
    return None


# -----------------------------------------------------------------------------
# Multi-GPU shard_map + halo wrapper.
#
# Every Pallas kernel in this codebase passes its state-shape input(s) to
# ``pl.pallas_call`` via ``BlockSpec(state.shape, lambda ...: (0, 0, 0, 0))``.
# That tells Pallas/Triton "each block program can read anywhere in the
# array", which is the correct (and fast) shape on a single device — but
# it makes the call entirely opaque to GSPMD.  When the input is sharded
# across a device mesh, XLA's only legal lowering is to ``all-gather`` the
# whole state on every device before each ``pallas_call``, which dominates
# every kernel hot-loop and kills strong scaling (~0.95× on the FD Pallas
# sound-wave benchmark before this fix).
#
# The fix is mechanical: wrap each ``pl.pallas_call`` in a ``shard_map``
# body that
#   1. ppermutes a halo of ``stencil_reach`` cells from each neighbour
#      shard along every sharded spatial axis (periodic ring),
#   2. concatenates [left_halo, local, right_halo] on each sharded axis,
#   3. calls the existing kernel on the local-padded shard (its modular
#      indexing wraps within the padded shape; halo cells provide the
#      correct neighbour values for interior reads),
#   4. strips the halo from the output.
#
# So the user-facing knob is just: ``pallas_mesh_context(mesh)`` around
# the JIT trace, plus each kernel calling ``_pallas_call_sharded`` instead
# of ``pl.pallas_call(...)(args)`` directly.  No kernel arithmetic changes.
# -----------------------------------------------------------------------------


_pallas_mesh_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "astronomix_pallas_mesh", default=None
)
_pallas_spec_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "astronomix_pallas_state_spec", default=None
)


@contextmanager
def pallas_mesh_context(mesh, spec=None):
    """Set the active mesh for Pallas kernel sharding.

    ``time_integration`` enters this context around the JIT trace whenever
    the user supplies a ``sharding`` argument.  Inside the context every
    Pallas kernel that calls ``_pallas_call_sharded`` will route through
    a ``shard_map`` + ppermute halo exchange instead of the bare
    ``pl.pallas_call``.

    ``spec``: the ``PartitionSpec`` of the ``(var, x, y, z)`` state (the
    user's ``sharding.spec``). Inside a JIT trace the kernel inputs are
    tracers, which carry no ``.sharding`` under ``AxisType.Auto`` meshes, so
    without it the wrapper had to guess ``P(*mesh.axis_names)`` -- right only
    for the 4-axis ``(var, x, y, z)`` benchmark mesh; for a 1-axis ``("x",)``
    mesh with ``P(None, "x")`` that guess shards NO spatial axis and every
    kernel silently fell back to the bare ``pallas_call`` (GSPMD all-gather
    of the full state per kernel).

    Callers that differentiate a sharded ``time_integration`` must hold this
    context around the OUTER trace as well: reverse-mode rules (custom_jvp
    primals, the checkpointed loop's backward sweep) are traced after
    ``time_integration`` has returned.

    When ``mesh`` is ``None`` (single-device run) or has size 1, the
    helper is a no-op — the kernel runs exactly as before.
    """
    token = _pallas_mesh_ctx.set(mesh)
    token_spec = _pallas_spec_ctx.set(spec)
    try:
        yield
    finally:
        _pallas_spec_ctx.reset(token_spec)
        _pallas_mesh_ctx.reset(token)


def _current_pallas_mesh():
    return _pallas_mesh_ctx.get()


def _current_pallas_spec():
    return _pallas_spec_ctx.get()


def sharded_roll(x, shift: int, axis: int):
    """``jnp.roll(x, shift, axis)`` along the SPLIT axis of the active Pallas
    mesh context as a shard_map + ppermute of the ``|shift|`` boundary planes,
    or ``None`` when there is nothing to do differently (no multi-device
    context, ``axis`` not the split axis, a shape that does not split evenly,
    ``|shift|`` beyond one shard): the caller then rolls as before.

    Why: GSPMD partitions the slice + concatenate of a periodic roll along a
    split axis as TWO all-to-all reshards of the whole local block, not as a
    halo exchange -- 900+ all-to-alls per 4D-Var gradient step at 256^3 on 4
    GPUs (``casa_4dvar_shard --hlo-audit``); a 20-yr 256^3 forward on 4 A100s
    took 205.8 s with them against 45.6 s on ONE GPU, and 26.2 s with this.
    The values are the same as the concatenate's (a pure data movement), so
    results are bitwise unchanged. (In a reverse sweep one ppermute per
    stencil shift is itself slow -- the native WENO tangent is therefore
    shard-local as a whole, ``_weno._native_tangent_sharded``.)

    The split axis is read from the context spec: for an array with as many
    dims as the ``(var, x, y, z)`` state spec, the spec's own position; with
    one dim fewer (a single field ``(x, y, z)``), shifted by one.
    """
    mesh = _current_pallas_mesh()
    spec = _current_pallas_spec()
    if mesh is None or mesh.size <= 1 or spec is None:
        return None
    mode = os.environ.get("ASTRONOMIX_SHARDED_ROLL", "auto")    # auto | 1 | 0 (GSPMD's concatenate)
    if mode == "0":
        return None
    names = [nm for nm in spec]
    split = [i for i, nm in enumerate(names) if nm is not None]
    if len(split) != 1:
        return None
    sa = split[0]
    name = names[sa]
    if isinstance(name, tuple):
        if len(name) != 1:
            return None
        name = name[0]
    nd = x.ndim
    if nd == len(names):
        pass
    elif nd == len(names) - 1 and sa >= 1:
        sa -= 1
    else:
        return None
    axis = axis % nd
    if axis != sa:
        return None
    ndev = mesh.shape[name]
    n = x.shape[axis]
    if ndev <= 1 or n % ndev:
        return None
    if ndev == 2 and mode != "1":
        # 2026-09-27: with 2 devices, rolls of this form inside the native WENO
        # TANGENT (a custom_jvp rule, transposed) gave a gradient 5 % off at a
        # few cells (4 devices: exact to 5e-7; the forward bitwise right; a
        # standalone roll-in-a-loop VJP exact on 2 GPUs too).
        # Review (shard_review/, state-level 128^3 tests of the 4D-Var solver):
        # deterministic, compile-dependent (XLA collective-permute combining
        # off changes it but does not remove it), in the bulk of the hot
        # ejecta, NOT on the seam planes -- so not a data-movement error, but
        # a different rounding of the GSPMD-partitioned native tangent that the
        # non-smooth tangent amplifies; 2-yr: vel. gradients 1-2 % (rel L2) vs a
        # 2.8-4.8 % change of the 1-GPU gradient under a 1e-7 input jitter.
        # With the tangent shard-local (``_native_tangent_sharded``) forcing
        # ASTRONOMIX_SHARDED_ROLL=1 on 2 devices gives a gradient BITWISE equal
        # to the default below (and 1.5e-7 of the 1-GPU one over 2 yr), so this
        # guard now only matters for native tangents that are not shard-local
        # (the MHD fluxes); 2-device meshes keep GSPMD's roll (slower: ~450
        # whole-block all-to-alls per 128^3 4D-Var gradient) unless
        # ASTRONOMIX_SHARDED_ROLL=1.
        return None
    loc = n // ndev
    s = int(shift) % n
    if s > n // 2:
        s -= n
    if s == 0:
        return x
    if abs(s) > loc:
        return None

    try:
        from jax.shard_map import shard_map  # jax >= 0.8
    except ImportError:  # jax < 0.8
        from jax.experimental.shard_map import shard_map

    pspec = [None] * nd
    pspec[sa] = name
    pspec = PartitionSpec(*pspec)
    right = [(j, (j + 1) % ndev) for j in range(ndev)]
    left = [(j, (j - 1) % ndev) for j in range(ndev)]

    def body(xl):
        if s > 0:           # out[i] = in[i - s]: the first s planes come from the left neighbour
            recv = jax.lax.ppermute(jax.lax.slice_in_dim(xl, loc - s, loc, axis=axis), name, perm=right)
            return jax.lax.concatenate([recv, jax.lax.slice_in_dim(xl, 0, loc - s, axis=axis)], axis)
        t = -s              # out[i] = in[i + t]: the last t planes come from the right neighbour
        recv = jax.lax.ppermute(jax.lax.slice_in_dim(xl, 0, t, axis=axis), name, perm=left)
        return jax.lax.concatenate([jax.lax.slice_in_dim(xl, t, loc, axis=axis), recv], axis)

    return shard_map(body, mesh=mesh, in_specs=(pspec,), out_specs=pspec, check_rep=False)(x)


def _round_halo_up_to_block(halo, block_shape) -> tuple[int, ...]:
    """Round each natural halo width up to a multiple of the corresponding
    Pallas block size.  The kernel's internal ``grid = (nx // bx, ...)``
    must remain block-divisible after halo padding, so we always grow the
    halo to the nearest block multiple."""
    out = []
    for h, b in zip(halo, block_shape, strict=False):
        h_i = int(h)
        b_i = max(int(b), 1)
        if h_i <= 0:
            out.append(0)
        else:
            q, r = divmod(h_i, b_i)
            out.append(b_i * (q + (1 if r else 0)))
    return tuple(out)


def _spatial_sharded_axes(mesh, pspec, ndim):
    """Return a list of ``(array_axis_idx, mesh_axis_name, num_dev)`` for
    every spatial array axis that is split across more than one device.
    The variable axis (index 0) is always skipped."""
    out = []
    for ax in range(1, ndim + 1):
        if ax >= len(pspec):
            break
        name = pspec[ax]
        if name is None:
            continue
        if isinstance(name, tuple):
            for nm in name:
                n = mesh.shape[nm]
                if n > 1:
                    out.append((ax, nm, n))
        else:
            n = mesh.shape[name]
            if n > 1:
                out.append((ax, name, n))
    return out


def _default_state_pspec(mesh, ndim) -> PartitionSpec:
    """Best-effort PartitionSpec when an input array's ``.sharding`` is an
    ``UnspecifiedValue`` (which can happen for intermediates inside a
    JIT trace).  Assumes the standard ``(VARAXIS, XAXIS, YAXIS, ZAXIS)``
    mesh emitted by ``pytests/_benchmark_utils.py::_build_sharding`` and
    by callers that mirror it."""
    axis_names = tuple(mesh.axis_names)
    return PartitionSpec(*axis_names[: 1 + ndim])


def _resolve_state_pspec(state, mesh) -> PartitionSpec:
    """
    Return the PartitionSpec of a (var, x, y, z) state array on ``mesh``.

    Concrete arrays carry their own ``NamedSharding``. Tracers carry no
    ``.sharding`` under ``AxisType.Auto`` meshes, so inside a traced
    ``time_integration(sharding=...)`` the spec registered with
    ``pallas_mesh_context`` is used, and the default ``(var, x, y, z)`` spec
    otherwise.

    Args:
        state: A state-shaped array (leading variable axis).
        mesh: The active Pallas mesh.

    Returns:
        The PartitionSpec of ``state``.
    """
    try:
        sharding = getattr(state, "sharding", None)
    except Exception:
        # Tracers raise "use jax.typeof(x)", not always as an AttributeError.
        sharding = None
    context_spec = _current_pallas_spec()
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    if context_spec is not None and len(context_spec) <= state.ndim:
        return PartitionSpec(*context_spec, *((None,) * (state.ndim - len(context_spec))))
    return _default_state_pspec(mesh, state.ndim - 1)


def _pallas_mesh_splits_axis(state, spatial_axis: int) -> bool:
    """
    Whether the active Pallas mesh distributes ``spatial_axis`` (0 = x) of
    ``state`` over more than one device.

    This is the condition under which ``_pallas_call_sharded`` exchanges halos
    along that axis; code paths that rely on such an exchange (for example a
    kept output halo) must be gated on it rather than on the number of
    visible devices, which says nothing about how the state is split.

    Args:
        state: A state-shaped array (leading variable axis).
        spatial_axis: The spatial axis, 0-based.

    Returns:
        True if the axis is split over several devices.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1:
        return False
    num_spatial_dims = state.ndim - 1
    pspec = _resolve_state_pspec(state, mesh)
    split_array_axes = {
        array_axis for array_axis, _, _ in _spatial_sharded_axes(mesh, pspec, num_spatial_dims)
    }
    return spatial_axis + 1 in split_array_axes


def _per_axis_halo(halo, num_spatial_dims: int) -> tuple:
    """
    Normalise a halo specification to one non-negative width per spatial axis.

    Short tuples are padded with zeros and long ones truncated, so ``(2,)``
    becomes ``(2, 0, 0)`` in 3D and ``(2, 1, 4)`` becomes ``(2, 1)`` in 2D.
    """
    widths = tuple(int(width) for width in halo) + (0,) * max(0, num_spatial_dims - len(halo))
    widths = widths[:num_spatial_dims]
    if any(width < 0 for width in widths):
        raise ValueError("Halo widths must be non-negative.")
    return widths


def _normalize_input_halos(input_halos, num_inputs: int, num_spatial_dims: int):
    """
    One per-axis halo width tuple per state input, or ``None`` if every input
    exchanges the full (block-rounded) halo.
    """
    if input_halos is None:
        return None
    if len(input_halos) != num_inputs:
        raise ValueError("input_halos must give one halo per state input.")
    return tuple(_per_axis_halo(halo, num_spatial_dims) for halo in input_halos)


def _normalize_output_halos(output_halo, num_outputs: int, num_spatial_dims: int):
    """
    One per-axis kept-halo tuple per state output. A single per-axis tuple
    applies to every output; ``None`` keeps no halo.
    """
    if output_halo is None:
        return tuple((0,) * num_spatial_dims for _ in range(num_outputs))
    if all(isinstance(width, int) for width in output_halo):
        output_halos = tuple(output_halo for _ in range(num_outputs))
    else:
        if len(output_halo) != num_outputs:
            raise ValueError("output_halo must give one halo per state output.")
        output_halos = tuple(output_halo)
    return tuple(_per_axis_halo(halo, num_spatial_dims) for halo in output_halos)


def _local_edge_padding(array, width: int, axis: int, *, left: bool):
    """
    ``width`` copies of the local edge plane of ``array`` along ``axis``.

    These fill the part of the block-rounded padding that an input does not
    exchange with its neighbours (see ``input_halos`` of
    ``_pallas_call_sharded``); kernels must never read them into a kept
    output.
    """
    if width <= 0:
        return None
    size = array.shape[axis]
    if left:
        edge = jax.lax.slice_in_dim(array, 0, 1, axis=axis)
    else:
        edge = jax.lax.slice_in_dim(array, size - 1, size, axis=axis)
    repetitions = [1] * array.ndim
    repetitions[axis] = int(width)
    return jnp.tile(edge, repetitions)


def _pad_axis_with_halo(array, axis, padded_width, exchanged_width, mesh_axis_name, num_devices):
    """
    Pad ``array`` by ``padded_width`` cells on both sides of ``axis``: the
    innermost ``exchanged_width`` cells come from the neighbouring shards
    (periodic ring), the rest are local edge copies.

    Args:
        array: The local shard.
        axis: The array axis to pad.
        padded_width: The (block-rounded) padding per side.
        exchanged_width: How many of those cells are real neighbour data.
        mesh_axis_name: The mesh axis the array axis is split over.
        num_devices: The number of devices along that mesh axis.

    Returns:
        The padded local array.
    """
    if exchanged_width > padded_width:
        raise ValueError("An input halo cannot exceed the padded shape halo.")

    pieces = []
    left_padding = _local_edge_padding(array, padded_width - exchanged_width, axis, left=True)
    if left_padding is not None:
        pieces.append(left_padding)

    if exchanged_width > 0:
        size = array.shape[axis]
        left_edge = jax.lax.slice_in_dim(array, 0, exchanged_width, axis=axis)
        right_edge = jax.lax.slice_in_dim(array, size - exchanged_width, size, axis=axis)
        to_left = [(device, (device - 1) % num_devices) for device in range(num_devices)]
        to_right = [(device, (device + 1) % num_devices) for device in range(num_devices)]
        # Each device sends its right edge to the right neighbour, which
        # installs it as its left halo; symmetrically for the right halo.
        left_halo = jax.lax.ppermute(right_edge, mesh_axis_name, perm=to_right)
        right_halo = jax.lax.ppermute(left_edge, mesh_axis_name, perm=to_left)
        pieces.extend((left_halo, array, right_halo))
    else:
        pieces.append(array)

    right_padding = _local_edge_padding(array, padded_width - exchanged_width, axis, left=False)
    if right_padding is not None:
        pieces.append(right_padding)

    return jnp.concatenate(pieces, axis=axis)


def _pallas_call_sharded(
    kernel_build_fn,
    state_inputs,
    other_args=(),
    *,
    halo,
    block_shape,
    num_state_outputs: int = 1,
    input_halos=None,
    output_halo=None,
):
    """Optionally wrap a Pallas-kernel build-and-call in ``shard_map``.

    Args:
        kernel_build_fn:
            Callable ``(state_inputs_local_padded..., other_args...) -> out``
            whose body builds and calls ``pl.pallas_call``.  When the call
            runs inside a ``shard_map`` body, each invocation sees the
            *local* (halo-padded) shape and the kernel's internal
            ``grid``/``BlockSpec`` are built for that shape automatically.
        state_inputs:
            Tuple of state-shape arrays that all share the same sharding
            (same ``PartitionSpec``).  Each one is padded with halo cells
            from neighbour shards along every sharded spatial axis.
        other_args:
            Tuple of replicated arrays (scalar dt, scalar gamma, ...)
            passed through as ``PartitionSpec()``.
        halo:
            Per-spatial-axis natural stencil reach ``(hx, hy, hz)``.
            Pointwise kernels pass ``(0, 0, 0)`` — they still get the
            ``shard_map`` (so the kernel runs locally on each shard),
            just with no ppermute.
        block_shape:
            Per-spatial-axis Pallas block size ``(bx, by, bz)``.  The
            halo is rounded up to the nearest block multiple so the
            padded shard remains block-divisible.
        num_state_outputs:
            Number of state-shape outputs of ``kernel_build_fn`` (1 for
            most kernels; >1 for the CT staged kernels which return
            tuples of single-channel arrays).
        input_halos:
            Optional per-input halo widths ``((hx, hy, hz), ...)``. Every input
            is still padded to the same block-rounded shape, but only the
            requested cells are exchanged with the neighbours; the rest are
            local edge copies. The kernel must not read those copies into any
            output it keeps. ``None`` exchanges the full padding for every
            input.
        output_halo:
            Optional halo width to keep on the state-shaped outputs, one
            per-axis tuple for all outputs or one per output. The default
            strips all padding. A kept halo lets a following local stencil
            reuse cells this kernel already computed instead of exchanging
            them again.

    Returns:
        Either ``kernel_build_fn(*state_inputs, *other_args)`` directly
        (single-device path) or the equivalent ``shard_map``-wrapped
        result with the halo stripped from each state-shape output.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1:
        return kernel_build_fn(*state_inputs, *other_args)

    state0 = state_inputs[0]
    ndim = state0.ndim - 1
    pspec = _resolve_state_pspec(state0, mesh)

    sharded_axes = _spatial_sharded_axes(mesh, pspec, ndim)
    if not sharded_axes:
        return kernel_build_fn(*state_inputs, *other_args)

    block_3 = tuple(block_shape) + (1,) * max(0, 3 - len(block_shape))
    halo_3 = tuple(halo) + (0,) * max(0, 3 - len(halo))
    requested_input_halos = _normalize_input_halos(input_halos, len(state_inputs), ndim)
    kept_output_halos = _normalize_output_halos(output_halo, int(num_state_outputs), ndim)

    # ``shape_halo`` is the block-rounded padding of every local array handed
    # to the kernel; ``exchanged_halos`` are the widths actually received from
    # the neighbours, per input.
    if requested_input_halos is None:
        shape_halo = _round_halo_up_to_block(halo_3[:ndim], block_3[:ndim])
        exchanged_halos = tuple(shape_halo for _ in state_inputs)
    else:
        largest_halo = tuple(
            max([int(halo_3[axis])] + [input_halo[axis] for input_halo in requested_input_halos])
            for axis in range(ndim)
        )
        shape_halo = _round_halo_up_to_block(largest_halo, block_3[:ndim])
        exchanged_halos = requested_input_halos

    for kept_halo in kept_output_halos:
        if any(kept > padded for kept, padded in zip(kept_halo, shape_halo)):
            raise ValueError("output_halo cannot exceed the padded shape halo.")

    try:
        from jax.shard_map import shard_map  # jax >= 0.8 (promoted out of experimental)
    except ImportError:  # jax < 0.8
        from jax.experimental.shard_map import shard_map

    def body(*all_args):
        state_arrays = list(all_args[: len(state_inputs)])
        others = all_args[len(state_inputs):]

        for array_axis_idx, mesh_axis_name, num_dev in sharded_axes:
            spatial_idx = array_axis_idx - 1
            if spatial_idx >= len(shape_halo):
                continue
            h = shape_halo[spatial_idx]
            if h <= 0:
                continue
            for i, arr in enumerate(state_arrays):
                state_arrays[i] = _pad_axis_with_halo(
                    arr,
                    array_axis_idx,
                    h,
                    exchanged_halos[i][spatial_idx],
                    mesh_axis_name,
                    num_dev,
                )

        # Re-enter the wrapper with mesh=None so the recursive
        # ``kernel_build_fn`` call goes through the no-wrap path.  Without
        # this, a kernel that calls ``_pallas_call_sharded`` from its body
        # would wrap itself forever.
        with pallas_mesh_context(None):
            out = kernel_build_fn(*state_arrays, *others)

        def _strip(o, kept_halo):
            for array_axis_idx, _, _ in sharded_axes:
                spatial_idx = array_axis_idx - 1
                if spatial_idx >= len(shape_halo):
                    continue
                stripped_width = shape_halo[spatial_idx] - kept_halo[spatial_idx]
                if stripped_width <= 0:
                    continue
                size = o.shape[array_axis_idx]
                o = jax.lax.slice_in_dim(
                    o, stripped_width, size - stripped_width, axis=array_axis_idx
                )
            return o

        if isinstance(out, tuple):
            return tuple(
                _strip(o, kept_halo) for o, kept_halo in zip(out, kept_output_halos, strict=True)
            )
        return _strip(out, kept_output_halos[0])

    state_specs = tuple(pspec for _ in state_inputs)
    other_specs = tuple(PartitionSpec() for _ in other_args)
    if num_state_outputs > 1:
        out_specs = tuple(pspec for _ in range(num_state_outputs))
    else:
        out_specs = pspec

    wrapped = shard_map(
        body,
        mesh=mesh,
        in_specs=state_specs + other_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    return wrapped(*state_inputs, *other_args)


# -----------------------------------------------------------------------------
# Differentiability: pair every Pallas entry with a native-JAX backward.
# -----------------------------------------------------------------------------
#
# Pallas kernels in this codebase use ``input_output_aliases`` for memory
# efficiency. JAX cannot transpose an aliased ``pl.pallas_call`` (``JVP with
# aliasing not supported``), so any path that hits a Pallas kernel is
# non-differentiable by default. We bridge that gap with a ``jax.custom_jvp``
# whose primal still calls the (aliased, fast) Pallas branch and whose
# tangent rule delegates to the equivalent native-JAX branch — which is
# already JVP-differentiable. Reverse-mode (``jax.grad``) is then derived by
# JAX via transposition.
#
# Forward simulation perf is unaffected: outside of AD the custom_jvp
# rule isn't invoked and the call collapses to the bare Pallas branch.
#
# Both branches must produce the same pytree-structured output. The Pallas
# guide promises bit-identical primal outputs for the existing kernels, so
# the gradient computed by transposing the native JVP at the Pallas-evaluated
# inputs is the correct gradient of the Pallas operation.
#
# Hand-rolled Pallas adjoint kernels can later replace the native tangent
# branch on a per-kernel basis without changing call sites.

def diffable_pallas_call(state, params, *, pallas_branch, native_branch):
    """Run ``pallas_branch(state, params)`` with a custom_jvp boundary that
    routes tangent computation through ``native_branch``.

    Both branches must accept the same positional ``(state, params)`` pair
    and produce the same pytree structure. Anything static (config,
    registered_variables, axis index, ...) should be closed over.

    Outside of AD the call collapses to ``pallas_branch(state, params)``
    directly — no overhead. Under ``jax.jvp`` / ``jax.jacfwd`` /
    ``jax.grad`` / ``jax.vjp`` / ``jax.jacrev`` the custom rule fires and
    the tangent goes through ``native_branch``.
    """
    @jax.custom_jvp
    def _f(s, p):
        return pallas_branch(s, p)

    @_f.defjvp
    def _f_jvp(primals, tangents):
        primal_out = pallas_branch(*primals)
        _, tangent_out = jax.jvp(native_branch, primals, tangents)
        return primal_out, tangent_out

    return _f(state, params)


def diffable_pallas_call_n(primals, *, pallas_branch, native_branch):
    """Same as :func:`diffable_pallas_call` but takes a tuple of arbitrary
    differentiable primals (so callers with more than two diff args, e.g.
    extra rhs/accumulator buffers, can still get a custom_jvp boundary)."""
    @jax.custom_jvp
    def _f(*args):
        return pallas_branch(*args)

    @_f.defjvp
    def _f_jvp(args, tangents):
        primal_out = pallas_branch(*args)
        _, tangent_out = jax.jvp(native_branch, args, tangents)
        return primal_out, tangent_out

    return _f(*primals)


def pallas_vjp_call(state, aux, *, pallas_forward, pallas_backward):
    """Run ``pallas_forward(state, aux)`` with a ``jax.custom_vjp`` boundary
    whose reverse rule is a *native Pallas adjoint kernel* ``pallas_backward``.

    Unlike :func:`diffable_pallas_call` (which routes the tangent — and hence
    the transposed gradient — through native JAX), this keeps the entire
    backward pass on the Pallas/GPU backend: ``pallas_backward(state, aux, cot)``
    returns the input cotangent ``d(loss)/d(state)`` directly from a
    hand-built adjoint kernel.

    Differentiates w.r.t. ``state`` only.  ``aux`` (e.g. the traced
    ``SimulationParams``) is threaded *through* the boundary and given a zero
    cotangent — it must be passed explicitly rather than closed over because
    ``jax.custom_vjp`` cannot capture traced values in its forward/backward
    closures (only static/concrete data — config, axis — may be closed over by
    the two branches).  Treating the physical constants as non-differentiable
    matches the inverse-problem regime (gradients w.r.t. the state, not params).

    NOTE: ``jax.custom_vjp`` supports reverse-mode only — ``jax.jvp`` /
    forward-mode AD on this boundary raises.  Use it for reverse-mode
    (``jax.grad`` / ``differentiation_mode = BACKWARDS``); for forward-mode keep
    :func:`diffable_pallas_call`.
    """
    @jax.custom_vjp
    def _f(s, a):
        return pallas_forward(s, a)

    def _f_fwd(s, a):
        return pallas_forward(s, a), (s, a)

    def _f_bwd(residual, cotangent):
        s, a = residual
        state_bar = pallas_backward(s, a, cotangent)

        def _zero(x):  # correctly-typed zero cotangent (float0 for non-inexact)
            x = jnp.asarray(x)
            if jnp.issubdtype(x.dtype, jnp.inexact):
                return jnp.zeros_like(x)
            return jnp.zeros(x.shape, dtype=jax.dtypes.float0)

        return (state_bar, jax.tree_util.tree_map(_zero, a))

    _f.defvjp(_f_fwd, _f_bwd)
    return _f(state, aux)
