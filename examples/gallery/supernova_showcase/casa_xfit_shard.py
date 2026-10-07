"""
Single-node multi-GPU sharding for ``casa_xfit`` and ``casa_4dvar``.

The domain is split into x-slabs: the ``(var, x, y, z)`` solver state is
``P(None, "x")`` on a 1-axis ``("x",)`` mesh with ``AxisType.Auto`` (the jax 0.10
default is Explicit, which the solver's ``with_sharding_constraint`` rejects),
3D fields are ``P("x")`` and sky-plane ``(..., x, z)`` arrays -- the line-of-sight
(y) column sums of the observation model -- are ``P(..., "x", None)``, so the whole
chain state -> plasma -> columns is shard-local and only 2D columns, 1D profiles
and scalars cross devices. GSPMD partitions everything that is not a Pallas
kernel; the FD Pallas kernels go through ``_pallas_call_sharded`` (shard_map +
ppermute halos), which needs the mesh AND the state spec in
``pallas_mesh_context`` for the whole OUTER trace (reverse-mode rules are
traced after ``time_integration`` has returned): every top-level jitted call
of the sharded path therefore runs under ``context()``.

Three things differ from the 1-device path (all only when a mesh is active;
with ``--gpus 1`` nothing here is used and the numbers are the old ones bit for
bit):

* **Lifted constants** (``Lifted``): arrays a jitted function closes over are
  embedded in the HLO as dense literals, replicated on every device (the 8 GB
  background state at 512^3; measured: jax 0.10 embeds closed-over jax.Arrays
  too). ``Lifted.jit`` passes them as sharded jit ARGUMENTS instead, rebinding
  the holders' attributes to the tracers during the trace.
* **Cone profiles** (``ShardedCones``): ``casa_pluto_diff.ConeProfiles.mean``
  gathers a flat cell-index list out of the raveled field -- under GSPMD an
  all-gather of the whole field. Here the same membership is two per-cell
  segment-id maps sharded like the state (a cell lies in at most two cones),
  i.e. a shard-local scatter-add into the (cone, bin) sums + one all-reduce.
* **B^1/2 prolongation + Gaussian smoothing** (``smooth_prolong``): one
  shard_map -- local 2x repeat, a 3D FFT over the sharded x axis as separable
  1D FFTs with one all-to-all (x-slabs <-> y-slabs) each way, and the
  separable transfer function.
"""
from contextlib import ExitStack, contextmanager
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

MESH = None
NDEV = 1
STATE = P(None, "x", None, None)
FIELD = P("x", None, None)
REPL = P()


def gpus_from_argv(flag="--gpus", default=1):
    """The ``--gpus N`` of the command line, read BEFORE jax initialises (autocvd)."""
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


def select_gpus(n):
    """autocvd(num_gpus=n) unless the queue (pq) set CUDA_VISIBLE_DEVICES."""
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        from autocvd import autocvd
        autocvd(num_gpus=n)


def activate(n):
    """Build the ``("x",)`` mesh over the first ``n`` devices (n > 1) and hook
    the observation model (``casa_jaxobs.SHARD``). Returns the mesh or None."""
    global MESH, NDEV
    if n <= 1:
        return None
    from jax.sharding import AxisType
    devs = jax.devices()
    if len(devs) < n:
        raise RuntimeError(f"--gpus {n}: only {len(devs)} devices visible ({devs})")
    jax.config.update("jax_use_shardy_partitioner", False)
    MESH = jax.make_mesh((n,), ("x",), axis_types=(AxisType.Auto,), devices=devs[:n])
    NDEV = n
    import casa_jaxobs
    casa_jaxobs.SHARD = sys.modules[__name__]
    print(f"[shard] mesh {dict(MESH.shape)} over {[str(d) for d in devs[:n]]}", flush=True)
    return MESH


def active():
    return MESH is not None and NDEV > 1


def sharding(spec):
    return NamedSharding(MESH, spec) if active() else None


def state_sharding():
    """The solver state's NamedSharding (``time_integration(sharding=...)``), or None."""
    return sharding(STATE)


def cols_spec(ndim):
    """(..., x, z) sky-plane arrays: split along x."""
    return P(*((None,) * (ndim - 2)), "x", None)


def constrain(a, spec):
    if not active():
        return a
    return jax.lax.with_sharding_constraint(a, NamedSharding(MESH, spec))


def cstate(a):
    return constrain(a, STATE) if active() and a.ndim == 4 else a


def cfield(a):
    return constrain(a, FIELD) if active() and a.ndim == 3 else a


def ccols(a):
    return constrain(a, cols_spec(a.ndim)) if active() and a.ndim >= 2 else a


def put(a, spec=None, dtype=None):
    """Host array -> device: sharded with ``spec`` (default: by rank -- 4D state,
    3D field, else replicated) when a mesh is active, else ``jnp.asarray``
    (the 1-device behaviour, unchanged)."""
    if not active():
        return jnp.asarray(a, dtype) if dtype is not None else jnp.asarray(a)
    a = np.asarray(a)
    if dtype is not None:
        a = a.astype(jnp.dtype(dtype))
    if spec is None:
        spec = {4: STATE, 3: FIELD}.get(a.ndim, REPL)
    return jax.device_put(a, NamedSharding(MESH, spec))


@contextmanager
def context():
    """The mesh + the astronomix Pallas mesh/state-spec context (no-op on one device)."""
    if not active():
        yield
        return
    from astronomix._pallas_helpers import pallas_mesh_context
    with ExitStack() as st:
        st.enter_context(MESH)
        st.enter_context(pallas_mesh_context(MESH, STATE))
        yield


def per_device_memory():
    """[(device, peak GB, in-use GB)] of the mesh devices (or device 0)."""
    devs = list(MESH.devices.ravel()) if active() else jax.devices()[:1]
    out = []
    for d in devs:
        try:
            ms = d.memory_stats() or {}
            out.append((str(d), ms.get("peak_bytes_in_use", 0) / 2 ** 30, ms.get("bytes_in_use", 0) / 2 ** 30))
        except Exception:
            out.append((str(d), float("nan"), float("nan")))
    return out


def device_limits_gb():
    """The allocator limit (GB) of each mesh device (or device 0)."""
    devs = list(MESH.devices.ravel()) if active() else jax.devices()[:1]
    out = []
    for d in devs:
        try:
            out.append((d.memory_stats() or {}).get("bytes_limit", 0) / 2 ** 30)
        except Exception:
            out.append(float("nan"))
    return out


# =============================================================================
# ============ ↓ Lifted constants ↓ ===========================================
# =============================================================================
class Lifted:
    """Big arrays read through ``holder.attr`` (or ``holder[key]`` for dicts)
    inside a traced function, passed as jit ARGUMENTS (sharded) instead of
    being captured as HLO constants. ``add`` places the array on the mesh;
    ``jit(fn)`` returns ``call(*args)`` that runs ``jit(transform(fn'))(*args,
    lifted)`` under ``context()``, where ``fn'`` rebinds the holders to the
    traced values for the duration of the trace. Without an active mesh
    ``jit`` is the plain ``jax.jit(transform(fn))`` (constants captured, as
    before)."""

    def __init__(self):
        self.items = []

    def add(self, holder, key, spec=None, dtype=None):
        if active():
            v = self._get(holder, key)
            if not (isinstance(v, jax.Array) and isinstance(getattr(v, "sharding", None), NamedSharding)
                    and v.sharding.mesh == MESH):
                v = put(np.asarray(v), spec, dtype)
            self._set(holder, key, v)
        self.items.append((holder, key))
        return self

    @staticmethod
    def _get(h, k):
        return h[k] if isinstance(h, dict) else getattr(h, k)

    @staticmethod
    def _set(h, k, v):
        if isinstance(h, dict):
            h[k] = v
        else:
            setattr(h, k, v)

    def values(self):
        return [self._get(h, k) for h, k in self.items]

    @contextmanager
    def bound(self, vals):
        old = self.values()
        for (h, k), v in zip(self.items, vals):
            self._set(h, k, v)
        try:
            yield
        finally:
            for (h, k), v in zip(self.items, old):
                self._set(h, k, v)

    def nbytes(self):
        return sum(int(np.prod(v.shape)) * jnp.dtype(v.dtype).itemsize for v in self.values())

    def jit(self, fn, transform=None, **jit_kw):
        transform = transform or (lambda f: f)
        if not active():
            return jax.jit(transform(fn), **jit_kw)

        def inner(*args):
            *a, lifted = args
            with self.bound(lifted):
                return fn(*a)
        jf = jax.jit(transform(inner), **jit_kw)

        def call(*args):
            with context():
                return jf(*args, self.values())
        call.lower = lambda *args: _lower(jf, args, self)
        return call


def _lower(jf, args, lifted):
    with context():
        return jf.lower(*args, lifted.values())
# =============================================================================
# ============ ↑ Lifted constants ↑ ===========================================
# =============================================================================


# =============================================================================
# ============ ↓ Shard-local reductions and transforms ↓ ======================
# =============================================================================
class ShardedCones:
    """``casa_pluto_diff.ConeProfiles`` with the membership as two per-cell
    segment-id maps (``seg0``, ``seg1``; ``nseg`` = none), sharded like the
    state: ``mean`` is a shard-local scatter-add + an all-reduce of the
    (cone x bin) sums instead of a gather out of the whole raveled field.
    Same membership, same counts; the sums differ from the 1-device gather
    only in their order (rounding)."""

    def __init__(self, cones, shape):
        cells = np.asarray(cones.cells, np.int64)
        segs = np.asarray(cones.segs, np.int64)
        ncell = int(np.prod(shape))
        order = np.argsort(cells, kind="stable")
        c, s = cells[order], segs[order]
        first = np.ones(c.size, bool)
        first[1:] = c[1:] != c[:-1]
        _, mult = np.unique(c, return_counts=True)
        if mult.max(initial=0) > 2:
            raise ValueError(f"a cell in {mult.max()} cones: ShardedCones assumes <= 2")
        self.nseg = int(cones.nseg)
        seg0 = np.full(ncell, self.nseg, np.int32)
        seg1 = np.full(ncell, self.nseg, np.int32)
        seg0[c[first]] = s[first]
        seg1[c[~first]] = s[~first]
        self.seg0 = put(seg0.reshape(shape), FIELD)
        self.seg1 = put(seg1.reshape(shape), FIELD)
        for k in ("angles", "rc", "nbins", "count", "valid"):
            setattr(self, k, getattr(cones, k))

    def mean(self, field):
        f = cfield(field)
        ns = self.nseg + 1
        tot = jax.ops.segment_sum(f.ravel(), self.seg0.ravel(), ns) \
            + jax.ops.segment_sum(f.ravel(), self.seg1.ravel(), ns)
        return (tot[:self.nseg] / self.count).reshape(len(self.angles), self.nbins)

    def lift(self, lifted):
        lifted.add(self, "seg0", FIELD).add(self, "seg1", FIELD)
        return self


def smooth_prolong(chi, sigma_cells):
    """``gaussian_filter_fft(prolong2(chi), sigma_cells)`` of ``casa_4dvar_control``
    for ``chi`` (..., m, m, m) split along x, as ONE shard_map: the 2x
    prolongation shard-locally, then the periodic Gaussian as separable 1D FFTs
    -- rfft(z), fft(y) locally, an all-to-all to y-slabs, fft(x), the
    separable transfer function, and back. (GSPMD alone turned both the
    x-split repeat and the reshard into all-gathers of the full field.)"""
    from jax.sharding import PartitionSpec as P_
    m = chi.shape[-3:]
    n = tuple(2 * k for k in m)
    if any(k % NDEV for k in (m[0], n[1])):
        raise ValueError(f"smooth_prolong: {m} not divisible over {NDEV} devices")
    lead = (None,) * (chi.ndim - 3)
    kx, ky = (2 * np.pi * np.fft.fftfreq(k) for k in n[:2])
    kz = 2 * np.pi * np.fft.rfftfreq(n[2])
    gx, gy, gz = (np.exp(-0.5 * sigma_cells ** 2 * k ** 2) for k in (kx, ky, kz))
    ny_loc = n[1] // NDEV
    ax = chi.ndim - 3                       # the x axis (y = ax + 1, z = ax + 2)

    def body(c):
        dt = c.dtype
        for a_ in (ax, ax + 1, ax + 2):
            c = jnp.repeat(c, 2, axis=a_)
        A = jnp.fft.rfft(c, axis=ax + 2)
        A = jnp.fft.fft(A, axis=ax + 1)
        A = jax.lax.all_to_all(A, "x", split_axis=ax + 1, concat_axis=ax, tiled=True)   # y-slabs
        A = jnp.fft.fft(A, axis=ax)
        j = jax.lax.axis_index("x")
        gyl = jax.lax.dynamic_slice_in_dim(jnp.asarray(gy, dt), j * ny_loc, ny_loc)
        G = jnp.asarray(gx, dt)[:, None, None] * gyl[None, :, None] * jnp.asarray(gz, dt)[None, None, :]
        A = jnp.fft.ifft(A * G, axis=ax)
        A = jax.lax.all_to_all(A, "x", split_axis=ax, concat_axis=ax + 1, tiled=True)   # x-slabs
        A = jnp.fft.ifft(A, axis=ax + 1)
        return jnp.fft.irfft(A, n=n[2], axis=ax + 2).astype(dt)

    spec = P_(*lead, "x", None, None)
    return jax.shard_map(body, mesh=MESH, in_specs=(spec,), out_specs=spec)(constrain(chi, spec))
# =============================================================================
# ============ ↑ Shard-local reductions and transforms ↑ ======================
# =============================================================================


# =============================================================================
# ============ ↓ HLO audit ↓ ==================================================
# =============================================================================
def collective_audit(compiled_text, min_elems=2 ** 20):
    """Collectives in an optimized HLO module whose largest RESULT array has >=
    ``min_elems`` elements: [(op, shape, count)] -- the all-gathers of full 3D
    fields are what makes a sharded run no smaller than a replicated one.

    The async forms (``all-gather-start``, ``all-to-all-start``,
    ``collective-permute-start``; combined ops) return TUPLES, e.g.
    ``(f32[5,16,64,64], f32[5,64,64,64])`` or ``((f32[2,64,64,128]), ...)``: every
    array shape in the result type counts, the largest decides. (Until
    2026-09-27 only a bare ``= f32[...] op(`` result matched, i.e. only the
    all-reduces: the 2-GPU audit reported "6 all-reduces" for a module with
    954 all-to-alls of whole local blocks.)"""
    import re
    rx_line = re.compile(r"=\s*(.*?)\s(all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter)"
                         r"(?:-start)?\(")
    rx_shape = re.compile(r"\b([a-z]\w*)\[([\d,]*)\]")
    hits = {}
    for line in compiled_text.splitlines():
        m = rx_line.search(line)
        if m is None:
            continue
        best, best_ne = None, -1
        for dt, dims in rx_shape.findall(m.group(1)):
            ne = int(np.prod([int(x) for x in dims.split(",") if x])) if dims else 1
            if ne > best_ne:
                best, best_ne = f"{dt}[{dims}]", ne
        if best is not None and best_ne >= min_elems:
            key = (m.group(2), best)
            hits[key] = hits.get(key, 0) + 1
    return sorted(((k[0], k[1], c) for k, c in hits.items()), key=lambda t: -t[2])
