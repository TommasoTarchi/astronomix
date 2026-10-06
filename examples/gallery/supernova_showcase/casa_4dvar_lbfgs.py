"""
Device L-BFGS for the Cas A 4D-Var (``casa_4dvar_robust --optimizer jax``).

Why (2026-10-05): the 448^3 run's optimiser was scipy L-BFGS-B on the host with
float64 n-vectors (n = 1.1e8: ~21 GB of workspace at maxcor 6, a 32-bit
workspace index that segfaulted at maxcor 10, plus host copies of z and g at
every evaluation) on a shared node that went down. Here every n-vector -- the
iterate, the gradient, the search direction and the m-pair history S, Y --
is a device array with the control's sharding (the objective's input), and
only scalars (J, g.d, s.y, norms) ever reach the host.

* ``VecOps``: the jitted vector kernels bound to one sharding (``None``: one
  device). Dot products are blocked float32 sums (blocks of ``block`` within
  each shard, recursively, then across shards), so the accuracy does not
  depend on how a backend orders one long reduction (``test_casa_4dvar_lbfgs``,
  n = 1e8 on XLA:CPU: relative error 3e-8 vs float64; XLA's own jnp.sum is as
  good there).
* ``History``: the fixed-size ring buffer (m, n) of (s, y) pairs (sharded
  along n), filled count + head on the host; ``direction`` is the jitted
  two-loop recursion masked for the filled count, gamma = s.y / y.y of the
  newest pair. A pair is skipped unless s.y > curv_eps * (-g.s) (L-BFGS-B's
  rule, with the float32 machine epsilon).
* ``wolfe_search``: strong-Wolfe line search (Nocedal & Wright Alg. 3.5 / 3.6,
  the bracketing + zoom of ``scipy.optimize.line_search``: cubic, then
  quadratic interpolation with safeguards, then bisection; c1 1e-4, c2 0.9)
  whose control flow runs on host scalars; every trial is one device
  value-and-gradient.
* ``minimize``: unconstrained L-BFGS with scipy L-BFGS-B's interface and stop
  messages (so ``casa_4dvar_robust``'s policies read them the same way):
  first step of length 1 (alpha = 1 / |d|), then alpha = 1; a failed line
  search with a non-empty memory resets the memory and retries along -g; a
  second failure is ``ABNORMAL_TERMINATION_IN_LNSRCH``; the relative-reduction
  test ``(f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol``.

Nothing here knows about the 4D-Var; ``fun(x) -> (f: host float, g: device
array)`` may raise to stop the run (the robust layer's _Stalled / _TinyStep /
_Budget exceptions pass straight through).
"""
import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

#: float32 machine epsilon (the vectors are float32 on the device)
EPS32 = float(np.finfo(np.float32).eps)


# =============================================================================
# ============ ↓ Vector kernels ↓ =============================================
# =============================================================================
def _blocked_sum_last(x, block):
    """Sum over the last axis in blocks of ``block`` (zero padded), recursively."""
    while x.shape[-1] > block:
        pad = (-x.shape[-1]) % block
        if pad:
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
        x = x.reshape(x.shape[:-1] + (x.shape[-1] // block, block)).sum(-1)
    return x.sum(-1)


class VecOps:
    """Jitted n-vector kernels for one ``sharding`` (a 1D NamedSharding, e.g.
    ``P("x")`` over the --gpus mesh, or None = the default device). All
    results that are n-vectors keep that sharding; reductions return device
    scalars (``float()`` them on the host)."""

    def __init__(self, n, dtype=jnp.float32, sharding=None, block=1024):
        self.n, self.dtype, self.block = int(n), jnp.dtype(dtype), int(block)
        self.sharding = sharding
        self.rows, self.s_rows, self.s_hist = 1, None, None
        if isinstance(sharding, NamedSharding) and len(sharding.spec) and sharding.spec[0] is not None:
            ax = sharding.spec[0]
            axes = ax if isinstance(ax, tuple) else (ax,)
            rows = int(np.prod([sharding.mesh.shape[a] for a in axes]))
            if self.n % rows:
                raise ValueError(f"n = {self.n} is not divisible by the {rows} shards of {sharding}")
            self.rows = rows
            self.s_rows = NamedSharding(sharding.mesh, P(ax, None))
            self.s_hist = NamedSharding(sharding.mesh, P(None, ax))
        elif sharding is not None:
            self.s_hist = sharding
        out = dict(out_shardings=sharding) if sharding is not None else {}
        self._dot = jax.jit(self.dot_traced)
        self._axpy = jax.jit(lambda a, x, y: (a * x + y).astype(self.dtype), **out)
        self._sub = jax.jit(lambda x, y: (x - y).astype(self.dtype), **out)
        self._scale = jax.jit(lambda a, x: (a * x).astype(self.dtype), **out)
        self._finite = jax.jit(lambda x: jnp.sum(~jnp.isfinite(x)))
        self._amax = jax.jit(lambda x: jnp.max(jnp.abs(x)))
        self._zeros = jax.jit(lambda: jnp.zeros((self.n,), self.dtype), **out)

    # ---- traced helpers (usable inside other jitted functions) ------------------
    def _rows(self, x):
        x = x.reshape(self.rows, self.n // self.rows)
        if self.s_rows is not None:
            x = jax.lax.with_sharding_constraint(x, self.s_rows)
        return x

    def dot_traced(self, a, b):
        """Blocked a.b (float32, or the vectors' wider dtype): blocks of ``block`` inside each
        shard, recursively, then over shards."""
        acc = jnp.promote_types(jnp.promote_types(a.dtype, b.dtype), jnp.float32)
        p = self._rows(a.astype(acc)) * self._rows(b.astype(acc))
        return _blocked_sum_last(_blocked_sum_last(p, self.block)[None, :], self.block)[0] \
            if self.rows > 1 else _blocked_sum_last(p, self.block)[0]

    # ---- host-facing --------------------------------------------------------------
    def put(self, x):
        """Host (or device) vector -> device, this dtype and sharding."""
        if isinstance(x, jax.Array) and x.dtype == self.dtype and (
                self.sharding is None or x.sharding == self.sharding):
            return x
        if isinstance(x, jax.Array):
            x = x.astype(self.dtype)
            return jax.device_put(x, self.sharding) if self.sharding is not None else x
        x = np.asarray(x).astype(self.dtype, copy=False)
        return jax.device_put(x, self.sharding) if self.sharding is not None else jnp.asarray(x)

    def zeros(self):
        return self._zeros()

    def dot(self, a, b):
        return float(self._dot(a, b))

    def norm(self, a):
        return math.sqrt(max(self.dot(a, a), 0.0))

    def dist(self, a, b):
        return self.norm(self._sub(a, b))

    def axpy(self, a, x, y):
        """a * x + y (a: host scalar)."""
        return self._axpy(jnp.asarray(a, self.dtype), x, y)

    def sub(self, x, y):
        return self._sub(x, y)

    def scale(self, a, x):
        return self._scale(jnp.asarray(a, self.dtype), x)

    def n_nonfinite(self, x):
        return int(self._finite(x))

    def amax(self, x):
        return float(self._amax(x))

    def to_host(self, x, dtype=np.float64):
        return np.asarray(jax.device_get(x)).astype(dtype)
# =============================================================================
# ============ ↑ Vector kernels ↑ =============================================
# =============================================================================


# =============================================================================
# ============ ↓ History + two-loop recursion ↓ ===============================
# =============================================================================
class History:
    """Ring buffer of the last ``m`` (s, y) pairs, (m, n) device arrays."""

    def __init__(self, ops, m, curv_eps=EPS32):
        self.ops, self.m, self.curv_eps = ops, int(m), float(curv_eps)
        out = dict(out_shardings=ops.s_hist) if ops.s_hist is not None else {}
        shape = (self.m, ops.n)
        self.S = jax.jit(lambda: jnp.zeros(shape, ops.dtype), **out)()
        self.Y = jax.jit(lambda: jnp.zeros(shape, ops.dtype), **out)()
        self.rho = np.zeros(self.m, np.float64)
        self.count = 0           # filled pairs
        self.head = 0            # next slot to write (the newest pair is head - 1)
        self.gamma = 1.0
        self.n_skip = 0
        self._push = jax.jit(lambda S, Y, s, y, i: (S.at[i].set(s), Y.at[i].set(y)), donate_argnums=(0, 1),
                             **({"out_shardings": (ops.s_hist, ops.s_hist)} if ops.s_hist is not None else {}))
        self._two = jax.jit(partial(_two_loop, ops.dot_traced),
                            **({"out_shardings": ops.sharding} if ops.sharding is not None else {}))

    def reset(self):
        self.count, self.head, self.gamma = 0, 0, 1.0
        self.rho[:] = 0.0

    def push(self, s, y, gs):
        """Add the pair (s, y) if s.y > curv_eps * (-g.s) (``gs`` = g_old.s); returns s.y, accepted."""
        sy, yy = self.ops.dot(s, y), self.ops.dot(y, y)
        if not (np.isfinite(sy) and np.isfinite(yy) and sy > self.curv_eps * max(-gs, 0.0) and sy > 0.0
                and yy > 0.0):
            self.n_skip += 1
            return sy, False
        self.S, self.Y = self._push(self.S, self.Y, s, y, jnp.int32(self.head))
        self.rho[self.head] = 1.0 / sy
        self.head = (self.head + 1) % self.m
        self.count = min(self.count + 1, self.m)
        self.gamma = sy / yy
        return sy, True

    def direction(self, g):
        """d = -H g (H: the L-BFGS inverse Hessian of the filled pairs, H0 = gamma I)."""
        if self.count == 0:
            return self.ops.scale(-1.0, g)
        dt = self.ops.dtype
        return self._two(g, self.S, self.Y, jnp.asarray(self.rho, dt), jnp.int32(self.count),
                         jnp.int32(self.head), jnp.asarray(self.gamma, dt))

    def newest(self):
        """Ring-buffer slots from the newest to the oldest filled pair (tests)."""
        return [(self.head - 1 - j) % self.m for j in range(self.count)]


def _two_loop(dot, g, S, Y, rho, count, head, gamma):
    m = S.shape[0]
    g = g.astype(S.dtype)

    def first(j, c):                  # newest -> oldest
        q, al = c
        i = (head - 1 - j) % m
        a = jnp.where(j < count, rho[i] * dot(jax.lax.dynamic_index_in_dim(S, i, 0, False), q), 0.0)
        q = (q - a * jax.lax.dynamic_index_in_dim(Y, i, 0, False)).astype(S.dtype)
        return q, al.at[j].set(a)

    q, al = jax.lax.fori_loop(0, m, first, (g, jnp.zeros((m,), S.dtype)))
    r = (gamma * q).astype(S.dtype)

    def second(k, r):                 # oldest -> newest
        j = m - 1 - k
        i = (head - 1 - j) % m
        b = rho[i] * dot(jax.lax.dynamic_index_in_dim(Y, i, 0, False), r)
        c = jnp.where(j < count, al[j] - b, 0.0)
        return (r + c * jax.lax.dynamic_index_in_dim(S, i, 0, False)).astype(S.dtype)

    r = jax.lax.fori_loop(0, m, second, r)
    return -r
# =============================================================================
# ============ ↑ History + two-loop recursion ↑ ===============================
# =============================================================================


# =============================================================================
# ============ ↓ Strong-Wolfe line search (host scalars) ↓ ====================
# =============================================================================
def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """Minimiser of the cubic through (a, fa, fpa), (b, fb), (c, fc) (scipy's _cubicmin), or None."""
    try:
        with np.errstate(all="ignore"):   # a negative radical / zero denominator -> nan / inf -> None
            C = fpa
            db, dc = b - a, c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.array([[dc ** 2, -db ** 2], [-dc ** 3, db ** 3]])
            A, B = d1 @ np.array([fb - fa - C * db, fc - fa - C * dc]) / denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
    except (ArithmeticError, FloatingPointError, ValueError):
        return None
    return float(xmin) if np.isfinite(xmin) else None


def _quadmin(a, fa, fpa, b, fb):
    """Minimiser of the quadratic through (a, fa, fpa), (b, fb) (scipy's _quadmin), or None."""
    try:
        db = b - a
        B = (fb - fa - fpa * db) / (db * db)
        xmin = a - fpa / (2.0 * B)
    except (ArithmeticError, FloatingPointError, ValueError):
        return None
    return float(xmin) if np.isfinite(xmin) else None


@dataclass
class LineSearch:
    alpha: float = None          # accepted step (None: failed)
    f: float = None
    df: float = None
    payload: object = None       # what phi returned for the accepted step (device x, g)
    n_eval: int = 0
    wolfe: bool = False          # strong Wolfe satisfied (else: an Armijo-only fallback or a failure)
    note: str = ""


def wolfe_search(phi, f0, df0, alpha1, *, c1=1e-4, c2=0.9, max_evals=20, alpha_max=1e10, extrap=4.0, xtol=0.1,
                 feps=EPS32):
    """Strong-Wolfe step along a descent direction. ``phi(alpha) -> (f, f', payload)``
    (host scalars + whatever the caller needs back for the accepted step; a
    non-finite f counts as +inf). Returns a ``LineSearch``; when the budget
    runs out the best Armijo point found (if any) is returned with
    ``wolfe=False``. Bracketing extrapolates a_new = a + extrap (a - a_prev)
    (4: Moré-Thuente's 1 -> 5 -> 21; scipy's line_search doubles), at most
    ``alpha_max``. The zoom stops once the bracket is narrower than ``xtol``
    times its end point (L-BFGS-B's dcsrch xtol 0.1): at the float32 noise
    floor of J the Armijo test is decided by rounding, and bisecting it to
    ``max_evals`` costs ~20 evaluations for nothing (CPU test: 40 of 76).
    For the same reason a zoom trial whose predicted decrease a |f'(0)| is
    below 4 ``feps`` max(|f0|, 1) (J's rounding; float32) is not evaluated
    ("rounding errors prevent progress", as dcsrch)."""
    if not (df0 < 0):
        return LineSearch(note=f"not a descent direction (g.d = {df0:.3g})")
    n = 0
    best = None                   # the lowest Armijo-satisfying trial: (alpha, f, df, payload)

    def ev(a):
        nonlocal n, best
        f, df, pl = phi(a)
        n += 1
        f = float(f) if np.isfinite(f) else np.inf
        df = float(df) if np.isfinite(df) else np.nan
        # (review 2026-10-05) only a trial with a finite gradient may become the fallback:
        # an Armijo point with a non-finite g.d was returned as the new iterate (NaN gradient)
        if f <= f0 + c1 * a * df0 and np.isfinite(df) and (best is None or f < best[1]):
            best = (a, f, df, pl)
        return f, df, pl

    def done(a, f, df, pl, note="strong Wolfe"):
        return LineSearch(alpha=a, f=f, df=df, payload=pl, n_eval=n, wolfe=True, note=note)

    def fallback(note):
        if best is not None:
            a, f, df, pl = best
            return LineSearch(alpha=a, f=f, df=df, payload=pl, n_eval=n, wolfe=False, note=note + "; Armijo point")
        return LineSearch(n_eval=n, note=note)

    def zoom(a_lo, a_hi, f_lo, f_hi, df_lo, pl_lo):
        a_rec, f_rec = 0.0, f0
        i = 0
        while n < max_evals:
            dal = a_hi - a_lo
            lo, hi = (a_lo, a_hi) if dal > 0 else (a_hi, a_lo)
            if abs(dal) <= xtol * max(abs(a_lo), abs(a_hi)):
                return fallback("xtol")
            aj = None
            if i > 0 and np.isfinite(f_hi) and np.isfinite(f_rec):
                cchk = 0.2 * abs(dal)
                aj = _cubicmin(a_lo, f_lo, df_lo, a_hi, f_hi, a_rec, f_rec)
                if aj is not None and (aj > hi - cchk or aj < lo + cchk):
                    aj = None
            if aj is None:
                qchk = 0.1 * abs(dal)
                aj = _quadmin(a_lo, f_lo, df_lo, a_hi, f_hi) if np.isfinite(f_hi) else None
                if aj is None or aj > hi - qchk or aj < lo + qchk:
                    aj = a_lo + 0.5 * dal
            if abs(aj - a_lo) <= 1e-12 * max(abs(a_lo), 1e-30):
                break
            if aj * abs(df0) < 4.0 * feps * max(abs(f0), 1.0):
                return fallback("rounding errors prevent progress")
            fj, dfj, plj = ev(aj)
            if fj > f0 + c1 * aj * df0 or fj >= f_lo:
                a_rec, f_rec = a_hi, f_hi
                a_hi, f_hi = aj, fj
            else:
                if not np.isfinite(dfj):
                    a_rec, f_rec = a_hi, f_hi
                    a_hi, f_hi = aj, fj
                    i += 1
                    continue
                if abs(dfj) <= -c2 * df0:
                    return done(aj, fj, dfj, plj)
                if dfj * (a_hi - a_lo) >= 0:
                    a_rec, f_rec = a_hi, f_hi
                    a_hi, f_hi = a_lo, f_lo
                else:
                    a_rec, f_rec = a_lo, f_lo
                a_lo, f_lo, df_lo, pl_lo = aj, fj, dfj, plj
            i += 1
        return fallback("zoom budget")

    a0, fa0, dfa0, pl0 = 0.0, f0, df0, None
    a1 = float(min(alpha1, alpha_max))
    i = 0
    while n < max_evals:
        f1, df1, pl1 = ev(a1)
        if f1 > f0 + c1 * a1 * df0 or (i > 0 and f1 >= fa0):
            return zoom(a0, a1, fa0, f1, dfa0, pl0)
        if not np.isfinite(df1):
            return zoom(a0, a1, fa0, np.inf, dfa0, pl0)
        if abs(df1) <= -c2 * df0:
            return done(a1, f1, df1, pl1)
        if df1 >= 0:
            return zoom(a1, a0, f1, fa0, df1, pl1)
        a2 = min(a1 + extrap * (a1 - a0), alpha_max)
        if a2 <= a1:
            return fallback("alpha_max")
        a0, fa0, dfa0, pl0 = a1, f1, df1, pl1
        a1 = a2
        i += 1
    return fallback("bracket budget")
# =============================================================================
# ============ ↑ Strong-Wolfe line search (host scalars) ↑ ====================
# =============================================================================


# =============================================================================
# ============ ↓ L-BFGS driver ↓ ==============================================
# =============================================================================
@dataclass
class Result:
    x: object
    fun: float
    jac: object
    nit: int
    nfev: int
    message: str
    success: bool
    history: object = None


def minimize(fun, x0, *, ops, maxiter=100, maxfun=None, maxcor=10, ftol=2.220446049250313e-09, gtol=1e-5,
             c1=1e-4, c2=0.9, max_ls=20, extrap=4.0, xtol=0.1, feps=EPS32, callback=None, verbose=False):
    """Unconstrained L-BFGS on the device. ``fun(x) -> (f: host float, g: device
    array like x)``; ``x0``: host or device vector (put on ``ops``' sharding);
    ``callback(x_k)`` after every iteration (device array). Stop messages as
    scipy's L-BFGS-B."""
    maxfun = maxfun if maxfun is not None else 15000
    x = ops.put(x0)
    f, g = fun(x)
    nfev, nit = 1, 0
    hist = History(ops, maxcor)
    msg = None
    retry_sd = False
    while True:
        if not np.isfinite(f):
            msg = "ABNORMAL_TERMINATION_IN_LNSRCH"      # non-finite at the iterate
            break
        if ops.amax(g) <= gtol:
            msg = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
            break
        if nit >= maxiter:
            msg = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
            break
        if nfev >= maxfun:
            msg = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
            break
        d = hist.direction(g)
        gd = ops.dot(g, d)
        if not (np.isfinite(gd) and gd < 0) and hist.count:
            hist.reset()
            d = hist.direction(g)
            gd = ops.dot(g, d)
        if not (np.isfinite(gd) and gd < 0):
            msg = "ABNORMAL_TERMINATION_IN_LNSRCH"
            break
        dn = math.sqrt(max(-gd if hist.count == 0 else ops.dot(d, d), 0.0))
        alpha1 = 1.0 / dn if hist.count == 0 else 1.0

        def phi(a, x=x, d=d):
            nonlocal nfev
            if nfev >= maxfun:
                raise _OutOfEvals
            xt = ops.axpy(a, d, x)
            ft, gt = fun(xt)
            nfev += 1
            return ft, (ops.dot(gt, d) if np.isfinite(ft) else np.nan), (xt, gt)

        try:
            ls = wolfe_search(phi, f, gd, alpha1, c1=c1, c2=c2, max_evals=max_ls, extrap=extrap, xtol=xtol,
                              feps=feps)
        except _OutOfEvals:
            msg = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
            break
        if verbose:
            print(f"[lbfgs] it {nit}: f {f:.6g} g.d {gd:.3g} alpha {ls.alpha} ({ls.n_eval} evals, {ls.note}); "
                  f"memory {hist.count}", flush=True)
        if ls.alpha is None:
            if hist.count and not retry_sd:          # refresh the memory, retry along -g
                hist.reset()
                retry_sd = True
                continue
            msg = "ABNORMAL_TERMINATION_IN_LNSRCH"
            break
        retry_sd = False
        x_new, g_new = ls.payload
        s = ops.sub(x_new, x)
        y = ops.sub(g_new, g)
        hist.push(s, y, ls.alpha * gd)
        del s, y
        f_old = f
        x, f, g = x_new, ls.f, g_new
        nit += 1
        if callback is not None:
            callback(x)
        if (f_old - f) / max(abs(f_old), abs(f), 1.0) <= ftol:
            msg = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"
            break
    return Result(x=x, fun=f, jac=g, nit=nit, nfev=nfev, message=msg,
                  success=msg.startswith("CONVERGENCE"), history=hist)


class _OutOfEvals(Exception):
    pass
# =============================================================================
# ============ ↑ L-BFGS driver ↑ ==============================================
# =============================================================================
