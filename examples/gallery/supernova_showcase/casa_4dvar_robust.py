"""
Robust L-BFGS for the Cas A 4D-Var (``casa_4dvar``) against discrete-path
roughness of J.

Why: J(z) is bitwise reproducible, yet float32-ulp changes of the control flip
discrete solver paths near the reverse shock (shock latch, cold-LLF /
cold-crush gates, CFL step count): J jumps by ~0.03-0.1 and the gradient at a
"lucky" low point is a spike (review_v noise probe: |g| 2.45 at z1 vs 0.20 at
z1 + 1 ulp). Plain L-BFGS-B keeps the luckiest evaluation as its best point
(winner's curse) and its line search ends up probing ulp-sized steps.

Policies (``Policy``; ``plain`` = the stage-2 behaviour, bit for bit the same
logic):

* ``confirm``: a new best is accepted only if it survives re-evaluation at
  ``confirm_n`` points z +- confirm_sigma * e_c (fixed antithetic
  directions, value only); the tracked, reported and checkpointed best is the
  CONFIRMED J = the median of the raw J and those values (robust to ONE lucky
  path among the three: with the mean of the two jittered values alone, a
  lucky pair made a confirmed best 0.05 below the point's fresh-jitter value
  and stalled the stage, policy test from z_stage0, 2026-09-26). The optimiser itself still sees the
  raw J and gradient; every (re)start begins at z_best + confirm_sigma * e_c
  (a normal-branch point, never the lucky one).
* ``spike_factor``: a gradient whose norm exceeds this factor times the median
  of the last finite evaluations' is a special-path spike (review_v: 12x
  larger, 70 % of |g|^2 in 100 cells at the reverse shock); the evaluation is
  replaced by one at a jittered neighbour z + spike_sigma * e (one extra
  gradient, only then; the smaller-|g| of the two is used). At the first
  evaluation of a run (no history yet) the neighbour is always evaluated and
  compared. WC short run 2026-09-26: one |g| = 289 point among |g| ~ 8-13
  sent the next L-BFGS step to J + 20 and cost 6 evaluations; the resumed
  run then started at z_best + 1e-6 e with |g| = 422 (the spike region is
  wider than 1e-6, hence spike_sigma 1e-4 per component).
* ``stall_restart``: a stall restarts L-BFGS from the confirmed best (a new
  start jitter, fresh memory) instead of ending the stage, until
  ``max_restarts`` is used up. So does a FACTR "CONVERGENCE" with iterations
  left (one non-decreasing step on the rough J, not a converged problem:
  s3-4dv-Rp stage 2 stopped at 10 of 25 iterations this way).
* ``min_step``: a line-search trial closer than ``min_step`` (whitened norm)
  to the current iterate ends the L-BFGS run (restart from the confirmed best
  with fresh memory, at most ``max_restarts`` times).
* ``nan_retries``: a non-finite J or gradient (review 2026-09-26: 29 of ~51
  evaluations on the R' trajectory had a NaN state gradient with a finite J,
  from the 2017.4 -> 2018.4 backward pass) is replaced by one at a seeded
  neighbour z + s_i * e_i, s_i = ``nan_sigmas`` (1e-5, 1e-5, 1e-4, 1e-4, ...),
  up to ``nan_retries`` times (the validation worker's ``run_patched.py``,
  now in the policy). Independently of it, a (re)start point that stays
  non-finite ends the stage at the best point so far instead of raising
  (the stage's very first evaluation still raises: nothing to keep).
* ``smooth``: L-BFGS on J_s(z) = J_b(z) + mean_k J_o(z + sigma_s e_k) with a
  FIXED set of antithetic directions e_k (common random numbers, per-component
  sd 1 on the rough, i.e. state, part of z) and gradient = the mean of the K
  gradients (K x the cost). The jitter enters only the model (observation /
  continuity / wind terms), not the quadratic background, so J_s has no
  sigma_s^2 |e|^2 offset. Best tracking uses J_s itself.

``Evaluator`` wraps the window's jitted ``vg(z, dz)`` / ``val(z, dz)`` (dz: the
model jitter; the background sees z) and caches the jitter directions on the
device. ``robust_minimize`` is the stage loop: L-BFGS, evals.jsonl records,
checkpoint callback, stall detector, restarts, a gradient-evaluation budget.

Optimisers (``--optimizer``; 2026-10-05, after compgpu12 went down under the
448^3 run with ~21 GB of float64 L-BFGS-B workspace + per-evaluation host
copies of z and g in this process):

* ``jax`` (default): ``casa_4dvar_lbfgs`` -- unconstrained L-BFGS with z, g,
  the search direction and the (s, y) history as device arrays sharded like
  the objective's input, a strong-Wolfe line search on host scalars, and an
  ``Evaluator`` in device mode (``ops=``): directions drawn by jax.random on
  the device, only J / chi2 / norms reach the host. No n-vector touches the
  host except the checkpoint (z_best, once per iteration). maxcor has no
  32-bit limit.
* ``scipy``: scipy L-BFGS-B on float64 host vectors, numpy directions (the
  stage-2..4 behaviour; maxcor capped at the 32-bit workspace limit).

The policies are the same in both; the trajectories differ (Moré-Thuente vs
the N&W zoom line search, other jitter draws).
"""
import argparse
import json
import os
import resource
import time
import zlib
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class Policy:
    confirm: bool = False
    confirm_n: int = 2
    confirm_sigma: float = 1e-6
    min_step: float = 0.0
    spike_factor: float = 0.0
    spike_sigma: float = 1e-4
    stall_restart: bool = False
    smooth_k: int = 0
    smooth_sigma: float = 3e-3
    seed: int = 0
    nan_retries: int = 0
    nan_sigmas: tuple = (1e-5, 1e-5, 1e-4, 1e-4)

    @property
    def smooth(self):
        return self.smooth_k > 0

    @property
    def name(self):
        if self.smooth:
            return f"smooth{self.smooth_k}_s{self.smooth_sigma:g}"
        if self.confirm or self.min_step or self.spike_factor:
            return ("confirm" if self.confirm else "raw") + (f"_min{self.min_step:g}" if self.min_step else "") \
                + (f"_spike{self.spike_factor:g}" if self.spike_factor else "")
        return "plain"


def policy_from_args(a):
    return Policy(confirm=bool(getattr(a, "confirm", False)), confirm_n=int(getattr(a, "confirm_n", 2)),
                  confirm_sigma=float(getattr(a, "confirm_sigma", 1e-6)),
                  min_step=float(getattr(a, "min_step", 0.0) or 0.0),
                  spike_factor=float(getattr(a, "spike_factor", 0.0) or 0.0),
                  spike_sigma=float(getattr(a, "spike_sigma", 1e-4)),
                  stall_restart=bool(getattr(a, "stall_restart", False)),
                  smooth_k=int(getattr(a, "smooth_k", 0) or 0),
                  smooth_sigma=float(getattr(a, "smooth_sigma", 3e-3)), seed=int(getattr(a, "jitter_seed", 0)),
                  nan_retries=int(getattr(a, "nan_retries", 0) or 0))


#: the stage-3 robust flags, the defaults of ``casa_4dvar --run`` since stage 4 (``--plain-policy``:
#: the stage-2 behaviour); fresh_n is casa_4dvar's own option
ROBUST_DEFAULTS = dict(confirm=True, min_step=1e-6, spike_factor=5.0, spike_sigma=1e-4, stall_restart=True,
                       nan_retries=4, fresh_n=2)
PLAIN_VALUES = dict(confirm=False, min_step=0.0, spike_factor=0.0, stall_restart=False, nan_retries=0, fresh_n=0)


def apply_plain_policy(a):
    """``--plain-policy``: reset the robust defaults to the stage-2 (plain L-BFGS-B) behaviour."""
    if getattr(a, "plain_policy", False):
        for k, v in PLAIN_VALUES.items():
            setattr(a, k, v)
    return a


def add_policy_arguments(ap, robust_defaults=False):
    """The policy options; ``robust_defaults``: ``ROBUST_DEFAULTS`` as the
    defaults (switch off with --no-confirm, --min-step 0, --spike-factor 0,
    --no-stall-restart, --nan-retries 0, or all at once with --plain-policy)."""
    D = ROBUST_DEFAULTS if robust_defaults else dict(PLAIN_VALUES, spike_sigma=1e-4)
    g = ap.add_argument_group("robust optimisation (casa_4dvar_robust)")
    g.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=D["confirm"],
                   help="accept a new best only if it survives re-evaluation at z +- confirm_sigma e")
    g.add_argument("--plain-policy", action="store_true",
                   help="the stage-2 plain L-BFGS-B: no confirm / min-step / spike guard / stall restart / "
                        "NaN retries / fresh-jitter scoring (overrides those options)")
    g.add_argument("--confirm-n", type=int, default=2)
    g.add_argument("--confirm-sigma", type=float, default=1e-6,
                   help="per-component jitter of the confirmation points (whitened)")
    g.add_argument("--min-step", type=float, default=D["min_step"],
                   help="end an L-BFGS run when a line-search trial is closer than this (whitened norm) "
                        "to the iterate (restart from the confirmed best); 0 = off")
    g.add_argument("--spike-factor", type=float, default=D["spike_factor"],
                   help="replace an evaluation whose |g| exceeds this x the recent median by one at a "
                        "jittered neighbour (0 = off; 5 recommended)")
    g.add_argument("--spike-sigma", type=float, default=D["spike_sigma"],
                   help="per-component jitter of the spike guard's neighbour (whitened)")
    g.add_argument("--stall-restart", action=argparse.BooleanOptionalAction, default=D["stall_restart"],
                   help="a stall restarts L-BFGS from the confirmed best (up to --max-restarts) "
                        "instead of ending the stage")
    g.add_argument("--smooth-k", type=int, default=0,
                   help="smoothed objective: K antithetic common-random-number jitters (even; 0 = off)")
    g.add_argument("--smooth-sigma", type=float, default=3e-3, help="per-component jitter sd (whitened)")
    g.add_argument("--jitter-seed", type=int, default=0)
    g.add_argument("--nan-retries", type=int, default=D["nan_retries"],
                   help="re-evaluate a non-finite J / gradient at up to this many seeded neighbours "
                        "(sigma 1e-5, 1e-5, 1e-4, 1e-4, ... per component, whitened); 0 = off; 4 recommended")
    g.add_argument("--optimizer", choices=("jax", "scipy"), default="jax",
                   help="jax: L-BFGS with every n-vector on the device(s), sharded like the control "
                        "(casa_4dvar_lbfgs); scipy: L-BFGS-B on float64 host vectors (the old path)")
    g.add_argument("--max-grad-evals", type=float, default=None,
                   help="per stage: stop after this many gradient-evaluation equivalents")
    return g


class Evaluator:
    """Jitter bookkeeping around ``vg(z, dz) -> ((J, chi2), g)`` and
    ``val(z, dz) -> (J, chi2)`` (z, dz: device arrays of ``size``);
    ``rough``: a boolean host mask of the components that are jittered
    (the state parts).

    ``ops`` (a ``casa_4dvar_lbfgs.VecOps``; ``--optimizer jax``): the device
    mode -- z, dz and g stay device arrays with ``ops.sharding`` (the
    objective's input), the jitter / neighbour directions are drawn ON the
    device (``jax.random``, keyed by (7919, crc32(tag), seed, i): same seeds,
    same directions; NOT the numpy directions of the host mode), and
    ``raw_vg_dev`` / ``smooth_vg_dev`` return J and chi2 as host scalars with g
    on the device. Without ``ops`` (``--optimizer scipy``) everything is as
    before: numpy directions, float64 host gradients."""

    #: direction families drawn once per run (cached on the device); the others
    #: (spike, nanguard, start: a new index per use) are drawn on demand -- caching
    #: them leaked one n-vector of device memory per evaluation
    CACHED = ("confirm", "smooth", "fresh")

    def __init__(self, vg, val, size, rough, dtype, policy, *, t_val_rel=0.2, ops=None):
        self.vg_j, self.val_j = vg, val
        self.size, self.dtype, self.policy = int(size), dtype, policy
        self.rough = np.asarray(rough, bool)
        self.ops = ops
        self.zero = ops.zeros() if ops is not None else jnp.zeros(self.size, dtype)
        self._dirs = {}
        self.n_grad = 0.0          # gradient-evaluation equivalents
        self.t_val_rel = t_val_rel
        if ops is not None:
            self._runs = _mask_runs(self.rough)
            self._gen = _direction_generator(self.size, ops.dtype, self._runs, ops.sharding)

    @property
    def device(self):
        return self.ops is not None

    def _cached(self, tag):
        return tag.startswith(self.CACHED)

    def direction(self, tag, i):
        """Fixed (seeded) white direction #i of family ``tag`` on the rough part, device array."""
        key = (tag, i)
        if key in self._dirs:
            return self._dirs[key]
        if self.device:
            k = jax.random.key(7919)
            for v in (zlib.crc32(tag.encode()), self.policy.seed, i):
                k = jax.random.fold_in(k, np.uint32(v))
            d = self._gen(k)
        else:
            rng = np.random.default_rng([7919, zlib.crc32(tag.encode()), self.policy.seed, i])
            dh = np.zeros(self.size, np.float32)
            dh[self.rough] = rng.standard_normal(int(self.rough.sum())).astype(np.float32)
            d = jnp.asarray(dh, self.dtype)
            del dh
        if self._cached(tag):
            self._dirs[key] = d
        return d

    def jitters(self, tag, n, sigma):
        """n antithetic jitters sigma * (+e_1, -e_1, +e_2, -e_2, ...)."""
        out = []
        for i in range((n + 1) // 2):
            e = self.direction(tag, i)
            if self.device:
                out += [self.ops.scale(sigma, e), self.ops.scale(-sigma, e)]
            else:
                out += [sigma * e, -sigma * e]
        return out[:n]

    def _z(self, z):
        return self.ops.put(z) if self.device else jnp.asarray(z, self.dtype)

    # ---- evaluations ------------------------------------------------------------------
    def _vg(self, z):
        (J, chi2), g = self.vg_j(jnp.asarray(z, self.dtype), self.zero)
        self.n_grad += 1
        return float(J), {k: float(v) for k, v in chi2.items()}, np.asarray(g, np.float64)

    def raw_vg(self, z):
        J, chi2, g = self._vg(z)
        p = self.policy
        for k in range(p.nan_retries):
            if np.isfinite(J) and np.all(np.isfinite(g)):
                break
            self.n_nan = getattr(self, "n_nan", 0) + 1
            sig = p.nan_sigmas[min(k, len(p.nan_sigmas) - 1)]
            zn = np.asarray(z, np.float64) + sig * np.asarray(self.direction("nanguard", self.n_nan), np.float64)
            J2, chi22, g2 = self._vg(zn)
            print(f"[nan-guard] non-finite (J {J}, {int(np.sum(~np.isfinite(g)))} non-finite gradient entries) "
                  f"-> neighbour #{k + 1} at sigma {sig:g}: J {J2:.4f}, |g| {np.linalg.norm(g2):.3g}", flush=True)
            J, chi2, g = J2, chi22, g2
        return J, chi2, g

    def _vg_dev(self, z):
        """(J, chi2 dict: host scalars, g: device array, number of non-finite g entries)."""
        (J, chi2), g = self.vg_j(self._z(z), self.zero)
        self.n_grad += 1
        J, chi2 = jax.device_get((J, chi2))
        return float(J), {k: float(v) for k, v in chi2.items()}, g, self.ops.n_nonfinite(g)

    def raw_vg_dev(self, z):
        """``raw_vg`` on the device: (J, chi2) host scalars, g the device array (no host copy)."""
        J, chi2, g, bad = self._vg_dev(z)
        p = self.policy
        for k in range(p.nan_retries):
            if np.isfinite(J) and bad == 0:
                break
            self.n_nan = getattr(self, "n_nan", 0) + 1
            sig = p.nan_sigmas[min(k, len(p.nan_sigmas) - 1)]
            zn = self.ops.axpy(sig, self.direction("nanguard", self.n_nan), self._z(z))
            J2, chi22, g2, bad2 = self._vg_dev(zn)
            print(f"[nan-guard] non-finite (J {J}, {bad} non-finite gradient entries) -> neighbour #{k + 1} at "
                  f"sigma {sig:g}: J {J2:.4f}, |g| {self.ops.norm(g2) if bad2 == 0 else float('nan'):.3g}",
                  flush=True)
            J, chi2, g, bad = J2, chi22, g2, bad2
        return J, chi2, g

    def raw_val(self, z, dz=None):
        J, chi2 = self.val_j(self._z(z), self.zero if dz is None else dz)
        self.n_grad += self.t_val_rel
        if self.device:
            J, chi2 = jax.device_get((J, chi2))
        return float(J), {k: float(v) for k, v in chi2.items()}

    def smooth_vg(self, z):
        """(J_s, mean chi2, g_s, member J list)."""
        if self.device:
            return self.smooth_vg_dev(z)
        p = self.policy
        zj = jnp.asarray(z, self.dtype)
        Js, gs, parts = [], None, None
        for dz in self.jitters("smooth", p.smooth_k, p.smooth_sigma):
            (J, chi2), g = self.vg_j(zj, dz)
            self.n_grad += 1
            Js.append(float(J))
            g = np.asarray(g, np.float64)
            gs = g if gs is None else gs + g
            c = {k: float(v) for k, v in chi2.items()}
            parts = c if parts is None else {k: parts[k] + c[k] for k in parts}
        k = len(Js)
        return float(np.mean(Js)), {kk: v / k for kk, v in parts.items()}, gs / k, Js

    def smooth_vg_dev(self, z):
        """``smooth_vg`` with the gradient mean accumulated on the device."""
        p = self.policy
        zj = self._z(z)
        Js, gs, parts = [], None, None
        for dz in self.jitters("smooth", p.smooth_k, p.smooth_sigma):
            (J, chi2), g = self.vg_j(zj, dz)
            self.n_grad += 1
            J, chi2 = jax.device_get((J, chi2))
            Js.append(float(J))
            gs = g if gs is None else self.ops.axpy(1.0, g, gs)
            c = {k: float(v) for k, v in chi2.items()}
            parts = c if parts is None else {k: parts[k] + c[k] for k in parts}
        k = len(Js)
        return float(np.mean(Js)), {kk: v / k for kk, v in parts.items()}, self.ops.scale(1.0 / k, gs), Js

    def smooth_val(self, z, sigma=None, k=None, tag="smooth"):
        p = self.policy
        Js = [self.raw_val(z, dz)[0] for dz in self.jitters(tag, k or p.smooth_k, sigma or p.smooth_sigma)]
        return float(np.mean(Js)), Js

    def confirm(self, z):
        p = self.policy
        Js = [self.raw_val(z, dz)[0] for dz in self.jitters("confirm", p.confirm_n, p.confirm_sigma)]
        return float(np.mean(Js)), Js

    def fresh(self, z, m=4, sigma=1e-6, seed=1):
        """J at m fresh tiny jitters (a different family than the run's): the
        honest value of a point, independent of which evaluations the run kept."""
        Js = [self.raw_val(z, dz)[0] for dz in self.jitters(f"fresh{seed}", m, sigma)]
        return dict(mean=float(np.mean(Js)), sd=float(np.std(Js)), J=Js)


def _mask_runs(mask):
    """[(lo, hi)] half-open index runs where the boolean host ``mask`` is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def _direction_generator(n, dtype, runs, sharding):
    """jit(key -> standard normal n-vector, zero outside ``runs``), produced with
    ``sharding`` (partitionable threefry: each device draws its own slab)."""
    if len(runs) > 64:
        raise ValueError(f"rough mask with {len(runs)} runs: use a device mask")

    def gen(key):
        e = jax.random.normal(key, (n,), dtype)
        i = jax.lax.iota(jnp.int32 if n < 2 ** 31 else jnp.int64, n)
        m = jnp.zeros((n,), bool)
        for lo, hi in runs:
            m = m | ((i >= lo) & (i < hi))
        return jnp.where(m, e, jnp.zeros((), dtype))
    return jax.jit(gen, **({"out_shardings": sharding} if sharding is not None else {}))


class _Stalled(Exception):
    pass


class _TinyStep(Exception):
    pass


class _Budget(Exception):
    pass


class _NonFiniteStart(Exception):
    pass


def mem_gb():
    """Peak GB in use: device 0, or the max over the mesh devices (casa_xfit_shard, --gpus)."""
    try:
        import casa_xfit_shard as SH
        if SH.active():
            return max(m for _, m, _ in SH.per_device_memory())
        ms = jax.devices()[0].memory_stats() or {}
        return float(ms.get("peak_bytes_in_use", 0)) / 2 ** 30
    except Exception:
        return float("nan")


#: hard cap on this process's host memory (GB): the optimiser (scipy L-BFGS-B, float64
#: host vectors) and JAX's host buffers live in CPU RAM next to other users' jobs on a shared
#: node (compgpu12 went down 2026-10-05 with a 448^3 run on it). Override with
#: CASA_4DVAR_MAX_HOST_GB; 0 disables the check.
MAX_HOST_GB = float(os.environ.get("CASA_4DVAR_MAX_HOST_GB", "300"))


def host_rss_gb():
    """Current resident set size of this process (GB, from /proc), with the peak as a fallback."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def check_host_memory(where):
    rss = host_rss_gb()
    if MAX_HOST_GB > 0 and rss > MAX_HOST_GB:
        raise MemoryError(f"[{where}] host RSS {rss:.1f} GB exceeds CASA_4DVAR_MAX_HOST_GB = {MAX_HOST_GB:.0f}")
    return rss


def robust_minimize(ev, z_start, *, n_iter, log_path=None, stage=0, it0=0, ckpt=None, obs_total=None,
                    stall_tol=0.05, stall_evals=10, max_restarts=2, maxcor=10, ftol=1e-9,
                    max_grad_evals=None, fmt=None, tag="", optimizer=None):
    """One stage of L-BFGS under ``ev.policy``. Returns a dict with z_best,
    best (confirmed / smoothed / raw J by policy), it, n_eval, n_grad, the
    stop reason and the per-evaluation history. ``ckpt(z_best, it)`` is called
    after every iteration.

    ``optimizer``: "scipy" = scipy L-BFGS-B on float64 host vectors (the
    stage-2..4 behaviour); "jax" = ``casa_4dvar_lbfgs.minimize`` with every
    n-vector on the device (``ev`` must be in device mode, ``ev.ops``); default:
    "jax" iff ``ev.ops`` is set. z_best is then a device array (``ckpt``
    receives it as one; ``np.asarray`` it to save)."""
    optimizer = optimizer or ("jax" if ev.device else "scipy")
    dev = optimizer == "jax"
    if dev and not ev.device:
        raise ValueError("optimizer 'jax' needs a device-mode Evaluator (ops=casa_4dvar_lbfgs.VecOps)")
    p = ev.policy
    fmt = fmt or (lambda c: "")
    obs_total = obs_total or (lambda c: float("nan"))
    if dev:
        import casa_4dvar_lbfgs as LB
        ops = ev.ops
        z_best0 = ops.put(z_start)
        dist = ops.dist
        gnorm = ops.norm
        add_dir = lambda z, s, e: ops.axpy(s, e, z)  # noqa: E731
        g_finite = lambda g: ops.n_nonfinite(g) == 0  # noqa: E731
        raw_vg = ev.raw_vg_dev
        keep = lambda z: z  # noqa: E731  (device arrays are immutable)
    else:
        z_best0 = np.array(z_start, np.float64)
        dist = lambda a, b: float(np.linalg.norm(a - b))  # noqa: E731
        gnorm = lambda g: float(np.linalg.norm(g))  # noqa: E731
        add_dir = lambda z, s, e: z + s * np.asarray(e, np.float64)  # noqa: E731
        g_finite = lambda g: bool(np.all(np.isfinite(g)))  # noqa: E731
        raw_vg = ev.raw_vg
        keep = np.array
    st = dict(it=it0, n_eval=0, best=np.inf, best_raw=np.inf, z_best=z_best0, last=None,
              since=0, z_iter=None, stop=None, hist=[])
    n0 = ev.n_grad

    def rec_write(rec):
        st["hist"].append(rec)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def fun(zz):
        if p.min_step and st["z_iter"] is not None:
            step_it = dist(zz, st["z_iter"])
            if 0.0 < step_it < p.min_step:
                raise _TinyStep(step_it)
        if max_grad_evals is not None and ev.n_grad - n0 >= max_grad_evals:
            raise _Budget
        t0 = time.time()
        spike = None
        if p.smooth:
            J, chi2, g, members = ev.smooth_vg(zz)
            ok = bool(np.isfinite(J) and g_finite(g))
            gn = gnorm(g) if ok else float("nan")
        else:
            J, chi2, g = raw_vg(zz)
            members = None
            ok = bool(np.isfinite(J) and g_finite(g))
            gn = gnorm(g) if (ok or not dev) else float("nan")
            recent = [h["gnorm"] for h in st["hist"][-6:] if h.get("finite")]
            first = not recent
            if p.spike_factor and (first or gn > p.spike_factor * float(np.median(recent))):
                zn = add_dir(zz, p.spike_sigma, ev.direction("spike", st["n_eval"]))
                Jn, chi2n, gnb = raw_vg(zn)
                okn = bool(np.isfinite(Jn) and g_finite(gnb))
                gnn = gnorm(gnb) if (okn or not dev) else float("nan")
                use = gnn < gn if not first else gn > p.spike_factor * gnn
                if use:
                    spike = dict(J=J, gnorm=gn, median=float(np.median(recent)) if recent else None,
                                 J_neighbour=Jn, gnorm_neighbour=gnn, first=first)
                    J, chi2, g, gn, ok = Jn, chi2n, gnb, gnn, okn
                print(f"[spike{' check' if first else ''}] |g| {spike['gnorm'] if spike else gn:.3g} "
                      f"(J {spike['J'] if spike else J:.4f}); "
                      f"neighbour J {Jn:.4f}, |g| {gnn:.3g} -> using the {'neighbour' if use else 'point'}",
                      flush=True)
                del zn, gnb
        st["n_eval"] += 1
        if not dev:
            ok = bool(np.isfinite(J) and np.all(np.isfinite(g)))
        rec = dict(stage=stage, tag=tag, policy=p.name, it=st["it"], eval=st["n_eval"], J=J,
                   gnorm=gn, parts=chi2, obs=obs_total(chi2), t_s=time.time() - t0,
                   peak_GB=mem_gb(), host_rss_GB=host_rss_gb(), finite=ok,
                   step=dist(zz, st["z_best"]),
                   step_iter=dist(zz, st["z_iter"]) if st["z_iter"] is not None else 0.0,
                   n_grad=ev.n_grad - n0, optimizer=optimizer)
        if members is not None:
            rec["members"] = members
        if spike is not None:
            rec["spike"] = spike
        check_host_memory(f"eval s{stage} #{st['n_eval']}")
        if not ok:
            rec_write(rec)
            print(f"[eval s{stage} it{st['it']} #{st['n_eval']}] NON-FINITE J {J}", flush=True)
            if st["last"] is None:
                if st["n_eval"] == 1:
                    raise FloatingPointError("J non-finite at the start point")
                raise _NonFiniteStart(J)
            J0, g0 = st["last"]
            return abs(J0) * 10.0 + 1e6, g0
        # ---- the best point: confirmed (policy.confirm), smoothed, or raw -------------
        Jc = J
        if p.confirm and not p.smooth and J < st["best"]:
            _, cj = ev.confirm(zz)
            Jc = float(np.median([J] + list(cj)))
            rec.update(J_confirm=Jc, confirm=cj)
        improved = Jc < st["best"]
        if Jc < st["best"] - stall_tol:
            st["since"] = 0
        else:
            st["since"] += 1
        if improved:
            st["best"], st["z_best"], st["best_raw"] = Jc, keep(zz), J
        rec.update(best=st["best"], accepted=bool(improved))
        rec_write(rec)
        print(f"[eval s{stage}{tag} it{st['it']} #{st['n_eval']}] J {J:.4f}"
              + (f" (conf {Jc:.4f})" if "J_confirm" in rec else "")
              + (f" (members {np.ptp(members):.3f} spread)" if members else "")
              + f" obs {rec['obs']:.1f} |g| {rec['gnorm']:.3g} |dz| {rec['step']:.3g} "
              f"({rec['t_s']:.0f} s, {rec['peak_GB']:.1f} GB, host {rec['host_rss_GB']:.1f} GB) "
              f"best {st['best']:.4f}: {fmt(chi2)}", flush=True)
        st["last"] = (J, g)
        if st["since"] >= stall_evals:
            raise _Stalled
        return J, g

    def callback(zk):
        st["it"] += 1
        st["z_iter"] = keep(zk)
        if ckpt is not None:
            ckpt(st["z_best"], st["it"])

    restarts = 0
    while st["it"] - it0 < n_iter and restarts <= max_restarts:
        left = n_iter - (st["it"] - it0)
        x_start = st["z_best"]
        if p.confirm and not p.smooth:
            # never (re)start on a possibly lucky point: a normal-branch neighbour
            x_start = add_dir(x_start, p.confirm_sigma, ev.direction("start", restarts))
        st["z_iter"] = keep(x_start)
        st["last"] = None
        try:
            if dev:
                r = LB.minimize(fun, x_start, ops=ops, maxiter=left, maxfun=int(2 * left + 5), maxcor=maxcor,
                                ftol=ftol, gtol=1e-12, callback=callback)
                name = "L-BFGS (device)"
            else:
                from scipy.optimize import minimize
                # scipy's L-BFGS-B indexes its workspace (2m + 5) n + 11 m^2 + 8 m with 32-bit
                # integers: at the 448^3 two-level control (n = 1.1e8) maxcor 10 overflows and
                # segfaults (2026-10-05). Cap the history so the workspace stays below 2^31.
                n_ctl = int(np.asarray(x_start).size)
                m_cap = max(1, int(((2 ** 31 - 1) // max(n_ctl, 1) - 5) // 2) - 1)
                if maxcor > m_cap:
                    print(f"[stage {stage}{tag}] L-BFGS-B history maxcor {maxcor} -> {m_cap} "
                          f"(n = {n_ctl}: 32-bit workspace limit)", flush=True)
                    maxcor = m_cap
                r = minimize(fun, x_start, jac=True, method="L-BFGS-B", callback=callback,
                             options=dict(maxiter=left, maxfun=int(2 * left + 5), maxcor=maxcor, ftol=ftol,
                                          gtol=1e-12))
                name = "L-BFGS-B"
            msg = str(r.message)
            print(f"[stage {stage}{tag}] {name}: {msg} (nit {r.nit}, nfev {r.nfev}); best {st['best']:.4f}",
                  flush=True)
            st["stop"] = msg
            factr_stall = "CONVERGENCE" in msg and "REDUCTION" in msg and p.stall_restart
            nit = int(r.nit)
            del r
            if nit >= left or ("ABNORMAL" not in msg and not factr_stall):
                break
        except _Stalled:
            st["stop"] = f"stalled ({stall_evals} evals without a decrease > {stall_tol})"
            print(f"[stage {stage}{tag}] {st['stop']}; best {st['best']:.4f}", flush=True)
            if not (p.stall_restart and restarts < max_restarts):
                break
            st["since"] = 0
            print(f"[stage {stage}{tag}] stall restart {restarts + 1} / {max_restarts}", flush=True)
        except _Budget:
            st["stop"] = f"budget ({max_grad_evals} gradient equivalents)"
            print(f"[stage {stage}{tag}] {st['stop']}; best {st['best']:.4f}", flush=True)
            break
        except _NonFiniteStart:
            st["stop"] = "non-finite J / gradient at a restart point"
            print(f"[stage {stage}{tag}] {st['stop']}: stage ends at the best point", flush=True)
            break
        except _TinyStep as e:
            st["stop"] = f"tiny step {float(e.args[0]):.3g} < {p.min_step:g}"
            print(f"[stage {stage}{tag}] {st['stop']}: restart from the best point", flush=True)
        restarts += 1
        if ckpt is not None:
            ckpt(st["z_best"], st["it"])
    st["restarts"] = restarts
    if st["n_eval"] == 0:
        # no evaluation (e.g. resumed at the stage's iteration budget): no best to report
        # (was J_best = inf -> "Infinity" in summary.json)
        st["best"] = st["best_raw"] = None
        st["stop"] = st["stop"] or "no iterations left"
    st["n_grad"] = ev.n_grad - n0
    st["policy"] = asdict(p)
    st["optimizer"] = optimizer
    st["last"] = None
    return st
