"""
CPU tests of the device L-BFGS (``casa_4dvar_lbfgs``) and the device path of
``casa_4dvar_robust``.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo>:<showcase> \\
        python -m pytest -q test_casa_4dvar_lbfgs.py            (SKIP_SLOW=1: skip the n = 1e8 dot)

Four fake CPU devices (XLA_FLAGS, set below before jax is imported) carry the
sharded tests; everything else runs on the default device.
"""
import json
import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
# ruff: noqa: E402

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from scipy.linalg import solve_banded
from scipy.optimize import minimize as sp_minimize

import casa_4dvar_lbfgs as LB
import casa_4dvar_robust as RB

CKPT_448 = "/export/data/lstorcks/casa_orlando150/work/ers/n448/run_4dv_R4b_jet/ckpt.npz"


# =============================================================================
# ============ ↓ Problems ↓ ===================================================
# =============================================================================
def rosen(x):
    """Extended (separable) Rosenbrock (Moré, Garbow & Hillstrom 1981): n/2 independent pairs."""
    a, b = x[0::2], x[1::2]
    return jnp.sum(100.0 * (b - a ** 2) ** 2 + (1.0 - a) ** 2)


def rosen_problem(dtype):
    vg = jax.jit(jax.value_and_grad(lambda x: rosen(x.astype(dtype))))
    counter = dict(n=0)

    def fun(x):
        counter["n"] += 1
        f, g = vg(x)
        return float(f), g
    return fun, counter


def spd_problem(n, seed=0):
    """f = 0.5 x.A x - b.x, A = diag(d) + c * (1D Laplacian): SPD, condition ~1e3."""
    rng = np.random.default_rng(seed)
    d = 10.0 ** rng.uniform(-1.0, 1.0, n)
    c = 5.0
    b = rng.standard_normal(n)
    ab = np.zeros((3, n))
    ab[0, 1:] = -c
    ab[1] = d + 2 * c
    ab[2, :-1] = -c
    x_star = solve_banded((1, 1), ab, b)

    def Ax(x, xp=jnp):
        lap = 2 * x - xp.concatenate([x[1:], xp.zeros(1, x.dtype)]) - xp.concatenate([xp.zeros(1, x.dtype), x[:-1]])
        return d * x + c * lap
    return d, b, Ax, x_star


def two_loop_ref(g, pairs, gamma):
    """Textbook two-loop recursion in float64 (pairs: oldest -> newest)."""
    q = g.copy()
    al = []
    for s, y in reversed(pairs):
        a = (s @ q) / (s @ y)
        al.append(a)
        q -= a * y
    r = gamma * q
    for (s, y), a in zip(pairs, reversed(al)):
        b = (y @ r) / (s @ y)
        r += s * (a - b)
    return -r
# =============================================================================
# ============ ↑ Problems ↑ ===================================================
# =============================================================================


def test_rosenbrock_vs_scipy():
    """n = 1000 extended Rosenbrock from the standard start (-1.2, 1, -1.2, 1, ...): the same minimiser as scipy
    L-BFGS-B, comparable evaluation counts."""
    n = 1000
    x0 = np.tile([-1.2, 1.0], n // 2)
    fun, cnt = rosen_problem(jnp.float32)
    ops = LB.VecOps(n, jnp.float32)
    r = LB.minimize(fun, x0, ops=ops, maxiter=500, maxfun=1000, maxcor=10, ftol=1e-12, gtol=1e-3)
    x = np.asarray(r.x, np.float64)
    sp = sp_minimize(lambda v: (float(rosen(jnp.asarray(v))), np.asarray(jax.grad(rosen)(jnp.asarray(v)),
                                                                         np.float64)),
                     x0, jac=True, method="L-BFGS-B", options=dict(maxiter=500, maxcor=10, ftol=1e-12, gtol=1e-3))
    print(f"\n[rosen] device: f {r.fun:.3g} nit {r.nit} nfev {r.nfev} ({r.message}); "
          f"scipy: f {sp.fun:.3g} nit {sp.nit} nfev {sp.nfev}")
    assert np.max(np.abs(x - 1.0)) < 2e-2, np.max(np.abs(x - 1.0))
    assert np.max(np.abs(sp.x - 1.0)) < 2e-2
    assert r.fun < 1e-4
    assert r.nfev <= 2.0 * sp.nfev + 20


def test_spd_quadratic_vs_scipy():
    """n = 1e5 SPD quadratic, the SAME float32 objective for both (the 4D-Var's J is float32 too):
    the same minimiser to the float32 noise floor of J, evaluation counts within 1.5x."""
    n = 100_000
    d, b, Ax, x_star = spd_problem(n)
    dj, bj = jnp.asarray(d, jnp.float32), jnp.asarray(b, jnp.float32)

    def Axj(x):
        return dj * x + 5.0 * (2 * x - jnp.concatenate([x[1:], jnp.zeros(1, x.dtype)])
                               - jnp.concatenate([jnp.zeros(1, x.dtype), x[:-1]]))
    vg = jax.jit(lambda x: (0.5 * jnp.dot(x, Axj(x)) - jnp.dot(bj, x), Axj(x) - bj))

    def fun(x):
        f, g = vg(x)
        return float(f), g
    ops = LB.VecOps(n, jnp.float32)
    r = LB.minimize(fun, np.zeros(n), ops=ops, maxiter=500, maxfun=1000, maxcor=10, ftol=1e-12, gtol=1e-4)
    x = np.asarray(r.x, np.float64)
    def fun_sp(v):
        f, g = vg(jnp.asarray(v, jnp.float32))
        return float(f), np.asarray(g, np.float64)
    sp = sp_minimize(fun_sp, np.zeros(n), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=500, maxcor=10, ftol=1e-12, gtol=1e-4))
    err = np.linalg.norm(x - x_star) / np.linalg.norm(x_star)
    err_sp = np.linalg.norm(sp.x - x_star) / np.linalg.norm(x_star)
    print(f"\n[spd] device: rel err {err:.2e} nit {r.nit} nfev {r.nfev} ({r.message}); "
          f"scipy: rel err {err_sp:.2e} nit {sp.nit} nfev {sp.nfev}")
    assert err < 2e-3 and err <= 1.5 * err_sp + 1e-5
    assert r.nfev <= 1.5 * sp.nfev + 10


@pytest.mark.parametrize("case", ["quadratic", "extrapolate", "zoom", "rosen_line"])
def test_strong_wolfe(case):
    """The accepted step satisfies the strong Wolfe conditions (c1 1e-4, c2 0.9)."""
    c1, c2 = 1e-4, 0.9
    if case == "quadratic":
        f = lambda a: (a - 3.0) ** 2  # noqa: E731
        df = lambda a: 2 * (a - 3.0)  # noqa: E731
        a1 = 1.0
    elif case == "extrapolate":          # minimum far out: bracketing must grow the step
        f = lambda a: (a - 40.0) ** 2 / 40.0  # noqa: E731
        df = lambda a: 2 * (a - 40.0) / 40.0  # noqa: E731
        a1 = 1.0
    elif case == "zoom":                 # first trial far past the minimum
        f = lambda a: np.log(1 + (a - 0.01) ** 2 * 1e4) - a * 1e-3  # noqa: E731
        df = lambda a: 2e4 * (a - 0.01) / (1 + (a - 0.01) ** 2 * 1e4) - 1e-3  # noqa: E731
        a1 = 1.0
    else:                                # Rosenbrock along -g from (-1.2, 1)
        x0 = np.array([-1.2, 1.0])
        g0 = np.asarray(jax.grad(rosen)(jnp.asarray(x0)), np.float64)
        dd = -g0
        f = lambda a: float(rosen(jnp.asarray(x0 + a * dd)))  # noqa: E731
        df = lambda a: float(np.asarray(jax.grad(rosen)(jnp.asarray(x0 + a * dd))) @ dd)  # noqa: E731
        a1 = 1.0 / np.linalg.norm(dd)
    f0, df0 = f(0.0), df(0.0)
    ls = LB.wolfe_search(lambda a: (f(a), df(a), a), f0, df0, a1, c1=c1, c2=c2, max_evals=30)
    assert ls.alpha is not None and ls.wolfe, ls
    a = ls.alpha
    assert f(a) <= f0 + c1 * a * df0 + 1e-12
    assert abs(df(a)) <= c2 * abs(df0) + 1e-12


def test_line_search_nonfinite_trial():
    """A non-finite trial (a NaN region past the minimum) is bracketed away by bisection."""
    f = lambda a: (a - 0.5) ** 2 if a < 2.0 else np.nan  # noqa: E731
    df = lambda a: 2 * (a - 0.5) if a < 2.0 else np.nan  # noqa: E731
    ls = LB.wolfe_search(lambda a: (f(a), df(a), None), f(0.0), df(0.0), 5.0, max_evals=30)
    assert ls.alpha is not None and ls.alpha < 2.0
    assert abs(df(ls.alpha)) <= 0.9 * abs(df(0.0))


def test_nonfinite_gradient_never_accepted():
    """(review 2026-10-05) A trial with a finite f but a NaN gradient must not become the
    iterate through the Armijo fallback (it did: x landed on the NaN-gradient point)."""
    vg = jax.jit(jax.value_and_grad(lambda x: jnp.sum((x - 1.0) ** 2)))

    def fun(x):
        f, g = vg(x)
        return float(f), (g * jnp.nan if float(x[0]) > 0.6 else g)
    r = LB.minimize(fun, np.zeros(4), ops=LB.VecOps(4, jnp.float32), maxiter=50, ftol=0.0, gtol=1e-12)
    assert np.all(np.isfinite(np.asarray(r.jac))) and float(np.asarray(r.x)[0]) <= 0.6


def test_ring_buffer_wraparound():
    """m = 3 ring buffer after 7 pushes: holds the last 3 pairs in ring order, and the
    two-loop direction equals the float64 textbook recursion on those pairs."""
    n, m = 64, 3
    rng = np.random.default_rng(1)
    ops = LB.VecOps(n, jnp.float32)
    h = LB.History(ops, m)
    A = np.diag(rng.uniform(0.5, 4.0, n))
    pairs = []
    for k in range(7):
        s = rng.standard_normal(n)
        y = A @ s
        sy, ok = h.push(ops.put(s), ops.put(y), gs=-abs(s @ y))
        assert ok
        pairs.append((s, y))
    assert h.count == 3 and h.head == 7 % 3
    S, Y = np.asarray(h.S, np.float64), np.asarray(h.Y, np.float64)
    for j, slot in enumerate(h.newest()):           # newest -> oldest
        s, y = pairs[-1 - j]
        np.testing.assert_allclose(S[slot], s, rtol=1e-6)
        np.testing.assert_allclose(Y[slot], y, rtol=1e-6)
    g = rng.standard_normal(n)
    gamma = pairs[-1][0] @ pairs[-1][1] / (pairs[-1][1] @ pairs[-1][1])
    d_ref = two_loop_ref(g, pairs[-3:], gamma)
    d = np.asarray(h.direction(ops.put(g)), np.float64)
    np.testing.assert_allclose(d, d_ref, rtol=2e-5, atol=2e-5 * np.abs(d_ref).max())
    # partially filled (count 2 of 3) after a reset: only the 2 newest pairs enter
    h.reset()
    for s, y in pairs[:2]:
        h.push(ops.put(s), ops.put(y), gs=-1.0)
    gamma2 = pairs[1][0] @ pairs[1][1] / (pairs[1][1] @ pairs[1][1])
    d2 = np.asarray(h.direction(ops.put(g)), np.float64)
    np.testing.assert_allclose(d2, two_loop_ref(g, pairs[:2], gamma2), rtol=2e-5,
                               atol=2e-5 * np.abs(d2).max())
    # curvature guard: s.y <= 0 is skipped
    s, y = pairs[0]
    _, ok = h.push(ops.put(s), ops.put(-y), gs=-1.0)
    assert not ok and h.count == 2


def test_sharded_vectors_stay_sharded():
    """4 fake devices, P("x"): the minimiser's iterate, gradient and history keep the
    control's sharding (nothing is gathered), and the result equals the 1-device one."""
    assert len(jax.devices()) >= 4, "needs the 4 fake CPU devices (XLA_FLAGS)"
    mesh = jax.make_mesh((4,), ("x",), axis_types=(jax.sharding.AxisType.Auto,), devices=jax.devices()[:4])
    sh = NamedSharding(mesh, P("x"))
    n = 4096
    d, b, Ax, x_star = spd_problem(n, seed=3)
    dj, bj = jnp.asarray(d, jnp.float32), jnp.asarray(b, jnp.float32)
    seen = []

    def make(sharding):
        def f(x):
            Axv = dj * x + 5.0 * (2 * x - jnp.concatenate([x[1:], jnp.zeros(1, x.dtype)])
                                  - jnp.concatenate([jnp.zeros(1, x.dtype), x[:-1]]))
            return 0.5 * jnp.dot(x, Axv) - jnp.dot(bj, x), Axv - bj
        vg = jax.jit(f, out_shardings=(None, sharding)) if sharding is not None else jax.jit(f)

        def fun(x):
            if sharding is not None:
                seen.append(x.sharding)
            fv, g = vg(x)
            return float(fv), g
        return fun
    ops4 = LB.VecOps(n, jnp.float32, sharding=sh)
    r4 = LB.minimize(make(sh), np.zeros(n), ops=ops4, maxiter=300, maxcor=8, ftol=1e-12, gtol=1e-4)
    ops1 = LB.VecOps(n, jnp.float32)
    r1 = LB.minimize(make(None), np.zeros(n), ops=ops1, maxiter=300, maxcor=8, ftol=1e-12, gtol=1e-4)
    assert all(s == sh for s in seen)
    assert r4.x.sharding == sh and r4.jac.sharding == sh
    assert r4.history.S.sharding == NamedSharding(mesh, P(None, "x"))
    # (the reduction order differs between 4 shards and 1 device: equal to float32 noise, then
    # the trajectories drift apart at the J noise floor; both reach the minimiser)
    for r in (r4, r1):
        assert np.linalg.norm(np.asarray(r.x, np.float64) - x_star) / np.linalg.norm(x_star) < 2e-3
    # the blocked dot: sharded == unsharded == float64 truth (to float32 accuracy)
    rng = np.random.default_rng(0)
    a, c = rng.standard_normal(n), rng.standard_normal(n)
    assert abs(ops4.dot(ops4.put(a), ops4.put(c)) - a @ c) < 1e-4 * np.linalg.norm(a) * np.linalg.norm(c)


@pytest.mark.skipif(os.environ.get("SKIP_SLOW") == "1", reason="SKIP_SLOW=1")
def test_dot_accuracy_1e8():
    """Blocked float32 dot at n = 1e8 vs the float64 truth (and the plain jnp.sum)."""
    n = 100_000_000
    rng = np.random.default_rng(5)
    a = (1.0 + rng.standard_normal(n, dtype=np.float32) * 0.1).astype(np.float32)
    b = (1.0 + rng.standard_normal(n, dtype=np.float32) * 0.1).astype(np.float32)
    truth = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    ops = LB.VecOps(n, jnp.float32)
    aj, bj = ops.put(a), ops.put(b)
    blocked = ops.dot(aj, bj)
    plain = float(jax.jit(lambda x, y: jnp.sum(x * y))(aj, bj))
    rel_b, rel_p = abs(blocked - truth) / abs(truth), abs(plain - truth) / abs(truth)
    print(f"\n[dot n=1e8] blocked rel err {rel_b:.2e}, plain jnp.sum rel err {rel_p:.2e}")
    assert rel_b < 1e-6


# =============================================================================
# ============ ↓ robust layer, device path ↓ ==================================
# =============================================================================
class ToyObjective:
    """J(z) = 0.5 |z|^2 (background) + 0.5 |A (z + dz) - b|^2 (model, jittered); optional
    injected spike / NaN at given evaluation numbers of the vg."""

    def __init__(self, n, seed=0, spike_at=(), nan_at=()):
        rng = np.random.default_rng(seed)
        self.d = jnp.asarray(rng.uniform(0.5, 3.0, n), jnp.float32)
        self.b = jnp.asarray(rng.standard_normal(n), jnp.float32)
        self.k = 0
        self.spike_at, self.nan_at = set(spike_at), set(nan_at)

        def obj(z, dz):
            r = self.d * (z + dz) - self.b
            chi2 = dict(b=jnp.sum(z ** 2), o=jnp.sum(r ** 2))
            return 0.5 * (chi2["b"] + chi2["o"]), chi2
        self._vg = jax.jit(jax.value_and_grad(obj, has_aux=True))
        self._val = jax.jit(obj)
        n_ = n
        self.z_star = np.asarray(self.d * self.b / (1.0 + self.d ** 2), np.float64)
        assert self.z_star.size == n_

    def vg(self, z, dz):
        self.k += 1
        (J, c), g = self._vg(z, dz)
        if self.k in self.spike_at:
            g = g * 50.0
        if self.k in self.nan_at:
            g = g.at[0].set(jnp.nan)
        return (J, c), g

    def val(self, z, dz):
        return self._val(z, dz)


@pytest.mark.parametrize("optimizer", ["jax", "scipy"])
def test_robust_minimize_paths(tmp_path, optimizer):
    """The robust policies on both optimisers: converges to the minimiser, confirm / spike /
    NaN retry fire, evals.jsonl has host_rss_GB, ckpt receives z_best each iteration."""
    n = 2000
    toy = ToyObjective(n, spike_at=(5,), nan_at=(8,))
    rough = np.zeros(n, bool)
    rough[:1500] = True
    pol = RB.Policy(confirm=True, min_step=1e-6, spike_factor=5.0, spike_sigma=1e-4, stall_restart=True,
                    nan_retries=4, seed=0)
    ops = LB.VecOps(n, jnp.float32) if optimizer == "jax" else None
    ev = RB.Evaluator(toy.vg, toy.val, n, rough, jnp.float32, pol, ops=ops)
    saved = []
    log = tmp_path / "evals.jsonl"
    st = RB.robust_minimize(ev, np.zeros(n), n_iter=25, log_path=log, stage=0,
                            ckpt=lambda zb, it: saved.append((np.asarray(zb, np.float64).copy(), it)),
                            stall_tol=1e-6, stall_evals=10, max_restarts=2, maxcor=10, ftol=1e-12,
                            optimizer=optimizer)
    zb = np.asarray(st["z_best"], np.float64)
    err = np.linalg.norm(zb - toy.z_star) / np.linalg.norm(toy.z_star)
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    print(f"\n[robust {optimizer}] rel err {err:.2e}, it {st['it']}, n_eval {st['n_eval']}, "
          f"n_grad {st['n_grad']:.1f}, stop {st['stop']}, spikes {sum('spike' in r for r in recs)}")
    assert err < 1e-3
    assert all("host_rss_GB" in r and r["optimizer"] == optimizer for r in recs)
    assert any("J_confirm" in r for r in recs)
    assert any("spike" in r for r in recs)          # the injected spike at vg #5 was replaced
    assert getattr(ev, "n_nan", 0) >= 1              # the injected NaN at vg #8 was retried
    assert all(r["finite"] for r in recs)
    assert saved and saved[-1][1] == st["it"]
    if optimizer == "jax":
        assert isinstance(st["z_best"], jax.Array)


def test_device_directions_reproducible():
    """Device jitter directions: zero off the rough mask, unit variance on it, the same for
    the same (tag, seed, i), different for another i; sharded like the control."""
    n = 40_000
    rough = np.zeros(n, bool)
    rough[100:30_000] = True
    rough[35_000:39_000] = True
    mesh = jax.make_mesh((4,), ("x",), axis_types=(jax.sharding.AxisType.Auto,), devices=jax.devices()[:4])
    sh = NamedSharding(mesh, P("x"))
    ops = LB.VecOps(n, jnp.float32, sharding=sh)
    pol = RB.Policy(seed=3)
    ev1 = RB.Evaluator(None, None, n, rough, jnp.float32, pol, ops=ops)
    ev2 = RB.Evaluator(None, None, n, rough, jnp.float32, pol, ops=ops)
    e1, e2 = ev1.direction("spike", 4), ev2.direction("spike", 4)
    e3 = ev1.direction("spike", 5)
    assert e1.sharding == sh
    a1, a3 = np.asarray(e1), np.asarray(e3)
    np.testing.assert_array_equal(a1, np.asarray(e2))
    assert np.all(a1[~rough] == 0) and abs(a1[rough].std() - 1) < 0.02
    assert np.corrcoef(a1[rough], a3[rough])[0, 1] < 0.05
    assert ("spike", 4) not in ev1._dirs and ("confirm", 0) not in ev1._dirs
    ev1.direction("confirm", 0)
    assert ("confirm", 0) in ev1._dirs


# =============================================================================
# ============ ↓ resume from the 448^3 run's checkpoint ↓ =====================
# =============================================================================
@pytest.mark.skipif(not os.path.exists(CKPT_448), reason="448^3 checkpoint not available")
def test_resume_448_ckpt_format(tmp_path):
    """The interrupted 448^3 run's ckpt.npz (scipy era) loads through casa_4dvar.load_ckpt,
    goes onto 4 (fake) devices sharded P("x") as the device optimiser holds it, and a
    device z_best saves back in the same format (z float64, stage, it, hist)."""
    import casa_4dvar as V
    z, stage, it, hist = V.load_ckpt(CKPT_448)
    assert z.dtype == np.float64 and z.size == 112_394_264 and (stage, it) == (0, 2)
    assert "--resume" in hist["argv"] and hist["val"] and not hist["stages"]
    mesh = jax.make_mesh((4,), ("x",), axis_types=(jax.sharding.AxisType.Auto,), devices=jax.devices()[:4])
    sh = NamedSharding(mesh, P("x"))
    ops = LB.VecOps(z.size, jnp.float32, sharding=sh)
    zd = ops.put(z)
    assert zd.sharding == sh and zd.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(zd), z.astype(np.float32))
    out = tmp_path / "ckpt.npz"
    V.save_ckpt(out, zd, stage, it + 1, hist)
    d0, d1 = np.load(CKPT_448), np.load(out)
    assert set(d0.files) == set(d1.files)
    assert d1["z"].dtype == np.float64 and d1["z"].shape == d0["z"].shape
    z2, s2, it2, h2 = V.load_ckpt(out)
    assert (s2, it2) == (0, 3) and h2["argv"] == hist["argv"]
    np.testing.assert_array_equal(z2, z.astype(np.float32).astype(np.float64))
