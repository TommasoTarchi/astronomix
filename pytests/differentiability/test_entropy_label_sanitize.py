"""
The shock-history label ``entropy_initial`` is reset where it is non-finite or
runs away (``sanitize_entropy_label``, applied in ``update_shock_history``).

The label is the only unbounded library scalar. On the Cas A R4b run (146 ->
2000 yr) the ratio recovery ``rho~ s0 / rho~`` of near-collapsed cells grew odd-
even pairs of labels (-3806 / +801 at 128^3, against a current entropy of
O(-5)); at 448^3 one overflowed, the NaN covered the whole label field, every
4D-Var gradient was NaN and the shock latch froze for the rest of the run.

Checked (x64, NATIVE_JAX, the 16^3 blast of ``test_fd_reverse_mode``):

* in-window labels pass bit for bit, NaN / +-inf / runaway ones become the
  current entropy, with a finite (zero) tangent and cotangent;
* ``update_shock_history`` with poisoned labels: identical everywhere else,
  finite, and the latch cannot fire on a reset label;
* a forward run seeded with a NaN label heals (every field finite, every label
  within the window), where the old code spread the NaN over the grid;
* the reverse-mode gradient through a runaway label is finite.

The forward bitwise-unchanged guarantee for clean runs is the reference hash of
``test_fd_reverse_mode.test_forward_default_unchanged_vs_reference``.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/differentiability/test_entropy_label_sanitize.py
"""
# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS", "") == "cpu":
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true")
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import importlib.util

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from astronomix import BACKWARDS, time_integration
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._fluid_equations._passive_scalars import (
    ENTROPY_LABEL_WINDOW,
    sanitize_entropy_label,
    specific_entropy,
    update_shock_history,
)

# the 16^3 Cas A-configured blast of the reverse-mode suite
_spec = importlib.util.spec_from_file_location(
    "_fd_reverse_mode", os.path.join(os.path.dirname(__file__), "test_fd_reverse_mode.py"))
RM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RM)

#: the label's index from the end of the state (entropy_initial, shocked_fraction,
#: time_since_shock, density_time)
I_LABEL = -4


def _bits(a):
    return np.asarray(a).view(np.int64)


def _rv_hydro(rv):
    return rv._replace(num_vars=rv.passive_scalar_index, passive_scalar_index=-1,
                       num_passive_scalars=0, passive_scalars_active=False,
                       shock_history_active=False)


def test_sanitize_unit():
    s_now = jnp.linspace(-30.0, 5.0, 9)
    w = ENTROPY_LABEL_WINDOW
    s0 = jnp.array([jnp.nan, jnp.inf, -jnp.inf, -3805.75, 800.81, 0.0, 0.0, -20.75, 5.0])
    s0 = s0.at[6].set(s_now[6] - (w - 1e-9))     # just inside the window
    out = sanitize_entropy_label(s0, s_now)
    reset = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], bool)
    np.testing.assert_array_equal(_bits(out)[reset], _bits(s_now)[reset])
    np.testing.assert_array_equal(_bits(out)[~reset], _bits(s0)[~reset])
    # tangent: the label's own inside, none on a reset cell (whatever s_now's)
    _, dout = jax.jvp(sanitize_entropy_label, (s0, s_now),
                      (jnp.ones_like(s0), 2.0 * jnp.ones_like(s_now)))
    np.testing.assert_array_equal(np.asarray(dout), np.where(reset, 0.0, 1.0))
    # cotangent: finite through NaN / inf labels
    g0, g1 = jax.grad(lambda a, b: jnp.sum(sanitize_entropy_label(a, b)), argnums=(0, 1))(s0, s_now)
    assert np.all(np.isfinite(g0)) and np.all(np.isfinite(g1))
    np.testing.assert_array_equal(np.asarray(g0), np.where(reset, 0.0, 1.0))
    np.testing.assert_array_equal(np.asarray(g1), 0.0)


def test_update_shock_history_poisoned_label():
    config, rv, state = RM._setup()
    rv_h = _rv_hydro(rv)
    # two steps in, so that there is a real shock to latch on
    st = time_integration(state, config, RM._params(t_end=2 * RM.DT_FIXED), rv)
    prim, hist = st[:rv.passive_scalar_index], st[-4:]
    s_now = specific_entropy(prim, RM.GAMMA, rv_h)

    def f(h):
        return update_shock_history(h, prim, 1e-3, RM.GAMMA, config.shock_entropy_jump,
                                    config, rv_h)
    clean = f(hist)
    assert np.all(np.abs(np.asarray(s_now - hist[0])) <= ENTROPY_LABEL_WINDOW)
    # the clean label is returned as is
    np.testing.assert_array_equal(_bits(clean[0]), _bits(hist[0]))

    # poison: NaN, inf and the two Cas A runaway values, the latter in CONVERGING,
    # not-yet-shocked cells, where the old code fired the latch on them
    vel = [prim[rv_h.velocity_index.x], prim[rv_h.velocity_index.y],
           prim[rv_h.velocity_index.z]]
    div_v = sum(_shift(v, -1, axis=a) - _shift(v, 1, axis=a) for a, v in enumerate(vel))
    cand = np.argwhere(np.asarray((div_v < 0) & (hist[1] == 0.0)))
    assert len(cand) >= 2
    cells = [(3, 3, 3), (12, 4, 9), tuple(cand[0]), tuple(cand[len(cand) // 2])]
    vals = [jnp.nan, -jnp.inf, -3805.75, -180.32]
    h0 = hist[0]
    for c, v in zip(cells, vals):
        h0 = h0.at[c].set(v)
    out = f(hist.at[0].set(h0))
    assert bool(jnp.all(jnp.isfinite(out)))
    mask = np.zeros(h0.shape, bool)
    for c in cells:
        mask[c] = True
    for k in range(4):
        np.testing.assert_array_equal(_bits(out[k])[~mask], _bits(clean[k])[~mask])
    np.testing.assert_array_equal(_bits(out[0])[mask], _bits(s_now)[mask])
    # a reset label has entropy rise 0, so it cannot latch
    np.testing.assert_array_equal(np.asarray(out[1])[mask], np.asarray(hist[1])[mask])
    # (the old code: s_now - (-3805) > jump with div v < 0 -> latched)
    old_rise = np.asarray(s_now)[mask] - np.asarray(h0)[mask]
    assert np.sum(old_rise[2:] > float(config.shock_entropy_jump)) == 2


def test_forward_heals_nan_label():
    config, rv, state = RM._setup()
    poisoned = state.at[I_LABEL, 8, 8, 8].set(jnp.nan).at[I_LABEL, 9, 8, 8].set(-3805.75)
    out = time_integration(poisoned, config, RM._params(), rv)
    assert bool(jnp.all(jnp.isfinite(out)))
    s_now = specific_entropy(out[:rv.passive_scalar_index], RM.GAMMA, _rv_hydro(rv))
    assert float(jnp.max(jnp.abs(s_now - out[I_LABEL]))) <= ENTROPY_LABEL_WINDOW


def test_grad_finite_through_runaway_label():
    config, rv, state = RM._setup(fixed_timestep=True, num_timesteps=RM.N_FIXED,
                                  differentiation_mode=BACKWARDS, ad_remat="stage")
    par = RM._params(t_end=RM.N_FIXED * RM.DT_FIXED)
    poisoned = state.at[I_LABEL, 8, 8, 8].set(-3805.75).at[I_LABEL, 9, 8, 8].set(800.81)
    w = jnp.asarray(np.random.default_rng(3).normal(size=state.shape))
    g = jax.jit(jax.grad(lambda s: jnp.vdot(w, time_integration(s, config, par, rv))))(poisoned)
    assert bool(jnp.all(jnp.isfinite(g)))
