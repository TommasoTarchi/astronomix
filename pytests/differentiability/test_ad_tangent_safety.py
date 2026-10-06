"""
Tangent-only safety of reverse-mode AD through the finite-difference solver
(stage-4 library adjoint work, 2026-09-26): the primal is bit for bit unchanged.

**Overflow-free WENO-JS weight derivative** (``_weno_omega_weights_ad``). JAX
differentiates ``alpha_0 / alpha_sum`` through ``alpha_sum^-2``; where every
smoothness indicator is large (a field with an O(1e5) jump across the stencil)
``alpha_sum`` falls below ~5e-20 and ``alpha_sum^-2`` overflows float32, so even
a ZERO cotangent gives ``0 * inf = NaN``. On the Cas A 4D-Var's R' trajectory
(128^3, float32) the unbounded ``entropy_initial`` label reached -6.7e4 in one
collapsed cell; the passive-scalar WENO backward made that NaN and the NaN
covered the whole state gradient within a year while J stayed finite. The
stencil below is the recorded one.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/differentiability/test_ad_tangent_safety.py
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

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from astronomix._finite_difference._interface_fluxes._weno_weights import (
    _weno_omega_weights,
    _weno_omega_weights_ad,
)
from astronomix._fluid_equations._passive_scalars import _weno5_left_biased

#: the passive-scalar ``entropy_initial`` ratio along x through the cell whose
#: WENO VJP was NaN (Cas A 4D-Var, R' z_stage3, 2017.4 -> 2018.4, step 2)
CASA_STENCIL = [-19.62940216064453, -19.648536682128906, -19.625972747802734,
                -19.603029251098633, -22.161638259887695, -778.392578125,
                -66914.5078125, -9.308792114257812, -18.60173988342285]


def _reconstruct(q, omega):
    return jnp.stack([_weno5_left_biased(q[i - 2], q[i - 1], q[i], q[i + 1], q[i + 2],
                                         1e-7, omega) for i in range(2, len(q) - 2)])


def test_weno_weights_ad_primal_bitwise():
    rng = np.random.default_rng(0)
    for dtype in (jnp.float32, jnp.float64):
        IS = [jnp.asarray(10.0 ** rng.uniform(-12, 12, size=4096), dtype) for _ in range(3)]
        for tiny in (1e-40, 1e-14):
            a = _weno_omega_weights(*IS, 1e-7, tiny)
            b = _weno_omega_weights_ad(*IS, 1e-7, tiny)
            for x, y in zip(a, b):
                assert x.dtype == y.dtype
                np.testing.assert_array_equal(np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8))
    q = jnp.asarray(CASA_STENCIL, jnp.float32)
    np.testing.assert_array_equal(np.asarray(_reconstruct(q, _weno_omega_weights)),
                                  np.asarray(_reconstruct(q, _weno_omega_weights_ad)))


def test_weno_weights_ad_tangent_exact():
    """Same derivative as autodiff of the plain weights wherever that is finite
    (x64), including a traced epsilon (the ``weno_epsilon_relative`` path)."""
    rng = np.random.default_rng(1)
    for scale in (1e-3, 1.0, 1e3):
        IS = [jnp.asarray(np.abs(rng.normal(size=2000)) * scale ** 2 * 10 ** rng.uniform(-6, 0, 2000))
              for _ in range(3)]
        dIS = [jnp.asarray(rng.normal(size=2000)) * IS[k] for k in range(3)]
        _, t_ref = jax.jvp(lambda a, b, c: _weno_omega_weights(a, b, c, 1e-7, 1e-40), IS, dIS)
        _, t_new = jax.jvp(lambda a, b, c: _weno_omega_weights_ad(a, b, c, 1e-7, 1e-40), IS, dIS)
        for x, y in zip(t_ref, t_new):
            np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=1e-11,
                                       atol=1e-13 * float(jnp.max(jnp.abs(x))))
    g_ref = jax.grad(lambda e: _weno_omega_weights(*IS, e, 1e-14)[0].sum())(jnp.asarray(1e-7))
    g_new = jax.grad(lambda e: _weno_omega_weights_ad(*IS, e, 1e-14)[0].sum())(jnp.asarray(1e-7))
    np.testing.assert_allclose(float(g_new), float(g_ref), rtol=1e-10)


def test_weno_weights_ad_no_nan_on_casa_stencil():
    """The recorded Cas A stencil: the plain weights' float32 VJP is NaN even for
    a zero cotangent; the overflow-free one is finite (and zero for zero)."""
    q = jnp.asarray(CASA_STENCIL, jnp.float32)
    y, vjp_plain = jax.vjp(lambda v: _reconstruct(v, _weno_omega_weights), q)
    assert not np.all(np.isfinite(np.asarray(vjp_plain(jnp.zeros_like(y))[0])))   # the bug
    y, vjp = jax.vjp(lambda v: _reconstruct(v, _weno_omega_weights_ad), q)
    np.testing.assert_array_equal(np.asarray(vjp(jnp.zeros_like(y))[0]), 0.0)
    g1 = np.asarray(vjp(jnp.ones_like(y))[0])
    assert np.all(np.isfinite(g1))
    # and it is the derivative: central differences in float64 on the same stencil
    q64 = jnp.asarray(CASA_STENCIL, jnp.float64)
    _, vjp64 = jax.vjp(lambda v: _reconstruct(v, _weno_omega_weights_ad), q64)
    g64 = np.asarray(vjp64(jnp.ones(5))[0])
    h = 1e-6
    fd = np.array([(float(jnp.sum(_reconstruct(q64.at[i].add(h * abs(CASA_STENCIL[i])), _weno_omega_weights)))
                    - float(jnp.sum(_reconstruct(q64.at[i].add(-h * abs(CASA_STENCIL[i])), _weno_omega_weights))))
                   / (2 * h * abs(CASA_STENCIL[i])) for i in range(len(CASA_STENCIL))])
    np.testing.assert_allclose(g64, fd, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(g1, g64, rtol=2e-3, atol=2e-4)


def test_passive_scalar_advection_vjp_finite_with_huge_label():
    """End to end through ``advect_passive_scalars`` (float32): an unbounded
    label holding the Cas A spike gives a finite VJP, zero for a zero cotangent."""
    from test_fd_reverse_mode import _setup
    from astronomix import BACKWARDS
    from astronomix._fluid_equations._passive_scalars import advect_passive_scalars
    config, rv, state = _setup(differentiation_mode=BACKWARDS)      # the masked (static) sub-step loop
    state = state.astype(jnp.float32)
    i_ent = rv.passive_scalar_index + 5            # entropy_initial (first history scalar)
    lab = jnp.full(state.shape[1:], -19.6, jnp.float32)
    lab = lab.at[5:14, 8, 8].set(jnp.asarray(CASA_STENCIL, jnp.float32))
    state = state.at[i_ent].set(lab)
    rv_h = rv._replace(num_vars=rv.passive_scalar_index, passive_scalar_index=-1,
                       num_passive_scalars=0, passive_scalars_active=False, shock_history_active=False)
    hydro = state[:rv.passive_scalar_index]
    scalars = state[rv.passive_scalar_index:]

    def f(s, h):
        return advect_passive_scalars(s, h, jnp.float32(2e-3), config.grid_spacing, config, rv_h)
    out, vjp = jax.vjp(f, scalars, hydro)
    assert np.all(np.isfinite(np.asarray(out)))
    gs, gh = vjp(jnp.zeros_like(out))
    np.testing.assert_array_equal(np.asarray(gs), 0.0)
    np.testing.assert_array_equal(np.asarray(gh), 0.0)
    gs, gh = vjp(jnp.ones_like(out))
    assert np.all(np.isfinite(np.asarray(gs))) and np.all(np.isfinite(np.asarray(gh)))
