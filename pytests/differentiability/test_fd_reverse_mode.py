"""
Reverse-mode AD through the finite-difference solver in its Cas A configuration.

The configuration is the one ``casa_xfit`` differentiates: FD/WENO hydro, the
dual-energy formalism, five composition passive scalars with physical bounds,
the library's shock history, redistribute positivity with the cold-crush blend
and the cold-LLF tangent. The FCT positivity-preserving flux is off: on XLA:CPU
its unrolled bisection makes the compile impractically large, and its weight is
``stop_gradient``-ed, so it does not change the structure of the linearisation.

What is checked (x64, NATIVE_JAX, a 16^3 periodic blast, a few steps):

* ``jax.grad`` through ``time_integration`` works under ``BACKWARDS`` (equinox
  checkpointed loop + the static, masked passive-scalar sub-cycling), and the
  gradient reaches the composition scalars;
* the dot-product test ``<J^T w, u> == w . (J u)``: on one fixed-step function
  (JVP and VJP of the same code), and between the ``FORWARDS`` JVP (while loop,
  traced sub-step count) and the ``BACKWARDS`` VJP (adaptive steps);
* the forward DEFAULT path is unchanged, bit for bit: against a reference hash
  recorded with the library before the reverse-mode work (compared bitwise
  only on the platform / jax version / XLA flags it was recorded with; to within
  round-off otherwise), and, when ``ASTRONOMIX_BASELINE_ROOT`` points at a
  directory holding an older ``astronomix/``, directly against that library in
  a subprocess;
* the masked sub-cycling reproduces the dynamic one when the flow needs several
  sub-steps; ``ad_remat`` does not change the gradient;
* the bound clamps keep derivative 1 at their bounds (``jnp.clip`` gives 0.5
  there, which halved the ejecta-fraction sensitivity every step), and the
  smooth shock latch leaves the primal unchanged while giving the latch a
  derivative;
* ``finalize_config`` refuses cosmic rays under the FD solver, and
  ``OPTIMAL_BACKEND`` resolves to NATIVE_JAX when JAX runs on the CPU.

Run on the CPU (fast compile flags are applied automatically there)::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/differentiability/test_fd_reverse_mode.py
"""

# ==== GPU selection ====
import os
import sys

if os.environ.get("JAX_PLATFORMS", "") == "cpu":
    # XLA:CPU compile economy: the unrolled FD step takes minutes and tens of GB
    # to optimise at the default LLVM level, seconds at level 0 (numerics differ
    # at round-off only, which is why the reference hash records the flags)
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true")
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import hashlib
import json
import subprocess
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from astronomix import (
    BACKWARDS,
    CARTESIAN,
    FINITE_DIFFERENCE,
    FORWARDS,
    NATIVE_JAX,
    OPTIMAL_BACKEND,
    PERIODIC_BOUNDARY,
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import POSITIVITY_REDISTRIBUTE

GAMMA = 5.0 / 3.0
N = 16
N_SCALARS = 5
T_END = 0.012          # 4 CFL steps of the blast below
N_FIXED = 3
#: fixed-step tests: well inside the CFL limit (the first adaptive step is
#: 0.008), where the coarse ejecta edge stays benign (a 0.004 step already
#: crushes one cell to rho ~ 23 and the linearisation to |J| ~ 1e19)
DT_FIXED = 0.0025


# -----------------------------------------------------------------------------
# problem
# -----------------------------------------------------------------------------
def _config(**extra):
    """casa_xfit's solver configuration (minus the FCT flux) on a 16^3 box.

    Only fields that already existed before the reverse-mode work are set by
    default, so the same function builds the reference run on an old library.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    kw = dict(
        solver_mode=FINITE_DIFFERENCE, dimensionality=3, geometry=CARTESIAN,
        first_order_fallback=False, box_size=1.0, num_cells=N,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        positivity_config=PositivityConfig(
            per_stage_mode=POSITIVITY_REDISTRIBUTE, per_step_mode=POSITIVITY_REDISTRIBUTE,
            preserving_flux=False, coldcrush_blend=True, coldcrush_blend_factor=8.0,
            nan_safe=True, vacuum_rest=True),
        dual_energy=True, weno_ad_frozen_weights=True, ad_tangent_llf_cold_factor=1000.0,
        num_passive_scalars=N_SCALARS, track_shock_history=True,
        passive_scalar_bounds=tuple((0.0, 1.0) for _ in range(N_SCALARS)),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False, num_checkpoints=8,
    )
    kw.update(extra)
    return SimulationConfig(**kw)


def _params(t_end=T_END):
    return SimulationParams(gamma=GAMMA, C_cfl=0.3, t_end=t_end,
                            minimum_density=1e-6, minimum_pressure=1e-8,
                            minimum_specific_pressure=1e-4)


def _setup(**extra):
    """A homologously expanding ejecta ball (C_ej = 1, exactly on the bound)
    driving a Mach ~8 shock into a cold ambient medium (C_ej = 0, likewise)."""
    config = _config(**extra)
    rv = get_registered_variables(config)
    x = (jnp.arange(N) + 0.5) / N - 0.5
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    r = jnp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    r_ej = 0.2
    inside = r < r_ej
    rho = jnp.where(inside, 5.0, 1.0) + 0.05 * jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
    vr = jnp.where(inside, 0.8 * r / r_ej, 0.0)
    rs = jnp.maximum(r, 1e-12)
    p = jnp.where(inside, 5e-2, 1e-2) * (1.0 + 0.1 * jnp.cos(2 * jnp.pi * Z))
    c_ej = jnp.where(inside, 1.0, 0.0)
    shell = jnp.clip(r / r_ej, 0.0, 1.0)
    scalars = jnp.stack([
        c_ej,
        c_ej * 0.4 * (1.0 - shell) + 0.001,
        c_ej * 0.3 * shell + 0.0007,
        0.25 + 0.2 * jnp.sin(2 * jnp.pi * X),
        0.28 - 0.1 * c_ej,
    ])
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=rho,
        velocity_x=vr * X / rs, velocity_y=vr * Y / rs, velocity_z=vr * Z / rs,
        gas_pressure=p, gamma=GAMMA, passive_scalars=scalars)
    config = finalize_config(config, state.shape)
    return config, rv, state


def _forward_final_state():
    config, rv, state = _setup()
    return np.asarray(time_integration(state, config, _params(), rv))


def _fingerprint(a):
    return dict(sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                sums=[float(v) for v in a.reshape(a.shape[0], -1).sum(axis=1)])


def _environment():
    return dict(platform=jax.devices()[0].platform, jax=jax.__version__,
                xla_flags=os.environ.get("XLA_FLAGS", ""))


#: Recorded with the library as it was before the reverse-mode work
#: (/export/data/lstorcks/casa_orlando150/work/astro_snapshot_2026_09_25),
#: via ``python test_fd_reverse_mode.py --reference`` (JAX_PLATFORMS=cpu, the
#: fast-compile XLA flags above).
REFERENCE = {
    "sha256": "a6d24871ae4f93ddf72da955f366a9f0cec696d753f786ea91f544c1af71a221",
    "sums": [4640.0, -0.22193946977395385, 1.625701838204182e-14, -2.8321156455169555e-15,
             54.49837673018368, 81.7475650952755, 165.73703846774563, 19.973211189561994,
             40.985465269972735, 1024.0047148154458, 1130.2896742284643,
             -19048.705637273328, 120.79094916818133, 1.3655649179425509,
             1.771269036395911],
    "env": {"platform": "cpu", "jax": "0.10.2",
            "xla_flags": "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"},
}


def _direction(state, rv, seed, scale=1e-3):
    """A smooth relative perturbation of density, velocity, pressure and every
    passive scalar (the dual-energy slot is re-derived from p, so left alone)."""
    rng = np.random.default_rng(seed)
    k = 2 * np.pi * np.arange(N) / N
    d = np.zeros(state.shape)
    for i in range(state.shape[0]):
        if rv.internal_energy_active and i == rv.internal_energy_index:
            continue
        a, b, c, ph = rng.normal(size=4)
        d[i] = (np.sin(a + k)[:, None, None] * np.cos(b + 2 * k)[None, :, None]
                * np.sin(c + k)[None, None, :] + 0.3 * np.cos(ph + 3 * k)[None, None, :])
    d = jnp.asarray(d)
    typical = jnp.maximum(jnp.abs(state), 1e-2)
    return scale * d * typical


# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------
def test_bound_clamps_keep_derivative():
    from astronomix._fluid_equations._passive_scalars import (
        _clip_keep_derivative, _max_keep_derivative)
    x = jnp.array([-1.0, -0.0, 0.0, 0.3, 1.0, 1.0 + 1e-12, 2.0, jnp.nan])
    # the primal is jnp.clip / jnp.maximum, bit for bit (signed zeros and NaN too)
    np.testing.assert_array_equal(np.asarray(_clip_keep_derivative(x, 0.0, 1.0)).view(np.int64),
                                  np.asarray(jnp.clip(x, 0.0, 1.0)).view(np.int64))
    np.testing.assert_array_equal(np.asarray(_max_keep_derivative(x, 0.0)).view(np.int64),
                                  np.asarray(jnp.maximum(x, 0.0)).view(np.int64))
    g = jax.grad(lambda v: jnp.sum(_clip_keep_derivative(v, 0.0, 1.0)))(x)
    np.testing.assert_array_equal(np.asarray(g), [0, 1, 1, 1, 1, 0, 0, 1])
    g = jax.grad(lambda v: jnp.sum(_max_keep_derivative(v, 0.0)))(x)
    np.testing.assert_array_equal(np.asarray(g), [0, 1, 1, 1, 1, 1, 1, 1])
    # what the fix is for: jnp.clip halves the derivative on the bound
    assert float(jax.grad(lambda v: jnp.clip(v, 0.0, 1.0))(0.0)) == 0.5
    # the bound's own derivative goes where the value comes from
    g_lo = jax.grad(lambda lo: jnp.sum(_clip_keep_derivative(x, lo, 1.0)))(0.0)
    assert float(g_lo) == 1.0          # only x = -1 is strictly below


def test_advection_keeps_bound_derivative():
    """At rest the scalar advection is the identity, so its tangent must be the
    identity too -- including the cells holding exactly 0 or 1 (the old clamp
    returned 0.5 there, 0.5^n after n steps)."""
    from astronomix._fluid_equations._passive_scalars import advect_passive_scalars
    config, rv, state = _setup()
    rv_h = rv._replace(num_vars=rv.passive_scalar_index, passive_scalar_index=-1,
                       num_passive_scalars=0, passive_scalars_active=False,
                       shock_history_active=False)
    hydro = state[:rv.passive_scalar_index]
    for ax in ("x", "y", "z"):
        hydro = hydro.at[getattr(rv.velocity_index, ax)].set(0.0)
    scalars = state[rv.passive_scalar_index:]
    tangent = jnp.ones_like(scalars)

    def f(s):
        return advect_passive_scalars(s, hydro, 1e-3, config.grid_spacing, config, rv_h)
    out, dout = jax.jvp(f, (scalars,), (tangent,))
    on_bound = np.asarray((scalars[0] == 0.0) | (scalars[0] == 1.0))
    assert on_bound.mean() > 0.5
    np.testing.assert_allclose(np.asarray(dout[0])[on_bound], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(dout), 1.0, rtol=0, atol=1e-12)


def test_shock_history_clamps_keep_derivative():
    """A never-shocked parcel has time_since_shock = density_time = 0 exactly;
    their clamp at 0 must not halve the transported derivative."""
    from astronomix._fluid_equations._passive_scalars import update_shock_history
    config, rv, state = _setup()
    n_hist = 4
    hist = state[-n_hist:]
    rv_h = rv._replace(num_vars=rv.passive_scalar_index, passive_scalar_index=-1,
                       num_passive_scalars=0, passive_scalars_active=False,
                       shock_history_active=False)
    prim = state[:rv.passive_scalar_index]

    def f(h):
        return update_shock_history(h, prim, 1e-3, GAMMA, config.shock_entropy_jump, config, rv_h)
    t = jnp.zeros_like(hist).at[1:].set(1.0)
    out, dout = jax.jvp(f, (hist,), (t,))
    fresh = np.asarray(out[1]) == 1.0
    assert (~fresh).mean() > 0.5
    # d sf = d sf_old where the latch did not fire (0 where it did: overwritten)
    np.testing.assert_array_equal(np.asarray(dout[1])[~fresh], 1.0)
    # d tss = d tss_old + dt * d sf
    np.testing.assert_allclose(np.asarray(dout[2])[~fresh], 1.0 + 1e-3, rtol=1e-14)


def test_smooth_shock_latch():
    """Primal unchanged (bitwise); the latch gets a derivative near shocks."""
    from astronomix._fluid_equations._passive_scalars import update_shock_history
    config, rv, state = _setup()
    config_s = config._replace(ad_smooth_shock_latch=True)
    rv_h = rv._replace(num_vars=rv.passive_scalar_index, passive_scalar_index=-1,
                       num_passive_scalars=0, passive_scalars_active=False,
                       shock_history_active=False)
    # evolve two steps so that there is a real shock to latch on
    st = time_integration(state, config, _params(t_end=2 * DT_FIXED), rv)
    prim, hist = st[:rv.passive_scalar_index], st[-4:]
    ip = rv.pressure_index

    def f(cfg):
        def g(p):
            return update_shock_history(hist, prim.at[ip].set(p), 1e-3, GAMMA,
                                        cfg.shock_entropy_jump, cfg, rv_h)
        return g
    t = prim[ip] * 1e-2
    out_h, d_h = jax.jvp(f(config), (prim[ip],), (t,))
    out_s, d_s = jax.jvp(f(config_s), (prim[ip],), (t,))
    np.testing.assert_array_equal(np.asarray(out_h).view(np.int64), np.asarray(out_s).view(np.int64))
    assert float(jnp.abs(d_h[1]).max()) == 0.0            # boolean latch: no derivative
    assert float(jnp.abs(d_s[1]).max()) > 1e-4            # surrogate: sensitivity near the shock
    assert bool(jnp.all(jnp.isfinite(d_s)))
    # quiescent, never-shocked gas (initial ambient: entropy rise 0, at rest)
    # must not leak surrogate derivative -- it would accumulate every step in
    # the carried fraction (default width: 0.4 % of the at-threshold response)
    prim0, hist0 = state[:rv.passive_scalar_index], state[-4:]
    _, d0 = jax.jvp(lambda p: update_shock_history(
        hist0, prim0.at[ip].set(p), 1e-3, GAMMA, config_s.shock_entropy_jump, config_s, rv_h),
        (prim0[ip],), (prim0[ip] * 1e-2,))
    ambient = np.asarray(state[rv.passive_scalar_index] == 0.0)
    at_threshold = 0.5 * 0.25 / config_s.ad_shock_latch_entropy_width * 1e-2
    assert np.abs(np.asarray(d0[1]))[ambient].max() < 0.01 * at_threshold


def test_masked_substeps_reverse_vs_dynamic_jvp():
    """The masked loop's VJP with n_sub > 1 (sub-steps taken AND skipped) is the
    transpose of the dynamic loop's JVP, through the scalar advection and the
    smooth-latch shock history (whose custom_jvp is transposed here)."""
    from astronomix._fluid_equations._passive_scalars import (
        _substep_count, _velocity_components, advect_passive_scalars, update_shock_history)
    from astronomix.option_classes.simulation_config import SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED
    cfgs = {}
    for mode in (SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED):
        cfgs[mode], rv, state = _setup(passive_scalar_cfl=0.02, max_passive_scalar_substeps=6,
                                       passive_scalar_substep_loop=mode, ad_smooth_shock_latch=True)
    rv_h = rv._replace(num_vars=rv.passive_scalar_index, passive_scalar_index=-1,
                       num_passive_scalars=0, passive_scalars_active=False,
                       shock_history_active=False)
    i0 = rv.passive_scalar_index
    prim, sc, dt = state[:i0], state[i0:], 4e-3
    c = cfgs[SUBSTEPS_DYNAMIC]
    n_sub = int(_substep_count(_velocity_components(prim, c, rv_h), c.grid_spacing, dt, c))
    assert 1 < n_sub < 6

    def F(cfg):
        def f(p, s, dt_):
            s2 = advect_passive_scalars(s, p, dt_, cfg.grid_spacing, cfg, rv_h)
            h = update_shock_history(s2[-4:], p, dt_, GAMMA, cfg.shock_entropy_jump, cfg, rv_h)
            return jnp.concatenate([s2[:-4], h], axis=0)
        return f
    rng = np.random.default_rng(0)
    up = _direction(state, rv, 5)[:i0]
    us = jnp.asarray(1e-3 * rng.normal(size=sc.shape))
    udt = 1e-4
    w = jnp.asarray(rng.normal(size=sc.shape))
    out_d, Ju = jax.jvp(F(cfgs[SUBSTEPS_DYNAMIC]), (prim, sc, dt), (up, us, udt))
    out_m, vjp = jax.vjp(F(cfgs[SUBSTEPS_MASKED]), prim, sc, dt)
    gp, gs, gdt = vjp(w)
    np.testing.assert_allclose(np.asarray(out_m), np.asarray(out_d), rtol=0,
                               atol=1e-13 * float(jnp.abs(out_d).max()))
    lhs = float(jnp.vdot(gp, up) + jnp.vdot(gs, us) + gdt * udt)
    rhs = float(jnp.vdot(w, Ju))
    assert abs(lhs - rhs) <= 1e-12 * abs(rhs), (lhs, rhs)


def test_forward_default_unchanged_vs_reference():
    """The default forward path (FORWARDS, dynamic sub-steps, no remat, hard
    latch) reproduces the pre-change library."""
    if REFERENCE is None:
        pytest.skip("no reference recorded (python test_fd_reverse_mode.py --reference)")
    a = _forward_final_state()
    fp = _fingerprint(a)
    same_env = all(REFERENCE["env"][k] == v for k, v in _environment().items())
    if same_env:
        assert fp["sha256"] == REFERENCE["sha256"]
    else:
        np.testing.assert_allclose(fp["sums"], REFERENCE["sums"], rtol=1e-9, atol=1e-12)


def test_forward_default_unchanged_vs_baseline_library():
    root = os.environ.get("ASTRONOMIX_BASELINE_ROOT")
    if not root:
        pytest.skip("set ASTRONOMIX_BASELINE_ROOT to a directory containing a baseline astronomix/")
    here = os.path.abspath(__file__)
    code = textwrap.dedent(f"""
        import importlib.util, json
        spec = importlib.util.spec_from_file_location("t", {here!r})
        t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
        print("FP" + json.dumps(t._fingerprint(t._forward_final_state())))
    """)
    env = dict(os.environ, PYTHONPATH=root)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout
    base = json.loads(out.split("FP", 1)[1].splitlines()[0])
    assert _fingerprint(_forward_final_state())["sha256"] == base["sha256"]


def test_masked_substeps_match_dynamic():
    """Force several sub-steps (small passive_scalar_cfl) and compare the
    reverse-differentiable masked loop against the traced-count one."""
    from astronomix.option_classes.simulation_config import SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED
    out = {}
    for mode in (SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED):
        config, rv, state = _setup(passive_scalar_cfl=0.02, max_passive_scalar_substeps=6,
                                   passive_scalar_substep_loop=mode)
        out[mode] = np.asarray(time_integration(state, config, _params(), rv))
    # the sub-cycling really happened: the result differs from a 1-sub-step run
    config, rv, state = _setup(max_passive_scalar_substeps=1)
    one = np.asarray(time_integration(state, config, _params(), rv))
    i0 = rv.passive_scalar_index
    assert np.abs(one[i0:] - out[SUBSTEPS_DYNAMIC][i0:]).max() > 1e-8
    # same arithmetic, compiled in different contexts: round-off apart
    for k in range(i0, one.shape[0]):
        a, b = out[SUBSTEPS_MASKED][k], out[SUBSTEPS_DYNAMIC][k]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13 * max(np.abs(b).max(), 1e-30))
    np.testing.assert_array_equal(out[SUBSTEPS_MASKED][:rv.passive_scalar_index],
                                  out[SUBSTEPS_DYNAMIC][:rv.passive_scalar_index])


def test_grad_backwards_and_dot_product_fixed_step():
    """JVP and VJP of ONE function (fixed steps, BACKWARDS config => masked
    sub-steps, scan time loop; stage rematerialisation on the reverse side,
    which the JVP does not see): <J^T w, u> == w . J u to round-off."""
    config, rv, state = _setup(fixed_timestep=True, num_timesteps=N_FIXED,
                               differentiation_mode=BACKWARDS, ad_remat="stage")
    par = _params(t_end=N_FIXED * DT_FIXED)

    def F(s):
        return time_integration(s, config, par, rv)
    u = _direction(state, rv, 1)
    w = jnp.asarray(np.random.default_rng(2).normal(size=state.shape))

    @jax.jit
    def both(s, d, c):
        return jax.jvp(F, (s,), (d,))[1], jax.vjp(F, s)[1](c)[0]
    Ju, JTw = both(state, u, w)
    assert bool(jnp.all(jnp.isfinite(JTw))) and bool(jnp.all(jnp.isfinite(Ju)))
    lhs, rhs = float(jnp.vdot(JTw, u)), float(jnp.vdot(w, Ju))
    assert abs(lhs - rhs) <= 1e-11 * max(abs(lhs), abs(rhs)), (lhs, rhs)
    # the gradient reaches the composition scalars and the history
    i0 = rv.passive_scalar_index
    assert float(jnp.abs(JTw[i0]).max()) > 0.0       # C_ej, mostly sitting on its bounds
    assert float(jnp.abs(JTw[-1]).max()) > 0.0       # density_time


def test_grad_adaptive_backwards_vs_forward_jvp():
    """jax.grad through the adaptive BACKWARDS loop (equinox checkpoints +
    masked sub-steps) against the FORWARDS JVP (while loop, traced count)."""
    config_b, rv, state = _setup(differentiation_mode=BACKWARDS)
    config_f, _, _ = _setup()
    par = _params()
    w = jnp.asarray(np.random.default_rng(3).normal(size=state.shape))

    def loss(cfg):
        return lambda s: jnp.vdot(w, time_integration(s, cfg, par, rv))
    val_b, g = jax.jit(jax.value_and_grad(loss(config_b)))(state)
    assert bool(jnp.all(jnp.isfinite(g)))
    i0 = rv.passive_scalar_index
    assert float(jnp.abs(g[i0]).max()) > 0.0
    for seed in (11, 12):
        u = _direction(state, rv, seed)
        val_f, jv = jax.jit(lambda s, d: jax.jvp(loss(config_f), (s,), (d,)))(state, u)
        vj = float(jnp.vdot(g, u))
        assert abs(float(val_f) - float(val_b)) <= 1e-12 * abs(float(val_f))
        assert abs(vj - float(jv)) <= 1e-10 * abs(float(jv)), (vj, float(jv))


def test_remat_same_gradient():
    """ad_remat changes what the backward stores, not what it computes ("stage"
    is covered by the dot-product test above; "axis" adds the per-axis and
    per-scalar checkpoints)."""
    grads, fwd = {}, {}
    # ("axis", 4): ad_remat_chunks -- each axis' increment as 4 checkpointed z / y slabs
    for remat, chunks in (("none", 1), ("axis", 1), ("axis", 4)):
        config, rv, state = _setup(fixed_timestep=True, num_timesteps=1,
                                   differentiation_mode=BACKWARDS, ad_remat=remat, ad_remat_chunks=chunks)
        w = jnp.asarray(np.random.default_rng(4).normal(size=state.shape))
        par = _params(t_end=DT_FIXED)
        key = remat if chunks == 1 else f"{remat}{chunks}"
        grads[key] = np.asarray(jax.jit(jax.grad(
            lambda s: jnp.vdot(w, time_integration(s, config, par, rv))))(state))
        fwd[key] = np.asarray(jax.jit(lambda s: time_integration(s, config, par, rv))(state))
    scale = np.abs(grads["none"]).max()
    np.testing.assert_allclose(grads["axis"], grads["none"], rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(grads["axis4"], grads["axis"], rtol=0, atol=1e-12 * scale)
    np.testing.assert_array_equal(fwd["axis4"], fwd["axis"])


def test_scalar_lean_same_gradient():
    """ad_scalar_lean (the passive-scalar block's reverse-mode memory: an
    equinox loop over the flow-derived sub-step count, checkpointed ratio
    recovery and shock history) changes what the backward stores, not what it
    computes -- with sub-cycling active (n_sub > 1) and on the adaptive,
    equinox-checkpointed path."""
    for fixed in (True, False):
        grads, fwd = {}, {}
        for lean in (False, True):
            extra = dict(fixed_timestep=True, num_timesteps=1) if fixed else {}
            config, rv, state = _setup(differentiation_mode=BACKWARDS, ad_remat="axis", ad_scalar_lean=lean,
                                       passive_scalar_cfl=0.02, max_passive_scalar_substeps=6, **extra)
            w = jnp.asarray(np.random.default_rng(5).normal(size=state.shape))
            par = _params(t_end=DT_FIXED if fixed else T_END)
            grads[lean] = np.asarray(jax.jit(jax.grad(
                lambda s: jnp.vdot(w, time_integration(s, config, par, rv))))(state))
            fwd[lean] = np.asarray(jax.jit(lambda s: time_integration(s, config, par, rv))(state))
        scale = np.abs(grads[False]).max()
        assert np.all(np.isfinite(grads[True]))
        np.testing.assert_allclose(grads[True], grads[False], rtol=0, atol=1e-11 * scale)
        np.testing.assert_allclose(fwd[True], fwd[False], rtol=1e-13, atol=0)


def test_config_guards():
    from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
    with pytest.raises(ValueError, match="cosmic"):
        finalize_config(_config(cosmic_ray_config=CosmicRayConfig(cosmic_rays=True)), (15, N, N, N))
    with pytest.raises(ValueError, match="ad_remat"):
        finalize_config(_config(ad_remat="everything"), (15, N, N, N))
    if jax.devices()[0].platform == "cpu":
        cfg = finalize_config(_config(backend_config=BackendConfig(backend=OPTIMAL_BACKEND)),
                              (15, N, N, N))
        assert cfg.backend_config.backend == NATIVE_JAX


if __name__ == "__main__":
    if "--reference" in sys.argv:
        a = _forward_final_state()
        print(json.dumps(dict(_fingerprint(a), env=_environment()), indent=1))
