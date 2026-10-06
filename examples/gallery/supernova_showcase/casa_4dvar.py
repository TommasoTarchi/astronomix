"""
Full-state 4D-Var of Cas A at the reference epoch 2000 (strong constraint,
observer frame): find the 2000 state x0 and the global observation parameters
whose evolution through 2000-2018 best reproduces the Chandra data, and score
the prediction of the held-out epochs 2019 and 2022.

Background: a ``casa_xfit --save-state`` file (the fit's evolved state at the
first epoch, 2000.08, with composition and shock history) and the theta it
carries (distance, sky roll / offset, emission, N_H map, synchrotron). Control
(``casa_4dvar_control``): the half-resolution whitened field chi of (delta ln
rho, delta v / v_ref, delta ln p) through a masked, Gaussian-smoothed B^(1/2);
a low-dimensional ln-density modulation of the unshocked CSM (Y_lm, l <= 2, +
a radial slope); the free globals (preconditioned by the xfit Gauss-Newton
scale, penalised by the xfit priors). Composition and shock history are carried.
With ``--warp`` (stage 4, ``casa_4dvar_warp``): a forward-shock displacement
control, a differentiable radial warp of the outer state (shocked shell + FS +
the CSM just ahead) by delta(n) = sum_{l <= 8} a_lm Y_lm (81 whitened
coefficients, pointwise prior rms ``--warp-sigma-arcsec`` 5" at the background
distance), applied to the background BEFORE the smooth increment; J gains
0.5 |xi_warp|^2 ('b_warp').

    J(z) = 0.5 |chi|^2 + 0.5 |xi_csm|^2 + 0.5 sum_k ((g_k - mu_k) / sigma_k)^2
           + 0.5 [wind_nh prior]^2 + 0.5 |r_train(x0, g)|^2

with r_train = casa_xfit.residual_parts (outline, proper motions, 2004 Doppler,
static / temporal images, static / temporal spectra) on the TRAINING epochs of
the current window (held out: 2019, 2022). The forward model is casa_xfit's
own (``make_forward_core``: the same solver config, observation chain and
output assembly), started from x0: a ``lax.scan`` over the epoch segments,
each segment (integrate to the next epoch, observe) under ``jax.checkpoint``,
the solver in ``differentiation_mode=BACKWARDS`` with ``ad_remat`` and a fixed
equinox checkpoint budget.

Tangents (``--tangent``): exact (weno_ad_frozen_weights off, cold-LLF tangent
factor 0) matches FD over <= 5 yr and explodes beyond (growth experiment
stage1/tlm/growth: x8e3 at 11 yr, x1e13 at 22 yr at 64^3); the approximations
(frozen WENO weights, cold-LLF 1000) stay bounded (growth 0.7-1.0) and agree
with FD at the L-BFGS step scale (h ~ 0.1). ``auto``: exact for windows <= 5.5
yr, else approximate. The primal is the same in both.

Modes:

    --taylor    Taylor / FD check of J along B^(1/2)-shaped directions (and the
                globals), with time and memory per gradient (test (a), (b))
    --run       L-BFGS-B (scipy, float64 host vectors) over the quasi-static
                window schedule ``--windows`` (e.g. 2004.5 2009.9 2014.5 2018.5),
                checkpoint every iteration, ``--resume``; validation (all-epoch
                forward: training and held-out chi2) at every stage boundary
    --eval      the validation forward only
    --policy-test plain confirm smooth
                the stage-3 robustness comparison: the same window / start
                point / gradient budget under several optimisation policies,
                each end point scored by J at fresh tiny jitters
    --wc        (with --run / --taylor) weak-constraint multiple shooting
                (``casa_4dvar_wc``): state controls at --wc-bounds, a Q-norm
                model-error penalty at each boundary, shared globals

Robust optimisation (``casa_4dvar_robust``, stage 3): J is bitwise
reproducible but rough at the ulp level (discrete solver paths near the
reverse shock), so plain L-BFGS keeps lucky low outliers. --confirm (a new best
must survive re-evaluation at z +- 1e-6 jitters; restarts begin at a jittered
neighbour), --min-step (stop line searches that probe ulp-sized steps),
--smooth-k K --smooth-sigma s (L-BFGS on the mean of K antithetic
common-random-number jitters of the model input; K x the cost). Since stage 4
the stage-3 robust flags are the DEFAULTS (``casa_4dvar_robust.ROBUST_DEFAULTS``:
--confirm, --min-step 1e-6, --spike-factor 5, --spike-sigma 1e-4,
--stall-restart, --nan-retries 4, --fresh-n 2); --plain-policy restores the
stage-2 behaviour (policy 'plain'), --no-confirm / --no-stall-restart / 0 values
switch single ones off.

    ./run.sh casa_4dvar.py --state W/stage2/state2000_Rstart_n128.npz --taylor --windows 2004.5
    ./run.sh casa_4dvar.py --state ... --run --windows 2004.5 2009.9 2014.5 2018.5 --iters 25 25 25 25 \
        --out-dir W/stage2/4dvar/run_A [--resume]

Stage-4 production (forward-shock warp; robust policy = default; clean hold-out):

    ./run.sh casa_4dvar.py --state W/state2000_Rp_n128.npz --jac W/xfit_Rp_jac.npz --pm-train-only \
        --holdout 2019 2022 --warp --run --windows 2018.5 --iters 15 \
        --init-z W/stage3/oos/run_Rp/z_stage3.npz --out-dir W/stage4/4dvar/run_warp --resume
    (from scratch: --windows 2004.5 2009.9 2014.5 2018.5 --iters 15 25 25 35 without --init-z;
     add --stage3 to score with the stage-3 likelihood that produced R' and run_Rp)

Multi-GPU (``--gpus N``, casa_xfit_shard; 2026-09-27): the state, the
background, the control transform and the observation model are split into
x-slabs over N GPUs of ONE node (the FD Pallas kernels as shard_map + halo
ppermutes, the B^1/2 FFT as an all-to-all transpose, the cone profiles as
shard-local segment sums; the background and the 3.2 GB observation tables are
jit arguments, not HLO constants). Strong-constraint, no-warp control only.
Validated at 128^3 against 1 GPU (J to 1.3e-5, gradient cosine 0.99999999;
``casa_4dvar_shard.py``). Run with NCCL_NVLS_ENABLE=0 on compgpu12:

    pq sub -t h200 -n 8 -- env NCCL_NVLS_ENABLE=0 ./run.sh casa_4dvar.py --gpus 8 \
        --state W/ers/shard/state2000_R2_n512.npz --jac W/xfit_R2_jac.npz ... --run ...

512^3 needs the H200 node: one gradient holds ~69 GB XLA temp + 5.5 GB arguments
per device at --ckpt 16 (8 x A100-40GB: out of memory; --ckpt 4 --remat-chunks 8
still ~51 GB). Per device, ~0.94 GB per equinox checkpoint (two copies live: equinox
sorts its checkpoint buffer), --remat-chunks N (checkpointed perpendicular slabs of
each axis' flux; exact) saves ~8 GB. On more than one GPU the Validator scores the
held-out epochs on replicated model outputs (else its compile ran > 1 h). Timings,
memory tables and the production command: /export/data/lstorcks/casa_orlando150/
work/ers/shard/REPORT.md.

Optimiser on the device (``--optimizer jax``, the default since 2026-10-05, after
compgpu12 went down under the 448^3 run): ``casa_4dvar_lbfgs`` keeps z, g, the
search direction and the L-BFGS history on the GPU(s), sharded ``P("x")`` like
the control (the objective takes z committed with that sharding and returns g
constrained to it), and the robust policies run on host scalars; the only
n-vector that reaches the host is z_best for ckpt.npz (same format, so scipy-era
checkpoints resume). ``--optimizer scipy``: the old float64 host L-BFGS-B.
``--lift-constants on`` (default): on one GPU the background, masks, CSM basis,
wind weight and the v2 tables are jit arguments too (a mesh always lifts them),
not ~2.6 GB of HLO literals per executable at 128^3; ``--const-audit`` prints
what a window's J still captures; ``--sharding-check`` compares J / g of the
replicated (scipy) and the sharded (jax) control input. Host RSS: logged at
setup, per evaluation (evals.jsonl host_rss_GB) and per stage, capped by
CASA_4DVAR_MAX_HOST_GB (casa_4dvar_robust). Tests:
``test_casa_4dvar_lbfgs.py`` (CPU) and /export/data/lstorcks/casa_orlando150/
work/ers/gpu_lbfgs/REPORT.md.

CPU smoke test: ``JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= ... --cpu-test --coarsen 32 --x64``
(NATIVE_JAX backend, no FCT flux limiter -- its XLA:CPU compile explodes).
"""

# ==== GPU selection ====
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and __name__ == "__main__":
    from autocvd import autocvd
    # --gpus N: the state, the background and the observation model split into
    # x-slabs over N GPUs of the node (casa_xfit_shard)
    autocvd(num_gpus=int(sys.argv[sys.argv.index("--gpus") + 1]) if "--gpus" in sys.argv else 1)
if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
# ruff: noqa: E402
# =======================

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

WORK = Path("/export/data/lstorcks/casa_orlando150/work")
CACHE = WORK / "stage2" / "4dvar" / "jaxcache"
jax.config.update("jax_compilation_cache_dir", str(CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 30.0)

from astronomix import BACKWARDS
import casa_pluto_diff as PD
import casa_xfit as X
import casa_xfit_shard as SH
import casa_xfit_state as XS
import casa_4dvar_control as C
import casa_4dvar_data as DA
import casa_4dvar_robust as RB
import casa_4dvar_warp as WP
import casa_4dvar_wc as WC

#: global (observer-frame / emission) parameters the 4D-Var controls by default;
#: the hydro / IC parameters of casa_xfit have no meaning at the reference epoch
GLOBALS = ("ln_D", "psi", "dw", "dn", "ln_A", "ln_nh", "ln_sync", "g_nh_w", "g_nh_n",
           "ln_kte", "ln_teq", "ln_fe", "lg_fmass", "ln_si")
#: the likelihood terms of casa_xfit.residual_parts that are constants here
#: (priors of the IC parameters) and are replaced by 'prior_glob'
DROP_PARTS = ("prior", "prior_extra", "conv_wall")
TANGENTS = {"exact": dict(weno_ad_frozen_weights=False, ad_tangent_llf_cold_factor=0.0),
            "approx": dict(weno_ad_frozen_weights=True, ad_tangent_llf_cold_factor=1000.0),
            # exact WENO-weight derivative, but the cold-knot flux derivative through LLF (stage 3)
            "semi": dict(weno_ad_frozen_weights=False, ad_tangent_llf_cold_factor=1000.0)}


# =============================================================================
# ============ ↓ Setup ↓ ======================================================
# =============================================================================
def likelihood_args(a, opts):
    """casa_xfit's likelihood settings (its CLI defaults)."""
    # + the multi-scale image terms (--fine-block; casa_xfit.fine_image_residuals; default off)
    return SimpleNamespace(sigma_model=a.sigma_model, doppler=True, spectra=True, img_weight=1.0 / 6.0,
                           sigma_static=0.5, sigma_temporal=0.075, sigma_spec_static=0.13,
                           sigma_spec_temporal=0.046, opts=opts, **DA.fine_likelihood_args(a))


def load_background(path, coarsen=None):
    """(fields, meta, layout, theta dict) of a casa_xfit --save-state file."""
    fields, meta = XS.load_state(path)
    # safety net: a non-finite / runaway shock-history label (pre-2026-10-04
    # backgrounds) makes every gradient NaN; re-seed it from the current entropy
    fields, n_reset = C.sanitize_entropy_label(fields, float(meta.get("gamma", 5.0 / 3.0)))
    if n_reset:
        print(f"[4dvar] entropy_initial: re-seeded {n_reset} non-finite/runaway cells "
              f"({n_reset / fields['rho'].size:.3%}) from the current entropy", flush=True)
    lay = meta["var_layout"]
    if coarsen and coarsen != fields["rho"].shape[0]:
        fields = C.block_average_state(fields, coarsen)
        meta = dict(meta, num_cells=np.int64(coarsen))
    th = dict(zip([str(s) for s in meta["names"]], np.asarray(meta["theta"], np.float64)))
    for k in X.PARAM_NAMES:
        th.setdefault(k, X.PRIOR[k][0])
    return fields, meta, lay, th


def globals_scale(names, jac_path=None):
    """Preconditioning scale per free global: the Gauss-Newton posterior width
    1 / sqrt(diag J^T J) of an xfit Jacobian dump (``--jac-dump``), else the
    prior width / 10."""
    s = {k: X.PRIOR[k][1] / 10.0 for k in names}
    if jac_path and Path(jac_path).exists():
        d = np.load(jac_path)
        Jm = np.asarray(d["J"], np.float64)
        h = np.sum(Jm ** 2, 0)
        for k in names:
            i = X.PARAM_NAMES.index(k)
            if h[i] > 0:
                s[k] = float(1.0 / np.sqrt(h[i]))
    return s


def solver_overrides(a, tangent):
    ov = dict(differentiation_mode=BACKWARDS, ad_remat=a.remat, num_checkpoints=a.ckpt,
              ad_remat_chunks=int(getattr(a, "remat_chunks", 1) or 1),
              ad_smooth_shock_latch=bool(a.smooth_latch), **TANGENTS[tangent])
    if a.cpu_test:
        from astronomix.option_classes.simulation_config import BackendConfig, NATIVE_JAX
        ov.update(backend_config=BackendConfig(backend=NATIVE_JAX),
                  positivity_config=X.fd_positivity(mode=X.POSITIVITY_REDISTRIBUTE)._replace(
                      preserving_flux=False))
    if getattr(a, "positivity", "redistribute") != "redistribute":
        # the 448^3 backgrounds need the mass-conserving mode (jetdbg 2026-10-03: REDISTRIBUTE
        # refills sub-floor cells without debiting the donors -> single-cell runaway); same
        # override as casa_xfit --positivity, the default leaves the solver bitwise unchanged
        from astronomix.option_classes.simulation_config import POSITIVITY_CONSERVATIVE
        pc = ov.get("positivity_config", X.fd_positivity(mode=POSITIVITY_CONSERVATIVE))
        ov["positivity_config"] = pc._replace(per_stage_mode=POSITIVITY_CONSERVATIVE,
                                              per_step_mode=POSITIVITY_CONSERVATIVE)
    return ov


class ArgLifted(SH.Lifted):
    """``casa_xfit_shard.Lifted`` that passes the registered arrays as jit
    ARGUMENTS on one device too (``--lift-constants on``): closed-over arrays
    are embedded in the HLO as literals -- host copies in every executable
    (vg, val, the validator) and in the persistent-cache blobs (1-GPU 128^3:
    3.4 GB of captured constants, mostly the v2 observation tables). With an
    active mesh it is the plain ``Lifted``."""

    def add(self, holder, key, spec=None, dtype=None):
        if SH.active():
            return super().add(holder, key, spec, dtype)
        v = self._get(holder, key)
        if not isinstance(v, jax.Array):
            v = jnp.asarray(np.asarray(v), dtype) if dtype is not None else jnp.asarray(np.asarray(v))
            self._set(holder, key, v)
        self.items.append((holder, key))
        return self

    def jit(self, fn, transform=None, **jit_kw):
        if SH.active():
            return super().jit(fn, transform, **jit_kw)
        transform = transform or (lambda f: f)

        def inner(*args):
            *a, lifted = args
            with self.bound(lifted):
                return fn(*a)
        jf = jax.jit(transform(inner), **jit_kw)

        def call(*args):
            return jf(*args, self.values())
        call.lower = lambda *args: jf.lower(*args, self.values())
        return call


def lift_v2_tables(lifted, img, min_bytes=8e6):
    """Register the big v2 observation tables (``img['v2']``, read at trace
    time; the same dict object in every epoch subset) with ``lifted`` -- what
    ``core.lift`` does on a mesh."""
    V = img.get("v2") if isinstance(img, dict) else None
    if not V:
        return lifted
    for k, v in list(V.items()):
        if getattr(v, "ndim", 0) > 0 and getattr(v, "nbytes", 0) > min_bytes:
            lifted.add(V, k)
    return lifted


def const_audit(win, top=12):
    """The closed-over arrays of the window's J (jaxpr constants = what the
    executables embed as HLO literals): total GB and the largest ones."""
    lifted = win.lifted
    vals = lifted.values()

    def inner(z, dz, vs):
        with lifted.bound(vs):
            return win.objective(z, dz)[0]
    zz = jax.ShapeDtypeStruct((win.ctrl.size,), win.dtype)
    with SH.context():
        cj = jax.make_jaxpr(inner)(zz, zz, vals)
    cs = [(tuple(getattr(c, "shape", ())), str(getattr(c, "dtype", type(c).__name__)),
           int(getattr(c, "nbytes", 0))) for c in cj.consts]
    cs.sort(key=lambda t: -t[2])
    tot = sum(c[2] for c in cs)
    out = dict(n_consts=len(cs), const_GB=tot / 1e9, lifted_GB=lifted.nbytes() / 1e9,
               n_lifted=len(lifted.items), top=[dict(shape=c[0], dtype=c[1], MB=c[2] / 1e6) for c in cs[:top]])
    print(f"[const-audit] {len(cs)} captured constants, {tot / 1e9:.3f} GB (lifted as arguments: "
          f"{len(lifted.items)} arrays, {out['lifted_GB']:.3f} GB); largest: "
          + ", ".join(f"{c[1]}{list(c[0])} {c[2] / 1e6:.0f} MB" for c in cs[:top]), flush=True)
    return out


def vec_sharding(n):
    """The device L-BFGS vectors' (and the objective input's) sharding: ``P("x")``
    over the --gpus mesh if n divides, replicated on the mesh otherwise, None on one device."""
    if not SH.active():
        return None
    from jax.sharding import PartitionSpec as P
    return SH.sharding(P("x") if n % SH.NDEV == 0 else P())


def make_evaluator(a, win, dtype, policy):
    """``casa_4dvar_robust.Evaluator`` for ``a.optimizer`` (jax: device mode, the
    vectors sharded like the objective input)."""
    ops = None
    if getattr(a, "optimizer", "scipy") == "jax":
        import casa_4dvar_lbfgs as LB
        ops = LB.VecOps(win.size, dtype, sharding=vec_sharding(win.size))
    return RB.Evaluator(win.vg, win.val, win.size, win.rough, dtype, policy, ops=ops)


def host(z):
    """A (device) control vector as a float64 host array."""
    return np.asarray(jax.device_get(z) if isinstance(z, jax.Array) else z, np.float64)


def load_ckpt(path):
    """``(z float64, stage, it, hist)`` of a ``save_ckpt`` checkpoint (both optimisers write the same format)."""
    d = np.load(path)
    return np.asarray(d["z"], np.float64), int(d["stage"]), int(d["it"]), json.loads(str(d["hist"]))


def sharding_check(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype, z0):
    """(J, g) of the first window at z0: the old call (z host -> replicated input, the gradient
    as XLA places it) vs the device optimiser's (z committed with ``vec_sharding``, the gradient
    constrained to it). Data movement only: the difference is the partitioner's reduction order."""
    tr, _ = DA.split_epochs(obs, t_end=a.windows[0], holdout=a.holdout)
    tangent = tangent_for(a, t0_year, float(np.asarray(obs["years"])[tr].max()))
    res = {}
    for mode in ("scipy", "jax"):
        win = Window(a, opts, meta, ctrl, theta_b, obs, img, tr, tangent=tangent, t0_year=t0_year, dtype=dtype)
        win.compile(vec_sharding(ctrl.size) if mode == "jax" else None)
        aa = SimpleNamespace(**dict(vars(a), optimizer=mode))
        ev = make_evaluator(aa, win, dtype, RB.Policy())
        t0 = time.time()
        if mode == "jax":
            J, chi2, g = ev.raw_vg_dev(ev.ops.put(z0))
            sh = str(g.sharding)
        else:
            J, chi2, g = ev.raw_vg(z0)
            sh = "host"
        res[mode] = (J, host(g))
        print(f"[sharding-check] {mode}: J {J:.6f}, |g| {np.linalg.norm(res[mode][1]):.6g}, g sharding {sh} "
              f"({time.time() - t0:.0f} s incl. compile)", flush=True)
        del win, ev, g
        jax.clear_caches()
    (J1, g1), (J2, g2) = res["scipy"], res["jax"]
    rel = float(np.linalg.norm(g2 - g1) / max(np.linalg.norm(g1), 1e-30))
    cos = float(g1 @ g2 / max(np.linalg.norm(g1) * np.linalg.norm(g2), 1e-30))
    print(f"[sharding-check] J_rel {abs(J2 - J1) / max(abs(J1), 1e-30):.3g}, grad rel_l2 {rel:.3g}, cos {cos:.12f}, "
          f"bitwise {'yes' if J1 == J2 and np.array_equal(g1, g2) else 'no'}", flush=True)
    return res


def log_rss(where):
    rss = RB.host_rss_gb()
    print(f"[host-rss] {where}: {rss:.2f} GB", flush=True)
    return rss


def tangent_for(a, t0_year, t_end):
    if a.tangent != "auto":
        return a.tangent
    return "exact" if (t_end - t0_year) <= a.exact_max_years else "approx"


class Window:
    """J and its gradient for one epoch set (a training window, or all epochs
    for validation): the casa_xfit forward core started from x0."""

    def __init__(self, a, opts, meta, ctrl, theta_b, obs, img, idx, *, tangent, t0_year, dtype):
        self.idx = np.asarray(idx)
        self.obs, self.img = DA.subset(obs, img, idx)
        self.labels = list(self.obs["epochs"])
        self.tangent = tangent
        ic = dict(box=float(meta["box"]), num_cells=int(meta["num_cells"]), age=float(meta["age"]))
        self.core = X.make_forward_core(ic, self.obs, self.img, opts=opts, ic_path=a.state,
                                        config_overrides=solver_overrides(a, tangent))
        self.ctrl, self.theta_b, self.dtype = ctrl, theta_b, dtype
        self.largs = likelihood_args(a, opts)
        core = self.core
        ys = np.sort(np.asarray(self.obs["years"], np.float64))
        dts = np.diff(np.concatenate([[t0_year], ys])) * core.yr
        if dts[0] < -1e-6 * core.yr:
            raise ValueError(f"first epoch {ys[0]} before the state's epoch {t0_year}")
        self.first_at_x0 = abs(dts[0]) < 1e-6 * core.yr
        self.t0_year = float(t0_year)
        self.dts = jnp.asarray(dts, dtype)
        # the Lee+14 pre-shock wind prior on the controlled CSM: sphere-mean n_H
        # at WIND_NH_PRIOR[0] pc of the never-shocked gas
        r = np.asarray(core.geom[0], np.float64)
        dx = core.box / core.n
        w = np.exp(-0.5 * ((r - PD.WIND_NH_PRIOR[0]) / dx) ** 2) * ctrl.u_csm_np
        self.wind_w = SH.put(w / max(w.sum(), 1e-30), SH.FIELD, dtype)
        # multi-GPU: the big arrays the traced J reads (background, masks, CSM basis,
        # wind weight, cone maps) become sharded jit arguments (casa_xfit_shard.Lifted);
        # --lift-constants on: on ONE device too (ArgLifted), plus the v2 observation tables
        # (3.2 GB), so the executables do not embed them as HLO literals (host RAM)
        lift1 = getattr(a, "lift_constants", "off") == "on" and not SH.active()
        self.lifted = ArgLifted() if lift1 else SH.Lifted()
        if SH.active() or lift1:
            ctrl.lift(self.lifted)
            self.lifted.add(self, "wind_w", SH.FIELD)
            core.lift(self.lifted)
        if lift1:
            lift_v2_tables(self.lifted, self.img)
        self.nh_per_rho = PD.csm_hydrogen_fraction(meta) * core.rho_c / PD.M_H_G
        self.wind_prior = bool(opts.wind_prior)
        self.wind_nh_mu, self.wind_nh_sd = PD.WIND_NH_PRIOR[1], PD.WIND_NH_PRIOR[2]
        self.prior = X.PRIOR_LEGACY if opts.priors == "legacy" else X.PRIOR
        self.drop_parts = DROP_PARTS

    def params(self, g):
        return {k: g[k] if k in g else jnp.asarray(self.theta_b[k], self.dtype) for k in X.PARAM_NAMES}

    def forward(self, x0, p):
        core = self.core
        integrate = core.integrator(x0.shape)
        xs = core.xs_all

        @jax.checkpoint
        def seg(st, dt, ep, pp):
            st = integrate(st, dt)
            return st, core.observer(pp)(st, ep)

        def body(st, xs_):
            dt, ep = xs_
            return seg(st, dt, ep, p)

        if self.first_at_x0:
            first = core.observer(p)(x0, jax.tree.map(lambda v: v[0], xs))
            _, rest = jax.lax.scan(body, x0, (self.dts[1:], jax.tree.map(lambda v: v[1:], xs)))
            outs = tuple(jnp.concatenate([f[None], b], 0) for f, b in zip(first, rest))
        else:
            _, outs = jax.lax.scan(body, x0, (self.dts, xs))
        return core.assemble(p, outs)

    def wind_nh(self, x0):
        return self.nh_per_rho * jnp.sum(self.wind_w * x0[self.ctrl.idx["rho"]])

    def parts(self, z, dz=None):
        """(dict of residual vectors, model, x0) at the control vector z; ``dz``:
        a model jitter (``casa_4dvar_robust``): the forward starts from the
        state of z + dz, the background terms see z."""
        ctrl = self.ctrl
        chi, xi_csm, xi_g = ctrl.split(z)
        xi_w = ctrl.warp_of(z)
        chi_f = ctrl.fine_of(z)
        if dz is None:
            x0 = ctrl.state(chi, xi_csm, xi_w, chi_f)
        else:
            chim, xim, _ = ctrl.split(z + dz)
            x0 = ctrl.state(chim, xim, ctrl.warp_of(z + dz), ctrl.fine_of(z + dz))
        x0 = SH.cstate(x0)
        g = ctrl.globals_of(xi_g)
        p = self.params(g)
        model = self.forward(x0, p)
        theta = jnp.stack([p[k] for k in X.PARAM_NAMES])
        parts = X.residual_parts(model, self.obs, self.img, theta, self.largs)
        for k in DROP_PARTS:
            parts.pop(k, None)
        if ctrl.gnames:
            parts["prior_glob"] = jnp.stack([(g[k] - self.prior[k][0]) / self.prior[k][1] for k in ctrl.gnames])
        if self.wind_prior:
            parts["wind_nh"] = jnp.atleast_1d((self.wind_nh(x0) - PD.WIND_NH_PRIOR[1]) / PD.WIND_NH_PRIOR[2])
        parts.update(ctrl.background_parts(chi, xi_csm, xi_w, chi_f))
        return parts, model, x0

    def objective(self, z, dz):
        parts, _, _ = self.parts(z, dz)
        chi2 = {k: jnp.sum(v ** 2) for k, v in parts.items()}
        return 0.5 * sum(chi2.values()), chi2

    def compile(self, vec_sharding=None):
        """``vg(z, dz)`` / ``val(z, dz)`` (dz: the model jitter, zeros for the plain J).
        ``vec_sharding`` (--optimizer jax on a mesh): the gradient is returned with
        the control's sharding (``P("x")``), as the device L-BFGS holds its vectors;
        z / dz arrive committed with it."""
        if vec_sharding is None:
            vg_t = lambda f: jax.value_and_grad(f, has_aux=True)  # noqa: E731
        else:
            def vg_t(f):
                vg = jax.value_and_grad(f, has_aux=True)

                def h(*args):
                    v, g = vg(*args)
                    return v, jax.lax.with_sharding_constraint(g, vec_sharding)
                return h
        self.vg = self.lifted.jit(self.objective, transform=vg_t)
        self.val = self.lifted.jit(self.objective)
        self.size = self.ctrl.size
        self.rough = np.zeros(self.size, bool)
        self.rough[:self.ctrl.n_chi] = True
        self.rough[self.ctrl.off_f:self.ctrl.off_f + self.ctrl.n_fine] = True
        return self
# =============================================================================
# ============ ↑ Setup ↑ ======================================================
# =============================================================================


# =============================================================================
# ============ ↓ Diagnostics ↓ ================================================
# =============================================================================
def mem_gb():
    """Peak GB in use: device 0, or the max over the mesh devices (--gpus)."""
    try:
        if SH.active():
            return max(m for _, m, _ in SH.per_device_memory())
        ms = jax.devices()[0].memory_stats() or {}
        return float(ms.get("peak_bytes_in_use", 0)) / 2 ** 30
    except Exception:            # CPU
        return float("nan")


def fmt_parts(chi2):
    return ", ".join(f"{k} {float(v):.1f}" for k, v in chi2.items())


def obs_total(chi2):
    return float(sum(float(v) for k, v in chi2.items() if not k.startswith(("b_", "prior_glob", "wind_nh"))))


class Validator:
    """All-epoch forward (value only): the full casa_xfit chi2 over every epoch,
    the chi2 over the training epochs of the full schedule, and the held-out
    epochs scored against the training set (``casa_4dvar_data.heldout_parts``)."""

    def __init__(self, win_all, train_idx, hold_idx):
        self.w = win_all
        self.train, self.hold = train_idx, hold_idx
        w = win_all

        def fn(z):
            parts, model, _ = w.parts(z)
            if SH.active():
                # multi-GPU: score the held-out epochs on REPLICATED model outputs (sky-plane
                # arrays, small). On the x-split ones the held-out graph (profiled calibrations,
                # spectrum nuisances) compiled for > 1 h at 128^3 on 2 GPUs (1 GPU: 250 s)
                model = jax.tree.map(lambda v: SH.constrain(v, SH.REPL) if hasattr(v, "ndim") else v, model)
            hp = DA.heldout_parts(model, w.obs, w.img, train_idx, hold_idx, w.largs)
            return ({k: jnp.sum(v ** 2) for k, v in parts.items()}, {k: jnp.sum(v ** 2) for k, v in hp.items()},
                    model["r_fs_arcsec"], model["r_rs_mean_arcsec"])
        self.fn = w.lifted.jit(fn)

    def __call__(self, z, label):
        t0 = time.time()
        full, hold, rfs, rrs = jax.block_until_ready(self.fn(jnp.asarray(z, self.w.dtype)))
        full = {k: float(v) for k, v in full.items()}
        hold = {k: float(v) for k, v in hold.items()}
        out = dict(label=label, full=full, full_obs_total=obs_total(full), heldout=hold,
                   heldout_total=float(sum(v for k, v in hold.items()
                                           if k in ("h_image_temporal", "h_spectrum", "h_outline_rel"))),
                   r_fs_mean_arcsec=[float(x) for x in np.nanmean(np.asarray(rfs), 1)],
                   r_rs_mean_arcsec=[float(x) for x in np.asarray(rrs)], t_s=time.time() - t0)
        print(f"[val {label}] all-epoch chi2 (obs terms) {out['full_obs_total']:.1f}: {fmt_parts(full)}", flush=True)
        print(f"[val {label}] held-out {', '.join(self.w.labels[i] for i in self.hold)}: "
              + ", ".join(f"{k} {v:.1f}" for k, v in hold.items()) + f" ({out['t_s']:.0f} s)", flush=True)
        return out
# =============================================================================
# ============ ↑ Diagnostics ↑ ================================================
# =============================================================================


# =============================================================================
# ============ ↓ Modes ↓ ======================================================
# =============================================================================
def direction_factory(ctrl, size, wc=None):
    """dirfn(kind, seed) -> float64 host direction of length ``size``: 'chi'
    (the 2000 state, white, per-component rms 1), 'globals', 'csm' (as
    ``Control.random_direction``) and, for the weak-constraint control, 'chik'
    (white on the boundary increments chi_1..chi_K)."""
    def dirfn(kind, seed):
        d = np.zeros(size)
        if kind == "chik":
            rng = np.random.default_rng(2000 + seed)
            d[ctrl.size:] = rng.normal(size=size - ctrl.size)
        else:
            d[:ctrl.size] = ctrl.random_direction(seed, what=kind)
        return d
    return dirfn


def taylor(a, win, ctrl, z0, out, dirfn=None, kinds=None, ev=None):
    """Taylor / FD test of J along random directions: B^(1/2)-shaped state
    increments (chi white, per-component rms 1), and the globals (+ CSM).
    ``ev`` (a ``casa_4dvar_robust.Evaluator`` with smooth_k > 0): also the
    smoothed-J central difference (J_s = mean over the fixed jitters, the
    reference for eps where plain FD is dominated by discrete-path jumps) and
    the smoothed gradient g_s . d."""
    size = getattr(win, "size", ctrl.size)
    dirfn = dirfn or direction_factory(ctrl, size)
    res = dict(window=win.labels, tangent=getattr(win, "tangent", None), n=int(ctrl.n), eps=a.eps, dirs=[],
               smooth=(dict(k=ev.policy.smooth_k, sigma=ev.policy.smooth_sigma) if ev is not None else None))
    zj = jnp.asarray(z0, win.dtype)
    zero = jnp.zeros(size, win.dtype)
    t0 = time.time()
    (J0, chi2), g = jax.block_until_ready(win.vg(zj, zero))
    res.update(t_grad_first_s=time.time() - t0, peak_GB_first=mem_gb())
    t0 = time.time()
    (J0, chi2), g = jax.block_until_ready(win.vg(zj, zero))
    res.update(t_grad_s=time.time() - t0, peak_GB=mem_gb(), J0=float(J0), parts0={k: float(v) for k, v in chi2.items()})
    g = np.asarray(g, np.float64)
    res["grad_norm"] = dict(chi=float(np.linalg.norm(g[:ctrl.n_chi])),
                            csm=float(np.linalg.norm(g[ctrl.n_chi:ctrl.n_chi + ctrl.n_csm])),
                            chik=float(np.linalg.norm(g[ctrl.size:])),
                            glob={k: float(v) for k, v in zip(ctrl.gnames, g[ctrl.off_g:ctrl.off_w])},
                            warp=float(np.linalg.norm(g[ctrl.off_w:ctrl.off_f])),
                            fine=float(np.linalg.norm(g[ctrl.off_f:ctrl.size])))
    gchi = g[:ctrl.n_chi].reshape(5, ctrl.m, ctrl.m, ctrl.m)
    res["grad_norm_by_field"] = {f: float(np.linalg.norm(gchi[i])) for i, f in enumerate(C.FIELDS)}
    res["nonfinite_by_field"] = {f: int(np.sum(~np.isfinite(gchi[i]))) for i, f in enumerate(C.FIELDS)}
    if not np.all(np.isfinite(g)):
        res["nonfinite"] = int(np.sum(~np.isfinite(g)))
        print(f"[taylor] NON-FINITE gradient: {res['nonfinite']} entries; by field {res['nonfinite_by_field']}",
              flush=True)
        if out:
            Path(out).write_text(json.dumps(res, indent=1))
        return res
    print(f"[taylor] J0 {float(J0):.4f}: {fmt_parts(chi2)}; grad {res['t_grad_first_s']:.0f} s first, "
          f"{res['t_grad_s']:.1f} s, peak {res['peak_GB']:.2f} GB; |g| {res['grad_norm']}", flush=True)
    t0 = time.time()
    jax.block_until_ready(win.val(zj, zero))
    res["t_val_first_s"] = time.time() - t0
    gs = None
    if ev is not None:
        t0 = time.time()
        Js0, _, gs, mem0 = ev.smooth_vg(z0)
        res.update(Js0=Js0, Js0_members=mem0, t_smooth_grad_s=time.time() - t0)
        print(f"[taylor] smoothed J_s0 {Js0:.4f} (members {', '.join(f'{v:.4f}' for v in mem0)}); "
              f"|g_s| {np.linalg.norm(gs):.4g}, |g| {np.linalg.norm(g):.4g}, cos(g, g_s) "
              f"{float(g @ gs / max(np.linalg.norm(g) * np.linalg.norm(gs), 1e-300)):.3f}", flush=True)
    if kinds is None:
        kinds = [("chi", s) for s in range(a.dirs)] + ([("globals", 0)] if ctrl.gnames else []) \
            + ([("csm", 0)] if ctrl.n_csm else []) + ([("neg_grad", 0)] if a.grad_dir else []) \
            + ([("warp", s) for s in range(getattr(a, "warp_dirs", 0))] if ctrl.n_warp else []) \
            + ([("neg_grad_warp", 0)] if ctrl.n_warp and a.grad_dir else []) \
            + ([("fine", s) for s in range(a.dirs)] if ctrl.n_fine else []) \
            + ([("neg_grad_fine", 0)] if ctrl.n_fine and a.grad_dir else [])
    if getattr(a, "taylor_kinds", None):
        kinds = [k for k in kinds if k[0] in a.taylor_kinds]
    # the whitened background block J_b = 0.5 |z[:nb]|^2 is quadratic, so its part of g.d (z0 . d) is
    # reproduced EXACTLY by central differences; away from z0 = 0 it can dominate g.d along white
    # directions and make the ratio look better (or worse) than the observation gradient is.
    # 'ratio_central_obs' removes it from both sides (review 2026-09-26).
    nb = ctrl.n_chi + ctrl.n_csm
    zb = np.asarray(z0, np.float64)
    for what, seed in kinds:
        if what == "neg_grad":
            # the steepest-descent state direction, UNIT norm: eps is the whitened step length itself
            # (L-BFGS steps are 0.3-2). NOT rms-1 like the white directions: g is concentrated (70 % of
            # |g|^2 in 100 of 1.3M components at a stage-1 best), so rms-1 scaling made eps = 0.3 a
            # |dz| = 343 step with ~40-sigma local ln rho increments (J NaN; review run 2026-09-26)
            d = np.zeros(size)
            gg = gs if gs is not None else g
            d[:ctrl.n_chi] = -gg[:ctrl.n_chi]
            d[ctrl.size:] = -gg[ctrl.size:]
            d /= max(float(np.linalg.norm(d)), 1e-300)
        elif what == "neg_grad_warp":
            # the steepest-descent direction of the warp coefficients alone, unit norm
            d = np.zeros(size)
            gg = gs if gs is not None else g
            d[ctrl.off_w:ctrl.off_f] = -gg[ctrl.off_w:ctrl.off_f]
            d /= max(float(np.linalg.norm(d)), 1e-300)
        elif what == "neg_grad_fine":
            # the steepest-descent direction of the fine-scale control level alone, unit norm
            d = np.zeros(size)
            gg = gs if gs is not None else g
            d[ctrl.off_f:ctrl.size] = -gg[ctrl.off_f:ctrl.size]
            d /= max(float(np.linalg.norm(d)), 1e-300)
        else:
            d = dirfn(what, seed)
        gd = float(g @ d)
        # the whitened quadratic background blocks: state + CSM, and the warp coefficients
        gb = float(zb[:nb] @ d[:nb] + zb[ctrl.off_w:ctrl.size] @ d[ctrl.off_w:ctrl.size])   # warp + fine
        row = dict(kind=what, seed=seed, gd=float(gd), gd_background=gb, fd={}, ratio_central={},
                   ratio_forward={}, ratio_central_obs={})
        if gs is not None:
            row.update(gsd=float(gs @ d), ratio_smooth={}, ratio_smooth_vs_gs={})
        eps_list = a.eps_warp if what in ("warp", "neg_grad_warp") and getattr(a, "eps_warp", None) else a.eps
        for eps in eps_list:
            t0 = time.time()
            Jp = float(win.val(jnp.asarray(z0 + eps * d, win.dtype), zero)[0])
            Jm = float(win.val(jnp.asarray(z0 - eps * d, win.dtype), zero)[0])
            c = (Jp - Jm) / (2 * eps)
            row["fd"][str(eps)] = dict(Jp=Jp, Jm=Jm, central=c, forward=(Jp - float(J0)) / eps)
            row["ratio_central"][str(eps)] = c / gd if gd != 0 else float("nan")
            row["ratio_central_obs"][str(eps)] = (c - gb) / (gd - gb) if gd != gb else float("nan")
            row["ratio_forward"][str(eps)] = (Jp - float(J0)) / eps / gd if gd != 0 else float("nan")
            extra = ""
            if gs is not None:
                Jsp, mp = ev.smooth_val(z0 + eps * d)
                Jsm, mm = ev.smooth_val(z0 - eps * d)
                cs = (Jsp - Jsm) / (2 * eps)
                row["fd"][str(eps)].update(Jsp=Jsp, Jsm=Jsm, central_smooth=cs, members_p=mp, members_m=mm)
                row["ratio_smooth"][str(eps)] = cs / gd if gd != 0 else float("nan")
                row["ratio_smooth_vs_gs"][str(eps)] = cs / row["gsd"] if row["gsd"] != 0 else float("nan")
                extra = (f"  smoothed central {cs:+.6g} (/g.d {row['ratio_smooth'][str(eps)]:.4f}, "
                         f"/g_s.d {row['ratio_smooth_vs_gs'][str(eps)]:.4f})")
            print(f"[taylor] {what}{seed} eps {eps:g}: g.d {gd:+.6g}  central {c:+.6g} "
                  f"(ratio {row['ratio_central'][str(eps)]:.4f}, obs-only "
                  f"{row['ratio_central_obs'][str(eps)]:.4f})  forward ratio "
                  f"{row['ratio_forward'][str(eps)]:.4f}  2nd-order rem {(Jp - float(J0) - eps * gd):+.3g}"
                  f"{extra} ({time.time() - t0:.0f} s)", flush=True)
        res["dirs"].append(row)
        if out:
            Path(out).write_text(json.dumps(res, indent=1))
    res["t_forward_s"] = res.get("t_val_first_s")
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def save_ckpt(path, z, stage, it, hist, extra=None):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, z=np.asarray(z, np.float64), stage=stage, it=it, hist=json.dumps(hist), **(extra or {}))
    os.replace(tmp, path)


def write_analysis_state(path, ctrl, win, z, meta, label):
    """The analysis x0 at 2000 in casa_xfit_state format (a casa_xfit --ic /
    casa_orlando --from-state restart), with the globals in its theta."""
    if SH.active():             # jitted, the background as sharded arguments
        x0 = np.asarray(ctrl.lift(SH.Lifted()).jit(ctrl.state_z)(jnp.asarray(z, win.dtype)))
    else:
        x0 = np.asarray(ctrl.state_z(jnp.asarray(z, win.dtype)))
    th = dict(win.theta_b)
    th.update(ctrl.globals_np(z))
    ic = {k: meta[k] for k in meta if k not in ("var_layout",)}
    ic.update(box=float(meta["box"]), num_cells=int(meta["num_cells"]), age=float(meta["ic_age"])
              if "ic_age" in meta else float(meta["age"]))
    XS.save_state(path, x0, win.core.rv, X.SCALAR_NAMES, ic=ic, theta=np.array([th[k] for k in X.PARAM_NAMES]),
                  names=X.PARAM_NAMES, epoch_year=float(meta["epoch_year"]), epoch_label=str(meta["epoch_label"]),
                  t_expl=float(meta["t_expl"]), options=dict(source="casa_4dvar", stage=label),
                  extra=dict(tag=f"4dvar {label}", z_globals=json.dumps(ctrl.globals_np(z)),
                             **({"z_warp": np.asarray(z, np.float64)[ctrl.off_w:ctrl.size],
                                 "warp_spec": json.dumps(dict(lmax=ctrl.warp.lmax, sigma_pc=ctrl.warp.sigma_pc,
                                                              taper=ctrl.warp.taper,
                                                              fold_frac=ctrl.warp.fold_frac))}
                                if ctrl.n_warp else {}),
                             **({"r_fs_pc": np.asarray(meta["r_fs_pc"])} if "r_fs_pc" in meta else {})))
    print(f"[4dvar] wrote {path}", flush=True)


def run(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype):
    """The strong-constraint quasi-static schedule (L-BFGS-B per window under
    the ``casa_4dvar_robust`` policy; ``plain`` = the stage-2 behaviour)."""
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    ck = od / "ckpt.npz"
    log = od / "evals.jsonl"
    policy = RB.policy_from_args(a)
    z = np.zeros(ctrl.size)
    stage0, it0, hist = 0, 0, dict(stages=[], val=[])
    if a.resume and ck.exists():
        z, stage0, it0, hist = load_ckpt(ck)
        print(f"[4dvar] resumed {ck}: stage {stage0} after {it0} iterations", flush=True)
    elif a.init_z:
        z = ctrl.pad_z(np.load(a.init_z)["z"])
    if z.size != ctrl.size:
        raise ValueError(f"control size {z.size} != {ctrl.size} (different options?)")
    hist.setdefault("argv", sys.argv)
    hist.setdefault("policy", policy.name)
    train_all, hold = DA.split_epochs(obs, t_end=max(a.windows), holdout=a.holdout)
    val = None
    if not a.no_validate:
        all_idx = np.arange(len(obs["epochs"]))
        tr_pos = np.searchsorted(all_idx, train_all)
        wv = Window(a, opts, meta, ctrl, theta_b, obs, img, all_idx, tangent="approx", t0_year=t0_year,
                    dtype=dtype)
        val = Validator(wv, tr_pos, hold)
        if not hist["val"]:
            hist["val"].append(val(z, "background"))
            save_ckpt(ck, z, stage0, it0, hist)
            # drop the validator's executable before the window's value / gradient ones are built
            # (as run_wc; review 2026-09-26: s3-4dv-Rp ran out of memory at 0.9 at the first
            # stage-0 gradient and when the validator was reloaded at the end of stage 0)
            jax.clear_caches()
    for s in range(stage0, len(a.windows)):
        t_end = a.windows[s]
        tr, _ = DA.split_epochs(obs, t_end=t_end, holdout=a.holdout)
        tangent = tangent_for(a, t0_year, float(np.asarray(obs["years"])[tr].max()))
        win = Window(a, opts, meta, ctrl, theta_b, obs, img, tr, tangent=tangent, t0_year=t0_year,
                     dtype=dtype)
        if a.const_audit:
            hist.setdefault("const_audit", {})[f"stage{s}"] = const_audit(win)
        win.compile(vec_sharding(ctrl.size) if a.optimizer == "jax" else None)
        n_it = a.iters[s] - (it0 if s == stage0 else 0)
        print(f"[stage {s}] window <= {t_end}: {len(tr)} training epochs {win.labels}; tangent {tangent}; "
              f"{n_it} iterations; policy {policy.name}; optimizer {a.optimizer}", flush=True)
        ev = make_evaluator(a, win, dtype, policy)
        t_stage = time.time()
        z_in = ev.ops.put(z) if ev.device else z.copy()
        log_rss(f"stage {s} start")
        fresh0 = ev.fresh(z_in, m=a.fresh_n) if a.fresh_n else None
        st = RB.robust_minimize(
            ev, z_in, n_iter=n_it, log_path=log, stage=s, it0=it0 if s == stage0 else 0,
            ckpt=lambda zb, it, s=s: save_ckpt(ck, zb, s, it, hist), obs_total=obs_total,
            stall_tol=a.stall_tol, stall_evals=a.stall_evals, max_restarts=a.max_restarts, maxcor=a.maxcor,
            ftol=a.ftol, max_grad_evals=a.max_grad_evals, fmt=fmt_parts, optimizer=a.optimizer)
        z_dev = st["z_best"]
        z = host(z_dev)
        rec = dict(stage=s, t_end=t_end, tangent=tangent, labels=win.labels, it=st["it"], n_eval=st["n_eval"],
                   J_best=st["best"], J_best_raw=st["best_raw"], stop=st["stop"], n_grad=st["n_grad"],
                   policy=st["policy"], optimizer=st["optimizer"], t_s=time.time() - t_stage,
                   host_rss_GB=log_rss(f"stage {s} end"))
        if a.fresh_n:
            rec.update(J_fresh_start=fresh0, J_fresh_best=ev.fresh(z_dev, m=a.fresh_n))
            print(f"[stage {s}] fresh-jitter J: start {fresh0['mean']:.4f} +- {fresh0['sd']:.4f} -> best "
                  f"{rec['J_fresh_best']['mean']:.4f} +- {rec['J_fresh_best']['sd']:.4f}", flush=True)
        hist["stages"].append(rec)
        np.savez(od / f"z_stage{s}.npz", z=z)
        write_analysis_state(od / f"state2000_4dvar_stage{s}.npz", ctrl, win, z, meta, f"stage{s}")
        # free the window's executables BEFORE the all-epoch validation (review 2026-09-26: OOM)
        del win, ev, st, z_dev, z_in
        jax.clear_caches()
        if val is not None:
            hist["val"].append(val(z, f"stage{s}"))
            jax.clear_caches()
        save_ckpt(ck, z, s + 1, 0, hist)
        (od / "summary.json").write_text(json.dumps(hist, indent=1))
    return z, hist


def compare_policies(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype):
    """Stage-3 test 1: the same window (``--windows``[0]) and start point
    (``--init-z``), one compiled J, several optimisation policies, each with the
    same gradient-evaluation budget; the honest score of each end point is the
    mean J over fresh tiny jitters (a family none of the runs used)."""
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tr, _ = DA.split_epochs(obs, t_end=a.windows[0], holdout=a.holdout)
    tangent = tangent_for(a, t0_year, float(np.asarray(obs["years"])[tr].max()))
    win = Window(a, opts, meta, ctrl, theta_b, obs, img, tr, tangent=tangent, t0_year=t0_year,
                 dtype=dtype).compile(vec_sharding(ctrl.size) if a.optimizer == "jax" else None)
    z0 = ctrl.pad_z(np.load(a.init_z)["z"]) if a.init_z else np.zeros(ctrl.size)
    out = dict(window=win.labels, tangent=tangent, init_z=a.init_z, budget=a.max_grad_evals, runs=[])
    specs = {"plain": RB.Policy(),
             "confirm": RB.Policy(confirm=True, min_step=a.min_step or 1e-6, spike_factor=a.spike_factor),
             "smooth": RB.Policy(smooth_k=a.smooth_k or 4, smooth_sigma=a.smooth_sigma),
             "smooth_confirm": RB.Policy(smooth_k=a.smooth_k or 4, smooth_sigma=a.smooth_sigma,
                                         min_step=a.min_step or 1e-6)}
    ev0 = RB.Evaluator(win.vg, win.val, win.size, win.rough, dtype, RB.Policy())
    t0 = time.time()
    jax.block_until_ready(win.vg(jnp.asarray(z0, dtype), ev0.zero))
    jax.block_until_ready(win.val(jnp.asarray(z0, dtype), ev0.zero))
    out["t_compile_s"] = time.time() - t0
    out["J_raw_start"] = ev0.raw_val(z0)[0]
    out["J_fresh_start"] = ev0.fresh(z0, m=a.fresh_n or 4)
    print(f"[policies] start: raw J {out['J_raw_start']:.4f}, fresh {out['J_fresh_start']}", flush=True)
    for name in a.policy_test:
        pol = specs[name] if name in specs else None
        if pol is None:
            raise ValueError(f"unknown policy {name}")
        ev = make_evaluator(a, win, dtype, pol)
        t0 = time.time()
        st = RB.robust_minimize(ev, z0, n_iter=a.iters[0], log_path=od / "evals.jsonl", stage=0, tag=f"-{name}",
                                obs_total=obs_total, stall_tol=a.stall_tol, stall_evals=a.stall_evals,
                                max_restarts=a.max_restarts, maxcor=a.maxcor, ftol=a.ftol,
                                max_grad_evals=a.max_grad_evals, fmt=fmt_parts, optimizer=a.optimizer)
        st["z_best"] = host(st["z_best"])
        t_run = time.time() - t0
        fr = ev0.fresh(st["z_best"], m=a.fresh_n or 4)
        acc = [h for h in st["hist"] if h.get("accepted")]
        row = dict(policy=name, spec=st["policy"], it=st["it"], n_eval=st["n_eval"], n_grad=st["n_grad"],
                   stop=st["stop"], restarts=st["restarts"], best=st["best"], best_raw=st["best_raw"],
                   J_fresh_best=fr, dJ_fresh=fr["mean"] - out["J_fresh_start"]["mean"], t_s=t_run,
                   step_norm=float(np.linalg.norm(st["z_best"] - z0)),
                   accepted=[dict(eval=h["eval"], J=h["J"], J_confirm=h.get("J_confirm"), best=h["best"],
                                  gnorm=h["gnorm"]) for h in acc],
                   J_trace=[h["J"] for h in st["hist"]])
        out["runs"].append(row)
        np.savez(od / f"z_{name}.npz", z=st["z_best"])
        print(f"[policies] {name}: best {st['best']:.4f} (raw {st['best_raw']:.4f}); fresh "
              f"{fr['mean']:.4f} +- {fr['sd']:.4f} (dJ {row['dJ_fresh']:+.4f}); {st['n_grad']:.1f} grad-equiv, "
              f"{t_run:.0f} s; stop: {st['stop']}", flush=True)
        (od / "policies.json").write_text(json.dumps(out, indent=1))
    return out


# ---- weak constraint ---------------------------------------------------------------------
class WCValidator:
    """All-epoch weak-constraint forward (value only): the WC J over every
    epoch and the held-out epochs PREDICTED by the last sub-window's trajectory
    (x_K forecast), scored against the training epochs."""

    def __init__(self, wcw, train_idx, hold_idx):
        self.w = wcw
        self.train, self.hold = train_idx, hold_idx
        w = wcw

        def fn(z, aux):
            chi2, model, _ = w.parts_chi2(z, jnp.zeros_like(z), aux)
            hp = DA.heldout_parts(model, w.obs, w.img, train_idx, hold_idx, w.base.largs)
            return chi2, {k: jnp.sum(v ** 2) for k, v in hp.items()}, model["r_fs_arcsec"]
        self.fn = jax.jit(fn)

    def __call__(self, z, label):
        t0 = time.time()
        full, hold, rfs = jax.block_until_ready(self.fn(jnp.asarray(z, self.w.dtype), self.w.aux))
        full = {k: float(v) for k, v in full.items()}
        hold = {k: float(v) for k, v in hold.items()}
        # obs terms only: obs_total_wc also drops the continuity chi2 (review 2026-09-26: obs_total
        # kept 'cont1..K', so the 'all-epoch WC chi2 (obs terms)' of s3-wc-prod included ~28-31 of it)
        out = dict(label=label, full=full, full_obs_total=obs_total_wc(full),
                   full_cont_total=float(sum(v for k, v in full.items() if k.startswith("cont"))), heldout=hold,
                   heldout_total=float(sum(v for k, v in hold.items()
                                           if k in ("h_image_temporal", "h_spectrum", "h_outline_rel"))),
                   r_fs_mean_arcsec=[float(x) for x in np.nanmean(np.asarray(rfs), 1)], t_s=time.time() - t0)
        print(f"[wcval {label}] all-epoch WC chi2 (obs terms) {out['full_obs_total']:.1f} (+ continuity "
              f"{out['full_cont_total']:.1f}): {fmt_parts(full)}", flush=True)
        print(f"[wcval {label}] held-out (x_K forecast) " + ", ".join(f"{k} {v:.1f}" for k, v in hold.items()
                                                                         if "_20" not in k), flush=True)
        return out


def obs_total_wc(chi2):
    return float(sum(float(v) for k, v in chi2.items()
                     if not k.startswith(("b_", "prior_glob", "wind_nh", "cont"))))


def wc_setup(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype, *, z_strong, od, grad=True):
    """(WCWindow over the training epochs, refs, bounds). The reference states
    are made once from ``z_strong`` and kept in ``od/wc_refs.npz`` (the chi_k
    are increments about them: a resume must reuse the same file)."""
    tr, _ = DA.split_epochs(obs, t_end=max(a.windows), holdout=a.holdout)
    tangent = tangent_for(a, t0_year, float(np.asarray(obs["years"])[tr].max()))
    base = Window(a, opts, meta, ctrl, theta_b, obs, img, tr, tangent=tangent, t0_year=t0_year, dtype=dtype)
    bounds = sorted(a.wc_bounds)
    rp = Path(od) / "wc_refs.npz"
    if rp.exists():
        d = np.load(rp)
        if not np.allclose(np.asarray(d["bounds"]), bounds):
            raise ValueError(f"{rp} was made for boundaries {d['bounds']}, not {bounds}")
        if not np.array_equal(np.asarray(d["z_strong"]), np.asarray(z_strong)):
            raise ValueError(f"{rp} was made from a different strong-constraint control")
        refs = np.asarray(d["refs"])
        print(f"[wc] reference states from {rp}", flush=True)
    else:
        t0 = time.time()
        refs = WC.make_refs(base, ctrl, z_strong, bounds, dtype)
        tmp = Path(str(rp) + ".tmp.npz")
        np.savez(tmp, refs=refs, bounds=np.asarray(bounds), z_strong=np.asarray(z_strong, np.float64))
        os.replace(tmp, rp)
        print(f"[wc] reference states made in {time.time() - t0:.0f} s -> {rp}", flush=True)
    _, xi_ref, _ = ctrl.split(np.asarray(z_strong, np.float64))
    wcw = WC.WCWindow(base, bounds, refs, xi_ref=xi_ref, alpha=a.wc_alpha, gcut=a.wc_gcut,
                      p_c_frac=a.wc_pc_frac, dtype=dtype).compile(grad=grad)
    wcw.tangent = tangent
    print(f"[wc] {wcw.K} boundaries {bounds}; steps {len(wcw.steps['t'])}; control {wcw.size} "
          f"(strong {ctrl.size} + {wcw.K} x {ctrl.n_chi}); alpha {wcw.alpha}, g_cut {wcw.gcut} "
          f"({wcw.n_keep} kept modes / field); mask fractions {[round(f, 3) for f in wcw.mask_frac]}; "
          f"p_c {[f'{v:.3g}' for v in wcw.p_c]}", flush=True)
    return wcw, refs, bounds


def run_wc(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype):
    """Weak-constraint 4D-Var over the training epochs <= max(--windows) from
    the strong-constraint control ``--init-z`` (and its forecast as the
    boundary references); one stage of ``--iters``[0] L-BFGS-B iterations
    under the robust policy; ``--resume`` continues from ``ckpt.npz``."""
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    ck = od / "ckpt.npz"
    log = od / "evals.jsonl"
    policy = RB.policy_from_args(a)
    rp = od / "wc_refs.npz"
    if rp.exists():
        z_strong = np.asarray(np.load(rp)["z_strong"], np.float64)
    elif a.init_z:
        z_strong = np.asarray(np.load(a.init_z)["z"], np.float64)[:ctrl.size]
    else:
        z_strong = np.zeros(ctrl.size)
    wcw, refs, bounds = wc_setup(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype, z_strong=z_strong,
                                 od=od)
    z = np.concatenate([z_strong, np.zeros(wcw.size - ctrl.size)])
    it0, hist = 0, dict(stages=[], val=[], wcval=[], argv=sys.argv, policy=policy.name, bounds=bounds,
                        alpha=a.wc_alpha, gcut=a.wc_gcut)
    if a.resume and ck.exists():
        d = np.load(ck)
        z = np.asarray(d["z"], np.float64)
        it0 = int(d["it"])
        hist = json.loads(str(d["hist"]))
        print(f"[wc] resumed {ck} after {it0} iterations", flush=True)
    elif a.init_z and np.load(a.init_z)["z"].size == wcw.size:
        z = np.asarray(np.load(a.init_z)["z"], np.float64)
    if z.size != wcw.size:
        raise ValueError(f"control size {z.size} != {wcw.size}")
    val = wval = None
    if not a.no_validate:
        all_idx = np.arange(len(obs["epochs"]))
        train_all, hold = DA.split_epochs(obs, t_end=max(a.windows), holdout=a.holdout)
        tr_pos = np.searchsorted(all_idx, train_all)
        wv = Window(a, opts, meta, ctrl, theta_b, obs, img, all_idx, tangent="approx", t0_year=t0_year,
                    dtype=dtype)
        val = Validator(wv, tr_pos, hold)
        wva = WC.WCWindow(wv, bounds, refs, xi_ref=wcw.xi_ref, alpha=a.wc_alpha, gcut=a.wc_gcut,
                          p_c_frac=a.wc_pc_frac, dtype=dtype)
        wval = WCValidator(wva, tr_pos, hold)
        if not hist["val"]:
            hist["val"].append(val(z[:ctrl.size], "start"))
            hist["wcval"].append(wval(z, "start"))
            save_ckpt(ck, z, 0, it0, hist)
        # drop the two validator executables before the WC value / gradient ones are built: four
        # >2 GB executables at once exhaust the A100's memory when CUDA graphs are instantiated
        # (s3-wc-short 2026-09-26: RESOURCE_EXHAUSTED at the first gradient); they recompile at the end
        jax.clear_caches()
    ev = make_evaluator(a, wcw, dtype, policy)
    t_stage = time.time()
    fresh0 = ev.fresh(z, m=a.fresh_n) if a.fresh_n else None
    if fresh0:
        print(f"[wc] start: fresh-jitter J {fresh0['mean']:.4f} +- {fresh0['sd']:.4f}", flush=True)
    n_it = a.iters[0] - it0
    st = RB.robust_minimize(ev, z, n_iter=n_it, log_path=log, stage=0, it0=it0, tag="-wc",
                            ckpt=lambda zb, it: save_ckpt(ck, zb, 0, it, hist), obs_total=obs_total_wc,
                            stall_tol=a.stall_tol, stall_evals=a.stall_evals, max_restarts=a.max_restarts,
                            maxcor=a.maxcor, ftol=a.ftol, max_grad_evals=a.max_grad_evals, fmt=fmt_parts,
                            optimizer=a.optimizer)
    z = host(st["z_best"])
    rec = dict(it=st["it"], n_eval=st["n_eval"], J_best=st["best"], J_best_raw=st["best_raw"], stop=st["stop"],
               n_grad=st["n_grad"], policy=st["policy"], t_s=time.time() - t_stage)
    if a.fresh_n:
        rec.update(J_fresh_start=fresh0, J_fresh_best=ev.fresh(z, m=a.fresh_n))
        print(f"[wc] fresh-jitter J: start {fresh0['mean']:.4f} -> best {rec['J_fresh_best']['mean']:.4f} "
              f"+- {rec['J_fresh_best']['sd']:.4f}", flush=True)
    hist["stages"].append(rec)
    save_ckpt(ck, z, 0, st["it"], hist)
    np.savez(od / "z_wc.npz", z=z)
    x0, Xs = wcw.states_np(z)
    np.savez(od / "wc_states.npz", x0=x0.astype(np.float32), X=Xs.astype(np.float32), bounds=np.asarray(bounds))
    write_analysis_state(od / "state2000_4dvar_wc.npz", ctrl, wcw.base, z[:ctrl.size], meta, "wc")
    if val is not None:
        del ev, wcw
        jax.clear_caches()
        hist["val"].append(val(z[:ctrl.size], f"it{st['it']}"))
        hist["wcval"].append(wval(z, f"it{st['it']}"))
    save_ckpt(ck, z, 0, st["it"], hist)
    (od / "summary.json").write_text(json.dumps(hist, indent=1))
    return z, hist
# =============================================================================
# ============ ↑ Modes ↑ ======================================================
# =============================================================================


def make_warp(a, fields, lay, core, theta_b, dtype):
    """The ``casa_4dvar_warp.Warp`` of the background (R_FS(n) from its hot
    indicator; prior sigma converted to pc at the background distance)."""
    names = [lay[k] for k in range(len(lay))]
    xb = np.stack([np.asarray(fields[nm], np.float64) for nm in names])
    hot = np.asarray(PD.shocked_indicator(xb, core.rv, core.t_per_code))
    d_kpc = float(np.exp(theta_b["ln_D"]))
    up = None
    if a.warp_upstream == "keep":
        I = {nm: k for k, nm in enumerate(names)}
        up = dict(rho=xb[I["rho"]], press=xb[I["press"]], t_per_code=core.t_per_code, i_rho=I["rho"],
                  i_press=I["press"], i_e=I.get("internal_energy"))
    w = WP.Warp(hot, core.box, core.n, sigma_pc=WP.sigma_pc_from_arcsec(a.warp_sigma_arcsec, d_kpc),
                lmax=a.warp_lmax, taper=tuple(a.warp_taper), fold_frac=a.warp_fold_frac, r_lfit=a.warp_r_lfit,
                dtype=dtype, kernel=a.warp_kernel, upstream=up, **WP.field_rules(names))
    print(f"[4dvar] {w.describe()} ({a.warp_sigma_arcsec:g} arcsec at {d_kpc:.3f} kpc)", flush=True)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", default=str(WORK / "stage2" / "state2000_Rstart_n128.npz"))
    ap.add_argument("--jac", default=str(WORK / "xfit_R_jac.npz"),
                    help="xfit Jacobian dump for the globals' preconditioning scale")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--taylor", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--windows", type=float, nargs="+", default=[2004.5, 2009.9, 2014.5, 2018.5],
                    help="quasi-static schedule: training epochs <= each end year")
    ap.add_argument("--iters", type=int, nargs="+", default=[25, 25, 25, 25])
    ap.add_argument("--holdout", nargs="*", default=["2019", "2022"])
    ap.add_argument("--tangent", choices=("auto", "exact", "approx", "semi"), default="approx",
                    help="approx (default): frozen WENO weights + cold-LLF 1000; exact explodes at 128^3 "
                         "even over 4 yr (|g| 4e16 in x64, NaN in f32); semi: exact WENO weights + cold-LLF "
                         "1000; auto: exact <= --exact-max-years")
    ap.add_argument("--exact-max-years", type=float, default=5.5)
    ap.add_argument("--remat", choices=("none", "stage", "axis"), default="axis")
    ap.add_argument("--ckpt", type=int, default=16, help="equinox checkpoints per integrate call")
    ap.add_argument("--remat-chunks", type=int, default=1,
                    help="--remat axis: each axis' flux as this many checkpointed slabs of a perpendicular "
                         "axis (memory; exact). 512^3 on 8 x A100-40GB needs it (see REPORT of ers/shard)")
    ap.add_argument("--smooth-latch", action="store_true", help="ad_smooth_shock_latch (straight-through)")
    ap.add_argument("--positivity", choices=("redistribute", "conservative"), default="redistribute",
                    help="solver positivity mode (casa_xfit --positivity); conservative for the 448^3 backgrounds")
    ap.add_argument("--ell", type=float, default=2.0, help="B^1/2 Gaussian sd, coarse cells")
    ap.add_argument("--fine-ctrl-ell", type=float, default=None,
                    help="two-level control: add a fine-scale white field chi_f on the same half-res grid, "
                         "smoothed with this sd (coarse cells, e.g. 0.5); default off (layout unchanged)")
    ap.add_argument("--fine-ctrl-sigma", type=float, default=0.3,
                    help="pointwise sd of the fine level as a fraction of --sigma (per field)")
    ap.add_argument("--sigma", type=float, nargs=5, default=[0.3, 0.5, 0.5, 0.5, 0.3],
                    metavar=("LNRHO", "VX", "VY", "VZ", "LNP"),
                    help="background sd: ln rho, v / v_ref (x3), ln p")
    ap.add_argument("--v-ref-kms", type=float, default=1000.0)
    ap.add_argument("--sigma-csm", type=float, default=0.3)
    ap.add_argument("--sigma-slope", type=float, default=0.5)
    ap.add_argument("--no-csm", action="store_true")
    ap.add_argument("--globals", nargs="*", default=list(GLOBALS), choices=X.PARAM_NAMES)
    # same dest as casa_xfit's --wind-prior {on,off}: one switch for both (review 2026-09-26: a separate
    # store_false flag on that dest turned '--wind-prior off' into the truthy string 'off', which kept
    # the controlled-CSM wind prior on while opts.wind_prior said off)
    ap.add_argument("--no-wind-prior", dest="wind_prior", action="store_const", const="off",
                    help="= --wind-prior off: no Lee+14 prior on the controlled CSM")
    ap.add_argument("--pm-train-only", action="store_true",
                    help="refit the registration proper motions from the per-epoch shifts WITHOUT the "
                         "held-out epochs (the default PMs are 2000-2022 slopes: they contain 2019/2022)")
    ap.add_argument("--sigma-model", type=float, default=5.0)
    ap.add_argument("--dirs", type=int, default=2, help="--taylor: random chi directions")
    ap.add_argument("--no-grad-dir", dest="grad_dir", action="store_false",
                    help="--taylor: skip the steepest-descent (-g, state part, unit norm) direction")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.03, 0.01, 0.003, 0.001])
    ap.add_argument("--out", default=None, help="--taylor / --eval json")
    ap.add_argument("--out-dir", default=None, help="--run: checkpoint, logs, analysis states")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-z", default=None, help="--run: start from this control (npz with z)")
    ap.add_argument("--maxcor", type=int, default=10,
                    help="L-BFGS history pairs (--optimizer scipy: capped at its 32-bit workspace limit)")
    ap.add_argument("--lift-constants", choices=("on", "off"), default="on",
                    help="one device: pass the background, masks, CSM basis, wind weight and the v2 "
                         "observation tables as jit arguments instead of HLO literals (host RAM; a mesh "
                         "always lifts them)")
    ap.add_argument("--const-audit", action="store_true",
                    help="--run: print the captured (closed-over) constants of each window's J")
    ap.add_argument("--sharding-check", action="store_true",
                    help="J and gradient of window --windows[0] at --init-z (or 0) with the control "
                         "replicated (--optimizer scipy) vs sharded like the device L-BFGS (jax); exit")
    ap.add_argument("--ftol", type=float, default=1e-9)
    ap.add_argument("--max-restarts", type=int, default=2)
    ap.add_argument("--stall-tol", type=float, default=0.05,
                    help="--run: a stage ends after --stall-evals evaluations without J decreasing by this")
    ap.add_argument("--stall-evals", type=int, default=10)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--fresh-n", type=int, default=RB.ROBUST_DEFAULTS["fresh_n"],
                    help="--run: score the start and best point of every stage by the mean J over this many "
                         "fresh tiny jitters (value only; 0 = off)")
    ap.add_argument("--policy-test", nargs="*", default=None,
                    choices=("plain", "confirm", "smooth", "smooth_confirm"),
                    help="compare optimisation policies on --windows[0] from --init-z with the same "
                         "--max-grad-evals budget (writes --out-dir/policies.json)")
    RB.add_policy_arguments(ap, robust_defaults=True)
    pw = ap.add_argument_group("forward-shock warp (casa_4dvar_warp; stage 4)")
    pw.add_argument("--warp", action="store_true",
                    help="add the forward-shock displacement control: a radial warp of the outer state "
                         "(shocked shell + FS + CSM just ahead) by delta(n) = sum_{l <= L} a_lm Y_lm")
    pw.add_argument("--warp-lmax", type=int, default=8)
    pw.add_argument("--warp-sigma-arcsec", type=float, default=5.0,
                    help="pointwise prior rms of the displacement (converted to pc at the background distance)")
    pw.add_argument("--warp-taper", type=float, nargs=4, default=list(WP.DEFAULT_TAPER),
                    metavar=("S0", "S1", "S2", "S3"),
                    help="radial taper in s = r / R_FS(n): 0 below s0, ramps to 1 at s1, 1 to s2, 0 at s3")
    pw.add_argument("--warp-fold-frac", type=float, default=0.85,
                    help="saturation of |delta| at this fraction of the fold limit")
    pw.add_argument("--warp-r-lfit", type=int, default=8, help="angular cut-off of the fitted R_FS(n)")
    pw.add_argument("--no-warp-mask", dest="warp_mask", action="store_false",
                    help="keep the B^1/2 remnant mask and CSM weight fixed (default: warped with the state)")
    pw.add_argument("--warp-upstream", choices=("keep", "resample"), default="keep",
                    help="keep: the unshocked CSM ahead of the shock keeps the background's radial profile "
                         "(gathered wind rescaled by its local power law); resample: plain resampling (the "
                         "taper's decay then compresses / stretches the wind ahead of the shock)")
    pw.add_argument("--warp-kernel", choices=("cubic", "linear"), default="cubic",
                    help="resampling: Catmull-Rom (C^1, exact AD; default) or trilinear (custom JVP)")
    pw.add_argument("--warp-dirs", type=int, default=2, help="--taylor: random warp directions")
    pw.add_argument("--eps-warp", type=float, nargs="+", default=[3.0, 1.0, 0.3, 0.1, 0.03],
                    help="--taylor: steps along the warp directions (whitened; the random ones have "
                         "per-component rms 1, i.e. eps x 5\" pointwise rms; neg_grad_warp is unit norm)")
    ap.add_argument("--taylor-kinds", nargs="*", default=None,
                    choices=("chi", "globals", "csm", "neg_grad", "warp", "neg_grad_warp"),
                    help="--taylor: only these direction kinds (default: all)")
    wg = ap.add_argument_group("weak constraint (casa_4dvar_wc)")
    wg.add_argument("--wc", action="store_true",
                    help="weak-constraint / multiple-shooting 4D-Var over the epochs <= max(--windows) "
                         "(--run, --taylor, --eval); references = the forecast of --init-z (strong control)")
    wg.add_argument("--wc-bounds", type=float, nargs="+", default=[2004.5, 2009.9, 2014.5],
                    help="sub-window boundaries (years; a state control at each)")
    wg.add_argument("--wc-alpha", type=float, default=0.3,
                    help="model-error amplitude: Q^1/2 = alpha x B^1/2 (same smoothing)")
    wg.add_argument("--wc-gcut", type=float, default=0.1,
                    help="the Q-norm keeps the modes with G(k) >= g_cut (reduced-rank inverse)")
    wg.add_argument("--wc-pc-frac", type=float, default=0.01,
                    help="continuity misfit in ln(p + p_c), p_c = this x the mean shocked-gas pressure")
    ap.add_argument("--coarsen", type=int, default=None, help="block-average the state (CPU tests)")
    ap.add_argument("--cpu-test", action="store_true")
    ap.add_argument("--trace-only", action="store_true", help="--taylor: trace value_and_grad and exit")
    ap.add_argument("--gpus", type=int, default=1,
                    help="split the state / background / observation model over this many GPUs of "
                         "one node (x-slabs, casa_xfit_shard); 1: the single-device path unchanged")
    X.add_fix_arguments(ap)
    DA.add_fine_arguments(ap)       # the multi-scale image likelihood (--fine-block; default off)
    a = ap.parse_args()
    SH.activate(a.gpus)
    if a.fine_ctrl_ell and a.wc:
        raise SystemExit("--fine-ctrl-ell is implemented for the strong-constraint control only")
    if SH.active() and (a.warp or a.wc):
        raise SystemExit("--gpus > 1 is implemented for the strong-constraint, no-warp control only")
    RB.apply_plain_policy(a)
    if a.warp and a.wc:
        raise SystemExit("--warp is not implemented for the weak-constraint control (--wc)")
    if len(a.iters) < len(a.windows):
        a.iters = a.iters + [a.iters[-1]] * (len(a.windows) - len(a.iters))
    a.ic = a.state
    for k in ("save_state", "no_history"):
        setattr(a, k, getattr(a, k, None))
    opts = X.resolve_options(a)
    print("[4dvar] options: " + ", ".join(f"{k}={v}" for k, v in vars(opts).items()), flush=True)
    dtype = jnp.float64 if a.x64 else jnp.float32
    fields, meta, lay, theta_b = load_background(a.state, a.coarsen)
    if opts.kdop == "fixed0":
        theta_b["ln_kdop"] = 0.0
    t0_year = float(meta["epoch_year"])
    log_rss("start")
    obs, img = DA.load_all(opts, table_dir=a.table_dir, pm_exclude=a.holdout if a.pm_train_only else None,
                           fine_block=a.fine_block)
    log_rss("data loaded")
    # the geometry of the solver grid (the control lives on it)
    ic = dict(box=float(meta["box"]), num_cells=int(meta["num_cells"]), age=float(meta["age"]))
    gs = globals_scale(a.globals, a.jac)
    core0 = X.make_forward_core(ic, *DA.subset(obs, img, [0]), opts=opts, ic_path=a.state,
                                config_overrides=solver_overrides(a, "approx"))
    warp = make_warp(a, fields, lay, core0, theta_b, dtype) if a.warp else None
    ctrl = C.Control(fields, lay, core0.geom, ell=a.ell, sigma=tuple(a.sigma), v_ref=a.v_ref_kms / 1000.0,
                     sigma_csm=a.sigma_csm, sigma_slope=a.sigma_slope, csm=not a.no_csm,
                     globals_b={k: theta_b[k] for k in a.globals}, globals_scale=gs, dtype=dtype,
                     r_ref=float(np.mean(meta["r_fs_pc"])) if "r_fs_pc" in meta else None,
                     warp=warp, warp_mask=a.warp_mask, fine_ell=a.fine_ctrl_ell, fine_sigma=a.fine_ctrl_sigma)
    del core0
    # the background fields (host, float32 x ~20 variables: ~7 GB at 448^3) are not read again:
    # the control holds what it needs (xb on the device, the masks)
    del fields
    log_rss("control built")
    print(f"[4dvar] background {a.state} (n = {ctrl.n}, epoch {meta['epoch_label']} = {t0_year:.4f}); "
          f"control: chi {ctrl.n_chi} + csm {ctrl.n_csm} + globals {len(ctrl.gnames)} "
          f"{'+ warp ' + str(ctrl.n_warp) + ' ' if ctrl.n_warp else ''}"
          f"{f'+ fine {ctrl.n_fine} (ell {ctrl.fine_ell}, x{ctrl.fine_sigma} sigma) ' if ctrl.n_fine else ''}"
          f"= {ctrl.size}; "
          f"mask volume fraction {ctrl.mask_np.mean():.3f}, r_ref {ctrl.r_ref:.3f} pc, B^1/2 norm "
          f"{ctrl.norm:.3f}; globals scale " + ", ".join(f"{k} {v:.3g}" for k, v in gs.items()), flush=True)
    z0 = np.zeros(ctrl.size)
    if a.init_z and not a.run and not a.wc:
        z_in = np.asarray(np.load(a.init_z)["z"])
        # (stage-4 review) the warp coefficients sit LAST in z, so a warp analysis read without --warp
        # (or with another --warp-lmax) would be silently truncated to the no-warp model here
        extra = z_in.size - ctrl.size
        if extra > 0 and extra in {(l + 1) ** 2 - ctrl.n_warp for l in range(1, 17)}:
            raise SystemExit(f"--init-z {a.init_z}: {z_in.size} components = control {ctrl.size} + {extra}, "
                             "the size of a forward-shock warp block: pass --warp (and its --warp-lmax)")
        z0 = ctrl.pad_z(z_in)[:ctrl.size]
    if a.taylor and a.wc:
        od = Path(a.out_dir or (WORK / "stage3" / "wc" / "taylor_refs"))
        od.mkdir(parents=True, exist_ok=True)
        z_in = np.asarray(np.load(a.init_z)["z"], np.float64) if a.init_z else np.zeros(ctrl.size)
        if (od / "wc_refs.npz").exists():
            z_strong = np.asarray(np.load(od / "wc_refs.npz")["z_strong"], np.float64)
        else:
            z_strong = z_in[:ctrl.size]
        if a.trace_only:          # no reference forecast: trace against the background as references
            base = Window(a, opts, meta, ctrl, theta_b, obs, img,
                          DA.split_epochs(obs, t_end=max(a.windows), holdout=a.holdout)[0], tangent=a.tangent
                          if a.tangent != "auto" else "approx", t0_year=t0_year, dtype=dtype)
            refs = np.stack([np.asarray(ctrl.xb)] * len(a.wc_bounds))
            wcw = WC.WCWindow(base, a.wc_bounds, refs, xi_ref=ctrl.split(z_strong)[1], alpha=a.wc_alpha,
                              gcut=a.wc_gcut, p_c_frac=a.wc_pc_frac, dtype=dtype).compile()
            zz = jnp.zeros(wcw.size, dtype)
            t0 = time.time()
            sh = jax.eval_shape(wcw.vg, zz, zz)
            print(f"[trace] WC value_and_grad traced in {time.time() - t0:.0f} s: grad {sh[1].shape}, parts "
                  + ", ".join(sorted(sh[0][1])), flush=True)
            return
        wcw, _, _ = wc_setup(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype, z_strong=z_strong, od=od)
        z0 = z_in if z_in.size == wcw.size else np.concatenate([z_strong, np.zeros(wcw.size - ctrl.size)])
        ev = (RB.Evaluator(wcw.vg, wcw.val, wcw.size, wcw.rough, dtype, RB.policy_from_args(a))
              if a.smooth_k else None)
        kinds = [("chi", s) for s in range(a.dirs)] + [("chik", s) for s in range(a.dirs)] \
            + [("globals", 0)] + ([("csm", 0)] if ctrl.n_csm else []) + ([("neg_grad", 0)] if a.grad_dir else [])
        print(f"[taylor] WC window {wcw.labels}, tangent {wcw.tangent}", flush=True)
        taylor(a, wcw, ctrl, z0, a.out, dirfn=direction_factory(ctrl, wcw.size), kinds=kinds, ev=ev)
    elif a.taylor:
        tr, _ = DA.split_epochs(obs, t_end=a.windows[0], holdout=a.holdout)
        tangent = tangent_for(a, t0_year, float(np.asarray(obs["years"])[tr].max()))
        win = Window(a, opts, meta, ctrl, theta_b, obs, img, tr, tangent=tangent, t0_year=t0_year,
                     dtype=dtype).compile()
        print(f"[taylor] window {win.labels}, tangent {tangent}", flush=True)
        if a.trace_only:
            t0 = time.time()
            zz = jnp.asarray(z0, dtype)
            sh = jax.eval_shape(win.vg, zz, jnp.zeros_like(zz))
            print(f"[trace] value_and_grad traced in {time.time() - t0:.0f} s: grad {sh[1].shape}, parts "
                  + ", ".join(sorted(sh[0][1])), flush=True)
            return
        ev = RB.Evaluator(win.vg, win.val, win.size, win.rough, dtype, RB.policy_from_args(a)) \
            if a.smooth_k else None
        taylor(a, win, ctrl, z0, a.out, ev=ev)
    if a.sharding_check:
        sharding_check(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype, z0)
        return
    if a.policy_test:
        compare_policies(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype)
    if a.eval:
        train_all, hold = DA.split_epochs(obs, t_end=max(a.windows), holdout=a.holdout)
        wv = Window(a, opts, meta, ctrl, theta_b, obs, img, np.arange(len(obs["epochs"])), tangent="approx",
                    t0_year=t0_year, dtype=dtype)
        res = Validator(wv, train_all, hold)(z0, "eval")
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
    if a.run:
        (run_wc if a.wc else run)(a, opts, meta, ctrl, theta_b, obs, img, t0_year, dtype)


if __name__ == "__main__":
    main()
