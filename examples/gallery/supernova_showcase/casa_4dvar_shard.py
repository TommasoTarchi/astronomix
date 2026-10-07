"""
Sharded (multi-GPU) 4D-Var gradient: validation and memory / time benchmark.

Builds exactly ``casa_4dvar``'s training Window (same flags; ``--gpus N`` splits
it over N GPUs of the node, casa_xfit_shard), compiles value_and_grad of J once
(AOT, so compile and run are timed separately), evaluates it ``--reps`` times
and writes J, the per-term chi2, the gradient, the timings, the compiled
per-device memory (XLA memory analysis) and the measured per-device peak to
``--dump`` (npz + a json next to it). ``--ref`` compares against another dump
(e.g. the 1-GPU run): relative J / per-term differences, gradient cosine and
relative L2 difference. ``--hlo-audit`` lists the large collectives of the
compiled module (all-gathers of full 3D fields are the thing to avoid).

    ./run.sh casa_4dvar_shard.py --gpus 2 --state W/state2000_R2_n128.npz --jac W/xfit_R2_jac.npz \\
        --pm-train-only --holdout 2019 2022 --windows 2004.5 --dump W/ers/shard/g128_2gpu.npz \\
        --ref W/ers/shard/g128_1gpu.npz --hlo-audit

CPU smoke test (4 virtual devices, NATIVE_JAX, 32^3):

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 ./run.sh casa_4dvar_shard.py \\
        --gpus 4 --cpu-test --coarsen 32 ...
"""
# ==== GPU selection ====
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=int(sys.argv[sys.argv.index("--gpus") + 1]) if "--gpus" in sys.argv else 1)
if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
# ruff: noqa: E402
# =======================
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import casa_4dvar as V
import casa_4dvar_data as DA
import casa_xfit as X
import casa_xfit_shard as SH
import casa_4dvar_control as C


def build(a):
    """casa_4dvar.main's setup up to the training Window of ``--windows[0]``."""
    opts = X.resolve_options(a)
    dtype = jnp.float64 if a.x64 else jnp.float32
    fields, meta, lay, theta_b = V.load_background(a.state, a.coarsen)
    if opts.kdop == "fixed0":
        theta_b["ln_kdop"] = 0.0
    t0_year = float(meta["epoch_year"])
    obs, img = DA.load_all(opts, table_dir=a.table_dir, pm_exclude=a.holdout if a.pm_train_only else None,
                           fine_block=getattr(a, "fine_block", 0))
    ic = dict(box=float(meta["box"]), num_cells=int(meta["num_cells"]), age=float(meta["age"]))
    gs = V.globals_scale(a.globals, a.jac)
    core0 = X.make_forward_core(ic, *DA.subset(obs, img, [0]), opts=opts, ic_path=a.state,
                                config_overrides=V.solver_overrides(a, "approx"))
    ctrl = C.Control(fields, lay, core0.geom, ell=a.ell, sigma=tuple(a.sigma), v_ref=a.v_ref_kms / 1000.0,
                     sigma_csm=a.sigma_csm, sigma_slope=a.sigma_slope, csm=not a.no_csm,
                     globals_b={k: theta_b[k] for k in a.globals}, globals_scale=gs, dtype=dtype,
                     r_ref=float(np.mean(meta["r_fs_pc"])) if "r_fs_pc" in meta else None,
                     fine_ell=a.fine_ctrl_ell, fine_sigma=a.fine_ctrl_sigma)
    del core0, fields
    tr, _ = DA.split_epochs(obs, t_end=a.windows[0], holdout=a.holdout)
    tangent = V.tangent_for(a, t0_year, float(np.asarray(obs["years"])[tr].max()))
    win = V.Window(a, opts, meta, ctrl, theta_b, obs, img, tr, tangent=tangent, t0_year=t0_year,
                   dtype=dtype).compile()
    return win, ctrl, dtype


def solver_objective(win, ctrl, years, dtype, grad=True):
    """(z, dz) -> (J, parts) [, g]: the SOLVER alone -- x0(z), integrate ``years``,
    J = sum(W1 ln rho + W2 ln p) with fixed random W (a solver-backward check
    without the observation model)."""
    core = win.core
    rv = core.rv
    n = ctrl.n
    rng = np.random.default_rng(3)
    hold = type("H", (), {})()
    hold.w = SH.put(rng.normal(size=(2, n, n, n)).astype(np.float32) / n ** 1.5, SH.STATE, dtype)
    L = SH.Lifted()
    if SH.active():
        ctrl.lift(L)
        L.add(hold, "w", SH.STATE)
    integrate = core.integrator(ctrl.xb.shape)
    dt = float(years) * core.yr

    def obj(z, dz):
        chi, xi, _ = ctrl.split(z)
        x0 = SH.cstate(ctrl.state(chi, xi, None, ctrl.fine_of(z)))
        # --solver-only 0: the control transform alone (no integration)
        x1 = integrate(x0, jnp.asarray(dt, dtype)) if float(years) > 0 else x0
        J = jnp.sum(hold.w[0] * jnp.log(x1[rv.density_index])) + jnp.sum(hold.w[1] * jnp.log(x1[rv.pressure_index]))
        return J, {"solver": J}
    f = L.jit(obj, transform=(lambda g: jax.value_and_grad(g, has_aux=True)) if grad else None)
    try:
        f.lifted_ref = L
    except AttributeError:      # a plain jax.jit object (one device): no lifted arguments
        pass
    return f


def fd_check(specs, win, ctrl, z, g):
    """Central differences of J (and of each chi2 term) along ``KIND[:EPS]``
    directions against g.d: ``c00`` (the unit CSM-monopole coefficient),
    ``chi`` / ``csm`` / ``globals`` (``Control.random_direction``, seed 0) and
    ``neg_grad`` (unit steepest-descent state direction)."""
    rows = []
    zero = jnp.zeros(ctrl.size, win.dtype)
    for spec in specs:
        kind, _, e = spec.partition(":")
        eps = float(e) if e else 0.01
        d = np.zeros(ctrl.size)
        if kind == "c00":
            d[ctrl.n_chi] = 1.0
        elif kind == "neg_grad":
            d[:ctrl.n_chi] = -g[:ctrl.n_chi]
            d /= max(float(np.linalg.norm(d)), 1e-300)
        else:
            d[:] = ctrl.random_direction(0, what=kind)[:ctrl.size]
        t0 = time.time()
        Jp, pp = win.val(jnp.asarray(z + eps * d, win.dtype), zero)
        Jm, pm = win.val(jnp.asarray(z - eps * d, win.dtype), zero)
        c, gd = (float(Jp) - float(Jm)) / (2 * eps), float(g @ d)
        row = dict(kind=kind, eps=eps, gd=gd, fd=c, ratio=c / gd if gd else float("nan"), Jp=float(Jp),
                   Jm=float(Jm), fd_parts={k: 0.5 * (float(pp[k]) - float(pm[k])) / (2 * eps) for k in pp},
                   t_s=time.time() - t0)
        print(f"[shard] FD {kind} eps {eps:g}: g.d {gd:.6g}, central {c:.6g}, ratio {row['ratio']:.5f}; "
              f"0.5 dchi2/de {json.dumps({k: round(v, 5) for k, v in row['fd_parts'].items()})}", flush=True)
        rows.append(row)
    return rows


def compare(res, g, ref_path):
    r = np.load(ref_path, allow_pickle=True)
    J0, g0 = float(r["J"]), np.asarray(r["g"], np.float64)
    parts0 = json.loads(str(r["parts"]))
    out = dict(ref=str(ref_path), J_ref=J0, J_rel=abs(res["J"] - J0) / max(abs(J0), 1e-300),
               parts_rel={k: abs(v - parts0[k]) / max(abs(parts0[k]), 1e-12) for k, v in res["parts"].items()
                          if k in parts0})
    n = min(g.size, g0.size)
    a, b = g[:n], g0[:n]
    out.update(grad_cos=float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)),
               grad_rel_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
               grad_norm=float(np.linalg.norm(a)), grad_norm_ref=float(np.linalg.norm(b)))
    nchi = res["n_chi"]
    for name, sl in (("chi", slice(0, nchi)), ("rest", slice(nchi, n))):
        aa, bb = a[sl], b[sl]
        out[f"grad_cos_{name}"] = float(aa @ bb / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-300))
    return out


def main():
    ap = V.argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # casa_4dvar's options that matter for one gradient
    ap.add_argument("--state", required=True)
    ap.add_argument("--jac", default=None)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--windows", type=float, nargs="+", default=[2004.5])
    ap.add_argument("--holdout", nargs="*", default=["2019", "2022"])
    ap.add_argument("--tangent", choices=("auto", "exact", "approx"), default="approx")
    ap.add_argument("--exact-max-years", type=float, default=5.5)
    ap.add_argument("--remat", choices=("none", "stage", "axis"), default="axis")
    ap.add_argument("--ckpt", type=int, default=16)
    ap.add_argument("--remat-chunks", type=int, default=1)
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra SimulationConfig overrides for the solver (memory experiments), "
                         "VALUE a Python literal, e.g. max_passive_scalar_substeps=2")
    ap.add_argument("--smooth-latch", action="store_true")
    ap.add_argument("--ell", type=float, default=2.0)
    ap.add_argument("--fine-ctrl-ell", type=float, default=None)
    ap.add_argument("--fine-ctrl-sigma", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, nargs=5, default=[0.3, 0.5, 0.5, 0.5, 0.3])
    ap.add_argument("--v-ref-kms", type=float, default=1000.0)
    ap.add_argument("--sigma-csm", type=float, default=0.3)
    ap.add_argument("--sigma-slope", type=float, default=0.5)
    ap.add_argument("--no-csm", action="store_true")
    ap.add_argument("--globals", nargs="*", default=list(V.GLOBALS), choices=X.PARAM_NAMES)
    ap.add_argument("--no-wind-prior", dest="wind_prior", action="store_const", const="off", default=None)
    ap.add_argument("--pm-train-only", action="store_true")
    ap.add_argument("--sigma-model", type=float, default=5.0)
    ap.add_argument("--coarsen", type=int, default=None)
    ap.add_argument("--cpu-test", action="store_true")
    # this script
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--init-z", default=None, help="evaluate at this control (npz with z); default 0")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--hlo-audit", action="store_true")
    ap.add_argument("--compile-only", action="store_true", help="XLA memory analysis only (memory planning)")
    ap.add_argument("--validator", action="store_true", help="compile / time casa_4dvar.Validator (all epochs) instead")
    ap.add_argument("--value-only", action="store_true", help="time/memory of J alone (no gradient)")
    ap.add_argument("--solver-only", type=float, default=None, metavar="YEARS",
                    help="J = the solver alone over YEARS (random-weight functional of ln rho, ln p)")
    ap.add_argument("--fd", nargs="*", default=None, metavar="KIND[:EPS]",
                    help="after the gradient: central FD of J along c00 | chi | csm | globals | neg_grad")
    ap.add_argument("--profile", default=None, help="write a jax.profiler trace of one extra evaluation here")
    X.add_fix_arguments(ap)
    DA.add_fine_arguments(ap)
    a = ap.parse_args()
    a.warp = a.wc = False
    a.ic = a.state
    for k in ("save_state", "no_history"):
        setattr(a, k, getattr(a, k, None))
    SH.activate(a.gpus)
    if a.override:            # e.g. max_passive_scalar_substeps=2 (memory planning)
        import ast
        extra = {k: ast.literal_eval(v) for k, v in (o.split('=', 1) for o in a.override)}
        _so = V.solver_overrides
        V.solver_overrides = lambda aa, tangent: {**_so(aa, tangent), **extra}
        print(f"[shard] solver overrides {extra}", flush=True)
    t_setup = time.time()
    win, ctrl, dtype = build(a)
    print(f"[shard] n = {ctrl.n}, window {win.labels}, control {ctrl.size}; lifted "
          f"{win.lifted.nbytes() / 2 ** 30:.2f} GiB; setup {time.time() - t_setup:.0f} s", flush=True)
    z = np.zeros(ctrl.size)
    if a.init_z:
        z = ctrl.pad_z(np.asarray(np.load(a.init_z)["z"], np.float64))[:ctrl.size]
    zj, zero = jnp.asarray(z, dtype), jnp.zeros(ctrl.size, dtype)
    fn = win.val if a.value_only else win.vg
    if a.validator:             # the all-epoch validation forward of casa_4dvar.run
        opts = X.resolve_options(a)
        obs, img = DA.load_all(opts, table_dir=a.table_dir, pm_exclude=a.holdout if a.pm_train_only else None,
                               fine_block=a.fine_block)
        train_all, hold = DA.split_epochs(obs, t_end=max(a.windows), holdout=a.holdout)
        all_idx = np.arange(len(obs["epochs"]))
        _, meta, _, theta_b = V.load_background(a.state, a.coarsen)
        wv = V.Window(a, opts, meta, ctrl, theta_b, obs, img, all_idx, tangent="approx",
                      t0_year=float(meta["epoch_year"]), dtype=dtype)
        vfn = V.Validator(wv, np.searchsorted(all_idx, train_all), hold).fn
        win = wv
        fn = type("VF", (), {"lower": staticmethod(lambda zz, dd: vfn.lower(zz))})()
    if a.solver_only is not None:
        # < 0: the window's own span (first to last training epoch from the state's epoch)
        yrs = a.solver_only if a.solver_only >= 0 else float(np.sum(np.asarray(win.dts, np.float64))) / win.core.yr
        print(f"[shard] solver-only: {yrs:.3f} yr", flush=True)
        fn = solver_objective(win, ctrl, yrs, dtype, grad=not a.value_only)
    t0 = time.time()
    lo = fn.lower(zj, zero)
    t_lower = time.time() - t0
    t0 = time.time()
    comp = lo.compile()
    t_compile = time.time() - t0
    ma = comp.memory_analysis()
    mem_an = None
    if ma is not None:
        mem_an = dict(temp_GB=ma.temp_size_in_bytes / 2 ** 30, arg_GB=ma.argument_size_in_bytes / 2 ** 30,
                      out_GB=ma.output_size_in_bytes / 2 ** 30, alias_GB=ma.alias_size_in_bytes / 2 ** 30,
                      code_GB=ma.generated_code_size_in_bytes / 2 ** 30)
    print(f"[shard] lower {t_lower:.0f} s, compile {t_compile:.0f} s; XLA memory analysis per device {mem_an}",
          flush=True)
    if a.compile_only:        # memory planning: the XLA memory analysis, no evaluation
        res = dict(gpus=a.gpus, n=int(ctrl.n), ckpt=a.ckpt, remat=a.remat, remat_chunks=a.remat_chunks, solver_only=a.solver_only,
                   value_only=bool(a.value_only), t_lower_s=t_lower, t_compile_s=t_compile, mem_analysis=mem_an,
                   limit_GB_per_device=SH.device_limits_gb(), lifted_GB=win.lifted.nbytes() / 2 ** 30)
        Path(a.dump).with_suffix(".json").write_text(json.dumps(res, indent=1))
        print(f"[shard] compile-only: wrote {Path(a.dump).with_suffix('.json')}", flush=True)
        return
    audit = None
    if a.hlo_audit:
        try:
            txt = comp.as_text()
        except Exception as e:      # > 2 GiB (the 1-device path embeds the tables as constants)
            print(f"[shard] no HLO text: {str(e)[:200]}", flush=True)
            txt = None
        if txt is not None:
            audit = SH.collective_audit(txt, min_elems=max(ctrl.n ** 3 // (2 * max(a.gpus, 1)), 4096))
            (Path(a.dump).with_suffix(".hlo.txt")).write_text(txt)
            print(f"[shard] large collectives (op, result shape, count): {audit[:30]}", flush=True)
    lifted = fn.lifted_ref if hasattr(fn, "lifted_ref") else win.lifted
    call = (lambda zz, dd: comp(zz, dd, lifted.values())) if SH.active() else comp
    times = []
    for _ in range(a.reps):
        t0 = time.time()
        out = jax.block_until_ready(call(zj, zero))
        times.append(time.time() - t0)
    if a.profile:               # one more evaluation under the profiler (perfetto json for analysis)
        with jax.profiler.trace(a.profile, create_perfetto_trace=True):
            jax.block_until_ready(call(zj, zero))
        print(f"[shard] profile written to {a.profile}", flush=True)
    if a.value_only:
        (J, chi2), g = out, None
    else:
        (J, chi2), g = out
    mem = SH.per_device_memory()
    res = dict(gpus=a.gpus, n=int(ctrl.n), window=win.labels, n_chi=int(ctrl.n_chi), size=int(ctrl.size),
               J=float(J), parts={k: float(v) for k, v in chi2.items()}, t_lower_s=t_lower,
               t_compile_s=t_compile, t_run_s=times, mem_analysis=mem_an,
               peak_GB_per_device=[m for _, m, _ in mem], devices=[d for d, _, _ in mem],
               limit_GB_per_device=SH.device_limits_gb(),
               lifted_GB=win.lifted.nbytes() / 2 ** 30, audit=audit, value_only=bool(a.value_only))
    print(f"[shard] J {res['J']:.6f}: {V.fmt_parts(chi2)}; run {', '.join(f'{t:.1f}' for t in times)} s; "
          f"peak GB/device {', '.join(f'{m:.2f}' for m in res['peak_GB_per_device'])}", flush=True)
    gh = None
    if g is not None:
        gh = np.asarray(g, np.float64)
        res["grad_norm"] = float(np.linalg.norm(gh))
        res["grad_finite"] = bool(np.all(np.isfinite(gh)))
    if a.ref and gh is not None and not Path(a.ref).exists():
        print(f"[shard] --ref {a.ref} not there (yet): compare later with --compare-only", flush=True)
    elif a.ref and gh is not None:
        res["compare"] = compare(res, gh, a.ref)
        print(f"[shard] vs {a.ref}: {json.dumps(res['compare'])}", flush=True)
    if a.fd and gh is not None:
        res["fd"] = fd_check(a.fd, win, ctrl, z, gh)
    np.savez(a.dump, J=res["J"], g=gh if gh is not None else np.zeros(0), parts=json.dumps(res["parts"]),
             res=json.dumps(res))
    Path(a.dump).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(f"[shard] wrote {a.dump}", flush=True)


if __name__ == "__main__":
    main()
