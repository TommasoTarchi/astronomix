"""
casa_xfit A/B at one theta for the stage-4 changes: ONE hydro run (the 146-yr IC
evolved through every epoch at ``theta``), re-observed with several option
sets, every chi2 part printed per variant (training / all epochs / held-out,
as ``casa_xfit --exclude-epochs``) and the differences against the stage-3
('old') likelihood.

The hydro does not depend on any stage-4 option (they change only the
observation model and the likelihood), so the solver runs once
(``forward.evolve``). Each variant then builds its own observer
(``make_forward_core``) and data terms (``attach_stage4``, ``apply_obs_masks``).

    pq sub -t a100 -n 1 --name s4-xfit-ab -- env PYTHONUNBUFFERED=1 \\
        XLA_FLAGS=--xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \\
        ./run.sh casa_xfit_compare.py --ic $W/pluto146_n128_solarcsm.npz --theta-json $W/xfit_Rp.json \\
        --exclude-epochs 2019 2022 --out $W/stage4/xfit/compare_Rp.json
"""
# ==== GPU selection ====
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and "--cpu-test" not in sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
# ruff: noqa: E402
# =======================
import argparse
import json
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import casa_pluto_diff as PD
import casa_xfit as X
import casa_xfit_obs2 as O2

NEW = {k: v[0] for k, v in X.STAGE4_OPTIONS.items()}
#: variant -> stage-4 overrides of the OLD (stage-3) options; "new" = every default
VARIANTS = {
    "old": {},
    "bkg_annulus": dict(background="annulus"),
    "bkg_particle": dict(background="particle"),
    "bkg_measured": dict(background="measured"),
    "outline_mask": dict(outline_mask="inner-arc"),
    "pm_mask300": dict(pm_mask_extra="300"),
    "heldout_sigma": dict(heldout_sigma="train"),
    "sync_trend": dict(sync_trend="on"),
    "gain": dict(spec_gain="profile"),
    "soft": dict(spec_soft="profile"),
    "gain_soft": dict(spec_gain="profile", spec_soft="profile"),
    "broad_vlos": dict(spec_broadening="vlos"),
    "broad_thermal": dict(spec_broadening="thermal"),
    "kte_fixed": dict(kte="fixed"),
    "new_kte_free": dict(NEW, kte="free"),
    "new": dict(NEW),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ic", required=True)
    ap.add_argument("--theta-json", required=True, help="casa_xfit --out json (theta + names)")
    ap.add_argument("--exclude-epochs", nargs="+", default=["2019", "2022"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--no-doppler", action="store_true")
    ap.add_argument("--summarize", nargs="*", default=["old", "new"], help="variants to run casa_xfit.summarize on")
    ap.add_argument("--save-models", default=None, help="npz prefix: save each variant's model dict")
    ap.add_argument("--cpu-test", action="store_true",
                    help="CPU pipeline test: NATIVE_JAX backend (tiny --ic only)")
    ap.add_argument("--table-dir", default=str(X.J.TABLE_DIR))
    a = ap.parse_args()
    largs = dict(sigma_static=0.5, sigma_temporal=0.075, img_weight=1.0 / 6.0, spectra=True,
                 sigma_spec_static=0.13, sigma_spec_temporal=0.046, sigma_model=5.0, doppler=not a.no_doppler)
    th_js = json.load(open(a.theta_json))
    tj = dict(zip(th_js["names"], th_js["theta"]))
    theta0 = np.array([float(tj.get(k, X.PRIOR[k][0])) for k in X.PARAM_NAMES])
    print(f"[ab] theta from {a.theta_json}; padded with prior means: "
          f"{[k for k in X.PARAM_NAMES if k not in tj]}", flush=True)
    dtype = jnp.float64 if a.x64 else jnp.float32
    ic_path = a.ic
    ic = dict(np.load(ic_path))
    overrides = None
    if a.cpu_test:
        from astronomix.option_classes.simulation_config import BackendConfig, NATIVE_JAX
        overrides = dict(backend_config=BackendConfig(backend=NATIVE_JAX))

    # ---- data: the union of every variant's needs, loaded once ----
    t0 = time.time()
    base = X.resolve_auto(X.default_options(stage3=True), ic_path)
    obs0 = PD.load_observations(pm_files=base.pm_files, pm_mask=base.pm_mask)
    obs0["doppler"] = O2.load_doppler_data()
    img0 = X.load_image_data(obs0["epochs"], obs0["years"], block=16, r_max=150.0, obs=base.obs,
                             history=base.history if hasattr(base, "history") else True,
                             table_dir=a.table_dir, spec_kinds=("C", "D", "D2"))
    img0["spec"] = X.load_spectrum_data(obs0["epochs"], img0)
    X.attach_responses(obs0, img0, base)
    print(f"[ab] data {time.time() - t0:.0f} s", flush=True)

    # ---- the hydro, once ----
    fwd = X.make_forward(ic_path, obs0, img0, opts=base, config_overrides=overrides)
    theta = jnp.asarray(theta0, dtype)
    t0 = time.time()
    st0, rest = jax.block_until_ready(jax.jit(fwd.evolve)(theta))
    diag = {k: v for k, v in jax.jit(fwd.ic_diag)(theta).items()}
    print(f"[ab] evolution {time.time() - t0:.0f} s; states {rest.shape}", flush=True)
    order, E = fwd.core.order, len(obs0["epochs"])

    def state_at(k):                         # k-th epoch in SORTED order
        return st0 if k == 0 else rest[k - 1]

    results = {}
    for name in a.variants:
        try:
            run_variant(name, a, results, obs0, img0, base, ic, ic_path, overrides, theta0, dtype, largs, diag,
                        state_at, E, fwd)
        except Exception:                                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[ab] variant {name} FAILED; continuing", flush=True)
            results[name] = dict(failed=True)
        json.dump(results, open(a.out, "w"), indent=1, default=float)
        jax.clear_caches()
    report(results, a)


def run_variant(name, a, results, obs0, img0, base, ic, ic_path, overrides, theta0, dtype, largs, diag, state_at,
                E, fwd):
    if True:
        t0 = time.time()
        ov = VARIANTS[name]
        o = X.resolve_auto(X.default_options(stage3=True, **ov), ic_path)
        o.history = getattr(base, "history", True)
        obs = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs0.items()}
        X.apply_obs_masks(obs, o)
        img = dict(img0)
        img["spec"] = dict(img0["spec"])
        X.attach_stage4(obs, img, o)
        th = np.array(theta0)
        if o.kte == "fixed":
            th[X.PARAM_NAMES.index("ln_kte")] = np.log(X.KTE_FIXED_KEV)
        thj = jnp.asarray(th, dtype)
        core = X.make_forward_core(ic, obs, img, opts=o, config_overrides=overrides,
                                   ic_path=ic_path)
        observe = jax.jit(lambda st, ep, t: core.observer(dict(zip(X.PARAM_NAMES, t)))(st, ep))
        outs = [observe(state_at(k), jax.tree.map(lambda v: v[k], core.xs_all), thj) for k in range(E)]
        outs = tuple(jnp.stack([ob[i] for ob in outs]) for i in range(len(outs[0])))
        model = core.assemble(dict(zip(X.PARAM_NAMES, thj)), outs)
        model.update(t_conv=diag["t_conv"], n_h_wind=diag["n_h_wind"], ylm_delta=diag["ylm_delta"],
                     **{k: v for k, v in diag.items() if k.startswith("ic_")})
        args = SimpleNamespace(**largs, opts=o)
        if a.exclude_epochs:
            excl = X.EpochExclusion(obs, img, a.exclude_epochs, args, o)
            ev = excl.evaluate(model, thj, name)
            if name in a.summarize:
                X.summarize(excl.model(model), excl.obs, excl.img, thj, excl.args, f"{name} [train]", fwd.budget0)
        else:
            parts = X.residual_parts(model, obs, img, thj, args)
            ev = dict(all={k: float(jnp.sum(v ** 2)) for k, v in parts.items()})
            ev["all_total"] = sum(ev["all"].values())
            if name in a.summarize:
                X.summarize(model, obs, img, thj, args, name, fwd.budget0)
        results[name] = dict(ev, options={k: getattr(o, k) for k in X.STAGE4_OPTIONS}, seconds=time.time() - t0)
        if a.save_models:
            np.savez_compressed(f"{a.save_models}_{name}.npz", theta=th, names=np.array(X.PARAM_NAMES),
                                epochs=np.array(obs["epochs"]),
                                **{k: np.asarray(v) for k, v in model.items()})
        print(f"[ab] {name}: {time.time() - t0:.0f} s", flush=True)


def report(results, a):
    # ---- Delta table against 'old' ----
    if "old" in results:
        b = results["old"]
        for sec in ("train", "all", "heldout"):
            if sec not in b:
                continue
            keys = [k for k in b[sec] if not any(k.endswith(f"_{e}") for e in (a.exclude_epochs or []))]
            print(f"\n[ab] Delta chi2 vs old ({sec}); old: " + ", ".join(f"{k} {b[sec][k]:.1f}" for k in keys))
            for name, r in results.items():
                if name == "old" or r.get("failed") or sec not in r:
                    continue
                dd = {k: r[sec].get(k, 0.0) - b[sec].get(k, 0.0) for k in set(keys) | set(r[sec])
                      if not any(k.endswith(f"_{e}") for e in (a.exclude_epochs or []))}
                tot = (r.get(f"{sec}_total", 0.0) - b.get(f"{sec}_total", 0.0))
                print(f"  {name:14s} total {tot:+8.1f} | " + ", ".join(f"{k} {v:+.1f}" for k, v in sorted(dd.items())
                                                                       if abs(v) > 0.05), flush=True)
    print(f"[ab] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
