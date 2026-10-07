"""
Line broadening in the integrated Cas A spectrum (stage-3 residual physics):
casa_xfit's spectra are computed with the Doppler terms OFF (band tables
(C, C), v_los off), so the ejecta's line-of-sight velocity spread (+-2000-5000
km/s in Cas A) and the ions' thermal spread never broaden the Si / S / Fe-K
lines. This computes, on fields saved by ``casa_resid_dump.py --save-fields``,
the thermal spectrum of each component (CSM / ejecta) (a) as the fit does,
(b) with every cell's v_los to second order (C + beta D + beta^2/2 D2), and
(c) plus the thermal ion spread, and writes the ratios per spectral bin.

    pq sub -t a100 -n 1 --name s3-resid-broad -- env PYTHONUNBUFFERED=1 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 ./run.sh casa_resid_broaden.py \
        --dump $W/stage3/physics/dump/dump_V3.npz --fields $W/stage3/physics/dump/fields_V3_{2000,2009,2019}.npz \
        --out $W/stage3/physics/dump/broaden_V3.npz
"""
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and "--cpu" not in sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

import casa_jaxobs as J
import casa_xfit as X
import casa_xfit_obs2 as O2
import casa_4dvar_data as DA
import casa_resid_dump as RD


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="casa_resid_dump npz (theta / names)")
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gain-kms", type=float, default=0.0,
                    help="also the thermal spectrum with EVERY cell at this line-of-sight velocity (first + second "
                         "order): (S - base) / beta is the spectrum's derivative w.r.t. an energy-scale (gain) error")
    ap.add_argument("--skip-broadening", action="store_true")
    X.add_fix_arguments(ap)
    a = ap.parse_args()
    d = np.load(a.dump, allow_pickle=True)
    a.ic = str(d["state"])
    for k in ("save_state", "no_history"):
        setattr(a, k, getattr(a, k, None))
    opts = X.resolve_options(a)
    obs, img = DA.load_all(opts, table_dir=a.table_dir)
    st = [J.load_binned_stack(i, "spec", history=opts.history, second_order=True, table_dir=a.table_dir)
          for i in O2.INSTRUMENTS]
    spec = {k: np.stack([s_[k] for s_ in st]) for k in ("C", "D", "D2")}
    del st
    if not np.allclose(spec["C"], np.asarray(img["v2"]["Cs"])):
        raise ValueError("the stacked C spec tables differ from casa_xfit's")
    sd = (jnp.asarray(spec["D"]), jnp.asarray(spec["D2"]))
    del spec
    f0 = np.load(a.fields[0])
    n = f0["rho"].shape[0]
    core = X.make_forward_core(dict(box=float(f0["box"]), num_cells=n, age=0.0), obs, img, opts=opts, ic_path=a.ic)
    D = RD.make_decomposer(core, img, opts, spec_doppler=sd)
    p = {k: jnp.asarray(v, jnp.float32) for k, v in zip([str(s) for s in d["names"]], np.asarray(d["theta"]))}
    years = np.asarray(obs["years"])
    out = dict(years=[], base=[], vlos=[], vlos_th=[], gain=[])
    keys = ["rho", "press", "vy", "C_ej", "C_Fe", "C_Si", "C_O", "C_He", "shocked_fraction", "time_since_shock",
            "density_time"]
    for fp in a.fields:
        t0 = time.time()
        f = np.load(fp)
        fo = {k: jnp.asarray(f[k]) for k in keys if k in f.files}
        e = int(np.argmin(np.abs(years - float(f["year"]))))
        wi = jnp.asarray(img["spec"]["inst_w"][e])
        rsp = np.asarray(img["spec"]["resp_corr"][e]) if img["spec"].get("resp_corr") is not None else 1.0
        row = {k: [] for k in ("base", "vlos", "vlos_th", "gain")}
        for m in ("csm", "ej"):
            row["base"].append(np.asarray(D.th_spec_fo[m](fo, wi, p)) * rsp)
            if not a.skip_broadening:
                row["vlos"].append(np.asarray(D.th_spec_broad[(m, False)](fo, wi, p)) * rsp)
                row["vlos_th"].append(np.asarray(D.th_spec_broad[(m, True)](fo, wi, p)) * rsp)
            if a.gain_kms:
                fg = dict(fo, vy=jnp.full_like(fo["vy"], a.gain_kms / 1000.0))   # code velocity = 1000 km/s
                row["gain"].append(np.asarray(D.th_spec_broad[(m, False)](fg, wi, p)) * rsp)
        for k, v in row.items():
            if v:
                out[k].append(np.stack(v))
        out["years"].append(float(f["year"]))
        key = "gain" if a.skip_broadening else "vlos_th"
        r = (row[key][0] + row[key][1]) / (row["base"][0] + row["base"][1])
        print(f"[broaden] {fp}: {time.time() - t0:.0f} s; total ratio {key} / base per bin: "
              + " ".join(f"{x:.4f}" for x in r), flush=True)
    np.savez(a.out, **{k: np.asarray(v) for k, v in out.items() if len(v)}, edges=X.SPEC_EDGES, gain_kms=a.gain_kms)
    print("[broaden] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
