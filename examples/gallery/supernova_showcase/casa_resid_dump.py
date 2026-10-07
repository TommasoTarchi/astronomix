"""
Residual-physics dump for the Cas A 2000-2022 models (stage 3, residual-physics
worker): evolve a saved 2000 state (casa_xfit --save-state, or a casa_4dvar
analysis state) through every epoch with EXACTLY the casa_4dvar validation
forward, and write per epoch

* the standard observation (and the casa_xfit residual vectors / chi2 parts),
* the image and integrated spectrum split into four additive components:
  thermal emission of the circumstellar part (forward-shocked), thermal emission
  of the ejecta part (reverse-shocked), synchrotron of freshly shocked CSM cells
  (forward-shock filaments) and synchrotron of freshly shocked ejecta cells
  (reverse-shock filaments),
* the synchrotron's time-dependence bookkeeping (radio anchor, fresh-cell
  weight, cutoff distribution),
* (``--variants``) thermal spectra of each component with the electron heating
  (kT_e0), the equilibration rate and the ionisation age (n_e t) rescaled, and
  synchrotron spectra with the acceleration efficiency eta rescaled -- the
  inputs of the offline chi2 scans in ``casa_resid_analyze.py``.

Nothing here changes the fit code: casa_xfit / casa_jaxobs are imported and the
component split only swaps ``casa_jaxobs.emitting_components`` while tracing.

GPU (pq) only at 128^3; ``--trace-only`` checks every function on the CPU:

    cd examples/gallery/supernova_showcase
    pq sub -t a100 -n 1 --name s3-resid-dump -- env PYTHONUNBUFFERED=1 \
        XLA_FLAGS=--xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        ./run.sh casa_resid_dump.py --states R=$W/state2000_R_n128.npz \
        V3=$W/stage2/4dvar/run_R/state2000_4dvar_stage3.npz --variants V3 --out-dir $W/stage3/physics/dump
"""
# ======================= GPU selection (before jax) =======================
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and "--cpu" not in sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# ===========================================================================

import argparse
import contextlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

import casa_jaxobs as J
import casa_xfit as X
import casa_4dvar as V
import casa_4dvar_control as C
import casa_4dvar_data as DA

SY = J.SY
COMP_NAMES = ("th_csm", "th_ej", "sy_csm", "sy_ej")
#: thermal-emission variants (kT_e0 factor, equilibration-rate factor, n_e t factor)
TH_VARIANTS = ([(1.0, 1.0, 1.0)]
               + [(f, 1.0, 1.0) for f in (0.5, 2.0, 4.0, 8.0)]
               + [(1.0, f, 1.0) for f in (0.25, 0.5, 2.0, 4.0)]
               + [(1.0, 1.0, f) for f in (0.25, 0.5, 2.0, 4.0)])
FE_VARIANTS = (0.0,)                  # Fe tracer scale (0 = the Fe line share by difference)
ETA_VARIANTS = (0.5, 1.0, 2.0, 4.0)   # synchrotron cutoff E_c ~ v_s^2 / eta
V_BINS = np.arange(0.0, 10001.0, 500.0)   # km/s, fresh-cell shock-speed histogram


# =============================================================================
# ============ ↓ Component split and synchrotron bookkeeping ↓ ================
# =============================================================================
_EMIT = J.emitting_components


@contextlib.contextmanager
def only_component(mode):
    """While tracing: ``casa_jaxobs.emitting_components`` returns only the
    circumstellar part (``"csm"``, weight 1 - C_ej) or only the ejecta phases
    (``"ej"``); ``"all"`` leaves it alone."""
    if mode == "all":
        yield
        return

    def sel(fields, **kw):
        comps = _EMIT(fields, **kw)
        if len(comps) == 1:                       # no sub-grid split: weight by C_ej
            c = jnp.clip(jnp.asarray(fields["C_ej"]), 0.0, 1.0)
            return [(fields, (1.0 - c) if mode == "csm" else c)]
        return comps[:1] if mode == "csm" else comps[1:]

    J.emitting_components = sel
    try:
        yield
    finally:
        J.emitting_components = _EMIT


def scan_slabs(fields, body, init, slab):
    """``casa_jaxobs.slab_scan`` for every key of ``fields`` (y slabs)."""
    n = fields["rho"].shape[1]
    xs = {k: jnp.moveaxis(jnp.asarray(v), 1, 0).reshape(n // slab, slab, v.shape[0], v.shape[2])
          for k, v in fields.items()}

    @jax.checkpoint
    def step(carry, sl):
        sl = {k: jnp.moveaxis(v, 0, 1) for k, v in sl.items()}
        return body(carry, sl), None

    carry, _ = jax.lax.scan(step, init, xs)
    return carry


def sync_split(fields, *, lecut, band_table, year, plasma_kw, eta=1.0, width_arcsec=2.0, gate_softness=0.15,
               diag_col=None, slab=8):
    """``casa_jaxobs.sync_columns`` (same formulas, same radio anchoring) split
    into the freshly shocked CSM (weight 1 - C_ej) and ejecta (C_ej) cells:
    ``(cols_csm, cols_ej, diag)``, each (n_band, x, z) in counts/s before the
    fitted efficiency. ``diag``: total radio weight, fresh radio weight per
    part, shock-speed histograms (radio- and band-``diag_col``-weighted) and the
    band-weighted mean ln E_cut per part."""
    lec = jnp.asarray(lecut)
    tab = jnp.asarray(band_table)
    lt = jnp.log(jnp.maximum(tab, 1e-38))
    vb = jnp.asarray(V_BINS[1:-1] * 1e5)
    nbin = len(V_BINS) - 1
    nb = tab.shape[-1]
    n = fields["rho"].shape[0]

    def body(carry, sl):
        cc, ce, d = carry
        pl = J.plasma(sl, "xrism_bulk", **(plasma_kw or {}))
        vs2 = J._VS2_PER_TI_MU * jnp.maximum(pl["T_i"], 0.0) / pl["mu_i"]
        w = jnp.where(pl["shocked"], pl["n_b"] * vs2 * 1e-16, 0.0)
        gate = J._WIDTH_CM * width_arcsec / jnp.maximum(jnp.sqrt(vs2) / 4.0, 1e3)
        tss = pl["t_shock_s"]
        fresh = jax.nn.sigmoid((gate - tss) / (gate_softness * gate)) * (tss > 0.0)
        ecut = SY.CUTOFF_KEV_AT_3000 * vs2 / 9.0e16 / eta
        x = jnp.clip((jnp.log(jnp.maximum(ecut, 1e-30)) - lec[0]) / (lec[1] - lec[0]), 0.0, lec.shape[0] - 1.0)
        i = jnp.minimum(jnp.floor(x).astype(jnp.int32), lec.shape[0] - 2)
        f = x - i
        rate = jnp.exp(lt[i] * (1.0 - f)[..., None] + lt[i + 1] * f[..., None])     # (x, s, z, b)
        c = jnp.clip(sl["C_ej"], 0.0, 1.0)
        wf = w * fresh
        cc = cc + jnp.einsum("xszb,xsz->bxz", rate, wf * (1.0 - c))
        ce = ce + jnp.einsum("xszb,xsz->bxz", rate, wf * c)
        vbin = jnp.searchsorted(vb, jnp.sqrt(vs2).ravel())
        rd = rate[..., diag_col] if diag_col is not None else jnp.ones_like(w)
        lne = jnp.log(jnp.maximum(ecut, 1e-30))
        d = dict(
            wsum=d["wsum"] + jnp.sum(w),
            wf_csm=d["wf_csm"] + jnp.sum(wf * (1.0 - c)), wf_ej=d["wf_ej"] + jnp.sum(wf * c),
            xb_csm=d["xb_csm"] + jnp.sum(wf * (1.0 - c) * rd), xb_ej=d["xb_ej"] + jnp.sum(wf * c * rd),
            lne_csm=d["lne_csm"] + jnp.sum(wf * (1.0 - c) * rd * lne),
            lne_ej=d["lne_ej"] + jnp.sum(wf * c * rd * lne),
            hv_radio_csm=d["hv_radio_csm"] + jnp.zeros(nbin).at[vbin].add((wf * (1.0 - c)).ravel()),
            hv_radio_ej=d["hv_radio_ej"] + jnp.zeros(nbin).at[vbin].add((wf * c).ravel()),
            hv_x_csm=d["hv_x_csm"] + jnp.zeros(nbin).at[vbin].add((wf * (1.0 - c) * rd).ravel()),
            hv_x_ej=d["hv_x_ej"] + jnp.zeros(nbin).at[vbin].add((wf * c * rd).ravel()),
            n_fresh=d["n_fresh"] + jnp.sum(fresh))
        return cc, ce, d

    z = jnp.zeros(())
    d0 = dict(wsum=z, wf_csm=z, wf_ej=z, xb_csm=z, xb_ej=z, lne_csm=z, lne_ej=z,
              hv_radio_csm=jnp.zeros(nbin), hv_radio_ej=jnp.zeros(nbin), hv_x_csm=jnp.zeros(nbin),
              hv_x_ej=jnp.zeros(nbin), n_fresh=z)
    cc, ce, d = scan_slabs(fields, body, (jnp.zeros((nb, n, n)), jnp.zeros((nb, n, n)), d0), slab)
    nu_s_nu = SY.RADIO_FREQ_GHZ * 1e9 * 1e-23 * SY.RADIO_FLUX_JY * \
        (1.0 - SY.RADIO_SECULAR_DECLINE_PER_YR) ** (year - SY.RADIO_EPOCH)
    k = nu_s_nu / (jnp.maximum(d["wsum"], 1e-30) * J._E_RADIO_KEV ** (1.0 - SY.RADIO_ALPHA))
    d["k_anchor"] = k
    return cc * k, ce * k, d
# =============================================================================
# ============ ↑ Component split and synchrotron bookkeeping ↑ ================
# =============================================================================


# =============================================================================
# ============ ↓ The decomposed observation (casa_xfit v2 observe) ↓ ==========
# =============================================================================
def make_decomposer(core, img, opts, spec_doppler=None):
    """Jitted functions of (state, ep, p [, knobs]) that reproduce casa_xfit's
    v2 ``observe`` split into components (see the module docstring).
    ``spec_doppler`` = (Ds, D2s), the first / second-order Doppler spectral
    tables stacked per instrument like ``Cs`` (``casa_xfit_obs2._stack("spec",
    ..., kinds=("C", "D", "D2"))``): adds ``th_spec_broad_fo`` -- the thermal
    spectrum with every cell's line-of-sight velocity (first + second order)
    and, optionally, the ions' thermal velocity spread (line broadening, which
    casa_xfit's spectra leave out: they use (C, C) with v_los off)."""
    Vt = img["v2"]
    box, n = core.box, core.n
    WX0, NZ0, _ = J.sky_geometry(box, n, X.D_REF_KPC)
    WX0, NZ0 = jnp.asarray(WX0, jnp.float32), jnp.asarray(NZ0, jnp.float32)
    lng = np.log(np.asarray(img["nh_grid"], np.float64))
    n_nh = len(lng)
    tables0 = Vt["tables0"]
    csm_solar = opts.csm_solar == "on"
    n_sp = Vt["keep"].shape[1]
    kt_interp = opts.kt_interp
    i_b4 = 4                                            # 4.2-6 keV (diagnostic band)
    i_mid = int(np.argmin(np.abs(lng - np.log(1.2))))   # an N_H node near 1.2e22 for the diagnostics

    def fold(T):
        return jnp.moveaxis(T, 0, -2).reshape(*T.shape[1:-1], n_nh * T.shape[-1])

    def geom(p):
        d_kpc = jnp.exp(p["ln_D"])
        scale = X.D_REF_KPC / d_kpc
        ps = jnp.deg2rad(p["psi"])
        sky_w = scale * (jnp.cos(ps) * WX0 - jnp.sin(ps) * NZ0) + p["dw"]
        sky_n = scale * (jnp.sin(ps) * WX0 + jnp.cos(ps) * NZ0) + p["dn"]
        lnh_map = p["ln_nh"] + (p["g_nh_w"] * sky_w + p["g_nh_n"] * sky_n) / 100.0
        R_sky = jnp.sqrt(sky_w ** 2 + sky_n ** 2)
        kp = jax.vmap(lambda kr: jnp.interp(R_sky, Vt["r_grid"], kr, right=0.0))(
            Vt["keep"].reshape(-1, Vt["keep"].shape[-1])).reshape(n_nh, n_sp, *R_sky.shape)
        return SimpleNamespace(scale=scale, sky_w=sky_w, sky_n=sky_n, lnh_map=lnh_map, R_sky=R_sky, kp=kp,
                               tau=X.nh_tents(lnh_map, lng), amp=jnp.exp(p["ln_A"]) * scale ** 2)

    def pkw_of(p, kte_f=1.0, teq_f=1.0, fe_f=1.0):
        k = dict(kT_e_shock_keV=jnp.exp(p["ln_kte"]) * kte_f, teq_scale=jnp.exp(p["ln_teq"]) * teq_f,
                 fe_scale=jnp.exp(p["ln_fe"]) * fe_f)
        if csm_solar:
            k["csm_solar"] = True
        return k

    def sg_of(p):
        return dict(chi=4.0, f_mass=jax.nn.sigmoid(p["lg_fmass"]))

    def mix(g, cols):
        return X.nh_mix(cols.reshape(n_nh, cols.shape[0] // n_nh, *cols.shape[1:]), g.lnh_map, lng, opts.nh_mode)

    def fields_of(st, p):
        fo = core.to_obs_fields(st)
        return X.scale_si(fo, p["ln_si"], csm_solar)

    def project(g, cols_mixed, wi, p):
        nb = cols_mixed.shape[0]
        K = jnp.tensordot(wi, Vt["halo_K"], 1)
        halo_ep = dict(Vt["halo_meta"], kernel=K.reshape(n_nh * nb, *K.shape[-2:]))
        cw = (g.tau[:, None] * cols_mixed[None]).reshape(n_nh * nb, *g.R_sky.shape)
        im = J.project_columns(cw, box_pc=box, distance_kpc=X.D_REF_KPC, halo=halo_ep, npix=img["npix"],
                               pix_arcsec=img["pix"], roll_deg=p["psi"], offset_arcsec=(p["dw"], p["dn"]),
                               sky_scale=g.scale)
        return im.reshape(n_nh, nb, *im.shape[-2:]).sum(0)

    def th_band(mode):
        def f(st, ep, p):
            g = geom(p)
            fo = fields_of(st, p)
            wi = ep["wi"]
            Cb = jnp.tensordot(wi, Vt["Cb"], 1); Db = jnp.tensordot(wi, Vt["Db"], 1)
            with only_component(mode):
                cols = J.band_columns(fo, tables0, box_pc=box, distance_kpc=X.D_REF_KPC, subgrid=sg_of(p),
                                      band_tables=(fold(Cb), fold(Db)), plasma_kw=pkw_of(p), kt_interp=kt_interp)
            cm = g.amp * mix(g, cols)                                       # (6, x, z)
            return project(g, cm, wi, p), cm.sum((1, 2))
        return jax.jit(f)

    def th_spec(mode):
        def f(st, ep, p, kte_f, teq_f, net_f, fe_f):
            g = geom(p)
            fo = fields_of(st, p)
            fo = dict(fo, density_time=fo["density_time"] * net_f)
            Cc = jnp.tensordot(ep["wi"], Vt["Cs"], 1)
            with only_component(mode):
                cc = J.band_columns(fo, tables0, box_pc=box, distance_kpc=X.D_REF_KPC, subgrid=sg_of(p),
                                    band_tables=(fold(Cc), fold(Cc)), plasma_kw=pkw_of(p, kte_f, teq_f, fe_f),
                                    v_los_kms=False, kt_interp=kt_interp)
            tot = (g.amp * cc).reshape(n_nh, n_sp, *g.R_sky.shape) * g.kp
            return jnp.sum(mix(g, tot.reshape(n_nh * n_sp, *g.R_sky.shape)), (1, 2))
        return jax.jit(f)

    def th_spec_broad(mode, thermal):
        Ds, D2s = spec_doppler

        def f(fo, wi, p):
            g = geom(p)
            fo = X.scale_si(fo, p["ln_si"], csm_solar)
            Cc, Dc, D2c = (jnp.tensordot(wi, T, 1) for T in (Vt["Cs"], Ds, D2s))
            with only_component(mode):
                cc = J.band_columns(fo, tables0, box_pc=box, distance_kpc=X.D_REF_KPC, subgrid=sg_of(p),
                                    band_tables=(fold(Cc), fold(Dc), fold(D2c)), plasma_kw=pkw_of(p),
                                    v_los_kms=True, thermal_broadening=thermal, kt_interp=kt_interp)
            tot = (g.amp * cc).reshape(n_nh, n_sp, *g.R_sky.shape) * g.kp
            return jnp.sum(mix(g, tot.reshape(n_nh * n_sp, *g.R_sky.shape)), (1, 2))
        return jax.jit(f)

    def th_spec_fo(mode):
        def f(fo, wi, p):
            g = geom(p)
            fo = X.scale_si(fo, p["ln_si"], csm_solar)
            Cc = jnp.tensordot(wi, Vt["Cs"], 1)
            with only_component(mode):
                cc = J.band_columns(fo, tables0, box_pc=box, distance_kpc=X.D_REF_KPC, subgrid=sg_of(p),
                                    band_tables=(fold(Cc), fold(Cc)), plasma_kw=pkw_of(p),
                                    v_los_kms=False, kt_interp=kt_interp)
            tot = (g.amp * cc).reshape(n_nh, n_sp, *g.R_sky.shape) * g.kp
            return jnp.sum(mix(g, tot.reshape(n_nh * n_sp, *g.R_sky.shape)), (1, 2))
        return jax.jit(f)

    @jax.jit
    def sync_band(st, ep, p, eta):
        g = geom(p)
        fo = fields_of(st, p)
        Sb = jnp.tensordot(ep["wi"], Vt["Sb"], 1)                       # (nh, ecut, 6)
        cc, ce, d = sync_split(fo, lecut=Vt["lecut"], band_table=fold(Sb), year=ep["year"],
                               plasma_kw=pkw_of(p), eta=eta, diag_col=i_mid * 6 + i_b4)
        es = jnp.exp(p["ln_sync"])
        mc, me = es * mix(g, cc), es * mix(g, ce)
        return project(g, mc, ep["wi"], p), project(g, me, ep["wi"], p), mc.sum((1, 2)), me.sum((1, 2)), d

    @jax.jit
    def sync_spec(st, ep, p, eta):
        g = geom(p)
        fo = fields_of(st, p)
        Sc = jnp.tensordot(ep["wi"], Vt["Ss"], 1)                       # (nh, ecut, 31)
        cc, ce, _ = sync_split(fo, lecut=Vt["lecut"], band_table=fold(Sc), year=ep["year"], plasma_kw=pkw_of(p),
                               eta=eta, slab=4)
        es = jnp.exp(p["ln_sync"])
        out = []
        for c in (cc, ce):
            tot = (es * c).reshape(n_nh, n_sp, *g.R_sky.shape) * g.kp
            out.append(jnp.sum(mix(g, tot.reshape(n_nh * n_sp, *g.R_sky.shape)), (1, 2)))
        return out[0], out[1]

    @jax.jit
    def cell_diag(st, p):
        """Radial / shock diagnostics of the state: mass, EM and 4.2-6 keV-free
        bookkeeping per part (shocked CSM, shocked ejecta), T_i of fresh cells."""
        fo = fields_of(st, p)
        pl = J.plasma(fo, "xrism_bulk", **pkw_of(p))
        c = jnp.clip(fo["C_ej"], 0.0, 1.0)
        sh = pl["shocked"].astype(jnp.float32)
        em = pl["n_e"] * pl["n_b"] * sh
        kT = pl["kT"]
        net = pl["net"]
        out = {}
        for nm, w in (("csm", (1.0 - c) * em), ("ej", c * em)):
            ws = jnp.maximum(jnp.sum(w), 1e-30)
            out[f"em_{nm}"] = jnp.sum(w)
            out[f"kTe_{nm}"] = jnp.sum(w * kT) / ws
            out[f"lnet_{nm}"] = jnp.sum(w * jnp.log10(jnp.maximum(net, 1e6))) / ws
            out[f"hist_lnet_{nm}"] = jnp.histogram(jnp.log10(jnp.maximum(net, 1e6)).ravel(),
                                                   bins=jnp.linspace(8.0, 13.0, 26), weights=w.ravel())[0]
            out[f"hist_kTe_{nm}"] = jnp.histogram(jnp.log10(jnp.maximum(kT, 1e-3)).ravel(),
                                                  bins=jnp.linspace(-1.5, 1.5, 31), weights=w.ravel())[0]
        return out

    extra = {}
    if spec_doppler is not None:
        extra = dict(th_spec_broad={(m, t): th_spec_broad(m, t) for m in ("csm", "ej") for t in (False, True)},
                     th_spec_fo={m: th_spec_fo(m) for m in ("csm", "ej")})
    return SimpleNamespace(**extra, th_band={m: th_band(m) for m in ("csm", "ej")},
                           th_spec={m: th_spec(m) for m in ("csm", "ej")}, sync_band=sync_band,
                           sync_spec=sync_spec, cell_diag=cell_diag,
                           sky=jax.jit(lambda p: (geom(p).sky_w, geom(p).sky_n)))
# =============================================================================
# ============ ↑ The decomposed observation ↑ =================================
# =============================================================================


def to_np(t):
    return jax.tree.map(lambda v: np.asarray(v), t)


def run_state(label, path, a, opts, obs, img, variants, out_dir):
    t_start = time.time()
    fields, meta, lay, theta_b = V.load_background(path, a.coarsen)
    if opts.kdop == "fixed0":
        theta_b["ln_kdop"] = 0.0
    t0_year = float(meta["epoch_year"])
    ic = dict(box=float(meta["box"]), num_cells=int(meta["num_cells"]), age=float(meta["age"]))
    a.state = path
    core0 = X.make_forward_core(ic, *DA.subset(obs, img, [0]), opts=opts, ic_path=path,
                                config_overrides=V.solver_overrides(a, "approx"))
    ctrl = C.Control(fields, lay, core0.geom, globals_b={k: theta_b[k] for k in V.GLOBALS},
                     globals_scale={k: 1.0 for k in V.GLOBALS}, dtype=jnp.float32,
                     r_ref=float(np.mean(meta["r_fs_pc"])) if "r_fs_pc" in meta else None)
    del core0
    E = len(obs["epochs"])
    win = V.Window(a, opts, meta, ctrl, theta_b, obs, img, np.arange(E), tangent="approx", t0_year=t0_year,
                   dtype=jnp.float32)
    core = win.core
    z0 = jnp.zeros(ctrl.size, jnp.float32)
    chi, xi_c, xi_g = ctrl.split(z0)
    x0 = ctrl.state(chi, xi_c)
    p = win.params(ctrl.globals_of(xi_g))
    D = make_decomposer(core, img, opts)
    integrate = jax.jit(core.integrator(x0.shape))
    observe = jax.jit(lambda st, ep, pp: core.observer(pp)(st, ep))
    xs = core.xs_all
    order = core.order
    years_sorted = np.asarray(obs["years"])[order]
    resp_img = img.get("resp_corr"); resp_spec = img["spec"].get("resp_corr")
    rec = {k: [] for k in ("outs", "img_comp", "spec_comp", "band_tot_comp", "sync_diag", "cell_diag",
                           "th_var", "fe_var", "sy_var")}
    st = x0
    dts = np.asarray(win.dts)
    for k in range(E):
        t0 = time.time()
        if not (k == 0 and win.first_at_x0):
            st = integrate(st, jnp.asarray(dts[k], jnp.float32))
        ep = jax.tree.map(lambda v: v[k], xs)
        e = int(order[k])                                     # index in obs order
        outs = observe(st, ep, p)
        rec["outs"].append(to_np(outs))
        ims, tots = [], []
        for m in ("csm", "ej"):
            im, tb = D.th_band[m](st, ep, p)
            ims.append(im); tots.append(tb)
        imc, ime, tbc, tbe, sd = D.sync_band(st, ep, p, jnp.float32(1.0))
        ims += [imc, ime]; tots += [tbc, tbe]
        ims = np.stack([np.asarray(i) for i in ims])
        if resp_img is not None:
            ims = ims * np.asarray(resp_img[e])[None]
        rec["img_comp"].append(ims.astype(np.float32))
        rec["band_tot_comp"].append(np.stack([np.asarray(t) for t in tots]))
        rec["sync_diag"].append(to_np(sd))
        rec["cell_diag"].append(to_np(D.cell_diag(st, p)))
        one = jnp.float32(1.0)
        sp = [D.th_spec[m](st, ep, p, one, one, one, one) for m in ("csm", "ej")]
        sc, se = D.sync_spec(st, ep, p, one)
        sp = np.stack([np.asarray(s) for s in sp + [sc, se]])
        rsp = np.asarray(resp_spec[e]) if resp_spec is not None else 1.0
        rec["spec_comp"].append(sp * rsp)
        if variants:
            tv = [[np.asarray(D.th_spec[m](st, ep, p, *(jnp.float32(x) for x in v), one)) * rsp
                   for m in ("csm", "ej")] for v in TH_VARIANTS]
            fv = [[np.asarray(D.th_spec[m](st, ep, p, one, one, one, jnp.float32(f))) * rsp
                   for m in ("csm", "ej")] for f in FE_VARIANTS]
            sv = [[np.asarray(s) * rsp for s in D.sync_spec(st, ep, p, jnp.float32(eta))] for eta in ETA_VARIANTS]
            rec["th_var"].append(np.asarray(tv)); rec["fe_var"].append(np.asarray(fv))
            rec["sy_var"].append(np.asarray(sv))
        if a.save_fields and obs["epochs"][e] in a.save_fields:
            fo = core.to_obs_fields(st)
            keep = {nm: np.asarray(st[ctrl.idx[nm]], np.float32) for nm in ("vx", "vz") if nm in ctrl.idx}
            np.savez(out_dir / f"fields_{label}_{obs['epochs'][e]}.npz",
                     **{kk: np.asarray(v, np.float32) for kk, v in fo.items()}, **keep,
                     box=core.box, year=years_sorted[k])
        im_ref = np.asarray(outs[3]) * (np.asarray(resp_img[e]) if resp_img is not None else 1.0)
        clo_i = ims.sum(0).sum((1, 2)) / np.maximum(im_ref.sum((1, 2)), 1e-30)
        clo_s = sp.sum(0) / np.maximum(np.asarray(outs[6]) * rsp, 1e-30)
        print(f"[{label}] closure image/band {np.round(clo_i, 4)}; spectrum min/max {clo_s.min():.4f}/{clo_s.max():.4f}",
              flush=True)
        print(f"[{label}] epoch {obs['epochs'][e]} ({years_sorted[k]:.2f}) done in {time.time() - t0:.1f} s; "
              f"band tot th_csm/th_ej/sy_csm/sy_ej (4.2-6) = "
              + " ".join(f"{rec['band_tot_comp'][-1][c, 4]:.3g}" for c in range(4))
              + f"; spec sum {rec['spec_comp'][-1].sum(1)}", flush=True)
    # ---- assemble exactly like the validator, residual parts ----
    outs = tuple(jnp.asarray(np.stack([o[i] for o in rec["outs"]])) for i in range(7))
    model = core.assemble(p, outs)
    theta = jnp.stack([p[kk] for kk in X.PARAM_NAMES])
    largs = V.likelihood_args(a, opts)
    parts = X.residual_parts(model, obs, img, theta, largs)
    chi2 = {kk: float(jnp.sum(v ** 2)) for kk, v in parts.items()}
    inv = np.argsort(order)                                    # sorted -> obs order
    save = dict(epochs=np.array(obs["epochs"]), years=np.asarray(obs["years"]), label=label, state=str(path),
                theta=np.asarray(theta), names=np.array(X.PARAM_NAMES),
                chi2=json.dumps(chi2), comp_names=np.array(COMP_NAMES),
                th_variants=np.array(TH_VARIANTS), fe_variants=np.array(FE_VARIANTS),
                eta_variants=np.array(ETA_VARIANTS), v_bins=V_BINS)
    for kk, v in parts.items():
        save[f"res_{kk}"] = np.asarray(v)
    for kk, v in model.items():
        if kk != "state0":
            save[f"model_{kk}"] = np.asarray(v)
    for kk in ("img_comp", "spec_comp", "band_tot_comp", "th_var", "fe_var", "sy_var"):
        if rec[kk]:
            save[kk] = np.stack(rec[kk])[inv]
    for kk in rec["sync_diag"][0]:
        save[f"sync_{kk}"] = np.stack([d[kk] for d in rec["sync_diag"]])[inv]
    for kk in rec["cell_diag"][0]:
        save[f"cell_{kk}"] = np.stack([d[kk] for d in rec["cell_diag"]])[inv]
    sw, sn = D.sky(p)
    save.update(sky_w=np.asarray(sw), sky_n=np.asarray(sn))
    np.savez_compressed(out_dir / f"dump_{label}.npz", **save)
    print(f"[{label}] chi2: " + ", ".join(f"{kk} {v:.1f}" for kk, v in chi2.items()) +
          f"; obs total {V.obs_total(chi2):.1f}; wall {time.time() - t_start:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="+", required=True, help="LABEL=path (casa_xfit_state files)")
    ap.add_argument("--variants", nargs="*", default=[], help="labels to run the emission-variant spectra for")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-fields", nargs="*", default=["2000", "2009", "2019"])
    ap.add_argument("--sigma-model", type=float, default=5.0)
    ap.add_argument("--coarsen", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="CPU (NATIVE_JAX): tests only")
    ap.add_argument("--trace-only", action="store_true", help="eval_shape every function on epoch 0 and exit")
    X.add_fix_arguments(ap)
    a = ap.parse_args()
    a.ic = a.states[0].split("=", 1)[1]
    for k in ("save_state", "no_history"):
        setattr(a, k, getattr(a, k, None))
    a.remat, a.ckpt, a.smooth_latch, a.cpu_test = "axis", 16, False, a.cpu
    opts = X.resolve_options(a)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obs, img = DA.load_all(opts, table_dir=a.table_dir)
    print(f"[dump] data loaded: epochs {list(obs['epochs'])}", flush=True)
    for spec in a.states:
        label, path = spec.split("=", 1)
        if a.trace_only:
            trace(label, path, a, opts, obs, img)
        else:
            run_state(label, path, a, opts, obs, img, label in a.variants, out_dir)


def trace(label, path, a, opts, obs, img):
    fields, meta, lay, theta_b = V.load_background(path, a.coarsen)
    theta_b["ln_kdop"] = 0.0
    ic = dict(box=float(meta["box"]), num_cells=int(meta["num_cells"]), age=float(meta["age"]))
    a.state = path
    core = X.make_forward_core(ic, obs, img, opts=opts, ic_path=path, config_overrides=V.solver_overrides(a, "approx"))
    lay = {int(k): v for k, v in lay.items()}
    x0 = jnp.asarray(np.stack([np.asarray(fields[lay[i]], np.float32) for i in range(len(lay))]))
    p = {k: jnp.asarray(theta_b[k], jnp.float32) for k in X.PARAM_NAMES}
    D = make_decomposer(core, img, opts)
    ep = jax.tree.map(lambda v: v[0], core.xs_all)
    one = jnp.float32(1.0)
    t0 = time.time()
    for nm, fn, args in (("th_band csm", D.th_band["csm"], (x0, ep, p)),
                         ("th_spec ej", D.th_spec["ej"], (x0, ep, p, one, one, one, one)),
                         ("sync_band", D.sync_band, (x0, ep, p, one)),
                         ("sync_spec", D.sync_spec, (x0, ep, p, one)),
                         ("cell_diag", D.cell_diag, (x0, p))):
        sh = jax.eval_shape(fn, *args)
        print(f"[trace] {nm}: {jax.tree.map(lambda s: s.shape, sh)} ({time.time() - t0:.0f} s)", flush=True)
    if a.coarsen:     # small grid: run it (checks the component closure against observe)
        observe = jax.jit(lambda st, e, pp: core.observer(pp)(st, e))
        outs = observe(x0, ep, p)
        ims = [D.th_band[m](x0, ep, p)[0] for m in ("csm", "ej")]
        s = D.sync_band(x0, ep, p, one)
        tot = ims[0] + ims[1] + s[0] + s[1]
        ref = outs[3]
        print("[trace] image closure per band (sum of components / observe):",
              np.asarray(tot.sum((1, 2)) / ref.sum((1, 2))), flush=True)
        sp = [D.th_spec[m](x0, ep, p, one, one, one, one) for m in ("csm", "ej")] + list(D.sync_spec(x0, ep, p, one))
        print("[trace] spectrum closure:", np.asarray(sum(sp) / outs[6]), flush=True)
        print("[trace] sync diag:", jax.tree.map(lambda v: np.asarray(v).round(4).tolist(), s[4]), flush=True)


if __name__ == "__main__":
    main()
