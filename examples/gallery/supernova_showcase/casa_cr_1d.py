"""
Cassiopeia A: is cosmic-ray back-reaction at the forward shock a key factor?

A cheap, decisive in-house 1D test of the CR hypothesis (Stage 1, worker
``cr``, 2026-09-25), so that the verdict of the literature audit
(``audit_2026_09_25/lit_cr.md``: "CRs are not the key missing factor") does
not rest on the literature alone.

**Model.** The calibrated 1D ejecta-into-wind explosion of
``casa_calibrate_1d.py`` (CALIBRATION.md Result 27: E = 2.43e51 erg,
M_ej = 3.0 Msun, n = 9 envelope, inner slope delta = 0.82, r^-2 wind
n_w = 0.925 cm^-3 at 2.5 pc + 0.1 cm^-3 floor, r0 = 0.05 pc, 4000 cells to
4 pc), run in the finite-volume solver with the two-fluid gas + CR model
(``astronomix._modules._cosmic_rays``, CRs as a gamma = 4/3 fluid advected
with the gas, no diffusion/streaming) and diffusive shock acceleration at the
FORWARD shock only (``shock_selection = OUTERMOST_SHOCK``; the Mach criterion
only ever flags shocks whose upstream is at larger radius, so the reverse
shock is never injected into either way). A fraction zeta of the energy
dissipated at the shock goes into CRs every step (Pfrommer et al. 2017);
the ``escape_fraction`` variant removes f_esc of that CR energy from the
system at injection (upstream escape: an energy sink at the shock).
Also an Orlando-like model (n = 7 flat-core envelope, E = 1.5e51, M_ej = 3.3,
n_w = 0.8).

**Measured** at 319 yr (Chandra 2000) and 341 yr (2022): r_FS, V_FS,
m = V t / R (local fit of ln r_FS vs ln t over +-15 yr), r_RS (outer edge of
the homologous ejecta), r_RS/r_FS, the FS - CD gap (CD = the Lagrangian mass
coordinate of the ejecta edge), post-shock density (outer 5 % of the shocked
wind), E_CR in the remnant, escaped energy and P_cr/P_th behind the shock.
For zeta = 0.1 and 0.3, E is re-tuned (secant in ln E) to recover the
zeta = 0 r_FS at 319 yr, and the changes are reported at fixed r_FS.

The CR injection itself is validated in ``pytests/cosmic_rays/`` (Pfrommer
tube to <0.7 %, spherical Sedov against the exact self-similar two-fluid
solution to 0.2 % in radius at 2001 cells).

Usage (CPU, float64; each run ~1-3 min, the suite runs them in parallel)::

    ./run.sh casa_cr_1d.py --suite                      # everything + table + figure
    ./run.sh casa_cr_1d.py --one --zeta 0.1 --f-esc 0.5 --out run.npz
    ./run.sh casa_cr_1d.py --report                     # table + figure from npz on disk
"""

# ==== precision / device ====
# 1D and cheap: float64 on the CPU (the GPUs of the login node belong to
# others). ``--gpu`` is honoured for completeness (autocvd only if the queue
# has not already set CUDA_VISIBLE_DEVICES).
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
if "--gpu" not in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as const

# jax
import jax.numpy as jnp

# astronomix
from astronomix import (
    SPHERICAL,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    HLL,
    NATIVE_JAX,
    POSITIVITY_HARD_FLOOR,
    BackendConfig,
    PositivityConfig,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    OUTERMOST_SHOCK,
    STRONGEST_SHOCK,
    CosmicRayConfig,
    CosmicRayParams,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)

# showcase helpers
from _common import GAMMA, MASS_PER_NUCLEUS, ejecta_radial_shape, snr_code_units
from casa_calibrate_1d import measure_snapshot, wind_number_density

DEFAULT_OUTDIR = Path("/export/data/lstorcks/casa_orlando150/work/stage1/cr/casa_cr_1d")
EPOCH_AGES = (319.0, 341.0)          # Chandra 2000 / 2022 at an explosion in 1681
PC_PER_YR_TO_KMS = float((1.0 * u.pc / u.yr).to(u.km / u.s).value)

# =============================================================================
# ============ ↓ Models ↓ =====================================================
# =============================================================================
MODELS = {
    # CALIBRATION.md Result 27 (casa_1d_map150_e319m.npz cfg_*)
    "fiducial": dict(energy_erg=2.4307e51, ejecta_mass_msun=3.0, envelope_slope=9.0,
                     inner_slope=0.8159, core_fraction=0.5, n_w=0.9247),
    # Orlando et al. (2016/2022)-like: steep n = 7 envelope, flat core
    "orlando": dict(energy_erg=1.5e51, ejecta_mass_msun=3.3, envelope_slope=7.0,
                    inner_slope=0.0, core_fraction=0.5, n_w=0.8),
}
COMMON = dict(r0=0.05, ejecta_temperature_K=100.0, taper_cells=3.0, r_fs_ref=2.5,
              n_c=0.1, wind_temperature_K=1e4, r_max=4.0, num_cells=4000, cfl=0.4,
              gamma=GAMMA, age_end_yr=358.0, num_snapshots=120,
              minimum_density=1e-6, minimum_pressure=1e-12)

#: gamma-ray bound on the CR proton energy in the remnant (lit_cr.md section 2,
#: rescaled to the post-shock n = 3.6 cm^-3)
W_P_BOUND_ERG = (0.5e50, 2.8e50)

#: Patnaude & Fesen (2009) Table 2: 1D CR-hydro at E = 2e51, M_ej = 2, n = 9
PATNAUDE_FESEN_2009 = dict(
    E_cr_frac=[0.0, 0.07, 0.17, 0.34, 0.50],
    m=[0.74, 0.73, 0.72, 0.66, 0.67],
    R_fs_pc=[2.79, 2.73, 2.64, 2.46, 2.22],
    V_kms=[6376, 6178, 5826, 5021, 4613],
)
# =============================================================================
# ============ ↑ Models ↑ =====================================================
# =============================================================================


# =============================================================================
# ============ ↓ One 1D run ↓ =================================================
# =============================================================================
def build(cfg):
    """1D spherical ejecta + wind IC with the two-fluid CR model switched on."""
    code_units = snr_code_units()
    zeta, f_esc = cfg["zeta"], cfg["f_esc"]
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=cfg["r_max"],
        num_cells=cfg["num_cells"],
        # same stabilisers as casa_calibrate_1d (see the comments there)
        first_order_fallback=True,
        positivity_config=PositivityConfig(
            per_step_mode=POSITIVITY_HARD_FLOOR, nan_safe=True, vacuum_rest=True),
        riemann_solver=HLL,
        # CRs are native-FV only; also avoids the nvidia-smi backend probe
        backend_config=BackendConfig(backend=NATIVE_JAX),
        # the CR fluid is ALWAYS on (zero CRs for zeta = 0), so every run takes
        # the identical solver path and the zeta = 0 run is the exact baseline
        cosmic_ray_config=CosmicRayConfig(
            cosmic_rays=True,
            diffusive_shock_acceleration=zeta > 0,
            shock_selection=OUTERMOST_SHOCK if cfg["selection"] == "outermost" else STRONGEST_SHOCK,
        ),
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(
            return_states=True, return_final_state=True,
            return_total_mass=True, return_total_energy=True),
        num_snapshots=cfg["num_snapshots"],
        progress_bar=False,
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)

    r = helper_data.geometric_centers
    dx = cfg["r_max"] / cfg["num_cells"]
    cell_vol = helper_data.cell_volumes
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n = float((const.k_B * cfg["wind_temperature_K"] * u.K / u.cm ** 3).to(code_units.code_pressure).value)

    n_amb = wind_number_density(r, n_w=cfg["n_w"], r_fs_ref=cfg["r_fs_ref"],
                                n_c=cfg["n_c"], r_cap=0.5 * cfg["r0"])
    rho_amb = n_amb * rho_per_n
    p_amb = n_amb * p_per_n

    E = float((cfg["energy_erg"] * u.erg).to(code_units.code_energy).value)
    M_ej = float((cfg["ejecta_mass_msun"] * u.Msun).to(code_units.code_mass).value)
    shape = ejecta_radial_shape(r, cfg["core_fraction"] * cfg["r0"], cfg["r0"], dx,
                                envelope_slope=cfg["envelope_slope"],
                                inner_slope=cfg["inner_slope"],
                                taper_cells=cfg["taper_cells"])
    m_ej = M_ej / jnp.sum(shape * cell_vol) * shape
    rho = rho_amb + m_ej
    s = jnp.sqrt(E / (0.5 * jnp.sum(m_ej ** 2 * r ** 2 / rho * cell_vol)))
    v = m_ej * s * r / rho
    p_cold = (rho / rho_per_n) * float(
        (const.k_B * cfg["ejecta_temperature_K"] * u.K / u.cm ** 3).to(code_units.code_pressure).value)
    p = p_amb * (1.0 - shape) + p_cold * shape

    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=rho, velocity_x=v, gas_pressure=p,
        cosmic_ray_pressure=jnp.zeros_like(r),
    )
    config = finalize_config(config, state.shape)

    t0_yr = float(((1.0 / s) * code_units.code_time).to(u.yr).value)
    t_end = float(((cfg["age_end_yr"] - t0_yr) * u.yr).to(code_units.code_time).value)
    params = SimulationParams(
        C_cfl=cfg["cfl"], gamma=cfg["gamma"], t_end=t_end,
        minimum_density=cfg["minimum_density"],
        minimum_pressure=cfg["minimum_pressure"],
        cosmic_ray_params=CosmicRayParams(
            diffusive_shock_acceleration_start_time=0.0,
            diffusive_shock_acceleration_efficiency=zeta,
            escape_fraction=f_esc,
        ),
    )

    # Lagrangian label of the contact discontinuity: the enclosed mass at the
    # outer edge of the ejecta-dominated region of the IC (1D flow preserves
    # mass ordering; numerical diffusion only smears the density around it)
    rho_np = np.asarray(rho)
    f_ej = np.asarray(m_ej) / rho_np
    m_cum = np.cumsum(rho_np * np.asarray(cell_vol))
    i_edge = int(np.flatnonzero(f_ej >= 0.5).max())
    info = dict(
        t0_yr=t0_yr, rho_per_n=rho_per_n, code_units=code_units,
        M_cd=float(m_cum[i_edge]),
        wind_mass_map=(m_cum, np.cumsum(np.asarray(rho_amb * cell_vol))),
        E_code_to_erg=float((1.0 * code_units.code_energy).to(u.erg).value),
    )
    return state, config, params, rv, helper_data, info


def _subcell_fs(r, rho, rho_amb, contrast=2.0):
    """Outermost radius where rho / rho_amb crosses ``contrast`` (interpolated)."""
    ratio = rho / rho_amb
    idx = np.flatnonzero(ratio > contrast)
    if idx.size == 0:
        return np.nan
    i = int(idx.max())
    if i + 1 >= r.size:
        return float(r[i])
    a, b = ratio[i], ratio[i + 1]
    return float(r[i] + (a - contrast) / (a - b) * (r[i + 1] - r[i]))


def measure(snaps, helper_data, rv, info, cfg):
    """Time series of every diagnostic, one entry per snapshot."""
    cu = info["code_units"]
    r = np.asarray(helper_data.geometric_centers)
    vol = np.asarray(helper_data.cell_volumes)
    dx = cfg["r_max"] / cfg["num_cells"]
    r_outer = r + 0.5 * dx
    states = np.asarray(snaps.states)
    age = info["t0_yr"] + (np.asarray(snaps.time_points) * cu.code_time).to(u.yr).value
    rho_amb = np.asarray(wind_number_density(
        jnp.asarray(r), n_w=cfg["n_w"], r_fs_ref=cfg["r_fs_ref"], n_c=cfg["n_c"],
        r_cap=0.5 * cfg["r0"])) * info["rho_per_n"]
    e2erg = info["E_code_to_erg"]

    keys = ("age", "r_fs", "r_fs_cell", "r_rs", "r_cd", "n_post", "compression",
            "x_post", "m_unshocked", "E_cr", "E_th", "E_kin", "E_tot")
    out = {k: np.full(states.shape[0], np.nan) for k in keys}
    out["age"] = age
    for k in range(states.shape[0]):
        st = states[k]
        rho = st[rv.density_index]
        v = st[rv.velocity_index]
        p_cr = np.asarray(cosmic_ray_pressure_from_n(st[rv.cosmic_ray_n_index]))
        p_th = st[rv.pressure_index] - p_cr
        m = measure_snapshot(r, rho, v, p_th, age_yr=age[k], cfg=cfg,
                             rho_per_n=info["rho_per_n"], code_units=cu,
                             wind_mass_map=info["wind_mass_map"])
        out["r_fs_cell"][k] = np.nan if m["r_fs"] is None else m["r_fs"]
        out["r_rs"][k] = np.nan if m["r_rs"] is None else m["r_rs"]
        out["n_post"][k] = np.nan if m["n_post"] is None else m["n_post"]
        out["m_unshocked"][k] = np.nan if m["m_unshocked"] is None else m["m_unshocked"]
        r_fs = _subcell_fs(r, rho, rho_amb)
        out["r_fs"][k] = r_fs
        m_cum = np.cumsum(rho * vol)
        out["r_cd"][k] = float(np.interp(info["M_cd"], m_cum, r_outer))
        if np.isfinite(r_fs):
            shell = (r > 0.95 * r_fs) & (r <= r_fs)
            if np.any(shell):
                out["x_post"][k] = float(np.sum(p_cr[shell] * vol[shell]) / np.sum(p_th[shell] * vol[shell]))
            near = (r > 0.85 * r_fs) & (r <= r_fs)
            if np.any(near):
                out["compression"][k] = float(np.max(rho[near] / rho_amb[near]))
        out["E_cr"][k] = float(np.sum(3.0 * p_cr * vol)) * e2erg
        out["E_th"][k] = float(np.sum(1.5 * p_th * vol)) * e2erg
        out["E_kin"][k] = float(np.sum(0.5 * rho * v ** 2 * vol)) * e2erg
    out["E_tot"] = out["E_cr"] + out["E_th"] + out["E_kin"]
    out["E_tot_snap"] = np.asarray(snaps.total_energy) * e2erg
    out["M_tot_snap"] = np.asarray(snaps.total_mass)
    return out


def at_age(series, age0, key, half_window=15.0, log_fit=False):
    """Local polynomial fit of ``series[key]`` around ``age0``.

    Returns ``(value, d value / d age)``; with ``log_fit`` the fit is
    ln(value) vs ln(age) (quadratic) and the derivative is dlnv/dlnt = m.
    """
    t = series["age"]
    y = series[key]
    sel = np.isfinite(y) & (np.abs(t - age0) <= half_window)
    if sel.sum() < 4:
        return np.nan, np.nan
    if log_fit:
        x = np.log(t[sel] / age0)
        c = np.polyfit(x, np.log(y[sel]), 2)
        return float(np.exp(c[2])), float(c[1])
    x = t[sel] - age0
    c = np.polyfit(x, y[sel], 1)
    return float(c[1]), float(c[0])


def summarise(series, cfg, e_tot0_erg, baseline_drift=None):
    """Scalar diagnostics at the two Chandra epochs."""
    res = {}
    for age0 in EPOCH_AGES:
        r_fs, m = at_age(series, age0, "r_fs", log_fit=True)
        v_fs = m * r_fs / age0 * PC_PER_YR_TO_KMS
        r_rs, drs = at_age(series, age0, "r_rs", half_window=10.0)
        r_cd, _ = at_age(series, age0, "r_cd", half_window=6.0)
        n_post, _ = at_age(series, age0, "n_post", half_window=6.0)
        e_cr = float(np.interp(age0, series["age"], series["E_cr"]))
        e_tot = float(np.interp(age0, series["age"], series["E_tot_snap"]))
        drift = 0.0 if baseline_drift is None else float(np.interp(age0, *baseline_drift))
        tag = f"{int(age0)}"
        res[tag] = dict(
            r_fs=r_fs, v_fs=v_fs, m=m, r_rs=r_rs, v_rs=drs * PC_PER_YR_TO_KMS,
            rs_over_fs=r_rs / r_fs, r_cd=r_cd, gap=(r_fs - r_cd) / r_fs,
            n_post=n_post,
            compression=float(np.interp(age0, series["age"], series["compression"])),
            x_post=float(np.interp(age0, series["age"], series["x_post"])),
            m_unshocked=float(np.interp(age0, series["age"], series["m_unshocked"])),
            E_cr=e_cr, E_cr_frac=e_cr / cfg["energy_erg"],
            # energy that left the system beyond the zeta = 0 run's own drift
            E_esc=(e_tot0_erg - e_tot) - drift,
        )
    return res


def run_one(cfg):
    """Build, run, measure; returns (series, summary, profiles)."""
    t_start = time.time()
    state, config, params, rv, helper_data, info = build(cfg)
    snaps = time_integration(state, config, params, rv)
    series = measure(snaps, helper_data, rv, info, cfg)
    e_tot0 = float(series["E_tot_snap"][0])
    # profiles at the snapshots nearest to the two epochs (for the figure)
    states = np.asarray(snaps.states)
    prof = {}
    for age0 in EPOCH_AGES:
        k = int(np.argmin(np.abs(series["age"] - age0)))
        st = states[k]
        p_cr = np.asarray(cosmic_ray_pressure_from_n(st[rv.cosmic_ray_n_index]))
        prof[f"{int(age0)}"] = dict(
            age=float(series["age"][k]), rho=st[rv.density_index] / info["rho_per_n"],
            v=st[rv.velocity_index] * 1000.0, p_th=st[rv.pressure_index] - p_cr, p_cr=p_cr)
    wall = time.time() - t_start
    return series, e_tot0, prof, np.asarray(helper_data.geometric_centers), info, wall


def save_run(path, cfg, series, e_tot0, prof, r, info, wall):
    flat = {f"series_{k}": v for k, v in series.items()}
    for tag, p in prof.items():
        for k, v in p.items():
            flat[f"prof{tag}_{k}"] = v
    np.savez_compressed(
        path, r=r, e_tot0=e_tot0, t0_yr=info["t0_yr"], wall_s=wall,
        cfg_json=json.dumps({k: v for k, v in cfg.items()}), **flat)


def load_run(path):
    d = np.load(path, allow_pickle=False)
    cfg = json.loads(str(d["cfg_json"]))
    series = {k[len("series_"):]: d[k] for k in d.files if k.startswith("series_")}
    prof = {}
    for tag in ("319", "341"):
        prof[tag] = {k[len(f"prof{tag}_"):]: d[k] for k in d.files if k.startswith(f"prof{tag}_")}
    return cfg, series, float(d["e_tot0"]), prof, d["r"], float(d["wall_s"])
# =============================================================================
# ============ ↑ One 1D run ↑ =================================================
# =============================================================================


# =============================================================================
# ============ ↓ Suite: sweep, re-tune, table, figure ↓ =======================
# =============================================================================
def run_name(model, zeta, f_esc, energy_erg=None, selection="outermost"):
    tag = f"{model}_z{zeta:.2f}_f{f_esc:.2f}"
    if energy_erg is not None and abs(energy_erg / MODELS[model]["energy_erg"] - 1.0) > 1e-12:
        tag += f"_E{energy_erg / 1e51:.9f}"
    if selection != "outermost":
        tag += f"_{selection}"
    return tag


def launch(outdir, model, zeta, f_esc, energy_erg=None, selection="outermost", n=None):
    """Run one configuration in a fresh subprocess (JAX + fork do not mix)."""
    name = run_name(model, zeta, f_esc, energy_erg, selection)
    path = outdir / f"{name}.npz"
    if path.exists():
        return path
    cmd = [sys.executable, __file__, "--one", "--model", model, "--zeta", str(zeta),
           "--f-esc", str(f_esc), "--selection", selection, "--out", str(path)]
    if energy_erg is not None:
        cmd += ["--energy-51", repr(float(energy_erg) / 1e51)]
    if n is not None:
        cmd += ["--n", str(n)]
    env = dict(os.environ)
    # a 4000-cell 1D run gains nothing from 192 threads; keep the runs polite
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    log = outdir / f"{name}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                            cwd=Path(__file__).resolve().parent).returncode
    if rc != 0 or not path.exists():
        raise RuntimeError(f"run {name} failed (rc={rc}); see {log}")
    return path


def retune_energy(outdir, model, zeta, f_esc, r_target, e_start, tol=2e-4, max_iter=6):
    """Secant in ln E so that r_FS(319 yr) = r_target."""
    def r_of(e):
        cfg, series, *_ = load_run(launch(outdir, model, zeta, f_esc, energy_erg=e))
        return at_age(series, 319.0, "r_fs", log_fit=True)[0]

    e_start = float(e_start)
    e0, r0 = e_start, r_of(e_start)
    # r_FS ~ E^0.3 in this regime: first step from that scaling
    e1 = float(e0 * (r_target / r0) ** (1.0 / 0.3))
    r1 = r_of(e1)
    for _ in range(max_iter):
        if abs(r1 / r_target - 1.0) < tol:
            break
        slope = (np.log(r1) - np.log(r0)) / (np.log(e1) - np.log(e0))
        e2 = float(e1 * np.exp((np.log(r_target) - np.log(r1)) / slope))
        e0, r0, e1, r1 = e1, r1, e2, r_of(e2)
    return e1


def suite(outdir, workers, models):
    outdir.mkdir(parents=True, exist_ok=True)
    zetas = (0.0, 0.05, 0.10, 0.15, 0.30)
    jobs = []
    for model in models:
        for z in zetas:
            jobs.append((model, z, 0.0))
            if z > 0:
                jobs.append((model, z, 0.5))
    # selection check: strongest vs outermost must agree (the RS is never flagged)
    jobs_extra = [("fiducial", 0.30, 0.0, None, "strongest")]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(launch, outdir, *j) for j in jobs]
        futs += [pool.submit(launch, outdir, m, z, f, e, s) for (m, z, f, e, s) in jobs_extra]
        for f in futs:
            f.result()
    print(f"[suite] sweep done in {time.time() - t0:.0f} s", flush=True)

    # re-tune E at fixed r_FS(319) for zeta = 0.1, 0.3 (with and without escape)
    retunes = {}
    tasks = []
    for model in models:
        cfg0, s0, *_ = load_run(outdir / f"{run_name(model, 0.0, 0.0)}.npz")
        r_target = at_age(s0, 319.0, "r_fs", log_fit=True)[0]
        for z in (0.10, 0.30):
            for f in (0.0, 0.5):
                tasks.append((model, z, f, r_target, MODELS[model]["energy_erg"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(retune_energy, outdir, m, z, f, rt, e): (m, z, f)
                for (m, z, f, rt, e) in tasks}
        for fut, key in futs.items():
            retunes["|".join(map(str, key))] = float(fut.result())
    (outdir / "retunes.json").write_text(json.dumps(retunes, indent=1))
    print(f"[suite] re-tunes done in {time.time() - t0:.0f} s", flush=True)


def collect(outdir, models):
    """Summaries of every run on disk, keyed like the table rows."""
    retunes = json.loads((outdir / "retunes.json").read_text()) if (outdir / "retunes.json").exists() else {}
    rows = []
    for model in models:
        base_path = outdir / f"{run_name(model, 0.0, 0.0)}.npz"
        if not base_path.exists():
            continue
        _, s_base, e0_base, *_ = load_run(base_path)
        drift = (s_base["age"], e0_base - s_base["E_tot_snap"])
        for path in sorted(outdir.glob(f"{model}_z*.npz")):
            cfg, series, e_tot0, prof, r, wall = load_run(path)
            e_fixed = abs(cfg["energy_erg"] / MODELS[model]["energy_erg"] - 1.0) < 1e-12
            key = "|".join(map(str, (model, cfg["zeta"], cfg["f_esc"])))
            retuned = (not e_fixed) and key in retunes and abs(retunes[key] / cfg["energy_erg"] - 1) < 1e-9
            if not e_fixed and not retuned:
                continue  # an intermediate secant iterate
            summ = summarise(series, cfg, e_tot0, baseline_drift=drift)
            rows.append(dict(model=model, zeta=cfg["zeta"], f_esc=cfg["f_esc"],
                             selection=cfg["selection"], E=cfg["energy_erg"],
                             retuned=retuned, wall=wall, path=str(path), **{
                                 f"{k}@{tag}": v for tag, d in summ.items() for k, v in d.items()}))
    rows.sort(key=lambda d: (d["model"], d["retuned"], d["selection"], d["f_esc"], d["zeta"]))
    return rows


def write_table(rows, outdir):
    cols = [("model", "{}"), ("zeta", "{:.2f}"), ("f_esc", "{:.1f}"), ("E", "{:.3e}"),
            ("retuned", "{}"), ("selection", "{}")]
    epoch_cols = [("r_fs", "{:.3f}"), ("v_fs", "{:.0f}"), ("m", "{:.3f}"), ("r_rs", "{:.3f}"),
                  ("rs_over_fs", "{:.3f}"), ("gap", "{:.3f}"), ("n_post", "{:.2f}"),
                  ("compression", "{:.2f}"), ("x_post", "{:.3f}"), ("E_cr", "{:.2e}"),
                  ("E_cr_frac", "{:.3f}"), ("E_esc", "{:.2e}"), ("m_unshocked", "{:.3f}")]
    header = [c for c, _ in cols] + [f"{c}@{t}" for t in ("319", "341") for c, _ in epoch_cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    csv = [",".join(header)]
    for row in rows:
        vals = [fmt.format(row[c]) for c, fmt in cols]
        vals += [fmt.format(row[f"{c}@{t}"]) for t in ("319", "341") for c, fmt in epoch_cols]
        lines.append("| " + " | ".join(vals) + " |")
        csv.append(",".join(str(row[c]) for c, _ in cols) + "," + ",".join(
            repr(row[f"{c}@{t}"]) for t in ("319", "341") for c, _ in epoch_cols))
    (outdir / "casa_cr_1d_table.md").write_text("\n".join(lines) + "\n")
    (outdir / "casa_cr_1d_table.csv").write_text("\n".join(csv) + "\n")
    return "\n".join(lines)


def make_figure(rows, outdir, model="fiducial"):
    """Changes relative to the model's own zeta = 0 run vs the CR energy fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # validated categorical slots 1-3 (dataviz reference palette); ink for text
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {0.0: blue, 0.5: orange}  # f_esc = 0 / 0.5
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "font.size": 9})
    sel = [r for r in rows if r["model"] == model and r["selection"] == "outermost"]
    base = next(r for r in sel if r["zeta"] == 0.0 and not r["retuned"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    (ax_m, ax_rs, ax_gap), (ax_rfs, ax_prof, ax_x) = axes

    def pts(f_esc, retuned):
        out = [base] + sorted(
            (r for r in sel if r["zeta"] > 0 and r["f_esc"] == f_esc and r["retuned"] == retuned),
            key=lambda r: r["zeta"])
        return out

    wp_lo, wp_hi = (w / base["E"] for w in W_P_BOUND_ERG)
    panels = (
        (ax_m, lambda p: p["m@319"] - base["m@319"], r"$\Delta m$ ($m = Vt/R$) at 319 yr", -0.05),
        (ax_rs, lambda p: p["rs_over_fs@319"] - base["rs_over_fs@319"],
         r"$\Delta(r_{\rm RS}/r_{\rm FS})$ at 319 yr", +0.05),
        (ax_gap, lambda p: p["gap@319"] / base["gap@319"] - 1.0,
         r"relative change of the FS$-$CD gap at 319 yr", None),
        (ax_rfs, lambda p: p["r_fs@319"] / base["r_fs@319"] - 1.0,
         r"relative change of $r_{\rm FS}$ at 319 yr", None),
    )
    for ax, fn, ylabel, threshold in panels:
        ax.axvspan(wp_lo, wp_hi, color="#9aa5b1", alpha=0.22, lw=0,
                   label=r"$\gamma$-ray bound $W_p$ = 0.5$-$2.8e50 erg")
        ax.axhline(0.0, color=ink2, lw=0.6)
        if threshold is not None:
            ax.axhline(threshold, color=ink, lw=0.9, ls=":", label=f"decisive threshold {threshold:+.2f}")
        for f_esc in (0.0, 0.5):
            fixed = pts(f_esc, False)
            ax.plot([p["E_cr_frac@319"] for p in fixed], [fn(p) for p in fixed], "-o",
                    color=colors[f_esc], ms=5, lw=2, label=f"fixed E, f_esc = {f_esc:g}")
            if f_esc == 0.0 and ax in (ax_rs, ax_gap):
                for p in fixed[1:]:
                    ax.annotate(f"\u03b6={p['zeta']:g}", (p["E_cr_frac@319"], fn(p)),
                                textcoords="offset points", xytext=(4, -11), fontsize=7, color=ink2)
            tuned = pts(f_esc, True)[1:]
            ax.plot([p["E_cr_frac@319"] for p in tuned], [fn(p) for p in tuned], "s",
                    mfc="white", mew=1.6, color=colors[f_esc], ms=8,
                    label=f"E re-tuned to the \u03b6=0 r_FS, f_esc = {f_esc:g}")
        ax.set_xlabel(r"$E_{\rm CR}$ in the remnant at 319 yr / $E_{\rm SN}$")
        ax.set_ylabel(ylabel)
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
    # Patnaude & Fesen 2009, Table 2 (E = 2e51, M_ej = 2, n = 9; FS injection)
    pf = PATNAUDE_FESEN_2009
    ax_m.plot(pf["E_cr_frac"], np.array(pf["m"]) - pf["m"][0], "^--", color=ink2, ms=6, lw=1,
              label="Patnaude & Fesen 2009 Tab. 2")
    ax_rfs.plot(pf["E_cr_frac"], np.array(pf["R_fs_pc"]) / pf["R_fs_pc"][0] - 1.0, "^--",
                color=ink2, ms=6, lw=1, label="Patnaude & Fesen 2009 Tab. 2")
    ax_m.set_ylim(-0.1, 0.01)
    ax_rs.set_ylim(-0.01, 0.06)
    ax_m.legend(fontsize=7, loc="lower left", frameon=False)
    ax_rfs.legend(fontsize=7, loc="lower left", frameon=False)
    ax_rs.legend(fontsize=7, loc="upper left", frameon=False)

    # profiles at 319 yr, fixed E, no escape
    prof_colors = {0.0: blue, 0.10: orange, 0.30: aqua}
    for r_ in sel:
        if r_["retuned"] or r_["f_esc"] != 0.0 or r_["zeta"] not in prof_colors:
            continue
        _, _, _, prof, rgrid, _ = load_run(r_["path"])
        p = prof["319"]
        lbl = f"\u03b6 = {r_['zeta']:g}"
        ax_prof.semilogy(rgrid, p["rho"], lw=2, color=prof_colors[r_["zeta"]], label=lbl)
        x_cr = np.where(p["p_th"] > 0, p["p_cr"] / np.maximum(p["p_th"], 1e-300), np.nan)
        ax_x.plot(rgrid, x_cr, lw=2, color=prof_colors[r_["zeta"]], label=lbl)
    r_lo, r_hi = 0.55 * base["r_fs@319"], 1.08 * base["r_fs@319"]
    ax_prof.set(xlabel="r [pc]", ylabel=r"$n$ [cm$^{-3}$] at 319 yr (fixed E)", xlim=(r_lo, r_hi),
                ylim=(0.2, 60))
    ax_x.set(xlabel="r [pc]", ylabel=r"$P_{\rm CR}/P_{\rm th}$ at 319 yr (fixed E)", xlim=(r_lo, r_hi))
    for ax in (ax_prof, ax_x):
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Cas A 1D ({model}, E = {base['E'] / 1e51:.2f}e51 erg): DSA at the forward "
                 f"shock, \u03b6 = 0-0.3, escape f_esc = 0 / 0.5 — changes vs \u03b6 = 0", color=ink)
    out = outdir / f"casa_cr_1d_{model}.png"
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return out


# =============================================================================
# ============ ↑ Suite: sweep, re-tune, table, figure ↑ =======================
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--one", action="store_true", help="run a single configuration")
    mode.add_argument("--suite", action="store_true", help="sweep + re-tune + report")
    mode.add_argument("--report", action="store_true", help="table + figure from disk")
    ap.add_argument("--model", default="fiducial", choices=sorted(MODELS))
    ap.add_argument("--models", default="fiducial,orlando")
    ap.add_argument("--zeta", type=float, default=0.0)
    ap.add_argument("--f-esc", type=float, default=0.0)
    ap.add_argument("--energy-51", type=float, default=None)
    ap.add_argument("--selection", default="outermost", choices=["outermost", "strongest"])
    ap.add_argument("--n", type=int, default=None, help="radial cells (default 4000)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    models = [m for m in args.models.split(",") if m]

    if args.one:
        cfg = dict(COMMON, **MODELS[args.model], model=args.model, zeta=args.zeta,
                   f_esc=args.f_esc, selection=args.selection)
        if args.energy_51 is not None:
            cfg["energy_erg"] = args.energy_51 * 1e51
        if args.n is not None:
            cfg["num_cells"] = args.n
        series, e_tot0, prof, r, info, wall = run_one(cfg)
        summ = summarise(series, cfg, e_tot0)
        print(f"[casa_cr_1d] {args.model} zeta={args.zeta} f_esc={args.f_esc} "
              f"E={cfg['energy_erg']:.4e}  t0={info['t0_yr']:.2f} yr  wall {wall:.0f} s")
        for tag, d in summ.items():
            print(f"  {tag} yr: " + "  ".join(f"{k}={v:.4g}" for k, v in d.items()))
        if args.out:
            save_run(args.out, cfg, series, e_tot0, prof, r, info, wall)
        return

    if args.suite:
        suite(outdir, args.workers, models)
    rows = collect(outdir, models)
    print(write_table(rows, outdir))
    for model in models:
        if any(r["model"] == model for r in rows):
            print("figure:", make_figure(rows, outdir, model))


if __name__ == "__main__":
    main()
