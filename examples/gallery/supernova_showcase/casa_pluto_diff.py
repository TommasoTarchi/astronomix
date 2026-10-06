"""
Differentiable Cas A from Orlando's 146-yr state: fit 22 years of Chandra outlines.

The initial condition is not built from a 1D profile here. It is S. Orlando's
3D MHD state of model W15-IIb-sh at 145.5 yr (``casa_pluto.py convert``),
which already carries the neutrino-driven explosion's structure. What the
fit varies are the physical quantities that are NOT settled by that state, as
smooth, traced transformations of it:

  ``ln_sv``   self-similar speed-up of everything inside the forward shock
              (v -> s v, p -> s^2 p): the remnant's energy at fixed structure.
              It also makes the IC's ejecta r / v = 145.5 yr / s, so the
              model's ballistic convergence date is t_expl + 145.5 (1 - 1/s)
              (``ballistic_convergence_date``; bounded below by the knots)
  ``ln_fw``   the unshocked wind density (the medium the next 200 yr run into)
  ``ln_fsh``  the Orlando et al. (2022) shell's density, and
  ``d_rsh``   its radius (pc): neither has been reached by 146 yr
  ``t_expl``  the explosion year: the model age of epoch e is e - t_expl
              (prior 1681 +- 19, Fesen et al. 2006, plus a one-sided wall on
              the ballistic convergence date >= 1671.3, Thorstensen+01)
  ``ln_D``    the distance (prior ln-normal 3.4 kpc +- 0.15 kpc, spanning
              Alarie+14 3.33 +- 0.10, Reed+95 3.4 -0.1/+0.3, Neumann+24 3.6)
  ``psi``     rotation of the model about the line of sight (deg); the
              orientation Orlando chose is the prior mean
  ``dip_*``   an ejecta velocity dipole v -> v (1 + C_ej dip . n_hat)
  ``ln_kdop`` Doppler-centroid dilution nuisance (casa_xfit freezes it at 0)
  ``ds_w``    change of the unshocked wind's radial slope, rho ~ r^-(2 + ds_w)
              about 2.5 pc: the mass-loss history, which sets how fast the
              blast decelerates at a given age
  ``wa_*``    wind density dipole rho_w -> rho_w exp(a . n_hat) / <exp(a . n_hat)>
              (smooth, positive, unit sphere mean; the old max(1 + a . n, 0.05)
              is ``wind_dipole="clip"``)

and, OUTSIDE ``PARAM_NAMES`` (``SIM_NAMES``, passed as ``sim=`` -- see below):

  ``ln_L``, ``ln_T``, ``ln_M``  the EXACT similarity map of the adiabatic problem
              (``casa_rescale``): r -> L r, t -> T t, rho -> (M / L^3) rho,
              v -> (L / T) v, p -> (M / L^3)(L / T)^2 p, applied to the whole
              state INCLUDING the wind and shell (wind rho r^2 x M / L, shell
              radius x L, mass x M) and the shock history. E -> M (L / T)^2 E,
              M_ej -> M M_ej, and the IC age becomes 145.5 T (``ic_age``), so
              the model's ballistic convergence date stays equal to t_expl.
              With them, FREEZE ``ln_sv`` AT 0: ``ln_sv`` is the member
              (L, T, M) = (1, 1/s, 1) without the age relabelling, which is why
              the ballistic-date wall was needed; the wall is redundant (it
              becomes t_expl >= 1671.3) once the age is relabelled. The other
              knobs act ON TOP of the scaled state (ln_fw relative to the scaled
              wind, d_rsh a shift of the scaled shell radius L r_sh).
              They are kept out of ``PARAM_NAMES`` on purpose: ``casa_xfit``
              builds its vector as ``PD.PARAM_NAMES + EXTRA`` and loads thetas
              by position, so appending here would shift every casa_xfit
              parameter (see ``transform_fields(sim=...)``).

and the data are the forward-shock radius per position angle, measured with
``casa_real_outline.outline`` on each Chandra epoch 2000-2022 (15 epochs x 36
cones, ~500 numbers), about the image centre RA0/DEC0. The model's explosion
centre sits at a sky offset from that point (the Thorstensen+01 expansion
centre, 13.8" E / 4.2" S), so the model radii are RE-CENTRED onto RA0/DEC0
before they are compared (``recentre_radii``; 2026-09-25 audit: without it the
wind dipole fits a coordinate offset). The fit is a Gauss-Newton /
Levenberg-Marquardt on forward-mode Jacobians -- one ``jax.jvp`` per parameter
through the whole 3D evolution, O(1) memory in the number of steps (see
``casa_diff.py`` for why forward mode is the right tool for few parameters and
a huge state).

THE OBSERVABLE HAS TO BE SMOOTH. The forward shock per cone is located where a
sigmoid indicator of the gas temperature falls, as an outer power mean
(``edge_radii``). The reverse shock is the OUTER power-mean edge of the
UNSHOCKED EJECTA (``rs_unshocked_edge``): the previous inner power mean of
the hot indicator was dominated by the sigmoid tail of floor-temperature gas
at the centre and sat at its own 0.25 r_FS cut (audit 2026-09-25, F1).

Usage::

    ./run.sh casa_pluto_diff.py --ic .../pluto146_n128.npz --forward   # one run, residuals
    ./run.sh casa_pluto_diff.py --ic ... --check-grad                  # JVP vs FD
    ./run.sh casa_pluto_diff.py --ic ... --fit --steps 6
    ./run.sh casa_pluto_diff.py --ic ... --forward --legacy            # the pre-audit likelihood
"""

# ==== GPU selection ====
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
# ruff: noqa: E402
# =======================

# general
import argparse
import ast
import json
import time
import warnings
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# units
from astropy import units as u

# astronomix
from astronomix import (
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# shared showcase helpers
from casa_pluto import radial_gaussian
from _common import (GAMMA, MASS_PER_NUCLEUS, POSITIVITY_REDISTRIBUTE, fd_positivity,
                     make_fd_config, snr_code_units)
import astropy.constants as const

WORK = Path("/export/data/lstorcks/casa_orlando150/work")
OBSERVED = WORK / "observed_outlines.npz"
#: registration proper motions (casa_expansion): fixed 130-210" window and the
#: +-12" window about each cone's outline. "ccopm" = re-registered with the
#: CCO's own proper motion added back (casa_expansion --cco-pm, 2026-09-25);
#: "legacy" = the CCO-locked registration, which subtracts that motion
PM_FILES = {
    "ccopm": (WORK / "expansion_ccopm.npz", WORK / "expansion_fs12_ccopm.npz"),
    "legacy": (WORK / "expansion.npz", WORK / "expansion_fs12.npz"),
}
EXPANSION_FIXED, EXPANSION_OUTLINE = PM_FILES["legacy"]
#: per-cone proper-motion error floor ("/yr): model-definition systematic
#: between a 3D shock radius and an image registration of the rim
PM_SIGMA_SYS = 0.03
#: global PM-scale nuisance: pm_model ~ (1 + eps) pm_data, eps ~ N(0, 0.08^2),
#: profiled analytically. The registration and the model's 3D edge differ by a
#: COHERENT ~10 % (audit fit_audit item 4), which the per-cone floor treated as
#: independent (the 30-cone mean pinned to ~1.7 %)
PM_SCALE_SIGMA = 0.08
#: cone-angle masks (theta = PA + 90, the code's image convention: theta from
#: west through north). "sw": the SW cones whose registration locks onto the
#: stationary / inward-moving reverse-shock features (PA 200, 230; 84 of Q2's
#: 208 motion chi2); "jet": the NE jet, PA 70-120 (Vink+22 mask it too)
CONE_MASKS = {"sw": (290.0, 320.0), "jet": (160.0, 170.0, 180.0, 190.0, 200.0, 210.0)}
#: Vink et al. (2022, ApJ 929, 57) Table 3: forward-shock expansion rate per
#: 20-deg sector about the Thorstensen expansion centre, 2000-2019, 4.2-6 keV,
#: in %/yr (statistical errors 0.001-0.003; pointing systematics ~0.01)
VINK22_PA = np.arange(10.0, 360.0, 20.0)
VINK22_RATE = np.array([0.1900, 0.2760, 0.1990, 0.2098, 0.2217, 0.2212, 0.2384, 0.2707, 0.2329,
                        0.1518, 0.2076, 0.2331, 0.1934, 0.2251, 0.2376, 0.2274, 0.1963, 0.1958])
VINK22_SIGMA = 0.012
VINK22_SIGMA_MODEL = 0.015
VINK22_YEARS = (1999.5, 2019.9)
#: X-ray rate: real 0.5-7 keV counts/s inside r < 200" (casa_observe), and the
#: rate per unit hot-gas emission measure / D^2 [code rho^2 pc^3 / kpc^2]
#: calibrated on the six full pyXSIM runs (Orlando as delivered, fits A, B:
#: 619/560/566 in 2000, 309/278/236 in 2022). The log error covers the proxy
#: scatter AND the sub-grid f_mass freedom the full forward model has.
OBSERVED_RATES = {"2000": (315.9, 580.0), "2022": (131.5, 275.0)}
RATE_SIGMA_LN = 0.15
#: After the solar-CSM correction (PLUTO150.md section 5b) the same full model
#: gives 0.754x (2000) and 0.730x (2022) the rate of the metal-rich-CSM runs
#: the calibration above was made on. The ABSOLUTE rate is further degenerate
#: with the sub-grid f_mass and moves 15-20 % per resolution doubling, so with
#: --rate-mode fading it enters only loosely and the 2022/2000 FADING RATIO,
#: which none of those move, carries the constraint.
OBSERVED_RATES_SOLAR = {"2000": (315.9, 580.0 * 0.754), "2022": (131.5, 275.0 * 0.730)}
RATE_SIGMA_LN_LOOSE = 0.30
#: measured Si He-alpha centroid velocities per sky sector, 2004 (km/s,
#: + = receding, mean-subtracted), and the per-sector error floor: centroid
#: systematics (gain, blending with the continuum) are ~50 km/s
DOPPLER_FILE = WORK / "doppler_si_2004.npz"
DOPPLER_SIGMA_SYS = 100.0
DOPPLER = ({"n": 24, **{k: np.asarray(v) for k, v in np.load(DOPPLER_FILE).items()}}
           if DOPPLER_FILE.exists() else {})
FADING_SIGMA_LN = 0.05
#: epochs whose image covers the whole outline (2006 / 2020 / 2023 are offset
#: pointings or CCO subarrays: 10-22 of 36 cones)
MIN_CONES_FOUND = 30
ARCSEC_PER_RAD = 206264.806

# ---- literature targets (audit 2026-09-25, lit_obs.md section 1) ----
#: Thorstensen et al. (2001) expansion centre, arcsec WEST / NORTH of RA0/DEC0
#: (23h23m27.77s +58d48'49.4", +-0.05 s / +-0.4"); prior width on the model's
#: explosion centre
COE_ARCSEC = (-13.8, -4.2)
COE_SIGMA_ARCSEC = 1.5
#: the undecelerated knot convergence date (Thorstensen+01): deceleration only
#: makes the true date LATER, so it bounds the model's ballistic date below
T_CONV_MIN, T_CONV_SIGMA = 1671.3, 0.9
#: one-sided soft wall sharpness: r = softplus(beta x) / beta (chi2 0.03 at x = 0)
WALL_BETA = 4.0
#: Lee et al. (2014): pre-shock n_H 0.89 +- 0.30 cm^-3 at the current outer
#: radius (~3 pc); (radius pc, n_H, sigma)
WIND_NH_PRIOR = (3.0, 0.89, 0.30)
#: r_RS / r_FS, remnant average (lit_obs R7 compilation: Gotthelf+01, Helder &
#: Vink 08, Arias+18, HL12, Orlando+16)
RS_RATIO_PRIOR = (0.66, 0.05)
M_H_G = 1.6735575e-24

PARAM_NAMES = ("ln_sv", "ln_fw", "ln_fsh", "d_rsh", "t_expl", "ln_D", "psi", "ds_w",
               "dip_x", "dip_y", "dip_z", "ln_kdop", "rot_x", "rot_y", "rot_z",
               "wa_x", "wa_y", "wa_z")
#: the exact similarity parameters (log L, log T, log M; ``casa_rescale``). NOT
#: part of PARAM_NAMES (casa_xfit appends its own parameters to PARAM_NAMES
#: and loads thetas positionally): pass them as ``transform_fields(sim=...)``
#: / ``make_forward(sim=...)``, or append them to the END of a caller's own
#: parameter vector (casa_xfit: to EXTRA_PRIOR, as ln_si / b_sx were)
SIM_NAMES = ("ln_L", "ln_T", "ln_M")
#: priors (2026-09-27 similarity analysis of the as-delivered 146 -> 540 yr run,
#: /export/data/lstorcks/casa_orlando150/work/ers/energy/REPORT.md): the
#: post-shell branch that fits r_FS(2000) = 158.9", the expansion rate (Vink+22
#: 0.218 %/yr -> T 1.21, in-house 0.200 %/yr -> T 1.31), t_expl 1681 +- 19 and
#: D 3.33 kpc has L 1.43 +- 0.05, T 1.21-1.31; ln_M from M_ej = 3.3 +- 0.55 Msun
#: (Orlando's 3.26 Msun; HL12 2-4). The pre-shell branch (L ~ T ~ 1.9, shell
#: still ahead in 2000) is excluded by the X-ray brightness (Delta chi2 ~ 20)
SIM_PRIOR = {"ln_L": (float(np.log(1.42)), 0.06), "ln_T": (float(np.log(1.25)), 0.06),
             "ln_M": (0.0, 0.17)}
SIM_FD_STEPS = {"ln_L": 0.02, "ln_T": 0.02, "ln_M": 0.05}
#: prior mean and width per parameter (Gaussian, in the fitted coordinates).
#: The widths on the physics parameters are deliberately loose: they regularise
#: directions the data cannot see, they are not meant to pull.
PRIOR_LEGACY = {
    "ln_sv": (0.0, 0.3), "ln_fw": (0.0, 0.7), "ln_fsh": (0.0, 1.0), "d_rsh": (0.0, 0.3),
    "t_expl": (1681.0, 19.0), "ln_D": (np.log(3.4), 0.06), "psi": (0.0, 30.0),
    "ds_w": (0.0, 0.5),
    # ejecta velocity dipole v -> v (1 + C_ej dip . n_hat): the explosion's
    # large-scale asymmetry, which the Doppler pattern and the outline see
    "dip_x": (0.0, 0.3), "dip_y": (0.0, 0.3), "dip_z": (0.0, 0.3),
    # measured Si centroid shift / true emission-weighted velocity (continuum
    # under the line dilutes the centroid): a nuisance scale
    "ln_kdop": (0.0, 0.5),
    # orientation of the EXPLOSION (the interior at 146 yr) relative to the
    # observer and the CSM, as a rotation vector in degrees. Orlando's
    # orientation is the prior mean, but only loosely: nothing he delivered
    # constrains the explosion's axes, and a rigid rotation of the evolved
    # remnant takes the Si Doppler correlation from -0.16 to +0.88 (PLUTO150.md)
    "rot_x": (0.0, 120.0), "rot_y": (0.0, 120.0), "rot_z": (0.0, 120.0),
    # wind density dipole (see ``wind_asymmetry``): the asymmetric
    # circumstellar medium our own calibrated track needed (A1 ~ 0.75)
    "wa_x": (0.0, 0.5), "wa_y": (0.0, 0.5), "wa_z": (0.0, 0.5),
}
#: the corrected priors (default): distance ln-normal about 3.4 kpc with
#: sigma 0.15 kpc (Alarie+14 3.33 +- 0.10, Reed+95 3.4 -0.1/+0.3, Neumann+24
#: 3.6 +- 0.1). The explosion date keeps Fesen+06's 1681 +- 19; its lower bound
#: enters as the ballistic-date wall (``physical_prior_residuals``)
PRIOR = dict(PRIOR_LEGACY, ln_D=(float(np.log(3.4)), 0.15 / 3.4))
THETA0 = np.array([PRIOR[k][0] for k in PARAM_NAMES])


# =============================================================================
# ============ ↓ Data ↓ =======================================================
# =============================================================================
def cone_mask(angles, spec):
    """Boolean mask of the cones named in ``spec`` ('none', 'sw', 'jet',
    'sw+jet'), by cone angle (theta = PA + 90)."""
    out = np.zeros(len(angles), bool)
    for key in [s for s in str(spec).split("+") if s and s != "none"]:
        for a in CONE_MASKS[key]:
            out |= np.abs(((np.asarray(angles) - a + 180.0) % 360.0) - 180.0) < 1e-6
    return out


def load_observations(path=OBSERVED, sigma_floor_arcsec=1.0, *, pm_files="ccopm", pm_mask="sw+jet"):
    """Per-cone forward-shock radii (arcsec) per usable epoch, and their errors.

    The per-cone error is estimated from the data: the scatter of each cone's
    radii about its own linear trend over 22 yr (measurement noise at the
    epochs' exposures), floored at ``sigma_floor_arcsec``.

    ``pm_files``: which registration proper motions (``PM_FILES``); ``pm_mask``:
    cones whose proper motion is dropped (``CONE_MASKS``; the outline radii of
    those cones are kept).
    """
    d = np.load(path)
    r = np.asarray(d["r_arcsec"], dtype=np.float64)
    years = np.asarray(d["years"], dtype=np.float64)
    good = np.isfinite(r).sum(1) >= MIN_CONES_FOUND
    r, years = r[good], years[good]
    epochs = [str(e) for e, g in zip(d["epochs"], good) if g]
    resid = np.full_like(r, np.nan)
    for k in range(r.shape[1]):
        ok = np.isfinite(r[:, k])
        if ok.sum() >= 4:
            c = np.polyfit(years[ok], r[ok, k], 1)
            resid[ok, k] = r[ok, k] - np.polyval(c, years[ok])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN cones
        sig_cone = np.sqrt(np.nanmean(resid ** 2, axis=0))
    sig = np.maximum(np.nan_to_num(sig_cone, nan=5.0), sigma_floor_arcsec)
    sigma = np.broadcast_to(sig, r.shape).copy()
    out = dict(epochs=epochs, years=years, angles=np.asarray(d["angles"], dtype=np.float64),
               r=r, sigma=sigma, mask=np.isfinite(r))
    # per-cone proper motions by profile registration (casa_expansion.py), kept
    # only where two registration windows agree to 0.1"/yr (the others lock
    # onto different features); same 10-degree cone centres as the outline
    fa, fb = PM_FILES[pm_files]
    if pm_files != "legacy" and not (fa.exists() and fb.exists()):
        print(f"[obs] WARNING: {fa.name} / {fb.name} missing -- using the LEGACY CCO-locked "
              f"proper motions (run casa_expansion.py --cco-pm)", flush=True)
        fa, fb = PM_FILES["legacy"]
        pm_files = "legacy"
    if fa.exists() and fb.exists():
        a, b = np.load(fa), np.load(fb)
        stable = (np.isfinite(a["pm"]) & np.isfinite(b["pm"])
                  & (np.abs(a["pm"] - b["pm"]) < 0.1))
        masked = cone_mask(out["angles"], pm_mask)
        out["pm"] = np.where(stable & ~masked, a["pm"], np.nan)
        out["pm_sigma"] = np.sqrt(np.nan_to_num(a["pm_err"], nan=1.0) ** 2
                                  + PM_SIGMA_SYS ** 2)
        out["pm_source"] = f"{pm_files} ({fa.name}), mask {pm_mask}: {int(masked.sum())} cones"
    out["rates"] = OBSERVED_RATES
    return out
# =============================================================================
# ============ ↑ Data ↑ =======================================================
# =============================================================================


# =============================================================================
# ============ ↓ The traced initial condition ↓ ===============================
# =============================================================================
def shell_template(r, X, Y, Z, ic, d_rsh, rho_c, L=None, M=None):
    """Orlando (2022) Eq. 1 shell at radius ``r_sh + d_rsh`` (code density),
    cell-averaged radially (see ``casa_pluto.radial_gaussian``).

    ``L``, ``M`` (traced, optional): the similarity-scaled shell -- radius,
    width and scale height x L, density x M / L^3 (mass x M); ``d_rsh`` is then
    a shift of the SCALED radius L r_sh. None = the legacy expression exactly.
    """
    th, ph = np.deg2rad(float(ic["ambient_theta_sh"])), np.deg2rad(float(ic["ambient_phi_sh"]))
    r_dot_D = X * np.cos(th) * np.cos(ph) - Y * np.sin(ph) + Z * np.sin(th) * np.cos(ph)
    sig = float(ic["ambient_sigma_sh"])
    rho_sh = float(ic["ambient_rho_sh"]) / rho_c
    dx = float(ic["box"]) / int(ic["num_cells"])
    if L is None and M is None:
        radial = radial_gaussian(r, float(ic["ambient_r_sh"]) + d_rsh, sig, dx,
                                 erf=jax.scipy.special.erf)
        return rho_sh * radial * jnp.exp(r_dot_D / float(ic["ambient_H_sh"]))
    L = 1.0 if L is None else L
    M = 1.0 if M is None else M
    radial = radial_gaussian(r, L * float(ic["ambient_r_sh"]) + d_rsh, L * sig, dx,
                             erf=jax.scipy.special.erf)
    return (M / L ** 3) * rho_sh * radial * jnp.exp(r_dot_D / (L * float(ic["ambient_H_sh"])))


#: precision of the coordinate matmuls (R K, R^T x) of the IC rotation / similarity
#: map. The GPU default for an f32 matmul is TF32 (10-bit mantissa) on A100 / H100:
#: R^T x then misplaces every resampling point by ~5e-4 x |x| (0.03 cells at 128^3,
#: 0.13 at 512^3), which changed the rotated interior in 3-4 % of the cells (C_ej by
#: up to 0.24, v by 0.7 code units at 512^3) vs the CPU on EVERY GPU run, 1 device or
#: 8 -- the "8-way sharded sim-path IC bug" of 2026-09-27 (ers/integ2/REPORT.md). The
#: CPU ignores the setting (bitwise unchanged there); on GPU it restores CPU agreement
#: to 2.5e-5 (f32 rounding). ``casa_rescale_test.test_coordinate_precision`` checks it.
COORD_PRECISION = jax.lax.Precision.HIGHEST


def rotation_matrix(rotvec_deg):
    """Rodrigues: rotation vector (degrees) -> 3x3 matrix, traced and smooth at 0."""
    w = jnp.deg2rad(jnp.asarray(rotvec_deg))
    th2 = jnp.sum(w ** 2)
    th = jnp.sqrt(th2 + 1e-30)
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    # sin(th)/th and (1 - cos th)/th^2 with their series near 0 (smooth JVP)
    a = jnp.where(th2 < 1e-8, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    b = jnp.where(th2 < 1e-8, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / (th2 + 1e-30))
    return jnp.eye(3) + a * K + b * jnp.matmul(K, K, precision=COORD_PRECISION)


def rotate_interior(fields, vector_keys, R, geom, box, n, r_cut=1.25, width=0.05):
    """Rotate the remnant interior (r < r_cut) rigidly by R; keep the exterior.

    Every scalar field f becomes f(R^T x) and the velocity v(x) -> R v(R^T x),
    by trilinear resampling (differentiable in R); a smooth radial mask blends
    the rotated interior into the untouched wind and shell. At 146 yr the
    forward shock is at 1.21 +- 0.02 pc, so r_cut = 1.25 pc takes the whole
    shocked region with it.
    """
    from jax.scipy.ndimage import map_coordinates
    r, X, Y, Z = geom
    P = jnp.stack([X, Y, Z], 0).reshape(3, -1)
    S = jnp.matmul(R.T, P, precision=COORD_PRECISION)     # source points (full f32 on GPU)
    idx = (S + 0.5 * box) / (box / n) - 0.5
    def samp(f):
        return map_coordinates(f, [idx[0], idx[1], idx[2]], order=1,
                               mode="nearest").reshape(f.shape)
    m = jax.nn.sigmoid((r_cut - r) / width)
    out = {}
    for k, f in fields.items():
        if k in vector_keys:
            continue
        out[k] = m * samp(f) + (1.0 - m) * f
    vs = [samp(fields[k]) for k in vector_keys]
    for i, k in enumerate(vector_keys):
        rot_v = R[i, 0] * vs[0] + R[i, 1] * vs[1] + R[i, 2] * vs[2]
        out[k] = m * rot_v + (1.0 - m) * fields[k]
    return out


def similarity_params(src):
    """(ln_L, ln_T, ln_M) from ``src``: None -> None (the legacy, unscaled path);
    a dict -> its ``SIM_NAMES`` entries (missing ones 0; None if it has none);
    a 3-sequence -> as is."""
    if src is None:
        return None
    if isinstance(src, dict):
        if not any(k in src for k in SIM_NAMES):
            return None
        return tuple(src.get(k, 0.0) for k in SIM_NAMES)
    return tuple(src[i] for i in range(3))


def similarity_scales(sim):
    """(L, T, M) = exp of ``similarity_params(sim)`` (traced), or None."""
    s3 = similarity_params(sim)
    if s3 is None:
        return None
    return tuple(jnp.exp(jnp.asarray(v)) for v in s3)


def ic_age(ic, sim=None):
    """The IC's age consistent with the similarity map: T x ic["age"] (145.5 yr
    for Orlando's state). THIS, not ic["age"], must start the evolution when
    ``sim`` is used (traced in ln_T)."""
    s3 = similarity_params(sim)
    if s3 is None:
        return float(ic["age"])
    return float(ic["age"]) * jnp.exp(jnp.asarray(s3[1]))


def similarity_energy_factor(sim):
    """E' / E = M (L / T)^2 of the similarity map (1 for sim None)."""
    sc = similarity_scales(sim)
    if sc is None:
        return 1.0
    L, T, M = sc
    return M * (L / T) ** 2


def ambient_source(ic, rho_c, r, X, Y, Z):
    """Orlando's analytic ambient (wind + shell, unscaled, code units) at the
    points (r, X, Y, Z): (rho, press) with the exterior T cap of
    ``casa_pluto.ambient_pressure`` (traced in the points)."""
    cu = snr_code_units()
    r_ref = float(ic["ambient_r_ref"])
    r_s = jnp.maximum(r, 1e-3)
    rho = float(ic["ambient_rho_w"]) / rho_c * (r_ref / r_s) ** 2 \
        + shell_template(r_s, X, Y, Z, ic, 0.0, rho_c)
    p_c = float((1.0 * cu.code_pressure).to(u.erg / u.cm ** 3).value)
    v_c = float((1.0 * cu.code_velocity).to(u.cm / u.s).value)
    p = float(ic["ambient_p_w"]) / p_c * (r_s / r_ref) ** float(ic["ambient_p_slope"])
    kt_over_m = float(const.k_B.cgs.value * 1e5 / (1.2889 * const.m_p.cgs.value)) / v_c ** 2
    return rho, jnp.minimum(p, rho * kt_over_m)


def similarity_resample(fields, vector_keys, R, L, geom, box, n, ic, rho_c, r_cut=1.25, width=0.05,
                        volumetric=("rho", "press"), velocity="primitive"):
    """f'(x) = f(R^T x / L) inside the (scaled) remnant, f(x / L) outside --
    ``rotate_interior`` composed with the similarity stretch in ONE trilinear
    resampling (differentiable in R and L). The factors of the map are applied
    by the caller.

    The resampling acts on CONSERVED densities: the ``volumetric`` fields (rho,
    press = (gamma - 1) e_int) as they are, every other field q (velocities,
    mass fractions, shock history) as rho q, divided by the resampled rho
    afterwards. Interpolating rho and C_ej separately loses ~5 % of the ejecta
    mass at 128^3 (the product of two interpolants at a clump edge), rho C_ej
    keeps it to the interpolation error of rho itself.

    Where x / L leaves the source box (L < 1) rho and press come from the
    analytic ambient at x / L (``ambient_source``); the per-mass fields keep
    their box-face values there (0, or the CSM composition, as in the IC).

    ``velocity``: the ``vector_keys`` are resampled as PRIMITIVE velocities
    ("primitive", default, as ``rotate_interior``) or as momenta rho v / rho
    ("momentum"). The momentum form hands a void cell next to a clump the
    clump's velocity, i.e. it SHARPENS the velocity jumps between clumps (at
    512^3 on the rescaled IC: 98k vs 58k cells with a neighbour |dv| > 2000
    km/s), and the 512^3 sharded forward then grew a 27,000 km/s rarefied
    ejecta cell within 2 yr and NaN'd by 2000; the primitive form is the one
    the 512^3 legacy runs were stable with. Mass (rho) and the mass fractions
    stay rho-weighted either way (the ejecta-mass point above).
    """
    from jax.scipy.ndimage import map_coordinates
    r, X, Y, Z = geom
    P = jnp.stack([X, Y, Z], 0).reshape(3, -1)
    h = box / n
    S_plain = P / L
    S_rot = jnp.matmul(R.T, P, precision=COORD_PRECISION) / L
    i_plain = (S_plain + 0.5 * box) / h - 0.5
    i_rot = (S_rot + 0.5 * box) / h - 0.5

    def samp(f, idx):
        return map_coordinates(f, [idx[0], idx[1], idx[2]], order=1,
                               mode="nearest").reshape(f.shape)
    m = jax.nn.sigmoid((r_cut * L - r) / (width * L))
    outside = (jnp.max(jnp.abs(S_plain), axis=0) > 0.5 * box - 0.5 * h).reshape(r.shape)
    amb_rho, amb_p = ambient_source(ic, rho_c, r / L, X / L, Y / L, Z / L)
    rho = fields["rho"]
    rho_ext_face = samp(rho, i_plain)
    out = {}
    for k, f in fields.items():
        if k in vector_keys:
            continue
        cons = f if k in volumetric else rho * f
        ext = samp(cons, i_plain)
        if k == "rho":
            ext = jnp.where(outside, amb_rho, ext)
        elif k == "press":
            ext = jnp.where(outside, amb_p, ext)
        elif k not in volumetric:
            ext = jnp.where(outside, amb_rho * ext / jnp.maximum(rho_ext_face, 1e-30), ext)
        out[k] = m * samp(cons, i_rot) + (1.0 - m) * ext
    mom = velocity == "momentum"
    wv = rho if mom else 1.0
    ms = [samp(wv * fields[k], i_rot) for k in vector_keys]
    for i, k in enumerate(vector_keys):
        rot_m = R[i, 0] * ms[0] + R[i, 1] * ms[1] + R[i, 2] * ms[2]
        out[k] = m * rot_m + (1.0 - m) * samp(wv * fields[k], i_plain)
    rho_new = jnp.maximum(out["rho"], 1e-30)
    return {k: (v if (k in volumetric or (k in vector_keys and not mom)) else v / rho_new)
            for k, v in out.items()}


def wind_mean_factor(a, form="exp"):
    """Sphere mean of the wind's angular factor g(n_hat) (traced in ``a``).

    ``exp``: 1 by construction. ``clip``: max(1 + A mu, 0.05) averages to 1 for
    A = |a| <= 0.95 and to 1 + (A - 0.95)^2 / (4 A) beyond (the floored cap
    ADDS mass).
    """
    if form == "exp":
        return jnp.ones((), jnp.asarray(a).dtype)
    A = jnp.sqrt(jnp.sum(jnp.asarray(a) ** 2) + 1e-12)
    return 1.0 + jnp.where(A > 0.95, (A - 0.95) ** 2 / (4.0 * A), 0.0)


def wind_asymmetry(a, X, Y, Z, r, form="exp"):
    """Angular factor of the unshocked wind density, rho_w -> rho_w g(n_hat).

    ``exp`` (default): g = exp(a . n) / <exp(a . n)>_sphere, with the sphere
    mean sinh|a| / |a| -- smooth, positive at every |a|, and ``ln_fw`` stays
    the sphere-mean density. To first order in a it equals the old form.
    ``clip``: the pre-2026-09-25 max(1 + a . n, 0.05), which floors a whole cap
    once |a| > 0.95 (Q2: |a| = 1.13, 8 % of the sky at the floor).
    """
    r_s = jnp.maximum(r, 1e-3)
    an = (a[0] * X + a[1] * Y + a[2] * Z) / r_s
    if form == "clip":
        return jnp.maximum(1.0 + an, 0.05)
    if form != "exp":
        raise ValueError(f"wind dipole form {form!r}")
    A2 = a[0] ** 2 + a[1] ** 2 + a[2] ** 2
    A = jnp.sqrt(A2 + 1e-12)
    mean = jnp.where(A2 < 1e-4, 1.0 + A2 / 6.0 + A2 ** 2 / 120.0, jnp.sinh(A) / A)
    return jnp.exp(an) / mean


def transform_fields(ic, theta, geom, rho_c, extra=(), wind_dipole="exp", sim=None, velocity="primitive",
                     jet=None):
    """Orlando's fields transformed by the physics parameters (traced).

    ``theta``: a ``PARAM_NAMES``-positional vector (longer vectors, e.g.
    casa_xfit's, are read up to ``len(PARAM_NAMES)``) or a dict by name.
    ``extra``: further scalar fields of ``ic`` (composition, shock history) to
    carry along -- rotated with the interior, otherwise unchanged (they are
    mass fractions or per-parcel histories, not densities).
    ``wind_dipole``: form of the wind's angular factor (``wind_asymmetry``).
    ``sim``: (ln_L, ln_T, ln_M) (``SIM_NAMES``; traced) -- the exact similarity
    map, applied FIRST to the whole state (interior, wind, shell, shock
    history: ``time_since_shock`` x T, ``density_time`` x M T / L^3); every
    other knob then acts on the scaled state. The evolution must start at
    ``ic_age(ic, sim)`` = T x 145.5 yr. ``None`` (default; also a dict
    ``theta`` without SIM_NAMES) is the legacy path, bitwise unchanged.
    ``velocity``: how the map resamples the velocities (``similarity_resample``;
    "primitive" default since 2026-09-27, "momentum" the first version).
    ``jet``: None (default, bitwise the old fields) or a dict of the
    ``casa_jet.JET_NAMES`` parameters: the Si-rich NE jet + SW counter-jet is
    ADDED last, in the rescaled, rotated frame, at the IC age ``ic_age(ic, sim)``
    (``casa_jet.add_jet``; its bookkeeping under ``out["_jet"]``). casa_xfit
    applies the same function itself, after its Y_lm modes.
    """
    p = dict(theta) if isinstance(theta, dict) else dict(zip(PARAM_NAMES, theta))
    if sim is None and isinstance(theta, dict):
        sim = similarity_params(theta)
    scales = similarity_scales(sim)
    r, X, Y, Z = geom
    box, n = float(ic["box"]), int(ic["num_cells"])
    keys = ("rho", "press", "vx", "vy", "vz", "C_ej", "shocked_fraction")
    base = {k: jnp.asarray(ic[k]) for k in keys + tuple(k for k in extra if k not in keys)}
    base["C_dop"] = jnp.asarray(doppler_tracer(ic))
    R = rotation_matrix(jnp.stack([p["rot_x"], p["rot_y"], p["rot_z"]]))
    # the rotation mask (r_cut 1.25 pc, width 0.05 pc) brackets the 146-yr forward
    # shock (1.21 pc) inside Orlando's shell (1.50 pc); an IC that is ALREADY
    # similarity-scaled (``casa_pluto convert --sim`` / ``casa_rescale``: its
    # ``similarity_L``) has both radii x L, so the mask scales with it (x 1.0,
    # i.e. bitwise the old constants, for an unscaled IC)
    L_ic = float(ic["similarity_L"]) if "similarity_L" in ic else 1.0
    cut = dict(r_cut=1.25 * L_ic, width=0.05 * L_ic)
    if scales is None:
        base = rotate_interior(base, ("vx", "vy", "vz"), R, geom, box, n, **cut)
        L = M = None
    else:
        L, T, M = scales
        f_rho, f_v = M / L ** 3, L / T
        # The shell the resampled state actually carries (and its d_rsh-shifted
        # copy) are resampled WITH the state from Orlando's template on the IC
        # grid; the shift there is d_rsh / L, i.e. d_rsh of the scaled radius
        # L r_sh. The analytic L-scaled template (shell_template(L=, M=)) is
        # narrower than the resampled shell, so subtracting it for ln_fsh /
        # d_rsh left a +- residual of 0.45 Msun at 128^3 and, through the density
        # floor below, CREATED 0.28 Msun (review 2026-09-27). With d_rsh = 0 the
        # two copies are identical, so ln_fsh = d_rsh = 0 is the pure map.
        base["_sh0"] = shell_template(r, X, Y, Z, ic, 0.0, rho_c)
        base["_sh1"] = shell_template(r, X, Y, Z, ic, p["d_rsh"] / L, rho_c)
        base = similarity_resample(base, ("vx", "vy", "vz"), R, L, geom, box, n, ic, rho_c,
                                   volumetric=("rho", "press", "_sh0", "_sh1"), velocity=velocity, **cut)
        sh_scaled = (base.pop("_sh0") * f_rho, base.pop("_sh1") * f_rho)
        base["rho"] = base["rho"] * f_rho
        base["press"] = base["press"] * f_rho * f_v ** 2
        for k in ("vx", "vy", "vz"):
            base[k] = base[k] * f_v
        if "time_since_shock" in base:
            base["time_since_shock"] = base["time_since_shock"] * T
        if "density_time" in base:
            base["density_time"] = base["density_time"] * f_rho * T
    rho = base["rho"]; press = base["press"]
    v = [base[k] for k in ("vx", "vy", "vz")]
    # unshocked circumstellar gas: no ejecta, never shocked
    w_amb = jnp.clip((1.0 - base["C_ej"]) * (1.0 - base["shocked_fraction"]), 0.0, 1.0)
    r_ref = float(ic["ambient_r_ref"])
    rho_w = float(ic["ambient_rho_w"]) / rho_c * (r_ref / jnp.maximum(r, 1e-3)) ** 2
    if scales is not None:
        rho_w = rho_w * (M / L)                       # the scaled wind: rho r^2 x M / L
    slope = (r_ref / jnp.maximum(r, 1e-3)) ** p["ds_w"]
    if scales is None:
        sh0 = shell_template(r, X, Y, Z, ic, 0.0, rho_c)
        sh1 = shell_template(r, X, Y, Z, ic, p["d_rsh"], rho_c)
    else:
        sh0, sh1 = sh_scaled
    asym = wind_asymmetry((p["wa_x"], p["wa_y"], p["wa_z"]), X, Y, Z, r, form=wind_dipole)
    d_amb = (jnp.exp(p["ln_fw"]) * slope * asym - 1.0) * rho_w + jnp.exp(p["ln_fsh"]) * sh1 - sh0
    rho_new = rho + w_amb * d_amb
    rho_new = jnp.maximum(rho_new, 1e-3 * rho)
    # ambient: scale p with rho (fixed T); interior: self-similar speed-up
    s = jnp.exp(p["ln_sv"])
    w_in = 1.0 - w_amb
    vel_scale = 1.0 + (s - 1.0) * w_in
    # large-scale ejecta asymmetry: a velocity dipole on the ejecta only
    r_safe = jnp.maximum(r, 1e-3)
    dip = (p["dip_x"] * X + p["dip_y"] * Y + p["dip_z"] * Z) / r_safe
    vel_scale = vel_scale * (1.0 + base["C_ej"] * dip)
    press_new = press * (rho_new / rho) * (1.0 + (s ** 2 - 1.0) * w_in)
    out = dict(rho=rho_new, vx=v[0] * vel_scale, vy=v[1] * vel_scale,
               vz=v[2] * vel_scale, press=press_new, C_dop=base["C_dop"],
               C_ej=base["C_ej"])
    out.update({k: base[k] for k in extra if k not in out})
    if jet is not None:
        import casa_jet
        t_ic = ic_age(ic, sim) * float((1.0 * u.yr).to(snr_code_units().code_time).value)
        out = casa_jet.add_jet(out, dict(jet, psi=p["psi"]), geom, t_ic, (box / n) ** 3)
    return out


def doppler_tracer(ic):
    """The scalar the Doppler observable is weighted by: the Si-group mass
    fraction, with the circumstellar part at SOLAR (``casa_pluto.recompose_csm``).

    The data are Si He-alpha line centroids, so the model velocity must be
    Si-weighted. Weighting by the ejecta tag instead (the first version) is
    dominated by Orlando's He envelope and gives the OPPOSITE correlation with
    the data (+0.31 vs -0.16 on the same fit-C state); with the old 4x solar
    CSM Si the shocked shell would dominate the Si weight instead.
    """
    from casa_pluto import solar_csm_composition
    from _common import CSM_COMPOSITION
    f_csm = np.clip(1.0 - np.asarray(ic["C_ej"], np.float64), 0.0, 1.0)
    return np.clip(np.asarray(ic["C_Si"], np.float64)
                   + f_csm * (solar_csm_composition()["Si"] - CSM_COMPOSITION["Si"]),
                   0.0, 1.0).astype(np.float32)


def make_initial_state(ic, theta, *, config, rv, geom, rho_c, wind_dipole="exp"):
    """Orlando's state transformed by the physics parameters (traced)."""
    f = transform_fields(ic, theta, geom, rho_c, wind_dipole=wind_dipole)
    scalars = f["C_dop"][None] if config.num_passive_scalars > 0 else None
    return construct_primitive_state(
        config=config, registered_variables=rv, density=f["rho"],
        velocity_x=f["vx"], velocity_y=f["vy"], velocity_z=f["vz"],
        gas_pressure=f["press"], gamma=GAMMA, passive_scalars=scalars)


def write_transformed_ic(ic_path, theta, out_path, wind_dipole="exp", sim=None):
    """Apply ``theta`` to a (composition-carrying) IC and save it in the same
    format, for ``casa_orlando.py --from-state`` production runs.

    The composition is carried as mass fractions, so changing the ambient
    density leaves it correct; the shock history is untouched (only unshocked
    gas and the interior's velocity/pressure are changed). NOTE: the casa_xfit
    Y_lm ejecta-density modes are NOT applied here.

    ``sim`` (ln_L, ln_T, ln_M): the similarity map as in ``transform_fields``;
    the file then carries age = T x age, the scaled ``ambient_*`` bookkeeping
    (``casa_rescale.scale_ambient``), the scaled shock history, and
    ``similarity_{L,T,M}``. (Trilinear resampling: for a conservative,
    resolution-matched production IC use ``casa_rescale.py`` or
    ``casa_pluto.py convert --sim``.)
    """
    ic = dict(np.load(ic_path))
    box, n = float(ic["box"]), int(ic["num_cells"])
    x = (np.arange(n) + 0.5) * box / n - box / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    geom = (np.sqrt(X ** 2 + Y ** 2 + Z ** 2), X, Y, Z)
    rho_c = float((1.0 * snr_code_units().code_density).to(u.g / u.cm ** 3).value)
    th = np.asarray(theta, dtype=np.float64)
    if similarity_params(sim) is not None:
        hist = tuple(k for k in ("C_Fe", "C_Si", "C_O", "C_He", "shocked_fraction",
                                 "time_since_shock", "density_time") if k in ic)
        f = transform_fields(ic, th, geom, rho_c, extra=hist, wind_dipole=wind_dipole, sim=sim)
        f.pop("C_dop", None)
        from casa_rescale import scale_ambient
        Ls, Ts, Ms = (float(v) for v in similarity_scales(sim))
        out = scale_ambient(ic, Ls, Ts, Ms)
        out.update({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
        for k in ("age", "age_target", "map_age"):
            if k in ic:
                out[k] = float(ic[k]) * Ts
        for k, v in zip("LTM", (Ls, Ts, Ms)):
            out[f"similarity_{k}"] = float(ic.get(f"similarity_{k}", 1.0)) * v
        out["similarity_age_unscaled"] = float(ic.get("similarity_age_unscaled", ic["age"]))
        out["similarity_method"] = np.array("linear (casa_pluto_diff.transform_fields)")
        for k in ("bx", "by", "bz"):
            out.pop(k, None)              # not resampled here (synchrotron only; see casa_rescale)
        out["fit_theta"] = np.asarray(theta, dtype=np.float64)
        out["fit_param_names"] = np.array(PARAM_NAMES)
        out["fit_sim"] = np.asarray([float(v) for v in similarity_params(sim)])
        out["fit_wind_dipole"] = np.array(wind_dipole)
        np.savez_compressed(out_path, **out)
        print(f"[diff] wrote transformed + similarity-scaled IC {out_path} (age {float(out['age']):.2f} yr)")
        return
    f = transform_fields(ic, th, geom, rho_c, wind_dipole=wind_dipole)
    f.pop("C_dop", None)
    out = dict(ic)
    out.update({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    # the explosion's orientation moves the composition and the shock history
    # with the interior, exactly as transform_fields moved the hydro fields
    p = dict(zip(PARAM_NAMES, th))
    R = rotation_matrix(jnp.stack([p["rot_x"], p["rot_y"], p["rot_z"]]))
    extra = {k: jnp.asarray(ic[k]) for k in ("C_Fe", "C_Si", "C_O", "C_He", "shocked_fraction",
                                             "time_since_shock", "density_time") if k in ic}
    extra = rotate_interior(extra, (), R, geom, box, n)
    out.update({k: np.asarray(v, dtype=np.float32) for k, v in extra.items()})
    out["fit_theta"] = np.asarray(theta, dtype=np.float64)
    out["fit_param_names"] = np.array(PARAM_NAMES)
    out["fit_wind_dipole"] = np.array(wind_dipole)
    np.savez_compressed(out_path, **out)
    print(f"[diff] wrote transformed IC {out_path}")


# ---- bookkeeping of the transformed IC (traced; audit 2026-09-25) ----------
def ballistic_convergence_date(t_expl, ln_sv, age0):
    """The model's kinematic explosion date: ``ln_sv`` multiplies the IC's
    velocities by s at fixed positions, so its ejecta r / v is age0 / s and they
    converge ballistically at t_expl + age0 (1 - 1/s). This, not t_expl, is what
    the optical knots' convergence date (>= 1671.3) constrains.

    With the similarity map pass the TRACED IC age ``ic_age(ic, sim)`` = 145.5 T
    as ``age0``: the scaled ejecta have r / v = 145.5 T / s, so with ln_sv
    frozen at 0 this is t_expl itself and the wall reduces to t_expl >= 1671.3."""
    return t_expl + age0 * (1.0 - jnp.exp(-ln_sv))


def csm_hydrogen_fraction(ic):
    """H mass fraction of the circumstellar gas: 1 - the IC's CSM composition
    (the pipeline groups He, O(+Ne+Mg), Si(+S+Ar+Ca), Fe)."""
    try:
        comp = ast.literal_eval(str(ic["csm_composition"]))
        return float(1.0 - sum(float(v) for v in comp.values()))
    except (KeyError, ValueError, SyntaxError):
        from casa_pluto import solar_csm_composition
        return float(1.0 - sum(solar_csm_composition().values()))


def wind_nh(ic, p, r_pc=WIND_NH_PRIOR[0], wind_dipole="exp", sim=None):
    """Sphere-mean PRE-SHOCK hydrogen density n_H (cm^-3) of the traced wind at
    ``r_pc``: X_H rho_w / m_H with rho_w = ambient_rho_w (r_ref / r)^(2 + ds_w)
    e^ln_fw <g> (the shell is ~1.5 pc and negligible at 3 pc). Lee+14's 0.89 is
    n_H (hydrogen nuclei), not the particle density n = rho / (mu m_H) that
    Orlando quotes (0.8 at mu = 1.29 is n_H 0.73 here)."""
    r_ref = float(ic["ambient_r_ref"])
    x_h = csm_hydrogen_fraction(ic)
    rho = float(ic["ambient_rho_w"]) * (r_ref / r_pc) ** 2 * (r_ref / r_pc) ** p["ds_w"] \
        * jnp.exp(p["ln_fw"]) * wind_mean_factor((p["wa_x"], p["wa_y"], p["wa_z"]), wind_dipole)
    # the similarity map scales the wind's rho r^2 by M / L (``sim``, or SIM_NAMES in ``p``)
    s3 = similarity_params(sim if sim is not None else p)
    if s3 is not None:
        rho = rho * jnp.exp(jnp.asarray(s3[2]) - jnp.asarray(s3[0]))
    return x_h * rho / M_H_G


def ic_budget(f, cell_vol_pc3, gamma=GAMMA):
    """Mass and energy of transformed IC fields (code units pc / Msun / 1000 km/s):
    dict of M_ej, M_tot (Msun), E_kin, E_th, E_kin_ej, E_tot (in 1e51 erg -- a
    float32-safe unit), traced."""
    e_unit = float((1.0 * u.Msun * (1000.0 * u.km / u.s) ** 2).to(u.erg).value) / 1e51
    rho = f["rho"]
    cej = jnp.clip(f["C_ej"], 0.0, 1.0)
    ek = 0.5 * rho * (f["vx"] ** 2 + f["vy"] ** 2 + f["vz"] ** 2)
    et = f["press"] / (gamma - 1.0)
    out = dict(M_ej=jnp.sum(cej * rho) * cell_vol_pc3, M_tot=jnp.sum(rho) * cell_vol_pc3,
               E_kin=jnp.sum(ek) * cell_vol_pc3 * e_unit, E_th=jnp.sum(et) * cell_vol_pc3 * e_unit,
               E_kin_ej=jnp.sum(cej * ek) * cell_vol_pc3 * e_unit)
    out["E_tot"] = out["E_kin"] + out["E_th"]
    return out


def format_budget(b, ref=None):
    s = (f"M_ej {float(b['M_ej']):.3f} Msun, M_box {float(b['M_tot']):.2f} Msun, "
         f"E_kin {float(b['E_kin']):.3f} (ejecta {float(b['E_kin_ej']):.3f}) + E_th "
         f"{float(b['E_th']):.3f} = E {float(b['E_tot']):.3f} x 1e51 erg")
    if ref is not None:
        s += (f"  [x{float(b['M_ej']) / float(ref['M_ej']):.3f} M_ej, "
              f"x{float(b['E_tot']) / float(ref['E_tot']):.3f} E vs the untransformed IC]")
    return s
# =============================================================================
# ============ ↑ The traced initial condition ↑ ===============================
# =============================================================================


# =============================================================================
# ============ ↓ Smooth observables ↓ =========================================
# =============================================================================
class ConeProfiles:
    """Static cone x radial-bin membership in the plane of the sky (x, z).

    Earth on -y (Orlando's convention, which ``casa_orlando`` shares); cones
    are +-width around each in-simulation position angle atan2(z, x), and a
    cell belongs to the plane of the sky if |y| / r < sin(10 deg).
    """

    def __init__(self, geom_np, *, n_angles=36, nbins=None, r_max=None, dx=None):
        r, X, Y, Z = (np.asarray(g) for g in geom_np)
        nbins = nbins or int(r_max / dx)
        self.angles = np.linspace(0.0, 360.0, n_angles, endpoint=False)
        width = 360.0 / n_angles
        pa = np.rad2deg(np.arctan2(Z, X)) % 360.0
        plane = (np.abs(Y) / np.maximum(r, 1e-12) < np.sin(np.deg2rad(10.0))) & (r < r_max)
        edges = np.linspace(0.0, r_max, nbins + 1)
        self.rc = 0.5 * (edges[:-1] + edges[1:])
        rbin = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
        cells, segs = [], []
        flat = np.arange(r.size).reshape(r.shape)
        for i, a in enumerate(self.angles):
            sel = plane & (np.abs(((pa - a + 180.0) % 360.0) - 180.0) < width)
            cells.append(flat[sel]); segs.append(i * nbins + rbin[sel])
        self.cells = jnp.asarray(np.concatenate(cells))
        self.segs = jnp.asarray(np.concatenate(segs))
        self.nseg = n_angles * nbins
        self.nbins = nbins
        cnt = np.bincount(np.concatenate(segs), minlength=self.nseg).astype(np.float64)
        self.count = jnp.asarray(np.maximum(cnt, 1.0))
        self.valid = jnp.asarray((cnt > 0).reshape(n_angles, nbins))

    def mean(self, field):
        tot = jax.ops.segment_sum(field.ravel()[self.cells], self.segs, self.nseg)
        return (tot / self.count).reshape(len(self.angles), self.nbins)


def shocked_indicator(state, rv, t_per_code, log_t_shock=7.0, width=0.1):
    """Smooth 0/1: is the gas hotter than 1e7 K?

    Temperature rather than a pressure contrast: the unshocked ejecta sit at the
    solver's 1e4 K specific floor, which at their density is far above the
    ambient pressure, so a pressure indicator switches on in the whole interior.
    Nothing unshocked is within a dex of 1e7 K (the wind is <= 1e5 K); the
    shocked CSM is at 1e8-1e9 K and the shocked ejecta above 1e7.
    """
    T = t_per_code * state[rv.pressure_index] / state[rv.density_index]
    x = (jnp.log10(jnp.maximum(T, 1.0)) - log_t_shock) / width
    return jax.nn.sigmoid(x)


#: reverse-shock estimators: "unshocked" (default, the tracer), "coldej" (cold
#: ejecta, C_ej (1 - hot): no shock-history scalar needed) and "legacy" (the
#: pre-2026-09-25 inner power mean of the hot indicator -- an ARTEFACT)
RS_ESTIMATORS = ("unshocked", "coldej", "legacy")


def unshocked_ejecta_profile(cones, c_ej, marker, eps=1e-3):
    """Per-(cone, radius) unshocked-ejecta fraction u = <C_ej (1 - m)> / <C_ej>.

    ``marker`` is the library's ``shocked_fraction`` (estimator "unshocked") or
    the hot indicator (estimator "coldej"). Where there are no ejecta (<C_ej> <<
    eps: the shocked and unshocked CSM) u -> 0 smoothly.
    """
    c = jnp.clip(c_ej, 0.0, 1.0)
    ej = cones.mean(c)
    un = cones.mean(c * (1.0 - jnp.clip(marker, 0.0, 1.0)))
    return un / (ej + eps)


#: gate of the unshocked-ejecta fraction u in ``rs_unshocked_edge``
RS_GATES = ("thrlin", "sigmoid")


def rs_unshocked_edge(u_prof, rc, valid, r_fs, k=12, u_width=0.05, gate="thrlin", u_min=0.2):
    """r_RS per cone: the OUTER power-mean edge of the unshocked ejecta,
    r_RS = ((k + 1) int g r^k dr)^(1 / (k + 1)) over r < r_FS (a one-bin
    sigmoid window), which -> the outer edge of g for large k, like r_FS.

    ``gate`` maps the unshocked-ejecta fraction u to the indicator g:

    * ``"thrlin"`` (default): g = u sigmoid((u - u_min) / u_width) -- LINEAR in
      u where u > u_min, so the edge bin's fractional u moves r_RS continuously
      at sub-bin resolution (d r_RS / d R_true = 0.99 +- 0.12 on a synthetic
      remnant), while the thin tails u < u_min of unshocked clumps beyond the
      reverse shock are cut (they carry weight (r / r_RS)^12);
    * ``"sigmoid"``: g = sigmoid((u - 1/2) / u_width), the audit's proposal
      (forward_physics F1). Same values to <= 0.003 in <r_RS / r_FS> on the
      evolved 256^3 states, but a near-binary g quantises r_RS to the radial
      bins: d r_RS / d R_true swings 0.26-1.9 across a bin.
    """
    dr = rc[1] - rc[0]
    u = jnp.clip(u_prof, 0.0, 1.0)
    if gate == "thrlin":
        g = u * jax.nn.sigmoid((u - u_min) / u_width)
    elif gate == "sigmoid":
        g = jax.nn.sigmoid((u - 0.5) / u_width)
    else:
        raise ValueError(f"r_RS gate {gate!r}")
    g = g * jax.nn.sigmoid((r_fs[:, None] - rc[None, :]) / dr)
    g = jnp.where(valid, g, 0.0)
    # floor rc[0]^(k+1) (as r_FS's r_in^(k+1)): a cone with no unshocked ejecta
    # (g == 0) would otherwise give 0^(1/(k+1)), whose JVP / VJP is inf * 0 =
    # NaN -- in reverse mode even when r_RS is unused (scan instantiates zero
    # cotangents). Changes a normal cone by ~(rc[0] / r_RS)^(k+1) ~ 1e-20.
    return (rc[0] ** (k + 1) + (k + 1) * jnp.sum(g * rc ** k, axis=1) * dr) ** (1.0 / (k + 1))


def edge_radii(w_prof, rc, valid, k=12, r_in_frac=0.25, u_prof=None, u_width=0.05, rs_gate="thrlin"):
    """Outer edge (r_FS) of the shocked region per cone, and the reverse shock.

    Power means rather than derivative-weighted means: for an indicator equal
    to 1 on ``[a, b]``, ``((k + 1) int w r^k dr)^(1 / (k + 1)) -> b`` as k
    grows, and holes inside the shell move it only logarithmically. The
    derivative-weighted version averaged every dip in the shocked layer into
    the answer.

    r_RS: with ``u_prof`` (``unshocked_ejecta_profile``) the outer edge of the
    unshocked ejecta (``rs_unshocked_edge``, the default everywhere the
    tracers exist). Without it, the LEGACY inner power mean of ``w`` from
    ``r_in_frac`` r_FS outward, ``((k - 1) int w r^-k dr)^(-1 / (k - 1))``,
    which is an ARTEFACT: below the sigmoid cut the r^-k weight outgrows the
    sigmoid tail, so floor-temperature and warm central gas dominate and the
    result sits at or below the cut (audit 2026-09-25).
    """
    w = jnp.where(valid, w_prof, 0.0)
    dr = rc[1] - rc[0]
    r_fs0 = ((k + 1) * jnp.sum(w * rc ** k, axis=1) * dr) ** (1.0 / (k + 1))
    # Second pass over the OUTER shell only. The shocked interior is the
    # Rayleigh-Taylor mixing layer, whose pointwise tangent grows exponentially
    # (chaotic; e-folding ~10-15 yr after 250 yr at 128^3): even at weight
    # (r / R)^k ~ 1e-4 it swamps d r_FS / d theta. The blast wave itself is
    # not chaotic, so integrate from a FROZEN inner radius r_in = 0.88 r_FS0
    # outward and fill the inside analytically (w = 1 there, as it is: the
    # contact discontinuity sits at ~0.75 r_FS).
    r_in = jax.lax.stop_gradient(0.88 * r_fs0)
    outer = jax.nn.sigmoid((rc[None, :] - r_in[:, None]) / (0.5 * dr))
    r_fs = (r_in ** (k + 1) + (k + 1) * jnp.sum(w * outer * rc ** k, axis=1) * dr) \
        ** (1.0 / (k + 1))
    if u_prof is not None:
        return r_fs, rs_unshocked_edge(u_prof, rc, valid, r_fs, k=k, u_width=u_width, gate=rs_gate)
    return r_fs, legacy_inner_edge(w, rc, r_fs, k=k, r_in_frac=r_in_frac)


def legacy_inner_edge(w, rc, r_fs, k=12, r_in_frac=0.25):
    """The pre-2026-09-25 r_RS (inner power mean of the hot indicator); kept to
    reproduce old numbers -- an estimator artefact, see ``edge_radii``."""
    dr = rc[1] - rc[0]
    inner = jax.nn.sigmoid((rc[None, :] - r_in_frac * r_fs[:, None]) / (2 * dr))
    return ((k - 1) * jnp.sum(w * inner * rc ** (-k), axis=1) * dr) ** (-1.0 / (k - 1))


def _binomial_smooth(q, passes):
    """``passes`` rounds of [1/4, 1/2, 1/4] along each spatial axis (periodic).

    Normalised, so the integral of every field is preserved exactly; ``passes``
    rounds are a Gaussian of variance passes / 2 cells^2.
    """
    for _ in range(passes):
        for ax in range(1, q.ndim):
            q = 0.25 * jnp.roll(q, 1, axis=ax) + 0.5 * q + 0.25 * jnp.roll(q, -1, axis=ax)
    return q


def tangent_filter(state, sigma_cells):
    """Identity on the state; a conservative low-pass on its TANGENT.

    Forward-mode derivatives through the RT mixing layer grow exponentially
    (chaos: grid-scale shear, e-folding a few yr at 128-256^3), and that
    growth lives at the grid scale. The derivatives the fit needs are
    large-scale: a shock's displacement enters the tangent as a spike of
    integral (delta rho) * (delta r), which a normalised smoothing preserves.
    Filtering the tangent between integration segments therefore damps the
    chaotic grid-scale part while keeping the integrals the observables are
    built from -- a regularised tangent, to be validated against finite
    differences (which see the ensemble-scale response), not assumed right.
    """
    passes = max(int(round(2.0 * sigma_cells ** 2)), 1)

    @jax.custom_jvp
    def f(q):
        return q

    @f.defjvp
    def f_jvp(primals, tangents):
        (q,), (dq,) = primals, tangents
        return q, _binomial_smooth(dq, passes)

    return f(state)


def periodic_interp(angles_deg, values, query_deg):
    """Linear interpolation on a periodic angular grid (traced in the query).
    The grid must be ``n`` equally spaced nodes starting at 0 deg."""
    n = angles_deg.shape[0]
    step = 360.0 / n
    q = (query_deg % 360.0) / step
    i0 = jnp.floor(q).astype(jnp.int32) % n
    f = q - jnp.floor(q)
    return values[..., i0] * (1.0 - f) + values[..., (i0 + 1) % n] * f


def periodic_interp_rows(angles_deg, values, query_deg):
    """``periodic_interp`` with one query row per value row: values (..., n),
    query (..., q) with the same leading shape -> (..., q)."""
    n = angles_deg.shape[0]
    step = 360.0 / n
    q = (query_deg % 360.0) / step
    i0 = jnp.floor(q).astype(jnp.int32) % n
    f = q - jnp.floor(q)
    v0 = jnp.take_along_axis(values, i0, axis=-1)
    v1 = jnp.take_along_axis(values, (i0 + 1) % n, axis=-1)
    return v0 * (1.0 - f) + v1 * f


def recentre_radii(radius_at, theta_deg, c_w, c_n, n_iter=2):
    """Rim radii about a sky point O from a rim known about another centre C.

    ``radius_at(phi)`` gives the rim radius about C at sky angle ``phi`` (deg,
    theta convention: from west through north), shape (E, q) for a query of
    shape (E, q); ``(c_w, c_n)`` = C - O in sky arcsec (west, north). Returns
    r'(theta) along the ray from O at angle theta, shape (E, len(theta)).

    First order: r' = r(theta) + c . n_hat(theta). ``n_iter`` fixed-point
    passes then evaluate the rim at the angle about C where the ray actually
    meets it and intersect exactly (r' = c.n + sqrt(r^2 - |c|^2 + (c.n)^2)),
    which removes the O(|c|^2 / r) and the angular-relabelling errors
    (|c| / r ~ 0.1 here). All smooth in the radii and in c.
    """
    theta_deg = jnp.asarray(theta_deg)
    th = jnp.deg2rad(theta_deg)
    ex, ey = jnp.cos(th), jnp.sin(th)
    cn = c_w * ex + c_n * ey
    c2 = c_w ** 2 + c_n ** 2
    rp = radius_at(theta_deg) + cn
    for _ in range(n_iter):
        px, py = rp * ex - c_w, rp * ey - c_n
        phi = jnp.rad2deg(jnp.arctan2(py, px))
        r = radius_at(phi)
        rp = cn + jnp.sqrt(jnp.maximum(r ** 2 - c2 + cn ** 2, 1e-6))
    return rp


def cone_m1(theta_deg, r):
    """m = 1 (dipole) amplitude and angle (theta convention) of r(theta), per row."""
    th = np.deg2rad(np.asarray(theta_deg))
    r = np.asarray(r, np.float64)
    r = np.where(np.isfinite(r), r, np.nanmean(r, axis=-1, keepdims=True))
    d = r - r.mean(-1, keepdims=True)
    c1 = 2 * np.mean(d * np.cos(th), -1); s1 = 2 * np.mean(d * np.sin(th), -1)
    return np.hypot(c1, s1), np.rad2deg(np.arctan2(s1, c1)) % 360.0


def vink22_model_rates(r_coe_arcsec, years):
    """Model fractional expansion rate (%/yr) per Vink+22 sector from radii
    about the expansion centre, r (E, 18), over the epochs 2000-2019."""
    yrs = np.asarray(years)
    sel = np.nonzero((yrs >= VINK22_YEARS[0]) & (yrs <= VINK22_YEARS[1]))[0]
    r = r_coe_arcsec[sel]
    yc = jnp.asarray(yrs[sel] - yrs[sel].mean())
    slope = jnp.sum(yc[:, None] * (r - r.mean(0)), axis=0) / jnp.sum(yc ** 2)
    return 100.0 * slope / r.mean(0)
# =============================================================================
# ============ ↑ Smooth observables ↑ =========================================
# =============================================================================


# =============================================================================
# ============ ↓ The forward model ↓ ==========================================
# =============================================================================
def make_forward(ic_path, obs, *, pa_flip=1.0, pa_offset=0.0, cfl=0.3,
                 tangent_sigma=0.0, filter_every_yr=0.0, ad_llf_cold=0.0,
                 centre="coe", rs_estimator="unshocked", wind_dipole="exp",
                 recentre_iter=2, config_overrides=None, sim=None):
    """``theta -> model radii (arcsec) at every (epoch, observed cone)``.

    ``sim``: fixed similarity parameters (ln_L, ln_T, ln_M) (``SIM_NAMES``);
    ``forward(theta, sim=...)`` overrides them per call (traced). With a
    similarity map the evolution starts at the traced ``ic_age`` = T x 145.5 yr.

    ``pa_flip``/``pa_offset`` map the simulation's position angle onto the
    image's: image PA = flip * sim PA + offset + psi. They are fixed by
    ``casa_observe`` on the same state (``--calibrate-pa``).

    ``centre``: "coe" places the model's explosion centre at the Thorstensen
    expansion centre and re-centres the model radii onto RA0/DEC0, where the
    data outline is measured; "grid" is the legacy comparison about the model
    centre. ``rs_estimator``: see ``RS_ESTIMATORS`` ("unshocked" adds C_ej and
    the library's shock history to the evolved state).
    """
    ic = dict(np.load(ic_path))
    box, n = float(ic["box"]), int(ic["num_cells"])
    sim_fixed = similarity_params(sim)
    # static (python) IC age: sizes the tangent-filter sub-segments below
    age0 = float(ic["age"]) * (float(np.exp(float(sim_fixed[1]))) if sim_fixed is not None else 1.0)
    cu = snr_code_units()
    rho_c = float((1.0 * cu.code_density).to(u.g / u.cm ** 3).value)
    yr = float((1.0 * u.yr).to(cu.code_time).value)
    tracers = rs_estimator in ("unshocked", "coldej")
    track = rs_estimator == "unshocked"
    # passive scalars: the Si-group fraction (Doppler), and C_ej for the
    # reverse-shock estimator; the shock history for "unshocked"
    n_sc = 2 if tracers else 1
    kw = dict(dual_energy=True, progress_bar=False, weno_ad_frozen_weights=True,
              positivity_config=fd_positivity(mode=POSITIVITY_REDISTRIBUTE),
              num_passive_scalars=n_sc, track_shock_history=track,
              passive_scalar_bounds=tuple((0.0, 1.0) for _ in range(n_sc)),
              ad_tangent_llf_cold_factor=float(ad_llf_cold))
    kw.update(config_overrides or {})
    config = make_fd_config(box, n, **kw)
    rv = get_registered_variables(config)
    hd = get_helper_data(config)
    c = hd.geometric_centers
    X, Y, Z = c[..., 0] - box / 2, c[..., 1] - box / 2, c[..., 2] - box / 2
    r = jnp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    geom = (r, X, Y, Z)
    cones = ConeProfiles(geom, r_max=0.5 * box, dx=box / n)
    cell_vol = (box / n) ** 3
    i0 = rv.passive_scalar_index
    i_hist = i0 + rv.num_passive_scalars - 4
    # Doppler sectors on the (x, z) sky plane (projection along y), annulus in
    # pc at a reference distance (membership is static; the distance enters the
    # model velocity only through nothing -- velocities are distance-free)
    xs = (np.arange(n) + 0.5) * box / n - box / 2
    PX, PZ = np.meshgrid(xs, xs, indexing="ij")
    ann = DOPPLER.get("annulus_arcsec", np.array([40.0, 170.0])) * 3.1e3 / ARCSEC_PER_RAD
    rp = np.hypot(PX, PZ)
    n_dop = int(DOPPLER.get("n", 24))
    sec = (np.rad2deg(np.arctan2(PZ, PX)) % 360.0 / (360.0 / n_dop)).astype(int) % n_dop
    sec = np.where((rp > ann[0]) & (rp < ann[1]), sec, n_dop)          # n_dop = discard
    sec_flat = jnp.asarray(sec.ravel())
    # T = t_per_code * p / rho, at mu = 0.6 (the threshold is a dex wide anyway)
    t_per_code = float((0.6 * const.m_p * cu.code_velocity ** 2 / const.k_B).to(u.K).value)
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(cu.code_density).value)
    p_per_n = float((const.k_B * 1e4 * u.K / u.cm ** 3).to(cu.code_pressure).value)
    base_params = SimulationParams(
        gamma=GAMMA, C_cfl=cfl, t_end=1.0,
        minimum_density=0.1 * rho_per_n * 1e-3, minimum_pressure=0.1 * p_per_n * 1e-2,
        minimum_specific_pressure=p_per_n / rho_per_n)
    years = jnp.asarray(obs["years"])
    order = np.argsort(obs["years"])
    # sub-segments per epoch segment for the tangent filter: fixed (static)
    # count, sized on the LONGEST segment (the first one, from the IC age)
    seg_len = np.diff(np.concatenate([[age0],
                                      np.sort(obs["years"]) - PRIOR["t_expl"][0]]))
    n_sub_first = (max(int(np.ceil(abs(seg_len[0]) / filter_every_yr)), 1)
                   if filter_every_yr > 0 else 1)
    n_sub_rest = (max(int(np.ceil(np.max(np.abs(seg_len[1:])) / filter_every_yr)), 1)
                  if filter_every_yr > 0 and len(seg_len) > 1 else 1)
    obs_angles = jnp.asarray(obs["angles"])
    extra = ("shocked_fraction", "time_since_shock", "density_time") if track else ()

    def forward(theta, sim=None):
        p = dict(zip(PARAM_NAMES, theta))
        s3 = similarity_params(sim) if sim is not None else sim_fixed
        f = transform_fields(ic, theta, geom, rho_c, extra=extra, wind_dipole=wind_dipole, sim=s3)
        # the IC's age: T x 145.5 yr under the similarity map (traced in ln_T)
        age_ic = ic_age(ic, s3)
        sc = [f["C_dop"]] + ([f["C_ej"]] if tracers else [])
        state = construct_primitive_state(
            config=config, registered_variables=rv, density=f["rho"],
            velocity_x=f["vx"], velocity_y=f["vy"], velocity_z=f["vz"],
            gas_pressure=f["press"], gamma=GAMMA, passive_scalars=jnp.stack(sc))
        for j, name in enumerate(extra):
            state = state.at[i_hist + 1 + j].set(f[name])
        budget = ic_budget(f, cell_vol)
        cfg = finalize_config(config, state.shape)
        ages = (years - p["t_expl"])[order]
        # one traced integrator inside a scan over the epoch segments, so the
        # while loop is compiled once rather than once per epoch
        dts = jnp.diff(jnp.concatenate([jnp.reshape(jnp.asarray(age_ic, dtype=ages.dtype), (1,)),
                                        ages])) * yr

        def integrate(st, dt, n_sub):
            """dt of evolution; with the tangent filter on, in n_sub pieces
            with the TANGENT low-passed between them (primal untouched)."""
            if not (tangent_sigma > 0 and filter_every_yr > 0):
                return time_integration(st, cfg, base_params._replace(t_end=dt), rv)
            h = dt / n_sub

            def sub(q, _):
                q = time_integration(q, cfg, base_params._replace(t_end=h), rv)
                return tangent_filter(q, tangent_sigma), None
            return jax.lax.scan(sub, st, None, length=n_sub)[0]

        def observe(st):
            hot = shocked_indicator(st, rv, t_per_code)
            w = cones.mean(hot)
            rc = jnp.asarray(cones.rc)
            u_prof = None
            if tracers:
                marker = st[i_hist + 1] if track else hot
                u_prof = unshocked_ejecta_profile(cones, st[i0 + 1], marker)
            r_fs, r_rs = edge_radii(w, rc, cones.valid, u_prof=u_prof)
            r_rs_legacy = legacy_inner_edge(jnp.where(cones.valid, w, 0.0), rc, r_fs)
            rho_s = st[rv.density_index]
            # hot-gas emission measure, the X-ray rate proxy (code rho^2 pc^3)
            em = jnp.sum(hot * rho_s ** 2) * cell_vol
            # Si-emission-weighted line-of-sight velocity per sky sector (the
            # scalar is the Si-group fraction, see doppler_tracer; observer at
            # -y: v_y > 0 = receding), km/s
            wej = hot * rho_s ** 2 * st[i0]
            num = jax.ops.segment_sum(jnp.sum(wej * st[rv.velocity_index.y], axis=1).ravel(),
                                      sec_flat, n_dop + 1)[:n_dop]
            den = jax.ops.segment_sum(jnp.sum(wej, axis=1).ravel(), sec_flat, n_dop + 1)[:n_dop]
            vlos = num / jnp.maximum(den, 1e-30) * 1e3
            return r_fs, r_rs, r_rs_legacy, em, vlos

        # the long first segment (IC age -> first epoch) and the short epoch
        # gaps get their own static sub-segment counts
        state = integrate(state, dts[0], n_sub_first)
        first = observe(state)

        def segment(st, dt):
            st = integrate(st, dt, n_sub_rest)
            return st, observe(st)

        _, rest = jax.lax.scan(segment, state, dts[1:])
        inv = np.argsort(order)
        r_fs_all, r_rs_all, r_rs_leg, em_all, vlos_all = (
            jnp.concatenate([a[None], b], axis=0)[inv] for a, b in zip(first, rest))
        d_pc = jnp.exp(p["ln_D"]) * 1e3
        cones_ang = jnp.asarray(cones.angles)

        def radius_at(phi):          # model radius (arcsec) about its centre at sky angle phi
            sim = (phi - pa_offset - p["psi"]) * pa_flip
            return periodic_interp_rows(cones_ang, r_fs_all, sim) / d_pc * ARCSEC_PER_RAD

        ang_e = jnp.broadcast_to(obs_angles, (r_fs_all.shape[0], obs_angles.shape[0]))
        if centre == "coe":
            # the explosion centre IS the expansion centre; the data outline is
            # about RA0/DEC0: c = C - O = the CoE's sky offset
            r_arc = recentre_radii(radius_at, ang_e, COE_ARCSEC[0], COE_ARCSEC[1], recentre_iter)
        elif centre == "grid":
            r_arc = radius_at(ang_e)
        else:
            raise ValueError(f"centre {centre!r}")
        vth = jnp.broadcast_to(jnp.asarray(VINK22_PA + 90.0), (r_fs_all.shape[0], len(VINK22_PA)))
        return dict(r_fs_arcsec=r_arc,
                    r_fs_pc=r_fs_all, r_rs_pc=r_rs_all, r_rs_legacy_pc=r_rs_leg,
                    r_fs_coe_arcsec=radius_at(vth),
                    em_over_d2=em_all / (d_pc * 1e-3) ** 2,
                    vlos_kms=vlos_all,
                    r_rs_mean_arcsec=jnp.mean(r_rs_all, axis=1) / d_pc * ARCSEC_PER_RAD,
                    t_conv=ballistic_convergence_date(p["t_expl"], p["ln_sv"], age_ic),
                    n_h_wind=wind_nh(ic, p, wind_dipole=wind_dipole, sim=s3),
                    ic_age=jnp.asarray(age_ic),
                    **{f"ic_{k}": v for k, v in budget.items()})

    return forward, dict(ic=ic, cones=cones, config=config)


def pm_residuals(model, obs, pm_scale_sigma=PM_SCALE_SIGMA):
    """Registration proper motions vs the model's slope of r_FS over the
    epochs, with the global PM-scale nuisance profiled (``pm_scale_sigma`` =
    None: no nuisance). Returns (residuals, eps_hat or None)."""
    yrs = jnp.asarray(obs["years"]); yc = yrs - yrs.mean()
    rm = model["r_fs_arcsec"]
    pm_model = jnp.sum(yc[:, None] * (rm - rm.mean(0)), axis=0) / jnp.sum(yc ** 2)
    ok = np.isfinite(obs["pm"])
    pd_ = jnp.asarray(np.nan_to_num(obs["pm"]))[ok]
    s = jnp.asarray(obs["pm_sigma"])[ok]
    d = pm_model[ok] - pd_
    if pm_scale_sigma is None:
        return d / s, None
    # pm_model ~ (1 + eps) pm_data, eps ~ N(0, sigma^2): the minimiser of
    # sum ((d - eps p) / s)^2 + (eps / sigma)^2
    w = 1.0 / s ** 2
    eps = jnp.sum(w * d * pd_) / (jnp.sum(w * pd_ ** 2) + 1.0 / pm_scale_sigma ** 2)
    return jnp.concatenate([(d - eps * pd_) / s, jnp.atleast_1d(eps / pm_scale_sigma)]), eps


def physical_prior_residuals(model, *, conv_wall=True, wind_prior=True):
    """The priors on derived physical quantities (dict of 1-element vectors):
    the one-sided wall on the ballistic convergence date (>= 1671.3) and the
    Lee+14 pre-shock wind density."""
    out = {}
    if conv_wall and "t_conv" in model:
        x = (T_CONV_MIN - model["t_conv"]) / T_CONV_SIGMA
        out["conv_wall"] = jnp.atleast_1d(jax.nn.softplus(WALL_BETA * x) / WALL_BETA)
    if wind_prior and "n_h_wind" in model:
        out["wind_nh"] = jnp.atleast_1d((model["n_h_wind"] - WIND_NH_PRIOR[1]) / WIND_NH_PRIOR[2])
    return out


def rs_ratio(model):
    """Remnant-average r_RS / r_FS per epoch (ratio of the cone means)."""
    return jnp.mean(model["r_rs_pc"], axis=1) / jnp.mean(model["r_fs_pc"], axis=1)


def residual_parts(model, obs, theta, *, sigma_model_arcsec=5.0, use_rate=True, use_pm=True,
                   prior=None, pm_scale_sigma=PM_SCALE_SIGMA, pm_target="registration",
                   rs_term=False, conv_wall=True, wind_prior=True, use_doppler=None):
    """Standardised residual vectors by term (dict, in likelihood order):
    outline shape, proper motions, X-ray rate, Doppler, r_RS / r_FS, priors.

    * outline: per cone, the epoch-mean outline radius against the model error
      ``sigma_model_arcsec`` (a model's shape error is the same at every epoch);
    * motion: per cone, the registration proper motion (``casa_expansion``)
      against the model's slope of r_FS over the epochs (``pm_scale_sigma``:
      the profiled global PM-scale nuisance). Falls back to the outline
      detector's per-epoch deviations if no registration file exists. With
      ``pm_target="vink22"``: Vink+22's per-sector fractional expansion rates
      about the expansion centre instead;
    * rate: log of the hot-gas EM proxy x calibration against the 2000 and
      2022 count rates;
    * rs: remnant-average r_RS / r_FS against 0.66 +- 0.05 (``rs_term``);
    * Gaussian priors (``prior``, default ``PRIOR``), and the physical priors
      (``physical_prior_residuals``).
    """
    prior = PRIOR if prior is None else prior
    use_doppler = obs.get("use_doppler") if use_doppler is None else use_doppler
    parts = {}
    m = np.asarray(obs["mask"])
    d = jnp.where(jnp.asarray(m), model["r_fs_arcsec"] - jnp.asarray(np.nan_to_num(obs["r"])), 0.0)
    n_ep = m.sum(0)
    use = n_ep >= 3
    mean_k = jnp.sum(d, axis=0) / np.maximum(n_ep, 1)
    parts["outline"] = (mean_k / sigma_model_arcsec)[use]
    if pm_target == "vink22":
        a_m = vink22_model_rates(model["r_fs_coe_arcsec"], obs["years"])
        sig = np.sqrt(VINK22_SIGMA ** 2 + VINK22_SIGMA_MODEL ** 2)
        parts["motion"] = (a_m - jnp.asarray(VINK22_RATE)) / sig
    elif use_pm and "pm" in obs:
        r_pm, _ = pm_residuals(model, obs, pm_scale_sigma)
        parts["motion"] = r_pm
    else:
        dev = (d - mean_k[None, :]) / jnp.asarray(obs["sigma"])
        parts["motion"] = dev[m & use[None, :]]
    if use_rate and "em_over_d2" in model:
        fading = obs.get("rate_mode", "absolute") == "fading"
        logr, rr = {}, []
        for ep, (rate, kappa) in obs["rates"].items():
            if ep in obs["epochs"]:
                e = obs["epochs"].index(ep)
                logr[ep] = jnp.log(kappa * model["em_over_d2"][e] / rate)
                rr.append(jnp.atleast_1d(
                    logr[ep] / (RATE_SIGMA_LN_LOOSE if fading else RATE_SIGMA_LN)))
        if fading and len(logr) == 2:
            a, b = logr.values()
            rr.append(jnp.atleast_1d((b - a) / FADING_SIGMA_LN))
        if rr:
            parts["rate"] = jnp.concatenate(rr)
    if use_doppler and "vlos_kms" in model and "v_kms" in DOPPLER:
        e = int(np.argmin(np.abs(np.asarray(obs["years"]) - float(DOPPLER["year"]))))
        p = dict(zip(PARAM_NAMES, theta))
        n_dop = len(DOPPLER["v_kms"])
        width = 360.0 / n_dop
        centres = (np.arange(n_dop) + 0.5) * width
        # the roll psi rotates the model on the sky; sectors -> data sectors
        # model value k sits at bin k's centre; querying at (k * width - psi)
        # on a grid whose node k is value k returns bin k exactly when psi = 0
        vm = periodic_interp(jnp.asarray(centres - 0.5 * width), model["vlos_kms"][e],
                             jnp.asarray(centres - 0.5 * width) - p["psi"])
        vm = jnp.exp(p["ln_kdop"]) * (vm - vm.mean())
        sig = np.sqrt(DOPPLER["v_err_kms"] ** 2 + DOPPLER_SIGMA_SYS ** 2)
        parts["doppler"] = (vm - jnp.asarray(DOPPLER["v_kms"])) / jnp.asarray(sig)
    if rs_term:
        parts["rs"] = jnp.atleast_1d((jnp.mean(rs_ratio(model)) - RS_RATIO_PRIOR[0]) / RS_RATIO_PRIOR[1])
    parts["prior"] = jnp.array([(theta[i] - prior[k][0]) / prior[k][1] for i, k in enumerate(PARAM_NAMES)])
    parts.update(physical_prior_residuals(model, conv_wall=conv_wall, wind_prior=wind_prior))
    return parts


def residuals(model, obs, theta, **kw):
    """``residual_parts`` concatenated (the vector Levenberg-Marquardt sees)."""
    return jnp.concatenate(list(residual_parts(model, obs, theta, **kw).values()))


def jacobian_fd(fun, theta, steps):
    """Central finite differences, one pair of forward passes per parameter.

    The fallback while the forward-mode tangent through the FD solver is NaN
    (see ``--check-grad``): at 128^3 a pass is ~40 s, so 7 parameters cost ~10
    min per Gauss-Newton step.
    """
    f = jax.jit(fun)
    val = f(theta)
    cols = []
    for i, h in enumerate(steps):
        e = jnp.zeros_like(theta).at[i].set(h)
        cols.append((f(theta + e) - f(theta - e)) / (2 * h))
    return val, jnp.stack(cols, axis=-1)


FD_STEPS = {"ln_sv": 0.02, "ln_fw": 0.05, "ln_fsh": 0.05, "d_rsh": 0.03,
            "t_expl": 2.0, "ln_D": 0.02, "psi": 2.0, "ds_w": 0.1,
            "dip_x": 0.05, "dip_y": 0.05, "dip_z": 0.05, "ln_kdop": 0.1,
            "rot_x": 3.0, "rot_y": 3.0, "rot_z": 3.0,
            "wa_x": 0.05, "wa_y": 0.05, "wa_z": 0.05}


def jacobian_jvp(fun, theta):
    cols = []
    for i in range(theta.shape[0]):
        t = jnp.zeros_like(theta).at[i].set(1.0)
        val, d = jax.jvp(fun, (theta,), (t,))
        cols.append(d)
    return val, jnp.stack(cols, axis=-1)
# =============================================================================
# ============ ↑ The forward model ↑ ==========================================
# =============================================================================


def summarize(model, obs, theta, label, sigma_model_arcsec=5.0, parts=None, show_doppler=True,
              show_budget=True, **res_kw):
    """Print the fit's terms and diagnostics. ``parts``: precomputed
    ``residual_parts`` (else computed here with ``res_kw``)."""
    if parts is None:
        parts = residual_parts(model, obs, theta, sigma_model_arcsec=sigma_model_arcsec, **res_kw)
    parts = {k: np.asarray(v) for k, v in parts.items()}
    rm = np.asarray(model["r_fs_arcsec"]); ro = obs["r"]; msk = obs["mask"]
    diff = np.where(msk, rm - ro, np.nan)
    yrs = obs["years"]
    mean_m = np.nanmean(np.where(msk, rm, np.nan), axis=1)
    mean_o = np.nanmean(np.where(msk, ro, np.nan), axis=1)
    sl_m = np.polyfit(yrs, mean_m, 1)[0]; sl_o = np.polyfit(yrs, mean_o, 1)[0]
    print(f"[{label}] theta = " + ", ".join(f"{k}={float(v):.4g}" for k, v in zip(PARAM_NAMES, theta)))
    print(f"[{label}] chi2 " + ", ".join(f"{k} {np.sum(v ** 2):.2f} / {v.size}" for k, v in parts.items())
          + f"; rms outline residual {np.sqrt(np.nanmean(diff ** 2)):.2f}\"")
    m1m, pam = cone_m1(obs["angles"], rm[0]); m1o, pao = cone_m1(obs["angles"], ro[0])
    print(f"[{label}] <r_FS> 2000: model {mean_m[0]:.1f}\" data {mean_o[0]:.1f}\"; "
          f"2022: model {mean_m[-1]:.1f}\" data {mean_o[-1]:.1f}\"; expansion model "
          f"{sl_m:.3f}\"/yr data {sl_o:.3f}\"/yr; m=1 2000 model {m1m:.1f}\" at theta {pam:.0f} "
          f"data {m1o:.1f}\" at theta {pao:.0f} (about RA0/DEC0; theta = PA + 90)")
    ok = msk.sum(0) >= 8
    sl_cm = [np.polyfit(yrs[msk[:, k]], rm[msk[:, k], k], 1)[0] for k in np.nonzero(ok)[0]]
    sl_co = [np.polyfit(yrs[msk[:, k]], ro[msk[:, k], k], 1)[0] for k in np.nonzero(ok)[0]]
    print(f"[{label}] per-cone proper motion median: model {np.median(sl_cm):.3f}\"/yr, "
          f"data {np.median(sl_co):.3f}\"/yr; corr(model, data) over cones "
          f"{np.corrcoef(sl_cm, sl_co)[0, 1]:.2f}")
    if "em_over_d2" in model:
        for ep, (rate, kappa) in obs["rates"].items():
            if ep in obs["epochs"]:
                e = obs["epochs"].index(ep)
                print(f"[{label}] X-ray rate {ep}: proxy {kappa * float(model['em_over_d2'][e]):.1f} "
                      f"vs Chandra {rate:.1f} counts/s")
    if "pm" in obs:
        yc = yrs - yrs.mean()
        pmm = np.sum(yc[:, None] * (rm - rm.mean(0)), axis=0) / np.sum(yc ** 2)
        ok = np.isfinite(obs["pm"])
        _, eps = pm_residuals(model, obs)
        print(f"[{label}] registration PM [{obs.get('pm_source', '?')}] ({ok.sum()} cones): model "
              f"mean {pmm[ok].mean():.3f} / median {np.median(pmm[ok]):.3f}, data mean "
              f"{obs['pm'][ok].mean():.3f} / median {np.median(obs['pm'][ok]):.3f}\"/yr; corr over "
              f"cones {np.corrcoef(pmm[ok], obs['pm'][ok])[0, 1]:.2f}; profiled PM scale "
              f"1 + eps = {1 + float(eps):.3f}")
    if "r_fs_coe_arcsec" in model:
        a_m = np.asarray(vink22_model_rates(model["r_fs_coe_arcsec"], yrs))
        print(f"[{label}] Vink+22 expansion rate about the CoE (18 sectors): model mean "
              f"{a_m.mean():.3f} %/yr, data {VINK22_RATE.mean():.3f}; corr "
              f"{np.corrcoef(a_m, VINK22_RATE)[0, 1]:+.2f}")
    if show_doppler and "vlos_kms" in model and "v_kms" in DOPPLER:
        e = int(np.argmin(np.abs(np.asarray(obs["years"]) - float(DOPPLER["year"]))))
        vm = np.asarray(model["vlos_kms"][e]); vm = vm - vm.mean()
        vd = np.asarray(DOPPLER["v_kms"])
        print(f"[{label}] Doppler 2004 (24 sectors, model's own frame): model rms {vm.std():.0f} "
              f"km/s, data rms {vd.std():.0f}; corr {np.corrcoef(vm, vd)[0, 1]:+.2f}")
    d_kpc = float(np.exp(dict(zip(PARAM_NAMES, theta))["ln_D"]))
    to_arc = ARCSEC_PER_RAD / (d_kpc * 1e3)
    rfs = np.asarray(model["r_fs_pc"]).mean(1); rrs = np.asarray(model["r_rs_pc"]).mean(1)
    line = "; ".join(f"{e} {rrs[i] * to_arc:.1f}\" ({rrs[i] / rfs[i]:.3f})"
                     for i, e in enumerate(obs["epochs"]))
    print(f"[{label}] <r_FS> (model frame) 2000 {rfs[0]:.3f} pc = {rfs[0] * to_arc:.1f}\", 2022 "
          f"{rfs[-1]:.3f} pc; <r_RS> per epoch (r_RS / r_FS): {line}  [target 0.66 +- 0.05; "
          f"Gotthelf+01 95.8 +- 9.7\"]")
    if "r_rs_legacy_pc" in model:
        leg = np.asarray(model["r_rs_legacy_pc"]).mean(1)
        print(f"[{label}] legacy (artefact) r_RS 2000 {leg[0] * to_arc:.1f}\" ({leg[0] / rfs[0]:.3f})")
    if "t_conv" in model:
        print(f"[{label}] ballistic convergence date {float(model['t_conv']):.1f} (>= {T_CONV_MIN}); "
              f"pre-shock wind n_H({WIND_NH_PRIOR[0]:.0f} pc) {float(model['n_h_wind']):.3f} cm^-3 "
              f"(Lee+14 {WIND_NH_PRIOR[1]} +- {WIND_NH_PRIOR[2]})")
    if show_budget and "ic_M_ej" in model:
        print(f"[{label}] IC budget: " + format_budget({k[3:]: model[k] for k in model
                                                       if k.startswith("ic_")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ic", required=True)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--check-grad", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fd-jacobian", action="store_true",
                    help="finite-difference Jacobian instead of forward-mode JVPs")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--theta", type=float, nargs="+", default=None,
                    help="parameter vector; shorter vectors (from before a parameter "
                         "was added) are padded with the prior means")
    ap.add_argument("--pa-flip", type=float, default=1.0)
    ap.add_argument("--pa-offset", type=float, default=0.0)
    ap.add_argument("--sigma-model", type=float, default=5.0, help="per-cone shape error of the model (arcsec)")
    ap.add_argument("--free", nargs="*", default=list(PARAM_NAMES), choices=PARAM_NAMES)
    ap.add_argument("--out", default=None, help="json with the fit trajectory")
    ap.add_argument("--ad-llf-cold", type=float, default=0.0, metavar="FACTOR",
                    help="tangent-only LLF linearisation on faces colder than FACTOR x "
                         "the 1e4 K floor (SimulationConfig.ad_tangent_llf_cold_factor)")
    ap.add_argument("--tangent-sigma", type=float, default=0.0, metavar="CELLS",
                    help="conservative low-pass of the forward-mode tangent (cells); "
                         "0 = off. Primal unchanged.")
    ap.add_argument("--filter-every", type=float, default=2.0, metavar="YR",
                    help="apply the tangent filter at most this many years apart")
    ap.add_argument("--rate-mode", choices=("absolute", "fading"), default="absolute",
                    help="'fading': solar-CSM calibration, loose absolute rate, and the "
                         "2022/2000 rate ratio as the constraint")
    ap.add_argument("--save-model", default=None, metavar="NPZ",
                    help="save the model observables at --theta (after the forward pass)")
    ap.add_argument("--write-ic", default=None, metavar="NPZ",
                    help="apply --theta to --ic and save it (CPU, no fit)")
    ap.add_argument("--doppler", action="store_true",
                    help="add the 2004 Si Doppler pattern to the likelihood")
    ap.add_argument("--sim", type=float, nargs=3, default=None, metavar=("LN_L", "LN_T", "LN_M"),
                    help="exact similarity map (casa_rescale), FIXED: r x L, t x T, masses x M, "
                         "IC age x T; freeze ln_sv at 0 with it")
    # ---- the 2026-09-25 audit fixes (defaults = corrected; --legacy = before) ----
    ap.add_argument("--legacy", action="store_true",
                    help="every fix below at its pre-2026-09-25 value (the old likelihood)")
    ap.add_argument("--centre", choices=("coe", "grid"), default=None,
                    help="model outline about RA0/DEC0 with the explosion at the expansion "
                         "centre (coe, default) or about the model centre (grid, legacy)")
    ap.add_argument("--rs-estimator", choices=RS_ESTIMATORS, default=None)
    ap.add_argument("--rs-term", action="store_true", help="r_RS / r_FS = 0.66 +- 0.05 in the likelihood")
    ap.add_argument("--wind-dipole", choices=("exp", "clip"), default=None)
    ap.add_argument("--priors", choices=("new", "legacy"), default=None)
    ap.add_argument("--no-conv-wall", action="store_true")
    ap.add_argument("--no-wind-prior", action="store_true")
    ap.add_argument("--pm-files", choices=tuple(PM_FILES), default=None)
    ap.add_argument("--pm-mask", default=None, help="none | sw | jet | sw+jet (default)")
    ap.add_argument("--pm-scale-sigma", type=float, default=None, help="0 = no PM-scale nuisance")
    ap.add_argument("--pm-target", choices=("registration", "vink22"), default="registration")
    args = ap.parse_args()
    L = args.legacy
    centre = args.centre or ("grid" if L else "coe")
    rs_est = args.rs_estimator or ("legacy" if L else "unshocked")
    wind_dipole = args.wind_dipole or ("clip" if L else "exp")
    priors = args.priors or ("legacy" if L else "new")
    pm_files = args.pm_files or ("legacy" if L else "ccopm")
    pm_mask = args.pm_mask or ("none" if L else "sw+jet")
    pm_scale = args.pm_scale_sigma if args.pm_scale_sigma is not None else (0.0 if L else PM_SCALE_SIGMA)
    res_kw = dict(prior=PRIOR_LEGACY if priors == "legacy" else PRIOR,
                  pm_scale_sigma=pm_scale if pm_scale > 0 else None, pm_target=args.pm_target,
                  rs_term=args.rs_term, conv_wall=not (L or args.no_conv_wall),
                  wind_prior=not (L or args.no_wind_prior))
    print(f"[diff] options: centre {centre}, r_RS {rs_est}, wind dipole {wind_dipole}, priors "
          f"{priors}, PM {pm_files} mask {pm_mask} scale {pm_scale}, target {args.pm_target}, "
          f"rs-term {args.rs_term}, wall {res_kw['conv_wall']}, wind prior {res_kw['wind_prior']}")
    if args.theta is not None and len(args.theta) < len(PARAM_NAMES):
        args.theta = list(args.theta) + [PRIOR[k][0] for k in PARAM_NAMES[len(args.theta):]]
    if args.write_ic:
        write_transformed_ic(args.ic, args.theta if args.theta is not None else THETA0,
                             args.write_ic, wind_dipole=wind_dipole, sim=args.sim)
        return

    obs = load_observations(pm_files=pm_files, pm_mask=pm_mask)
    obs["rate_mode"] = args.rate_mode
    obs["use_doppler"] = args.doppler
    if args.rate_mode == "fading":
        obs["rates"] = OBSERVED_RATES_SOLAR
    print(f"[diff] {len(obs['epochs'])} epochs {obs['epochs'][0]}-{obs['epochs'][-1]}, "
          f"{int(obs['mask'].sum())} cone radii; median per-cone sigma "
          f"{np.median(obs['sigma']):.2f}\"")
    forward, aux = make_forward(args.ic, obs, pa_flip=args.pa_flip, pa_offset=args.pa_offset,
                                tangent_sigma=args.tangent_sigma,
                                filter_every_yr=args.filter_every,
                                ad_llf_cold=args.ad_llf_cold, centre=centre,
                                rs_estimator=rs_est, wind_dipole=wind_dipole, sim=args.sim)
    dtype = jnp.float64 if args.x64 else jnp.float32
    theta = jnp.asarray(args.theta if args.theta is not None else THETA0, dtype=dtype)
    free = np.array([k in args.free for k in PARAM_NAMES])

    def resid_fun(th):
        return residuals(forward(th), obs, th, sigma_model_arcsec=args.sigma_model, **res_kw)

    t0 = time.time()
    model = jax.jit(forward)(theta)
    jax.block_until_ready(model)
    print(f"[diff] forward pass {time.time() - t0:.1f} s")
    summarize(model, obs, theta, "start", args.sigma_model, **res_kw)
    if args.save_model:
        np.savez(args.save_model, theta=np.asarray(theta), names=np.array(PARAM_NAMES),
                 years=obs["years"], angles=obs["angles"], r_obs=obs["r"], mask=obs["mask"],
                 pm_obs=obs.get("pm", np.full(36, np.nan)),
                 **{k: np.asarray(v) for k, v in model.items()})
        print(f"[diff] wrote {args.save_model}")

    if args.check_grad:
        f = jax.jit(lambda th: jnp.sum(resid_fun(th) ** 2))
        # FD noise floor: is the forward pass reproducible? GPU scatter-adds are
        # not bitwise deterministic, and 200 yr of chaotic evolution amplifies
        # that into chi2 differences an FD step then divides by 2h
        f_a, f_b = float(f(theta)), float(f(theta))
        print(f"[diff] repeat forward: chi2 {f_a:.6f} vs {f_b:.6f} (diff {f_a - f_b:+.3e}); "
              f"FD noise ~ |diff| / 2h = {abs(f_a - f_b) / 0.04:.3e} at h = 0.02")
        t0 = time.time()
        _, J = jax.jit(lambda th: jacobian_jvp(f, th))(theta)
        print(f"[diff] JVP gradient ({time.time() - t0:.1f} s): {np.asarray(J)}")
        for i, k in enumerate(PARAM_NAMES):
            h = FD_STEPS[k]
            e = jnp.zeros_like(theta).at[i].set(h)
            fd = (f(theta + e) - f(theta - e)) / (2 * h)
            print(f"   d chi2 / d {k:7s}: JVP {float(J[i]):+.4e}  FD {float(fd):+.4e}")

    if args.fit:
        if args.fd_jacobian:
            steps = [FD_STEPS[k] for k in PARAM_NAMES]
            jac = lambda th: jacobian_fd(resid_fun, th, steps)      # noqa: E731
        else:
            jac = jax.jit(lambda th: jacobian_jvp(resid_fun, th))
        lam = args.damping
        hist = []
        for it in range(args.steps):
            t0 = time.time()
            rvec, J = jac(theta)
            rvec, J = np.asarray(rvec, np.float64), np.asarray(J, np.float64)
            J[:, ~free] = 0.0
            # a column can be non-finite when a perturbed forward run crashes
            # (e.g. at a large velocity dipole): freeze that parameter for this
            # step instead of letting one NaN poison the whole step
            bad_col = ~np.all(np.isfinite(J), axis=0)
            if bad_col.any():
                print(f"[fit {it}] non-finite Jacobian column(s) "
                      f"{[PARAM_NAMES[i] for i in np.nonzero(bad_col)[0]]}: frozen this step")
                J[:, bad_col] = 0.0
            chi = float(rvec @ rvec)
            A = J.T @ J; g = J.T @ rvec
            fz = ~free | bad_col
            A[fz, fz] = 1.0
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            trial = theta + jnp.asarray(step, dtype=dtype)
            chi_trial = float(jnp.sum(jax.jit(resid_fun)(trial) ** 2))
            print(f"[fit {it}] chi2 {chi:.1f} -> {chi_trial:.1f} (lambda {lam:.2g}, "
                  f"{time.time() - t0:.0f} s); step " +
                  ", ".join(f"{k} {s:+.3g}" for k, s in zip(PARAM_NAMES, step)))
            hist.append(dict(it=it, chi2=chi, theta=[float(x) for x in theta],
                             chi2_trial=chi_trial, step=step.tolist()))
            if chi_trial < chi:
                theta = trial; lam = max(lam / 3, 1e-3)
            else:
                lam *= 4
        # Laplace approximation around the result
        rvec, J = jac(theta)
        J = np.asarray(J, np.float64); J[:, ~free] = 0.0
        A = J.T @ J; A[~free, ~free] = 1.0
        cov = np.linalg.inv(A)
        model = jax.jit(forward)(theta)
        summarize(model, obs, theta, "fit", args.sigma_model, **res_kw)
        print("[fit] Laplace 1-sigma: " + ", ".join(
            f"{k} {float(theta[i]):.4g} +- {np.sqrt(cov[i, i]):.3g}"
            for i, k in enumerate(PARAM_NAMES) if free[i]))
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        print("[fit] correlation matrix:\n" + np.array2string(corr, precision=2, suppress_small=True))
        if args.out:
            Path(args.out).write_text(json.dumps(dict(
                history=hist, theta=[float(x) for x in theta], cov=cov.tolist(),
                names=PARAM_NAMES, ic=args.ic,
                options=dict(centre=centre, rs_estimator=rs_est, wind_dipole=wind_dipole,
                             priors=priors, pm_files=pm_files, pm_mask=pm_mask,
                             pm_scale=pm_scale, pm_target=args.pm_target),
                model_r_fs_arcsec=np.asarray(model["r_fs_arcsec"]).tolist(),
                model_r_fs_pc=np.asarray(model["r_fs_pc"]).tolist(),
                model_r_rs_pc=np.asarray(model["r_rs_pc"]).tolist()), indent=1))


if __name__ == "__main__":
    main()
