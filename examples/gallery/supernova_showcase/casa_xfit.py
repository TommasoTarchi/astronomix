"""
Differentiable Cas A against the X-ray images of every Chandra epoch 2000-2023.

``casa_pluto_diff`` fits Orlando's 146-yr state to smooth summaries of the
data (forward-shock outline, proper motions, a Doppler proxy, an
emission-measure rate proxy). This goes to the data themselves: the forward
model evolves the full state -- hydro, the ejecta/composition scalars
(``C_ej``, ``C_Fe``, ``C_Si``, ``C_O``, ``C_He``, solar CSM) and the library's
shock history (``shocked_fraction``, ``time_since_shock``, ``density_time``) --
from 146 yr through every usable epoch, and at each epoch observes the state
with the differentiable X-ray model (``casa_jaxobs``: NEI plasma, the epoch's
ACIS response, interstellar absorption, the dust halo) in six bands.

Per epoch the response is the epoch's detector (ACIS-I for the 2022 I3
pointing, else ACIS-S) with the ACIS-S tables interpolated linearly in time
between the tabulated cycles (cy0 = 1999, cy10 = 2009, cy22 = 2021): the
filter contamination grows smoothly, and the nearest-cycle choice
under-absorbs mid-cycle epochs (casa_observe.instrument_for_epoch).

Likelihood (a residual vector for Levenberg-Marquardt):

* band images, block-summed to ``--block`` x 1.97" pixels (default 16 = 31.5"),
  split into a STATIC part (the epoch-mean brightness per block, model error
  ``--sigma-static``) and a TEMPORAL part (each epoch's block brightness
  relative to the block's own mean, ``--sigma-temporal``) -- see
  ``image_residuals`` for why comparing the epochs independently is wrong;
* the forward-shock outline per cone and the registration proper motions
  (``casa_pluto_diff.residual_parts``), with the model radii RE-CENTRED from
  the model's explosion centre (dw, dn) onto RA0/DEC0, where the data are
  measured, and a profiled global PM-scale nuisance;
* the 2004 Si Doppler pattern, the model's MEASURED centroid statistic
  (continuum dilution included), in sky sectors about RA0/DEC0 (traced roll,
  distance and offset);
* optional: r_RS / r_FS = 0.66 +- 0.05 (``--rs-term``);
* Gaussian priors, a one-sided wall on the ballistic convergence date and the
  Lee+14 pre-shock wind density.

Parameters: ``casa_pluto_diff.PARAM_NAMES`` plus the sky offset of the
explosion centre (``dw``, ``dn``, arcsec west / north of RA0/DEC0; prior: the
Thorstensen+01 expansion centre (-13.8, -4.2) +- 1.5"), a global X-ray
amplitude ``ln_A`` (sub-grid clumping, the one emission nuisance), the
absorbing column ``ln_nh`` (traced through the tables' own N_H grid, ln-linear
extrapolation outside it), and the ejecta log-density modes (mass-neutral).

The 2026-09-25 audit fixes are the defaults; ``--legacy`` restores every one of
them to its old value (the likelihood fits A-Q2 used), and each has its own
flag.

Stage-2 observation chain (defaults; ``--legacy`` = ``--obs v1 --responses
tables``): ``--obs v2`` wires the obs_tables v2 tables (``casa_xfit_obs2``:
binned tables on the analysis bins with the NEI T_e-history axis, the v2 N_H
grid 0.5-4e22 mixed ln-linearly in N_H (``--nh-mode geo``), the halo per N_H
node, log-kT, solar CSM for ``*_solarcsm`` ICs, the exact-window Doppler
moments at the Doppler epoch with the v2 data file); ``--responses ciao``
multiplies the predicted images / spectra by the CIAO per-epoch corrections
(``casa_xfit_responses``: exposure-map geometry x folded ARF x RMF ratio);
``--save-state`` writes the evolved state at the first epoch
(``casa_xfit_state``); ``ln_si`` scales the ejecta Si-group yield.

Stage-4 residual physics (2026-09-26, ``STAGE4_OPTIONS``; defaults = new,
``--stage3`` or ``--legacy`` = the refit R / R' behaviour, each with its own
flag): ``--background measured`` (casa_xfit_bkg: per-epoch, per-band particle
background with the out-of-time readout-streak events removed, plus the streak
itself, instead of BKG_RATE); ``--outline-mask inner-arc`` (cones 260, 270, 10
out of the outline) and ``--pm-mask-extra 300``; ``--heldout-sigma train``
(held-out outline with a training-only per-cone sigma and a data-only
detector-jump cut); ``--sync-trend on`` (parameter ``b_sx``, the X-ray
synchrotron trend in %/yr on top of the radio anchor, prior -1.0 +- 0.5);
``--spec-gain profile`` / ``--spec-soft profile`` (per-epoch energy scale,
prior 0.2 %, and soft-bin calibration, prior 5 %, profiled in closed form);
``--spec-broadening thermal`` (line-of-sight velocity to second order plus the
thermal ion spread in the v2 spectra: +4.3 GB of D / D2 tables); ``--kte
fixed`` (kT_e0 = 0.3 keV, ln_kte frozen). ``casa_xfit_compare`` evaluates
every variant on one hydro run.

    ./run.sh casa_xfit.py --ic .../pluto146_n128_solarcsm.npz --theta ... --forward
    ./run.sh casa_xfit.py --ic ... --theta ... --forward --legacy
    ./run.sh casa_xfit.py --ic ... --theta ... --fit --steps 4 --free ...

``--gpus N`` (casa_xfit_shard) splits ONE forward over N GPUs of a node (x-slabs;
the IC fields, geometry and tables as sharded jit arguments); with
``--save-state F --state-only`` it only evolves to the first epoch and writes
the state (the 512^3 background of the sharded 4D-Var):

    pq sub -t a100 -n 8 -- env NCCL_NVLS_ENABLE=0 ./run.sh casa_xfit.py --ic W/ers/shard/pluto146_n512_solarcsm.npz \
        --theta $(cat W/ers/shard/theta_R2.txt) --doppler --spectra --obs v2 --responses ciao \
        --gpus 8 --state-only --save-state W/ers/shard/state2000_R2_n512.npz
"""

# ==== GPU selection ====
import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and __name__ == "__main__":
    from autocvd import autocvd
    # --gpus N: one forward sharded over N GPUs (casa_xfit_shard); --devices N:
    # N independent FD-Jacobian forwards
    _n = [int(sys.argv[sys.argv.index(f) + 1]) for f in ("--gpus", "--devices") if f in sys.argv]
    autocvd(num_gpus=max(_n + [1]))
if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
# ruff: noqa: E402
# =======================

import argparse
import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u
import astropy.constants as const

from astronomix import (SimulationParams, construct_primitive_state, finalize_config,
                        get_helper_data, get_registered_variables, time_integration)
from _common import (GAMMA, MASS_PER_NUCLEUS, POSITIVITY_REDISTRIBUTE, fd_positivity,
                     make_fd_config, snr_code_units)
import _plasma as P
import casa_jaxobs as J
import casa_jet as JT
import casa_pluto_diff as PD
import casa_xfit_obs2 as O2
import casa_xfit_responses as XR
import casa_xfit_shard as SH
import casa_xfit_state as XS

#: multi-GPU: ``make_forward`` stores ``casa_pluto_diff.doppler_tracer(ic)`` (numpy,
#: so not traceable once the IC fields are jit arguments) under this key, and
#: ``transform_fields`` picks it up through the wrapper below (values identical;
#: without the key the original function runs, i.e. every 1-device path).
_C_DOP_KEY = "_casa_xfit_C_dop"


def _doppler_tracer_lifted(ic, _orig=PD.doppler_tracer):
    return ic[_C_DOP_KEY] if _C_DOP_KEY in ic else _orig(ic)


PD.doppler_tracer = _doppler_tracer_lifted

# =============================================================================
# ============ ↓ Parameters and data ↓ ========================================
# =============================================================================
EXTRA_PRIOR = {"dw": (PD.COE_ARCSEC[0], PD.COE_SIGMA_ARCSEC),
               "dn": (PD.COE_ARCSEC[1], PD.COE_SIGMA_ARCSEC),
               "ln_A": (0.0, 0.5),
               "ln_nh": (np.log(1.2), 0.2),
               # synchrotron efficiency over the radio-anchored prediction
               # (casa_observe --sync-norm; ~5 on the fit-I state, CALIBRATION
               # Result 26 found 10-25 with the observed filament width)
               "ln_sync": (np.log(5.0), 1.5),
               # absorbing-column gradient across the remnant, d ln N_H per
               # 100" west / north (N_H rises from ~1.2 to ~2e22 toward the W/SW)
               "g_nh_w": (0.0, 0.3), "g_nh_n": (0.0, 0.3),
               # emission physics the hydro does not set (observation model only):
               # post-shock kT_e (Ghavamian+07: 0.3 keV, extrapolated to metal
               # ejecta), a multiplier on the Coulomb equilibration rate, the
               # Fe-group yield, and the sub-grid dense-phase mass fraction (logit)
               "ln_kte": (np.log(0.3), 1.0), "ln_teq": (0.0, 1.0), "ln_fe": (0.0, 0.5),
               "lg_fmass": (float(np.log(0.34 / 0.66)), 1.0)}
#: field-level modes: ejecta log-density perturbations at 146 yr on real
#: spherical harmonics l = 1..3 (Cartesian forms of n_hat, O(1) amplitude),
#: delta ln rho = C_ej (sum a_lm Y_lm - delta), pressure unchanged; delta
#: renormalises the ejecta mass to the unperturbed one (``--ylm-mass``)
YLM = {
    "ej11x": lambda x, y, z: x, "ej11y": lambda x, y, z: y, "ej10": lambda x, y, z: z,
    "ej22xy": lambda x, y, z: 3 * x * y, "ej21yz": lambda x, y, z: 3 * y * z,
    "ej20": lambda x, y, z: 1.5 * z * z - 0.5, "ej21xz": lambda x, y, z: 3 * x * z,
    "ej22c": lambda x, y, z: 1.5 * (x * x - y * y),
    "ej33s": lambda x, y, z: y * (3 * x * x - y * y), "ej32xyz": lambda x, y, z: 5 * x * y * z,
    "ej31s": lambda x, y, z: y * (5 * z * z - 1), "ej30": lambda x, y, z: 0.5 * z * (5 * z * z - 3),
    "ej31c": lambda x, y, z: x * (5 * z * z - 1), "ej32c": lambda x, y, z: 2.5 * z * (x * x - y * y),
    "ej33c": lambda x, y, z: x * (x * x - 3 * y * y),
}
EXTRA_PRIOR.update({k: (0.0, 0.3) for k in YLM})
#: Si-group (Si, S, Ar, Ca tracer) yield multiplier on the EJECTA part of the
#: tracer (the solar CSM part is left alone): with the v2 tables the Si / S
#: lines are still 1.17 / 1.09 of the model relative to the continuum at fixed
#: parameters (obs_tables report, item 9) -- a nucleosynthesis yield the hydro
#: does not care about, as ``ln_fe``. Appended last so older thetas stay valid.
EXTRA_PRIOR["ln_si"] = (0.0, 0.5)
#: X-ray synchrotron secular trend ON TOP of the radio anchor (``--sync-trend``,
#: stage 4): the synchrotron columns are multiplied by exp(0.01 b_sx (year -
#: SYNC_T_REF)), b_sx in %/yr. The anchored model fades -1.0 to -1.1 %/yr at
#: 4.2-6 keV; Patnaude+11 measure -1.5 +- 0.17. The stage-3 scan gave Delta chi2
#: -14 (R) to -19 (V3) at -1 %/yr, and a loss-limited cutoff following the
#: model's own shock deceleration supplies <= -0.2 %/yr (stage3/physics
#: REPORT 1b). Appended last so older thetas stay valid (padded with -1.0).
EXTRA_PRIOR["b_sx"] = (-1.0, 0.5)
#: the EXACT similarity map of the adiabatic problem (``casa_pluto_diff.SIM_NAMES``,
#: ``casa_rescale``; 2026-09-27): ln_L, ln_T, ln_M are the TOTAL log scales of
#: the state relative to Orlando's as-delivered 146-yr one (r x L, t x T, masses
#: x M; E x M (L / T)^2, IC age 145.5 T). An IC that is already scaled (its
#: ``similarity_{L,T,M}`` stamp: ``casa_pluto convert --sim``) contributes its own
#: scales, and the traced map applies only the remainder (``ic_similarity_ref``).
#: The path is switched on by ``opts.similarity`` ("on": a scale parameter is free
#: or differs from the IC's own; main's ``--similarity auto``); off, the forward
#: is bitwise the old one and the three priors are not in the likelihood. With it
#: on, ln_sv is frozen at 0 (it is the member (1, 1/s, 1) without the age
#: relabelling); the ballistic-date wall stays: with the relabelled age it is the
#: Thorstensen+01 bound t_expl >= 1671.3 (``--no-sim-wall`` drops it, as refit R3
#: did; review 2026-09-27). Priors: ``PD.SIM_PRIOR``.
#: Appended last so older thetas stay valid (padded with the IC's own scales).
SIM_NAMES = PD.SIM_NAMES
EXTRA_PRIOR.update(PD.SIM_PRIOR)
#: the Si-rich NE jet + SW counter-jet added to the IC (``casa_jet``; ``--jet on``,
#: 2026-10-01): M_NE (ln Msun), tip speed (1000 km/s), sky PA and inclination
#: (deg), ln half-angle (deg), ln M_SW / M_NE. Off (default) the forward is
#: bitwise the old one, the parameters sit at their prior means, frozen, and
#: their priors are not in the likelihood. Appended last so older thetas stay
#: valid (padded with the prior means).
JET_NAMES = JT.JET_NAMES
EXTRA_PRIOR.update(JT.JET_PRIOR)
#: the pre-2026-09-25 centre prior: 0 +- 10" about RA0/DEC0 (the image centre,
#: 14.4" WNW of the expansion centre)
EXTRA_PRIOR_LEGACY = dict(EXTRA_PRIOR, dw=(0.0, 10.0), dn=(0.0, 10.0))
PARAM_NAMES = PD.PARAM_NAMES + tuple(EXTRA_PRIOR)
PRIOR = {**PD.PRIOR, **EXTRA_PRIOR}
PRIOR_LEGACY = {**PD.PRIOR_LEGACY, **EXTRA_PRIOR_LEGACY}
FD_STEPS = {**PD.FD_STEPS, "dw": 1.0, "dn": 1.0, "ln_A": 0.05, "ln_nh": 0.05, "ln_sync": 0.1,
            "g_nh_w": 0.05, "g_nh_n": 0.05, "ln_kte": 0.1, "ln_teq": 0.1, "ln_fe": 0.05,
            "lg_fmass": 0.1, **{k: 0.1 for k in YLM}, "ln_si": 0.05, "b_sx": 0.1, **PD.SIM_FD_STEPS,
            **JT.JET_FD_STEPS}
THETA0 = np.array([PRIOR[k][0] for k in PARAM_NAMES])

SCALAR_NAMES = ("C_ej",) + tuple(f"C_{s}" for s in P.TRACKED_SPECIES)
HISTORY = ("shocked_fraction", "time_since_shock", "density_time")
D_REF_KPC = 3.0                  # the observation model's reference distance
#: the pre-2026-09-25 hard-coded column grid (1e22); the grid is now read from
#: the tables themselves (``table_nh_grid``)
NH_GRID_LEGACY = np.array([0.8, 1.0, 1.2, 1.5, 2.0])
NH_GRID = NH_GRID_LEGACY
S_CYCLE_YEAR = {0: 1999.0, 10: 2009.0, 22: 2021.0}
BKG_RATE = 1e-6                  # counts/s per 1.97" pixel per band (floor)
#: per-band prior width of the per-epoch calibration amplitude (ln): the ACIS
#: contamination makes the soft band's time dependence uncertain at ~10 %
CALIB_SIGMA = (0.10, 0.03, 0.03, 0.03, 0.03, 0.03)
#: spectral bins with upper edge <= this are dropped at ACIS-I epochs (2022),
#: as the image term drops the ACIS-I 0.5-1.5 keV band (``bw``)
ACISI_SOFT_KEV = 1.5

#: the audit fixes: option -> (new default, legacy value)
FIX_OPTIONS = {
    "rs_estimator": ("unshocked", "legacy"),
    "centre": ("sky", "grid"),
    "doppler_frame": ("sky", "sim"),
    "priors": ("new", "legacy"),
    "conv_wall": (True, False),
    "wind_prior": (True, False),
    "spec_acisi_soft": ("exclude", "include"),
    "wind_dipole": ("exp", "clip"),
    "ylm_mass": ("neutral", "free"),
    "kdop": ("fixed0", "free"),
    "pm_files": ("ccopm", "legacy"),
    "pm_scale_sigma": (PD.PM_SCALE_SIGMA, 0.0),
    "pm_mask": ("sw+jet", "none"),
    "nh_mode": ("extrap", "clamp"),
}
#: the 2026-09-25 (stage 2) observation-model chain: option -> (new, legacy)
OBS_OPTIONS = {
    "obs": ("v2", "v1"),                 # obs_tables v2 (INTEGRATION.md) vs the v1 channel tables
    "responses": ("ciao", "tables"),     # CIAO per-epoch corrections (casa_xfit_responses) vs none
    "csm_solar": ("auto", "off"),        # auto: on for v2 with a *_solarcsm IC
    "kt_interp": ("auto", "linear"),     # auto: log for v2, linear for v1
}
#: the 2026-09-26 (stage 4) residual-physics changes (stage3/physics REPORT,
#: "Model changes for the next fit round"): option -> (new default, old). The
#: old value is the stage-3 behaviour (refit R / R'); ``--legacy`` and
#: ``--stage3`` set every one of them to it.
STAGE4_OPTIONS = {
    # 1. measured per-epoch, per-band instrumental background in images and
    #    spectra (casa_xfit_bkg: particle background with the out-of-time
    #    readout-streak events removed, plus the streak itself) instead of BKG_RATE
    "background": ("measured", "rate"),
    # 2. outline cones where casa_real_outline locks onto an inner arc (the
    #    hard-band filament is 10-25" further out, where the model is) dropped
    #    from the outline; cone 300 (PA 210, reverse-shock-contaminated
    #    registration like 290 / 320) dropped from the proper motions; held-out
    #    outline scored with a per-cone sigma from the TRAINING epochs only
    "outline_mask": ("inner-arc", "none"),
    "pm_mask_extra": ("300", "none"),
    "heldout_sigma": ("train", "data"),      # train | train-nocut | data
    # 3. X-ray synchrotron secular trend b_sx on top of the radio anchor
    "sync_trend": ("on", "off"),
    # 4. per-epoch spectral nuisances, profiled analytically: energy scale
    #    (prior GAIN_SIGMA, linearised through the line-shift derivative) and a
    #    soft-bin (upper edge <= SOFT_KEV) calibration factor (prior SOFT_SIGMA)
    "spec_gain": ("profile", "off"),
    "spec_soft": ("profile", "off"),
    # 5. line broadening in the v2 spectra: every cell's line-of-sight velocity
    #    to second order (C + beta D + beta^2 / 2 D2) plus the ions' thermal
    #    spread ("thermal"), without the thermal spread ("vlos"), or none ("off")
    "spec_broadening": ("thermal", "off"),
    # 6. post-shock kT_e fixed at KTE_FIXED_KEV (no leverage on any bin: stage-3
    #    kT_e0 variants x0.5-2 move no spectral bin by > 2 %)
    "kte": ("fixed", "free"),
}
#: cone angles (theta = PA + 90) of ``--outline-mask inner-arc``
OUTLINE_MASK_CONES = {"none": (), "inner-arc": (260.0, 270.0, 10.0),
                      # + the cones a model jet's bow shock pushes out (casa_jet; NE PA 50-90, SW PA 250-270):
                      # there the model's 3D shocked-gas edge follows the jet, the data's broadband-decline
                      # edge the rim beside it -- different observables (the jet itself: --jet-img on)
                      "inner-arc+jet": (260.0, 270.0, 10.0) + JT.JET_OUTLINE_CONES}
#: --heldout-sigma train: held-out cone-epochs whose DATA radius leaves the
#: cone's training-epoch linear trend by more than this (arcsec) are detector
#: jumps (casa_real_outline switching features: 20-47" in 1-4 yr, i.e. >= 10^5
#: km/s) and are dropped from the held-out outline for every model alike
HELDOUT_JUMP_ARCSEC = 5.0
KTE_FIXED_KEV = 0.3
SYNC_T_REF = 2010.0
GAIN_SIGMA = 0.002
SOFT_SIGMA = 0.05
SOFT_KEV = 1.1
#: uniform line-of-sight velocity (km/s) of the one-sided difference that gives
#: d(spectrum)/d(energy scale): beta = 0.002 = the gain prior width
GAIN_DV_KMS = 599.584916


def default_options(legacy=False, stage3=False, **over):
    """The fix options as a namespace (``FIX_OPTIONS``, ``OBS_OPTIONS``,
    ``STAGE4_OPTIONS``), for callers that do not go through ``main``'s argparse.
    ``stage3``: the stage-4 options at their old (refit R / R') values, the
    rest at the defaults."""
    o = {k: v[1] if legacy else v[0] for k, v in {**FIX_OPTIONS, **OBS_OPTIONS}.items()}
    o.update({k: v[1] if (legacy or stage3) else v[0] for k, v in STAGE4_OPTIONS.items()})
    o.update(rs_term=False, pm_target="registration", recentre_iter=2, bkg_file=None, bkg_model=None,
             similarity="off", jet="off", jet_img="off")
    if not legacy:
        o["nh_mode"] = "auto"            # geo for v2, extrap for v1 (``resolve_obs_options``)
    o.update(over)
    return SimpleNamespace(**o)


def ic_similarity_ref(ic):
    """{ln_L, ln_T, ln_M} of the IC ITSELF (dict or npz path): the log of its
    ``similarity_{L,T,M}`` stamp (``casa_pluto convert --sim`` / ``casa_rescale``),
    0 for Orlando's unscaled state. The similarity parameters are total scales;
    the traced map applies ``p - ref`` (so a rescaled IC at its own scales is the
    identity of the map), and thetas without them are padded with ``ref``."""
    if isinstance(ic, (str, Path)):
        with np.load(ic) as d:
            ic = {k: d[k] for k in d.files if k.startswith("similarity_")}
    return {f"ln_{c}": float(np.log(float(ic[f"similarity_{c}"]))) if f"similarity_{c}" in ic else 0.0
            for c in "LTM"}


def stage4(opts, key):
    """A stage-4 option of ``opts``, its OLD value if ``opts`` predates them
    (a namespace built by an older caller)."""
    return getattr(opts, key, STAGE4_OPTIONS[key][1])


def spec_table_kinds(opts):
    """The v2 spectral table kinds the options need: C always, D for the gain
    derivative or the broadening, D2 for the broadening."""
    if getattr(opts, "obs", "v1") != "v2":
        return ("C",)
    if stage4(opts, "spec_broadening") != "off":
        return ("C", "D", "D2")
    if stage4(opts, "spec_gain") != "off":
        return ("C", "D")
    return ("C",)


def resolve_auto(opts, ic_path=None):
    """Concrete values for the ``"auto"`` observation options (in place):
    ``nh_mode`` geo for v2 / extrap for v1, ``kt_interp`` log for v2 / linear
    for v1, ``csm_solar`` on for v2 with a ``*_solarcsm`` IC (INTEGRATION.md
    2b: correct only for those)."""
    if getattr(opts, "obs", "v1") not in ("v1", "v2"):
        raise ValueError(f"obs {opts.obs!r}")
    if opts.nh_mode == "auto":
        opts.nh_mode = "geo" if opts.obs == "v2" else "extrap"
    if opts.kt_interp == "auto":
        opts.kt_interp = "log" if opts.obs == "v2" else "linear"
    if opts.csm_solar == "auto":
        opts.csm_solar = "on" if (opts.obs == "v2" and ic_solar_csm(ic_path)) else "off"
    if opts.obs == "v1":
        # the gain derivative and the broadening need the v2 spectral D / D2 tables
        for k in ("spec_gain", "spec_broadening"):
            if getattr(opts, k, "off") != "off":
                print(f"[xfit] --obs v1: {k} {getattr(opts, k)} -> off (needs the v2 tables)", flush=True)
                setattr(opts, k, "off")
    return opts


def ic_solar_csm(ic_path):
    """Whether the IC's circumstellar composition is solar: a ``*_solarcsm``
    file (``casa_pluto recompose``), or any npz carrying recompose's
    ``csm_composition`` stamp at the solar values -- e.g. a ``--save-state``
    file, which copies the stamp but not the ``_solarcsm`` name (review
    2026-09-25: the name test alone switched the solar CSM OFF for the 2000
    state the 4D-Var starts from)."""
    if not ic_path:
        return False
    if "solarcsm" in Path(ic_path).name:
        return True
    try:
        with np.load(ic_path) as d:
            if "csm_composition" not in d.files:
                return False
            comp = ast.literal_eval(str(d["csm_composition"]))
    except (OSError, ValueError, SyntaxError):
        return False
    sun = J.solar_csm_tracers()
    return all(k in comp and abs(float(comp[k]) - float(v)) <= 1e-4 * abs(float(v)) for k, v in sun.items())


def table_nh_grid(table_dir=J.TABLE_DIR, instruments=None):
    """The N_H grid (1e22) the emissivity tables were computed on, read from
    the files; every instrument must share it."""
    instruments = instruments or INSTRUMENTS
    grids = {i: np.asarray(np.load(Path(table_dir) / f"emissivity_{i}.npz")["nh"], np.float64)
             for i in instruments}
    g0 = next(iter(grids.values()))
    for i, g in grids.items():
        if g.shape != g0.shape or not np.allclose(g, g0):
            raise ValueError(f"N_H grids differ between tables: {i} {g} vs {g0}")
    return g0


def resample_nh(arr, nh_src, nh_dst, axis=0):
    """Resample a table along its N_H axis (host numpy): linear in ln N_H
    inside the source grid (as ``load_tables``), ln(value) linear in N_H
    outside it (as ``nh_mix``)."""
    nh_src = np.asarray(nh_src, np.float64); nh_dst = np.asarray(nh_dst, np.float64)
    if nh_src.shape == nh_dst.shape and np.allclose(nh_src, nh_dst):
        return arr
    a = np.moveaxis(np.asarray(arr, np.float64), axis, 0)
    out = []
    ls = np.log(nh_src)
    for v in nh_dst:
        if v < nh_src[0] or v > nh_src[-1]:
            i, j = (0, 1) if v < nh_src[0] else (-2, -1)
            t = (v - nh_src[i]) / (nh_src[j] - nh_src[i])
            ok = (a[i] > 0) & (a[j] > 0)
            out.append(np.where(ok, np.exp(np.log(np.where(ok, a[i], 1.0)) * (1 - t)
                                           + np.log(np.where(ok, a[j], 1.0)) * t), a[i if t < 0 else j]))
        else:
            k = int(np.clip(np.searchsorted(ls, np.log(v)) - 1, 0, len(ls) - 2))
            w = (np.log(v) - ls[k]) / (ls[k + 1] - ls[k])
            out.append((1 - w) * a[k] + w * a[k + 1])
    return np.moveaxis(np.stack(out), 0, axis).astype(np.asarray(arr).dtype)


def epoch_responses(detnam, year):
    """[(soxs instrument, weight)] for an epoch: ACIS-I cy22, or ACIS-S linear
    in time between the bracketing tabulated cycles."""
    chips = [int(c) for n in detnam for c in str(n).replace("ACIS-", "") if c.isdigit()]
    if chips and max(chips) <= 3:
        return [("chandra_acisi_cy22", 1.0)]
    cy = sorted(S_CYCLE_YEAR, key=S_CYCLE_YEAR.get)
    yrs = np.array([S_CYCLE_YEAR[c] for c in cy])
    y = float(np.clip(year, yrs[0], yrs[-1]))
    k = int(np.clip(np.searchsorted(yrs, y) - 1, 0, len(yrs) - 2))
    w = (y - yrs[k]) / (yrs[k + 1] - yrs[k])
    return [(f"chandra_aciss_cy{cy[k]}", 1.0 - w), (f"chandra_aciss_cy{cy[k + 1]}", w)]


def load_image_data(epochs, years, *, block, r_max=150.0, table_dir=J.TABLE_DIR, nh_grid=None, obs="v1",
                    history=True, spec_kinds=("C",), fine_block=0, jet_bins=None):
    """Per epoch: block-summed counts, the block mask, exposure, pixel mask,
    and the stacked band tables (n_nh, c, t, n, band) of its response, on the
    tables' own N_H grid (``nh_grid``; default: read from ``table_dir``).

    ``obs="v2"``: no per-epoch tables; ``out["v2"]`` holds every v2 table per
    instrument (``casa_xfit_obs2.load_tables``) and ``out["inst_w"]`` (E, 4)
    the epochs' instrument weights, mixed inside the forward model. ``spec_kinds``
    (v2): the spectral table kinds (``spec_table_kinds(opts)``: + D / D2 for the
    stage-4 gain derivative / line broadening).

    ``fine_block`` (pixels, a divisor of ``block``; 0 = off): also the counts
    summed in fine blocks (``counts_fine`` (E, band, nf, nf), ``fmask`` fine
    blocks >= 90 % unmasked, ``fine_xy`` their centres in arcsec W / N of
    RA0/DEC0) for the multi-scale image term (``fine_image_residuals``).

    ``jet_bins`` (``--jet-img on``; ``casa_jet.JET_PROFILE``): the jet term's
    sky-PA bins in an annulus beyond the rim (``casa_jet.jet_bin_index``):
    ``jet_counts`` (E, band, n_bin), ``jet_pm`` (E, npix^2) the usable pixels,
    ``jet_B`` (n_bin, npix^2) the pixel -> bin one-hot (``jet_image_residuals``)."""
    if obs == "v2":
        v2 = O2.load_tables(table_dir, history=history, spec_kinds=spec_kinds)
        nh_grid = v2["nh"]
    else:
        nh_grid = table_nh_grid(table_dir) if nh_grid is None else np.asarray(nh_grid)
    cache, scache = {}, {}

    def inst_tables(inst):
        if inst not in cache:
            cache[inst] = [J.band_tables_of(J.load_tables(inst, nh, table_dir=table_dir)) for nh in nh_grid]
        return cache[inst]

    def sync_tables(inst):
        if inst not in scache:
            lec, S = J.load_sync_tables(inst, table_dir=table_dir)
            nh_s = np.asarray(np.load(Path(table_dir) / f"sync_{inst}.npz")["nh"])
            scache[inst] = (lec, resample_nh(S, nh_s, nh_grid, axis=0))
        return scache[inst]

    out = dict(counts=[], exposure=[], pmask=[], bmask=[], Cb=[], Db=[], Sb=[], resp=[], bw=[])
    for label, year in zip(epochs, years):
        d = np.load(J.DATA_DIR / f"bands_{label}.npz")
        npix = d["counts"].shape[-1]
        pix = float(d["pixel_arcsec"])
        ax = (np.arange(npix) - 0.5 * (npix - 1)) * pix
        NN, WW = np.meshgrid(ax, ax, indexing="ij")
        cw = -(J.CCO_RADEC[0] - J.RA0) * np.cos(np.deg2rad(J.DEC0)) * 3600.0
        cn = (J.CCO_RADEC[1] - J.DEC0) * 3600.0
        pm = ((np.asarray(d["edge"]) <= 0.02) & (np.hypot(WW, NN) < r_max)
              & (np.hypot(WW - cw, NN - cn) >= 6.0))
        if jet_bins is not None:
            jb = JT.jet_bin_index(WW, NN, *jet_bins)
            jpm = (np.asarray(d["edge"]) <= 0.02) & (jb >= 0)
            nbj = int(jb.max()) + 1
            B = (jb.ravel()[None, :] == np.arange(nbj)[:, None]).astype(np.float32)
            out.setdefault("jet_counts", []).append(
                np.asarray(d["counts"], np.float64).reshape(len(J.BANDS), -1) * jpm.ravel() @ B.T)
            out.setdefault("jet_pm", []).append(jpm.ravel().astype(np.float32))
            out["jet_B"] = B
        nb = npix // block
        bs = lambda a: a.reshape(*a.shape[:-2], nb, block, nb, block).sum((-3, -1))  # noqa: E731
        out["counts"].append(bs(np.asarray(d["counts"]) * pm))
        # keep blocks that are >= 90 % unmasked (the model is masked identically)
        out["bmask"].append(bs(pm.astype(float)) >= 0.9 * block ** 2)
        if fine_block:
            if block % fine_block:
                raise ValueError(f"fine_block {fine_block} must divide block {block}")
            nf = npix // fine_block
            fs = lambda a: a.reshape(*a.shape[:-2], nf, fine_block, nf, fine_block).sum((-3, -1))  # noqa: E731
            out.setdefault("counts_fine", []).append(fs(np.asarray(d["counts"]) * pm))
            out.setdefault("fmask", []).append(fs(pm.astype(float)) >= 0.9 * fine_block ** 2)
            if "fine_xy" not in out:
                out["fine_xy"] = np.stack([fs(WW) / fine_block ** 2, fs(NN) / fine_block ** 2])
        out["pmask"].append(pm)
        out["exposure"].append(float(d["exposure"]))
        resp = epoch_responses(d["detnam"], year)
        out["resp"].append(resp)
        # the ACIS-I (FI chip) soft band is 35 % off the S-array epochs' trend
        # at the fiducial model -- a response systematic, not remnant physics:
        # excluded rather than absorbed into N_H
        out["bw"].append(np.array([0.0 if (resp[0][0].startswith("chandra_acisi") and k == 0) else 1.0
                                   for k in range(len(J.BANDS))]))
        out.setdefault("inst_w", []).append(O2.inst_weights(resp))
        if obs == "v2":
            continue
        Cb = sum(w * jnp.stack([t[0] for t in inst_tables(i)]) for i, w in resp)
        Db = sum(w * jnp.stack([t[1] for t in inst_tables(i)]) for i, w in resp)
        out["Cb"].append(Cb); out["Db"].append(Db)
        out["Sb"].append(jnp.asarray(sum(w * sync_tables(i)[1] for i, w in resp)))
    if obs == "v2":
        for k in ("Cb", "Db", "Sb"):
            del out[k]
    fine_xy = out.pop("fine_xy", None)
    jet_B = out.pop("jet_B", None)
    out = {k: (v if k == "resp" else np.stack(v) if k not in ("Cb", "Db", "Sb") else jnp.stack(v))
           for k, v in out.items()}
    out.update(npix=npix, pix=pix, block=block, nh_grid=np.asarray(nh_grid), table_dir=Path(table_dir),
               obs=obs)
    if fine_block:
        out.update(fine_block=int(fine_block), fine_xy=fine_xy)
    if jet_B is not None:
        out["jet_B"] = jet_B
    if obs == "v2":
        out.update(v2=v2, lecut=v2["lecut"])
    else:
        out["lecut"] = scache[resp[0][0]][0]
    return out
SPEC_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
SPEC_EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)     # analysis bins (keV)
INSTRUMENTS = ("chandra_aciss_cy0", "chandra_aciss_cy10", "chandra_aciss_cy22", "chandra_acisi_cy22")


def load_spectrum_data(epochs, img):
    """The integrated spectra (r < 200", 50 eV bins; casa_observe.real_epoch_spectrum)
    on SPEC_EDGES, and the channel-resolved tables to model them.

    Channel tables are kept per INSTRUMENT (and mixed per epoch inside the
    forward model): stacked per epoch they would be ~9 GB. ``acis_i`` flags
    the ACIS-I epochs (their soft bins are dropped from the likelihood).
    """
    table_dir = img.get("table_dir", J.TABLE_DIR)
    nh_grid = img.get("nh_grid", NH_GRID_LEGACY)
    if img.get("obs", "v1") == "v2":
        # v2: the tables ARE on SPEC_EDGES (no channel rebinning, no overlap
        # map O); C / S / keep are in img["v2"] (the forward model mixes them)
        if not np.allclose(J.SPEC_EDGES, SPEC_EDGES):
            raise ValueError("casa_jaxobs.SPEC_EDGES != casa_xfit.SPEC_EDGES")
        out = spectrum_counts(epochs, img)
        out.update(n_bin=len(SPEC_EDGES) - 1)
        return out
    t0 = J.load_tables(INSTRUMENTS[0], 1.2, table_dir=table_dir)
    ch = np.asarray(t0["ch_edges"])
    sel = np.nonzero((ch[1:] > SPEC_EDGES[0] - 0.1) & (ch[:-1] < SPEC_EDGES[-1] + 0.1))[0]
    lo, hi = ch[sel], ch[sel + 1]
    O = np.clip(np.minimum(SPEC_EDGES[1:, None], hi[None]) - np.maximum(SPEC_EDGES[:-1, None], lo[None]),
                0.0, None) / (hi - lo)[None]                       # (n_bin, n_sel)
    C = np.stack([np.stack([np.asarray(J.load_tables(i, nh, table_dir=table_dir)["C"])[..., sel]
                            for nh in nh_grid])
                  for i in INSTRUMENTS]).astype(np.float32)      # (inst, nh, c, t, n, sel)

    def sync_raw(i):
        d = np.load(Path(table_dir) / f"sync_{i}.npz")
        return resample_nh(d["S"][..., sel], d["nh"], nh_grid, axis=0)
    S = np.stack([sync_raw(i) for i in INSTRUMENTS]).astype(np.float32)
    halo = J.load_halo(1.2, table_dir=table_dir)
    e_mid = 0.5 * (lo + hi)
    keep = np.stack([np.interp(e_mid, halo["e_grid"], halo["aperture_keep"][:, k])
                     for k in range(len(halo["r_grid"]))], 1)      # (sel, n_r)
    out = spectrum_counts(epochs, img)
    out.update(C=jnp.asarray(C), S=jnp.asarray(S), O=jnp.asarray(O, jnp.float32),
               keep=jnp.asarray(keep, jnp.float32), r_grid=jnp.asarray(halo["r_grid"], jnp.float32),
               n_sel=len(sel), n_bin=len(SPEC_EDGES) - 1)
    return out


def spectrum_counts(epochs, img):
    """The data side of the spectra: counts on SPEC_EDGES (from the 50 eV epoch
    spectra; 0.7 is a multiple of 0.05, so the edges are exact), exposure,
    which epochs have one, their instrument weights and ACIS-I flags."""
    counts, expo, has, wi, acis_i = [], [], [], [], []
    for e, label in enumerate(epochs):
        path = SPEC_DIR / f"epoch_{label}_spectrum.npz"
        w = np.zeros(len(INSTRUMENTS))
        for inst, ww in img["resp"][e]:
            w[INSTRUMENTS.index(inst)] += ww
        wi.append(w)
        acis_i.append(img["resp"][e][0][0].startswith("chandra_acisi"))
        if path.exists():
            d = np.load(path)
            eb = np.asarray(d["ebins"])
            idx = np.searchsorted(np.round(eb, 3), SPEC_EDGES)
            if not np.allclose(np.round(eb, 3)[idx], SPEC_EDGES):
                raise ValueError(f"{path}: SPEC_EDGES are not on its energy bins")
            cs = np.concatenate([[0.0], np.cumsum(d["counts"])])
            counts.append(cs[idx[1:]] - cs[idx[:-1]]); expo.append(float(d["exposure"])); has.append(True)
        else:
            counts.append(np.zeros(len(SPEC_EDGES) - 1)); expo.append(1.0); has.append(False)
    return dict(counts=np.stack(counts), exposure=np.array(expo), has=np.array(has),
                inst_w=jnp.asarray(np.stack(wi), jnp.float32), acis_i=np.array(acis_i))


# =============================================================================
# ============ ↑ Parameters and data ↑ ========================================
# =============================================================================


# =============================================================================
# ============ ↓ Observation-model pieces owned by the fit ↓ ==================
# =============================================================================
def _geo(a, b, t):
    """a^(1 - t) b^t elementwise (ln-linear through a at t = 0 and b at t = 1);
    ``a`` where either is not positive (no emission in that column)."""
    ok = (a > 0) & (b > 0)
    la = jnp.log(jnp.where(ok, a, 1.0)); lb = jnp.log(jnp.where(ok, b, 1.0))
    return jnp.where(ok, jnp.exp(la + t * (lb - la)), a)


def nh_mix(cols, lnh_map, lng, mode="extrap"):
    """Columns at the tabulated N_H nodes -> the per-sky-column N_H map.

    ``cols`` (n_nh, k, x, z) at ln N_H nodes ``lng``; ``lnh_map`` (x, z).
    Inside the grid: tent weights in ln N_H (linear in the columns, as before).
    Outside (``mode="extrap"``): ln(column) linear in N_H through the two edge
    nodes -- exact for single-energy absorption; on the current tables it
    reproduces a held-out edge node to <= 7.5 % (soft band) and <= 0.2 %
    (>= 1.5 keV), where the old clamp was off by up to x1.8. ``"clamp"``: the
    pre-2026-09-25 behaviour (zero N_H gradient outside the grid).
    """
    n_nh = len(lng)
    if mode == "geo":
        return nh_mix_geo(cols, lnh_map, lng)
    w = jnp.stack([jnp.interp(lnh_map, jnp.asarray(lng), jnp.asarray(np.eye(n_nh)[j]))
                   for j in range(n_nh)])                                   # (n_nh, x, z)
    inside = jnp.einsum("jkxz,jxz->kxz", cols, w)
    if mode == "clamp":
        return inside
    nh = np.exp(np.asarray(lng))
    N = jnp.exp(lnh_map)
    t_lo = jnp.minimum((N - nh[0]) / (nh[1] - nh[0]), 0.0)[None]
    t_hi = jnp.maximum((N - nh[-2]) / (nh[-1] - nh[-2]), 1.0)[None]
    below = _geo(cols[0], cols[1], t_lo)
    above = _geo(cols[-2], cols[-1], t_hi)
    return jnp.where((lnh_map < lng[0])[None], below,
                     jnp.where((lnh_map > lng[-1])[None], above, inside))


def nh_mix_geo(cols, lnh_map, lng):
    """``nh_mix`` with ln(column) linear in N_H between (and beyond) the two
    bracketing nodes: a^(1 - t) b^t, t = (N - N_j) / (N_j+1 - N_j) -- exact for
    single-energy absorption, and the same form as the ``extrap`` branch
    outside the grid (so value and slope are continuous at the edges). The
    tent weights in ln N_H of ``extrap`` / ``clamp`` over-estimate the absorbed
    soft bins between nodes (exp(-sigma N) is convex): with the v2 grid by up
    to +19 % at 0.7-0.9 keV (obs_tables review r5); holding out every other
    v2 node (twice the spacing) the band-table error drops from 14-29 % to
    3-8 % (stage-2 ``nh_interp_test.log``). Where either node is not positive
    (no emission, or a signed moment) it falls back to the linear tent."""
    n_nh = len(lng)
    lg = jnp.asarray(lng)
    nh = jnp.asarray(np.exp(np.asarray(lng)))
    j = jnp.clip(jnp.searchsorted(lg, lnh_map) - 1, 0, n_nh - 2)                  # (x, z)
    t = (jnp.exp(lnh_map) - nh[j]) / (nh[j + 1] - nh[j])
    idx = jnp.broadcast_to(j[None, None], (1,) + cols.shape[1:])
    a = jnp.take_along_axis(cols, idx, axis=0)[0]
    b = jnp.take_along_axis(cols, idx + 1, axis=0)[0]
    ok = (a > 0) & (b > 0)
    la = jnp.log(jnp.where(ok, a, 1.0)); lb = jnp.log(jnp.where(ok, b, 1.0))
    tl = jnp.clip((lnh_map - lg[j]) / (lg[j + 1] - lg[j]), 0.0, 1.0)[None]
    return jnp.where(ok, jnp.exp(la + t[None] * (lb - la)), a + tl * (b - a))


def nh_tents(lnh_map, lng):
    """(n_nh, x, z) tent weights in ln N_H, clamped to the edge nodes outside
    the grid: which N_H node's halo kernel a column is scattered with."""
    n_nh = len(lng)
    return jnp.stack([jnp.interp(lnh_map, jnp.asarray(lng), jnp.asarray(np.eye(n_nh)[j]))
                      for j in range(n_nh)])


def scale_si(fields, ln_si, csm_solar):
    """The Si-group tracer with its EJECTA part scaled by exp(ln_si). With a
    solar CSM the CSM part is (1 - C_ej) X_Si,sun capped at the tracer
    (``casa_jaxobs.element_fractions``' split, same where-clips); otherwise the
    ejecta part is C_ej C_Si."""
    s = jnp.exp(ln_si)
    X = fields["C_Si"]
    c_ej = jnp.where(fields["C_ej"] < 0.0, 0.0, jnp.where(fields["C_ej"] > 1.0, 1.0, fields["C_ej"]))
    if csm_solar:
        a = (1.0 - c_ej) * J.solar_csm_tracers()["Si"]
        x_csm = jnp.where(a <= X, a, X)
        return dict(fields, C_Si=x_csm + s * (X - x_csm))
    return dict(fields, C_Si=X * (1.0 + (s - 1.0) * c_ej))


def doppler_moment_maps(fields, tables, *, box_pc, distance_kpc, band=J.SI_BAND,
                        subgrid=dict(chi=4.0, f_mass=0.34), split="xrism_bulk", slab=None,
                        plasma_kw=None):
    """Per sky column (x, z) the zeroth and first photon-energy moments about E0
    in ``band`` (the data's centroid statistic, continuum dilution included):
    ``casa_jaxobs.doppler_sectors`` up to its sector reduction, so the sectors
    can be drawn in the traced SKY frame (``sky_sector_velocities``).
    Returns (S (2, n, n), E0)."""
    if hasattr(J, "doppler_moment_maps"):                  # if the obs model exposes it
        return J.doppler_moment_maps(fields, tables, box_pc=box_pc, distance_kpc=distance_kpc,
                                     band=band, subgrid=subgrid, split=split, slab=slab,
                                     plasma_kw=plasma_kw)
    n = fields["rho"].shape[0]
    norm = J.flux_norm((box_pc / n * J.PC_CM) ** 3, distance_kpc)
    E0 = 0.5 * (band[0] + band[1])
    e = np.asarray(tables["e_ch"])
    m = jnp.asarray(((e >= band[0]) & (e < band[1])).astype(np.float32))
    de = jnp.asarray((e - E0).astype(np.float32))
    M0, M1 = jnp.einsum("ctnh,h->ctn", tables["C"], m), jnp.einsum("ctnh,h->ctn", tables["C"], m * de)
    D0, D1 = jnp.einsum("ctnh,h->ctn", tables["D"], m), jnp.einsum("ctnh,h->ctn", tables["D"], m * de)
    ncomp = len(tables["names"])
    cidx = jnp.arange(ncomp)[:, None, None, None, None]
    MD = jnp.stack([M0, M1, D0, D1], -1)                                  # (c, t, n, 4)

    def body(S, sl):
        beta = sl["vy"] * 1e3 / J.C_KMS
        for fcomp, ew in J.emitting_components(sl, split=split, **subgrid):
            pl = J.plasma(fcomp, split, **(plasma_kw or {}))
            w = J.component_weights(pl, tables, norm, ew)
            ii, jj, ww = J.corners(tables, pl)
            t = jnp.einsum("cqxszk,cxsz,qxsz->kxsz", MD[cidx, ii[None], jj[None]], w, ww)
            S = S + jnp.stack([t[0] + beta * t[2], t[1] + beta * t[3]]).sum(axis=2)
        return S

    return J.slab_scan(fields, slab, body, jnp.zeros((2, n, n))), E0


def doppler_moments_v2(fo, V, wi, *, box, amp, ln_sync, year, lnh_map, lng, subgrid, plasma_kw,
                       kt_interp="log"):
    """The data's Si centroid statistic per sky column, (M0, M1) (2, x, z): the
    v2 native-channel moments in exactly [1.78, 1.94] keV of the epoch's
    instrument mix ``wi`` (``casa_xfit_obs2``), thermal (x ``amp``) plus the
    synchrotron dilution, mixed over the N_H nodes with linear tents (M1 is
    signed). The tables hold M1' = M1 + DOP_SHIFT M0 > 0 (the log
    interpolations need a positive table); M1 is restored here."""
    n_nh = len(lng)

    def fold(T):
        return jnp.moveaxis(T, 0, -2).reshape(*T.shape[1:-1], n_nh * T.shape[-1])

    Cd = jnp.tensordot(wi, V["Cd"], 1); Dd = jnp.tensordot(wi, V["Dd"], 1)
    Sd = jnp.tensordot(wi, V["Sd"], 1)
    th = J.doppler_columns(fo, V["tables_dop0"], box_pc=box, distance_kpc=D_REF_KPC, subgrid=subgrid,
                           plasma_kw=plasma_kw, kt_interp=kt_interp, band_tables=(fold(Cd), fold(Dd)))
    sy = J.sync_columns(fo, lecut=V["lecut"], band_table=fold(Sd), year=year, plasma_kw=plasma_kw)
    tot = amp * th + jnp.exp(ln_sync) * sy                                    # (n_nh * 2, x, z)
    S2 = nh_mix(tot.reshape(n_nh, 2, *tot.shape[1:]), lnh_map, lng, "clamp")
    return jnp.stack([S2[0], S2[1] - V["dop_shift"] * S2[0]])


def sky_sector_velocities(Sxz, E0, sky_w, sky_n, *, cell_arcsec, n_sec=24, annulus=(40.0, 170.0)):
    """Centroid velocity per SKY sector about RA0/DEC0 (the data's sectors:
    sector k spans [k, k + 1) x 360 / n_sec deg in theta = atan2(north, west)),
    with the columns at their traced sky positions (roll, distance, offset).
    Membership is soft on the scale of a column (sigmoids in angle and in the
    annulus radius), so the statistic is smooth in (psi, D, dw, dn)."""
    R = jnp.sqrt(sky_w ** 2 + sky_n ** 2)
    th = jnp.rad2deg(jnp.arctan2(sky_n, sky_w)) % 360.0
    soft_r = 0.5 * cell_arcsec
    wa = jax.nn.sigmoid((R - annulus[0]) / soft_r) * jax.nn.sigmoid((annulus[1] - R) / soft_r)
    half = 180.0 / n_sec
    centres = jnp.asarray((np.arange(n_sec) + 0.5) * 2 * half)
    dl = ((th[None] - centres[:, None, None] + 180.0) % 360.0) - 180.0
    soft_a = jnp.rad2deg(0.5 * cell_arcsec / jnp.maximum(R, cell_arcsec))[None]
    W = (jax.nn.sigmoid((dl + half) / soft_a) - jax.nn.sigmoid((dl - half) / soft_a)) * wa[None]
    S = jnp.einsum("kxz,sxz->ks", Sxz, W, precision=jax.lax.Precision.HIGHEST)
    s0, s1 = S[0], S[1]
    valid = s0 > 0.0
    dE = jnp.where(valid, s1 / jnp.where(valid, s0, 1.0), 0.0)
    mean = jnp.sum(jnp.where(valid, dE, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
    return jnp.where(valid, -J.C_KMS * (dE - mean) / E0, 0.0)
# =============================================================================
# ============ ↑ Observation-model pieces owned by the fit ↑ ==================
# =============================================================================


# =============================================================================
# ============ ↓ The forward model ↓ ==========================================
# =============================================================================
def make_forward_core(ic, obs, img, *, cfl=0.3, ad_llf_cold=0.0, subgrid=dict(chi=4.0, f_mass=0.34),
                      opts=None, config_overrides=None, ic_path=None):
    """What ``make_forward`` builds that does not depend on how the FIRST
    epoch's state is made: the solver config, the geometry, the observation
    model of one epoch for a parameter dict (``core.observer(p)`` ->
    ``observe(state, ep)``), the per-epoch inputs (``core.xs_all``, in sorted
    epoch order) and the assembly of the per-epoch outputs into the model dict
    (``core.assemble(p, outs)``). ``casa_4dvar`` starts from a saved 2000
    state with these same pieces.

    ``ic``: a dict with at least ``box``, ``num_cells``, ``age`` (and the wind
    bookkeeping ``casa_pluto_diff.wind_nh`` reads, for ``make_forward``);
    ``ic_path``: its file (for the ``"auto"`` options)."""
    opts = resolve_auto(opts or default_options(), ic_path)
    v2 = opts.obs == "v2"
    # the data dicts must have been built for the same chain (review 2026-09-25:
    # a caller outside ``main`` with the v2 defaults and v1 ``img`` failed with a
    # bare KeyError, and --responses ciao without ``attach_responses`` or the v2
    # model against the legacy Doppler data were silently not what opts says)
    if img.get("obs", "v1") != opts.obs:
        raise ValueError(f"img was loaded for --obs {img.get('obs', 'v1')}, opts say {opts.obs} "
                         f"(load_image_data(..., obs=opts.obs))")
    if getattr(opts, "responses", "tables") != "tables" and "resp_corr" not in img:
        raise ValueError(f"opts.responses={opts.responses!r} but img has no resp_corr: call attach_responses")
    if v2 and "doppler" not in obs:
        raise ValueError("--obs v2: obs['doppler'] must be the v2 data (casa_xfit_obs2.load_doppler_data())")
    box, n = float(ic["box"]), int(ic["num_cells"])
    cu = snr_code_units()
    rho_c = float((1.0 * cu.code_density).to(u.g / u.cm ** 3).value)
    yr = float((1.0 * u.yr).to(cu.code_time).value)
    kw = dict(dual_energy=True, progress_bar=False, weno_ad_frozen_weights=True,
              positivity_config=fd_positivity(mode=POSITIVITY_REDISTRIBUTE),
              num_passive_scalars=len(SCALAR_NAMES), track_shock_history=True,
              passive_scalar_bounds=tuple((0.0, 1.0) for _ in SCALAR_NAMES),
              ad_tangent_llf_cold_factor=float(ad_llf_cold))
    kw.update(config_overrides or {})
    config = make_fd_config(box, n, **kw)
    rv = get_registered_variables(config)
    # multi-GPU (casa_xfit_shard.activate): the geometry split like the state,
    # the cone profiles as shard-local segment maps, the solver sharded
    sharding = SH.state_sharding()
    hd = get_helper_data(config)        # (sharded: re-placed from the host below, no device-to-device copy)
    c = hd.geometric_centers
    X, Y, Z = c[..., 0] - box / 2, c[..., 1] - box / 2, c[..., 2] - box / 2
    r = jnp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    geom = (r, X, Y, Z)
    cones = PD.ConeProfiles(geom, r_max=0.5 * box, dx=box / n)
    if SH.active():
        geom = tuple(SH.put(np.asarray(g), SH.FIELD) for g in geom)
        cones = SH.ShardedCones(cones, (n, n, n))
    cell_vol = (box / n) ** 3
    t_per_code = float((0.6 * const.m_p * cu.code_velocity ** 2 / const.k_B).to(u.K).value)
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(cu.code_density).value)
    p_per_n = float((const.k_B * 1e4 * u.K / u.cm ** 3).to(cu.code_pressure).value)
    base_params = SimulationParams(
        gamma=GAMMA, C_cfl=cfl, t_end=1.0,
        minimum_density=0.1 * rho_per_n * 1e-3, minimum_pressure=0.1 * p_per_n * 1e-2,
        minimum_specific_pressure=p_per_n / rho_per_n)
    order = np.argsort(obs["years"])
    inv = np.argsort(order)
    obs_angles = jnp.asarray(obs["angles"])
    table_dir = img.get("table_dir", J.TABLE_DIR)
    if v2:
        V = img["v2"]
        tables0 = V["tables0"]                  # grids (kT x rho, n_e t) for the corner lookup
        halo = None
        if opts.doppler_frame != "sky":
            raise ValueError("--obs v2 needs --doppler-frame sky (the v2 statistic is in the data's sectors)")
    else:
        tables0 = J.load_tables("chandra_aciss_cy0", 1.2, table_dir=table_dir)  # component names; Doppler moments
        halo = J.load_halo(1.2, table_dir=table_dir)
    kt_interp = opts.kt_interp
    csm_solar = opts.csm_solar == "on"
    i0 = rv.passive_scalar_index
    i_hist = i0 + rv.num_passive_scalars - 4
    years_sorted = jnp.asarray(np.asarray(obs["years"])[order])
    spec = img["spec"]
    wi_all = spec["inst_w"][order]
    has_all = jnp.asarray(spec["has"][order])
    n_bin = int(spec.get("n_bin", len(SPEC_EDGES) - 1))
    dop_data = obs.get("doppler", PD.DOPPLER)
    e_dop = int(np.argmin(np.abs(np.asarray(obs["years"]) - float(dop_data["year"]))))
    dop_all = jnp.asarray(np.arange(len(order))[order] == e_dop)
    if v2:
        xs_all = dict(year=years_sorted, wi=wi_all, has=has_all, dop=dop_all)
    else:
        xs_all = dict(Cb=img["Cb"][order], Db=img["Db"][order], Sb=img["Sb"][order], year=years_sorted,
                      wi=wi_all, has=has_all)
    resp_img = img.get("resp_corr")
    resp_spec = spec.get("resp_corr")
    lng = np.log(np.asarray(img.get("nh_grid", NH_GRID_LEGACY), np.float64))
    WX0, NZ0, _ = J.sky_geometry(box, n, D_REF_KPC)
    cell0_arcsec = float(WX0[1, 0] - WX0[0, 0])
    WX0, NZ0 = jnp.asarray(WX0, jnp.float32), jnp.asarray(NZ0, jnp.float32)
    dop_ann = tuple(float(a) for a in dop_data.get("annulus_arcsec", (40.0, 170.0)))
    n_dop = int(dop_data.get("n", 24))
    cones_ang = jnp.asarray(cones.angles)
    vth = np.asarray(PD.VINK22_PA + 90.0)
    rs_est = opts.rs_estimator
    # stage-4 observation-model options
    sync_trend = stage4(opts, "sync_trend") == "on"
    kte_fixed = stage4(opts, "kte") == "fixed"
    broadening = stage4(opts, "spec_broadening")
    gain_on = stage4(opts, "spec_gain") == "profile"
    if broadening not in ("off", "vlos", "thermal"):
        raise ValueError(f"spec_broadening {broadening!r}")
    if v2 and broadening != "off" and "D2s" not in img["v2"]:
        raise ValueError("spec_broadening needs the D / D2 spectral tables: load_image_data(..., "
                         "spec_kinds=spec_table_kinds(opts))")
    if v2 and gain_on and "Ds" not in img["v2"]:
        raise ValueError("spec_gain needs the D spectral tables: load_image_data(..., "
                         "spec_kinds=spec_table_kinds(opts))")
    gain_beta = GAIN_DV_KMS / J.C_KMS

    def to_obs_fields(st):
        f = dict(rho=st[rv.density_index], press=st[rv.pressure_index], vy=st[rv.velocity_index.y])
        for k, name in enumerate(SCALAR_NAMES):
            f[name] = st[i0 + k]
        for j, name in enumerate(HISTORY):
            f[name] = st[i_hist + 1 + j]
        return {k: SH.cfield(v) for k, v in f.items()} if SH.active() else f

    def observer(p):
        """``observe(state, ep) -> (r_fs, r_rs, r_rs_leg, image, v, sync_frac,
        spectrum)`` of one epoch (``ep``: one slice of ``xs_all``) for the
        observer-frame / emission parameters in ``p``."""
        d_kpc = jnp.exp(p["ln_D"])
        scale = D_REF_KPC / d_kpc
        # traced N_H MAP: ln N_H = ln_nh + a gradient on the sky; the columns are
        # computed at every tabulated N_H (folded into the band axis) and mixed
        # per sky column (``nh_mix``: tents inside the grid, ln-linear outside;
        # ``geo``: ln-linear in N_H throughout)
        ps = jnp.deg2rad(p["psi"])
        sky_w = scale * (jnp.cos(ps) * WX0 - jnp.sin(ps) * NZ0) + p["dw"]
        sky_n = scale * (jnp.sin(ps) * WX0 + jnp.cos(ps) * NZ0) + p["dn"]
        lnh_map = p["ln_nh"] + (p["g_nh_w"] * sky_w + p["g_nh_n"] * sky_n) / 100.0
        amp = jnp.exp(p["ln_A"]) * scale ** 2
        n_nh = len(lng)
        pkw = dict(kT_e_shock_keV=jnp.zeros_like(p["ln_kte"]) + KTE_FIXED_KEV if kte_fixed
                   else jnp.exp(p["ln_kte"]), teq_scale=jnp.exp(p["ln_teq"]),
                   fe_scale=jnp.exp(p["ln_fe"]))

        def ln_sync_at(year):
            """ln of the synchrotron efficiency at ``year``: ln_sync plus, with
            --sync-trend on, the X-ray secular trend b_sx (%/yr) about SYNC_T_REF
            on top of the radio anchor."""
            if sync_trend:
                return p["ln_sync"] + 0.01 * p.get("b_sx", EXTRA_PRIOR["b_sx"][0]) * (year - SYNC_T_REF)
            return p["ln_sync"]

        def sync_amp(year):
            return jnp.exp(ln_sync_at(year))
        if csm_solar:
            pkw["csm_solar"] = True
        sg = dict(subgrid, f_mass=jax.nn.sigmoid(p["lg_fmass"]))

        def fold(T):                     # (n_nh, ..., b) -> (..., n_nh * b)
            return jnp.moveaxis(T, 0, -2).reshape(*T.shape[1:-1], n_nh * T.shape[-1])

        R_sky = jnp.sqrt(sky_w ** 2 + sky_n ** 2)

        def mix(cols):                   # (n_nh * k, x, z) -> (k, x, z)
            return nh_mix(cols.reshape(n_nh, cols.shape[0] // n_nh, *cols.shape[1:]),
                          lnh_map, lng, opts.nh_mode)

        def obs_fields(st):
            fo = to_obs_fields(st)
            return scale_si(fo, p["ln_si"], csm_solar) if "ln_si" in p else fo

        def spectrum(fo, wi, year):
            """Counts/s per analysis bin in the 200" aperture (halo aperture-keep
            per pixel and channel, N_H map, synchrotron)."""
            Cc = jnp.tensordot(wi, spec["C"], 1)                          # (nh, c, t, n, sel)
            Sc = jnp.tensordot(wi, spec["S"], 1)                          # (nh, ecut, sel)
            cc = J.band_columns(fo, tables0, box_pc=box, distance_kpc=D_REF_KPC, subgrid=sg,
                                band_tables=(fold(Cc), fold(Cc)), plasma_kw=pkw, v_los_kms=False,
                                kt_interp=kt_interp)
            sc = J.sync_columns(fo, lecut=img["lecut"], band_table=fold(Sc), year=year, plasma_kw=pkw)
            tot = mix(amp * cc + sync_amp(year) * sc)                     # (sel, x, z)
            keep = jax.vmap(lambda kr: jnp.interp(R_sky, spec["r_grid"], kr, right=0.0))(spec["keep"])
            return spec["O"] @ jnp.sum(tot * keep, (1, 2))

        def radii(st):
            hot = PD.shocked_indicator(st, rv, t_per_code)
            w = cones.mean(hot)
            rc = jnp.asarray(cones.rc)
            u_prof = None
            if rs_est in ("unshocked", "coldej"):
                marker = st[i_hist + 1] if rs_est == "unshocked" else hot
                u_prof = PD.unshocked_ejecta_profile(cones, st[i0], marker)
            r_fs, r_rs = PD.edge_radii(w, rc, cones.valid, u_prof=u_prof)
            r_rs_leg = PD.legacy_inner_edge(jnp.where(cones.valid, w, 0.0), rc, r_fs)
            return r_fs, r_rs, r_rs_leg

        def observe(st, ep):
            r_fs, r_rs, r_rs_leg = radii(st)
            fo = obs_fields(st)
            cols = mix(J.band_columns(fo, tables0, box_pc=box, distance_kpc=D_REF_KPC,
                                      subgrid=sg, band_tables=(fold(ep["Cb"]), fold(ep["Db"])), plasma_kw=pkw,
                                      kt_interp=kt_interp))
            # the synchrotron is anchored to the radio flux: no distance scaling
            sync = sync_amp(ep["year"]) * mix(J.sync_columns(
                fo, lecut=img["lecut"], band_table=fold(ep["Sb"]), year=ep["year"], plasma_kw=pkw))
            sync_frac = sync.sum((1, 2)) / (amp * cols.sum((1, 2)) + sync.sum((1, 2)))
            image = J.project_columns(amp * cols + sync, box_pc=box, distance_kpc=D_REF_KPC, halo=halo,
                                      npix=img["npix"], pix_arcsec=img["pix"],
                                      roll_deg=p["psi"], offset_arcsec=(p["dw"], p["dn"]),
                                      sky_scale=scale)
            if opts.doppler_frame == "sky":
                Sxz, E0 = doppler_moment_maps(fo, tables0, box_pc=box, distance_kpc=D_REF_KPC,
                                              subgrid=sg, plasma_kw=pkw)
                v = sky_sector_velocities(Sxz, E0, sky_w, sky_n, cell_arcsec=cell0_arcsec * scale,
                                          n_sec=n_dop, annulus=dop_ann)
            else:
                v, _ = J.doppler_sectors(fo, tables0, box_pc=box, distance_kpc=D_REF_KPC, subgrid=sg,
                                         plasma_kw=pkw)
            sp = jax.lax.cond(ep["has"], lambda: spectrum(fo, ep["wi"], ep["year"]),
                              lambda: jnp.zeros(spec["O"].shape[0]))
            return r_fs, r_rs, r_rs_leg, image, v, sync_frac, sp

        if v2:
            tau = nh_tents(lnh_map, lng)                                   # halo-kernel weights
            n_sp = V["keep"].shape[1] if "keep" in V else n_bin
            if "keep" in V:
                kp = jax.vmap(lambda kr: jnp.interp(R_sky, V["r_grid"], kr, right=0.0))(
                    V["keep"].reshape(-1, V["keep"].shape[-1])).reshape(n_nh, n_sp, *R_sky.shape)

            def spectrum_v2(fo, wi, year):
                """Counts/s per analysis bin (SPEC_EDGES) in the 200" aperture: the
                v2 bin tables, the aperture keep of each N_H node's halo, N_H map.

                --spec-broadening vlos / thermal: every cell's line-of-sight
                velocity to second order (C + beta D + beta^2 / 2 D2), + the ions'
                thermal spread (stage-3 physics item 2: the Fe-K wing pattern).
                --spec-gain profile: also returns d(spectrum)/d(energy scale), a
                one-sided difference in a uniform extra line-of-sight velocity
                beta = GAIN_DV_KMS / c (an energy-scale error is a uniform line
                shift), under stop_gradient (a nuisance sensitivity)."""
                Cc = jnp.tensordot(wi, V["Cs"], 1)                        # (nh, c, t, n, 31)
                Sc = jnp.tensordot(wi, V["Ss"], 1)                        # (nh, ecut, 31)
                sc = J.sync_columns(fo, lecut=V["lecut"], band_table=fold(Sc), year=year, plasma_kw=pkw)
                if broadening != "off" or gain_on:
                    Dc = jnp.tensordot(wi, V["Ds"], 1)
                    tabs = (fold(Cc), fold(Dc))
                    if broadening != "off":
                        tabs = tabs + (fold(jnp.tensordot(wi, V["D2s"], 1)),)

                def thermal(f, tables, v_los, therm):
                    return J.band_columns(f, tables0, box_pc=box, distance_kpc=D_REF_KPC, subgrid=sg,
                                          band_tables=tables, plasma_kw=pkw, v_los_kms=v_los,
                                          thermal_broadening=therm, kt_interp=kt_interp)

                def total(cc):
                    tot = (amp * cc + sync_amp(year) * sc).reshape(n_nh, n_sp, *R_sky.shape) * kp
                    return jnp.sum(mix(tot.reshape(n_nh * n_sp, *R_sky.shape)), (1, 2))

                if broadening == "off":
                    sp = total(thermal(fo, (fold(Cc), fold(Cc)), False, False))
                else:
                    sp = total(thermal(fo, tabs, True, broadening == "thermal"))
                if not gain_on:
                    return sp
                dv = GAIN_DV_KMS / 1e3                                       # code velocity: 1000 km/s
                if broadening == "off":
                    fg = dict(fo, vy=jnp.full_like(fo["vy"], dv))
                    sg_ = total(thermal(fg, tabs, True, False))
                else:
                    sg_ = total(thermal(dict(fo, vy=fo["vy"] + dv), tabs, True, broadening == "thermal"))
                return sp, jax.lax.stop_gradient((sg_ - sp) / gain_beta)

            def doppler_v2(fo, wi, year):
                S2 = doppler_moments_v2(fo, V, wi, box=box, amp=amp, ln_sync=ln_sync_at(year), year=year,
                                        lnh_map=lnh_map, lng=lng, subgrid=sg, plasma_kw=pkw,
                                        kt_interp=kt_interp)
                return sky_sector_velocities(S2, V["E0"], sky_w, sky_n, cell_arcsec=cell0_arcsec * scale,
                                             n_sec=n_dop, annulus=dop_ann)

            def observe(st, ep):                                            # noqa: F811
                r_fs, r_rs, r_rs_leg = radii(st)
                fo = obs_fields(st)
                wi = ep["wi"]
                Cb = jnp.tensordot(wi, V["Cb"], 1); Db = jnp.tensordot(wi, V["Db"], 1)
                Sb = jnp.tensordot(wi, V["Sb"], 1)
                cols = mix(J.band_columns(fo, tables0, box_pc=box, distance_kpc=D_REF_KPC, subgrid=sg,
                                          band_tables=(fold(Cb), fold(Db)), plasma_kw=pkw,
                                          kt_interp=kt_interp))                       # (6, x, z)
                sync = sync_amp(ep["year"]) * mix(J.sync_columns(
                    fo, lecut=V["lecut"], band_table=fold(Sb), year=ep["year"], plasma_kw=pkw))
                sync_frac = sync.sum((1, 2)) / (amp * cols.sum((1, 2)) + sync.sum((1, 2)))
                nb = cols.shape[0]
                # each column scattered with the halo of ITS N_H: the N_H-mixed
                # column split over the bracketing nodes' kernels (tents), all
                # 8 x 6 kernels in one projection (nh-major, halo_nh_stacked)
                K = jnp.tensordot(wi, V["halo_K"], 1)
                halo_ep = dict(V["halo_meta"], kernel=K.reshape(n_nh * nb, *K.shape[-2:]))
                cw = (tau[:, None] * (amp * cols + sync)[None]).reshape(n_nh * nb, *R_sky.shape)
                image = J.project_columns(cw, box_pc=box, distance_kpc=D_REF_KPC, halo=halo_ep,
                                          npix=img["npix"], pix_arcsec=img["pix"], roll_deg=p["psi"],
                                          offset_arcsec=(p["dw"], p["dn"]), sky_scale=scale)
                image = image.reshape(n_nh, nb, *image.shape[-2:]).sum(0)
                v = jax.lax.cond(ep["dop"], lambda: doppler_v2(fo, wi, ep["year"]), lambda: jnp.zeros(n_dop))
                if gain_on:
                    # the 8th output: d(spectrum)/d(energy scale) (``assemble``: spectra_dg)
                    sp, sp_dg = jax.lax.cond(ep["has"], lambda: spectrum_v2(fo, wi, ep["year"]),
                                             lambda: (jnp.zeros(n_bin), jnp.zeros(n_bin)))
                    return r_fs, r_rs, r_rs_leg, image, v, sync_frac, sp, sp_dg
                sp = jax.lax.cond(ep["has"], lambda: spectrum_v2(fo, wi, ep["year"]),
                                  lambda: jnp.zeros(n_bin))
                return r_fs, r_rs, r_rs_leg, image, v, sync_frac, sp

        return observe

    def integrator(shape):
        """``integrate(state, dt)`` for states of ``shape``."""
        cfg = finalize_config(config, shape)

        def integrate(st, dt):
            if sharding is None:
                return time_integration(st, cfg, base_params._replace(t_end=dt), rv)
            return SH.cstate(time_integration(SH.cstate(st), cfg, base_params._replace(t_end=dt), rv,
                                              sharding=sharding))
        return integrate

    def assemble(p, outs):
        """The model dict from the per-epoch observe() outputs, stacked over the
        epochs in SORTED order (``outs``: the 7 arrays of ``observe``)."""
        r_fs, r_rs, r_rs_leg, images, vlos, sync_frac, spectra = (a[inv] for a in outs[:7])
        # --spec-gain profile: the 8th output, d(spectrum)/d(energy scale)
        spectra_dg = outs[7][inv] if len(outs) > 7 else None
        # the CIAO per-epoch responses (--responses ciao): exposure-map geometry
        # x band-area ratio per pixel, folded-response ratio per spectral bin
        if resp_img is not None:
            images = images * resp_img
        if resp_spec is not None:
            spectra = spectra * resp_spec
            if spectra_dg is not None:
                spectra_dg = spectra_dg * resp_spec
        d_kpc = jnp.exp(p["ln_D"])
        d_pc = d_kpc * 1e3

        def radius_at(phi):      # model r_FS (arcsec) about the explosion centre at sky angle phi
            return PD.periodic_interp_rows(cones_ang, r_fs, phi - p["psi"]) / d_pc * PD.ARCSEC_PER_RAD

        ang_e = jnp.broadcast_to(obs_angles, (r_fs.shape[0], obs_angles.shape[0]))
        if opts.centre == "sky":
            # the explosion centre is at (dw, dn) from RA0/DEC0, the data's centre
            r_arc = PD.recentre_radii(radius_at, ang_e, p["dw"], p["dn"], opts.recentre_iter)
        elif opts.centre == "grid":
            r_arc = radius_at(ang_e)
        else:
            raise ValueError(f"centre {opts.centre!r}")
        # about the expansion centre, in Vink+22's sectors (for --pm-target vink22)
        r_coe = PD.recentre_radii(radius_at, jnp.broadcast_to(jnp.asarray(vth), (r_fs.shape[0], len(vth))),
                                  p["dw"] - PD.COE_ARCSEC[0], p["dn"] - PD.COE_ARCSEC[1],
                                  opts.recentre_iter)
        out = dict(r_fs_arcsec=r_arc, r_fs_pc=r_fs, r_rs_pc=r_rs, r_rs_legacy_pc=r_rs_leg,
                   r_fs_coe_arcsec=r_coe,
                   r_rs_mean_arcsec=jnp.mean(r_rs, axis=1) / d_pc * PD.ARCSEC_PER_RAD,
                   images=images, vlos_kms=vlos, sync_frac=sync_frac, spectra=spectra)
        if spectra_dg is not None:
            out["spectra_dg"] = spectra_dg
        return out

    core = SimpleNamespace(opts=opts, config=config, rv=rv, geom=geom, box=box, n=n, cell_vol=cell_vol,
                           rho_c=rho_c, yr=yr, t_per_code=t_per_code, rho_per_n=rho_per_n, p_per_n=p_per_n,
                           base_params=base_params, order=order, inv=inv, xs_all=xs_all, i0=i0,
                           i_hist=i_hist, cones=cones, observer=observer, integrator=integrator,
                           assemble=assemble, to_obs_fields=to_obs_fields, sharding=sharding)

    def lift(lifted):
        """Register the core's big traced constants (the cone segment maps) with
        a ``casa_xfit_shard.Lifted`` (multi-GPU: jit arguments, not HLO literals)."""
        if SH.active():
            cones.lift(lifted)
            if v2:              # the v2 tables (3.2 GB, read as V[...] at trace time): replicated arguments
                for k, v in list(V.items()):
                    if getattr(v, "ndim", 0) > 0 and getattr(v, "nbytes", 0) > 8e6:
                        lifted.add(V, k, SH.REPL)
        return lifted
    core.lift = lift
    return core


def make_forward(ic_path, obs, img, *, cfl=0.3, ad_llf_cold=0.0, subgrid=dict(chi=4.0, f_mass=0.34),
                 opts=None, config_overrides=None):
    """``theta -> model observables`` (dict) at every epoch.

    ``opts``: the fix options (``default_options``; default: the corrected
    ones). ``config_overrides``: extra ``SimulationConfig`` fields (e.g. the
    NATIVE_JAX backend for CPU tests). The returned function carries
    ``forward.ic_diag(theta)``: the IC bookkeeping (M_ej, energies, the
    ballistic date, the wind n_H) without the evolution.

    ``opts.obs == "v2"``: the obs_tables v2 chain (``casa_xfit_obs2``; binned
    tables with the NEI history axis, the v2 N_H grid and halo per N_H node,
    log-kT, solar CSM, the exact-window Doppler moments at the Doppler epoch
    only). ``img["resp_corr"]`` / ``img["spec"]["resp_corr"]`` (``--responses
    ciao``) multiply the predicted images / spectra. The output carries
    ``state0``, the full solver state at the FIRST epoch (``--save-state``;
    unused outputs are dead code under jit).

    The observation model, the solver config and the output assembly are
    ``make_forward_core`` (shared with ``casa_4dvar``); this adds the traced
    146-yr initial condition and the evolution to the first epoch.
    """
    ic = dict(np.load(ic_path))
    core = make_forward_core(ic, obs, img, cfl=cfl, ad_llf_cold=ad_llf_cold, subgrid=subgrid, opts=opts,
                             config_overrides=config_overrides, ic_path=ic_path)
    opts, config, rv = core.opts, core.config, core.rv
    # the traced functions read the IC fields and the geometry through H, so that
    # the multi-GPU path can pass them as sharded jit arguments (forward.lifted)
    H = SimpleNamespace(ic=ic, geom=core.geom)
    lifted = SH.Lifted()
    if SH.active():
        n3 = (int(ic["num_cells"]),) * 3
        H.ic = ic = dict(ic)
        # transform_fields' doppler_tracer(ic) is numpy: precomputed (identical values)
        ic[_C_DOP_KEY] = PD.doppler_tracer(ic)
        for k in list(ic):
            if np.ndim(ic[k]) == 3 and tuple(np.shape(ic[k])) == n3:
                lifted.add(ic, k, SH.FIELD)
        H.geom = SimpleNamespace(r=core.geom[0], X=core.geom[1], Y=core.geom[2], Z=core.geom[3])
        for k in ("r", "X", "Y", "Z"):
            lifted.add(H.geom, k, SH.FIELD)
        core.lift(lifted)

    def geom_of():
        g = H.geom
        return (g.r, g.X, g.Y, g.Z) if isinstance(g, SimpleNamespace) else g
    cell_vol, rho_c, yr = core.cell_vol, core.rho_c, core.yr
    age0 = float(ic["age"])
    # the similarity map (``SIM_NAMES``): traced only with opts.similarity "on"
    sim_on = getattr(opts, "similarity", "off") == "on"
    sim_ref = ic_similarity_ref(ic)

    def sim_of(p):
        """(ln L, ln T, ln M) of the map applied to THIS IC (None: the old path)."""
        return tuple(p[k] - sim_ref[k] for k in SIM_NAMES) if sim_on else None

    def age_ic(p):
        """The IC's age: ic["age"] (a float, the old path) or T_map x ic["age"] (traced)."""
        return PD.ic_age(ic, sim_of(p)) if sim_on else age0

    def head(p, dtype):
        a0 = age_ic(p)
        return jnp.array([a0], dtype=dtype) if not sim_on else jnp.reshape(a0, (1,)).astype(dtype)

    def pd_params(p):
        """p restricted to ``PD.PARAM_NAMES`` (PD.wind_nh reads SIM_NAMES from a dict)."""
        return {k: p[k] for k in PD.PARAM_NAMES}
    years = jnp.asarray(obs["years"])
    order = core.order
    xs_all = core.xs_all
    i_hist = core.i_hist
    # the untransformed IC, for the budget ratios
    ic0 = {k: np.asarray(ic[k], np.float64) for k in ("rho", "vx", "vy", "vz", "press", "C_ej")}
    budget0 = {k: float(v) for k, v in PD.ic_budget(ic0, cell_vol).items()}
    jet_on = getattr(opts, "jet", "off") == "on"

    def initial_fields(theta):
        p = dict(zip(PARAM_NAMES, theta))
        r, X, Y, Z = geom = geom_of()
        f = PD.transform_fields(H.ic, theta, geom, rho_c, extra=SCALAR_NAMES[1:] + HISTORY,
                                wind_dipole=opts.wind_dipole, sim=sim_of(p))
        if SH.active():
            f = {k: SH.cfield(v) for k, v in f.items()}
        # the field modes, on the ejecta only (after the rotation: they live
        # in the frame of the rotated explosion's current position)
        rs = jnp.maximum(r, 1e-3)
        nx, ny, nz = X / rs, Y / rs, Z / rs
        dln = sum(p[k] * fn(nx, ny, nz) for k, fn in YLM.items())
        cej = jnp.clip(f["C_ej"], 0.0, 1.0)
        rho = f["rho"]
        delta = jnp.zeros((), rho.dtype)
        if opts.ylm_mass == "neutral":
            # rho -> rho exp(C_ej (dln - delta)), delta such that the ejecta
            # mass sum(C_ej rho) is unchanged: Newton (3 steps: quadratic
            # convergence from 0; exact 0 when every mode is 0)
            m0 = jnp.sum(cej * rho)
            for _ in range(3):
                ex = rho * jnp.exp(cej * (dln - delta))
                delta = delta + (jnp.sum(cej * ex) - m0) / jnp.maximum(jnp.sum(cej ** 2 * ex), 1e-30)
        f["rho"] = rho * jnp.exp(cej * (dln - delta))
        if jet_on:
            # the NE jet + SW counter-jet (``casa_jet``), AFTER the similarity map, the
            # rotation and the Y_lm modes: in the frame the observer sees (jet_pa is
            # the sky PA through psi), with its mass exact; bookkeeping in f["_jet"]
            f = JT.add_jet(f, p, geom, age_ic(p) * yr, cell_vol)
        return f, p, delta

    def ic_diag(theta):
        f, p, delta = initial_fields(theta)
        out = {f"ic_{k}": v for k, v in PD.ic_budget(f, cell_vol).items()}
        out.update({f"jet_{k}": v for k, v in f.get("_jet", {}).items()})
        out.update(ylm_delta=delta,
                   t_conv=PD.ballistic_convergence_date(p["t_expl"], p["ln_sv"], age_ic(p)),
                   n_h_wind=PD.wind_nh(ic, pd_params(p), wind_dipole=opts.wind_dipole, sim=sim_of(p)))
        if sim_on:
            out.update(ic_age=age_ic(p), e_factor=PD.similarity_energy_factor(
                tuple(p[k] for k in SIM_NAMES)))
        return out

    def forward(theta):
        f, p, delta = initial_fields(theta)
        state = construct_primitive_state(
            config=config, registered_variables=rv, density=f["rho"],
            velocity_x=f["vx"], velocity_y=f["vy"], velocity_z=f["vz"], gas_pressure=f["press"],
            gamma=GAMMA, passive_scalars=jnp.stack([f[k] for k in SCALAR_NAMES]))
        # entropy_initial is seeded from the state; the rest is Orlando's history
        for j, name in enumerate(HISTORY):
            state = state.at[i_hist + 1 + j].set(f[name])
        state = SH.cstate(state)
        budget = PD.ic_budget(f, cell_vol)
        integrate = core.integrator(state.shape)
        ages = (years - p["t_expl"])[order]
        dts = jnp.diff(jnp.concatenate([head(p, ages.dtype), ages])) * yr
        observe = core.observer(p)

        state = integrate(state, dts[0])
        state0 = state
        first = observe(state, jax.tree.map(lambda a: a[0], xs_all))

        def segment(st, xs):
            dt, ep = xs
            st = integrate(st, dt)
            return st, observe(st, ep)

        _, rest = jax.lax.scan(segment, state, (dts[1:], jax.tree.map(lambda a: a[1:], xs_all)))
        model = core.assemble(p, tuple(jnp.concatenate([a[None], b], 0) for a, b in zip(first, rest)))
        return dict(model, t_conv=PD.ballistic_convergence_date(p["t_expl"], p["ln_sv"], age_ic(p)),
                    n_h_wind=PD.wind_nh(ic, pd_params(p), wind_dipole=opts.wind_dipole, sim=sim_of(p)),
                    ylm_delta=delta, state0=state0, **{f"ic_{k}": v for k, v in budget.items()},
                    **{f"jet_{k}": v for k, v in f.get("_jet", {}).items()},
                    **(dict(ic_age=age_ic(p)) if sim_on else {}))

    def evolve(theta):
        """(first-epoch state, the states of the LATER epochs (E - 1, n_var, n,
        n, n)), in SORTED epoch order: ``forward``'s evolution without the
        observation, so that one hydro run can be observed with several
        observation models (``casa_xfit_compare``)."""
        f, p, delta = initial_fields(theta)
        state = construct_primitive_state(
            config=config, registered_variables=rv, density=f["rho"],
            velocity_x=f["vx"], velocity_y=f["vy"], velocity_z=f["vz"], gas_pressure=f["press"],
            gamma=GAMMA, passive_scalars=jnp.stack([f[k] for k in SCALAR_NAMES]))
        for j, name in enumerate(HISTORY):
            state = state.at[i_hist + 1 + j].set(f[name])
        integrate = core.integrator(state.shape)
        ages = (years - p["t_expl"])[order]
        dts = jnp.diff(jnp.concatenate([head(p, ages.dtype), ages])) * yr
        state = integrate(state, dts[0])

        def segment(st, dt):
            st = integrate(st, dt)
            return st, st

        _, rest = jax.lax.scan(segment, state, dts[1:])
        return state, rest

    def first_state(theta, years=None, jitter=None):
        """The full solver state at the FIRST epoch only (``--state-only``: the
        ``--save-state`` file without the evolution through the later epochs and
        without any observation) and the first epoch's r_FS / r_RS profiles."""
        f, p, delta = initial_fields(theta)
        if jitter:              # diagnostics: relative density noise (rounding-sensitivity reference)
            f["rho"] = f["rho"] * (1.0 + jitter * jax.random.normal(jax.random.PRNGKey(0), f["rho"].shape,
                                                                     f["rho"].dtype))
        state = construct_primitive_state(
            config=config, registered_variables=rv, density=f["rho"],
            velocity_x=f["vx"], velocity_y=f["vy"], velocity_z=f["vz"], gas_pressure=f["press"],
            gamma=GAMMA, passive_scalars=jnp.stack([f[k] for k in SCALAR_NAMES]))
        for j, name in enumerate(HISTORY):
            state = state.at[i_hist + 1 + j].set(f[name])
        state = SH.cstate(state)
        integrate = core.integrator(state.shape)
        ages = (years_all - p["t_expl"])[order]
        dt0 = (ages[0] - age_ic(p)) * yr if years is None else jnp.asarray(years * yr, ages.dtype)
        state = integrate(state, dt0)
        r_fs, r_rs = core.observer(p)(state, jax.tree.map(lambda a: a[0], xs_all))[:2]
        return state, r_fs, r_rs

    forward.evolve = evolve
    forward.first_state = first_state
    years_all = years
    forward.ic_diag = ic_diag
    forward.sim_on, forward.sim_ref = sim_on, sim_ref
    forward.lifted = lifted
    forward.budget0 = budget0
    forward.opts = opts
    forward.rv = rv
    forward.ic = ic
    forward.core = core
    return forward
# =============================================================================
# ============ ↑ The forward model ↑ ==========================================
# =============================================================================


# =============================================================================
# ============ ↓ Likelihood ↓ =================================================
# =============================================================================
def image_residuals(model, img, args):
    """Static + temporal image residuals, per band and block.

    The model's knot-scale structure is wrong in the same way at every epoch
    (the 146-yr state fixes where the knots are), so comparing each epoch's
    image independently counts ONE static mismatch 15 times and lets it outvote
    every kinematic constraint (fit K: image chi2 -45 % while the outline,
    proper-motion and Doppler terms all degraded). The images are therefore
    split into what they measure:

    * static -- the exposure-weighted epoch-mean rate per block, as a log
      residual with model error ``sigma_static`` (plus Poisson);
    * temporal -- each epoch's block rate relative to that block's own mean,
      ln(n_e / n_mean) - ln(lambda_e / lambda_mean), with ``sigma_temporal``:
      expansion (rim blocks brighten or fade as the shock crosses them), the
      fading, the ionisation history -- and the knot placement cancels to
      first order. The epoch-dependent response is inside lambda_e.
    """
    expo = jnp.asarray(img["exposure"])[:, None, None, None]
    lam = (model["images"] + image_background(img)) * expo * jnp.asarray(img["pmask"])[:, None]
    b, nb = img["block"], img["npix"] // img["block"]
    lam = lam.reshape(*lam.shape[:-2], nb, b, nb, b).sum((-3, -1))              # (E, band, nb, nb)
    n = jnp.asarray(img["counts"])
    ok = jnp.asarray(img["bmask"])[:, None] & (jnp.asarray(img["bw"])[:, :, None, None] > 0)
    okf = ok.astype(lam.dtype)
    # static: exposure-weighted mean rate over the epochs a block is valid in
    N_tot = jnp.sum(n * okf, 0); L_tot = jnp.sum(lam * okf, 0)
    good = (jnp.sum(okf, 0) >= 3) & (N_tot > 25)
    # the logs take their argument through a where as well (same primal): a
    # masked block has L_tot = 0 and log(0)'s derivative, times the where's
    # zero cotangent, is NaN in REVERSE mode (casa_4dvar); JVP never saw it
    L_safe = jnp.where(good, L_tot, 1.0)
    r_static = jnp.where(good, (jnp.log(jnp.maximum(N_tot, 1.0)) - jnp.log(L_safe)) /
                         jnp.sqrt(args.sigma_static ** 2 + 1.0 / jnp.maximum(N_tot, 1.0)), 0.0)
    # temporal: per epoch, relative to the block's own mean
    E_tot = jnp.sum(expo[..., 0, 0][:, :, None, None] * okf, 0)                 # exposure sum
    n_rel = jnp.log(jnp.maximum(n, 1.0) / expo * E_tot / jnp.maximum(N_tot, 1.0))
    use = ok & good[None] & (n > 10)
    l_rel = jnp.log(jnp.where(use, lam / expo * E_tot / L_safe, 1.0))
    sig2 = args.sigma_temporal ** 2 + 1.0 / jnp.maximum(n, 1.0)
    d = jnp.where(use, n_rel - l_rel, 0.0)
    # per-(epoch, band) calibration amplitude, profiled analytically: the
    # response interpolation between tabulated cycles is not exact (fit M's
    # soft band dips 0.95 -> 0.81 -> 0.91 over the decade, the signature of
    # linear-in-time contamination), and an error common to every block must
    # not be fitted with the remnant's age. Prior widths: CALIB_SIGMA.
    w = jnp.where(use, 1.0 / sig2, 0.0)
    sa = jnp.asarray(CALIB_SIGMA)[None, :]
    a_hat = jnp.sum(w * d, (-2, -1)) / (jnp.sum(w, (-2, -1)) + 1.0 / sa ** 2)   # (E, band)
    r_temp = jnp.where(use, (d - a_hat[..., None, None]) / jnp.sqrt(sig2), 0.0)
    return r_static.ravel(), jnp.concatenate([r_temp.ravel(), (a_hat / sa).ravel()])


def jet_image_residuals(model, img, args):
    """{'jet_image': ...} with ``--jet-img on`` (else {}): the jets beyond the rim.

    The block image term stops at --r-max-img (150"), inside the forward shock,
    so it never sees the NE jet / SW counter-jet that pierce it; adding the
    jet sectors' 31.5" blocks instead puts the RIM into the temporal term (the
    shock crossing those blocks: +339 chi2 from 14 blocks on R3w, 2026-10-01).
    This term is the exposure-weighted epoch-mean rate per band in the sky-PA
    bins of an annulus BEYOND the rim (``casa_jet.JET_PROFILE``; on- and
    off-jet bins, so the jet's PA, width and brightness), as a log residual
    with model error ``casa_jet.JET_SIGMA`` plus Poisson -- static only (the
    model jet is a smooth stream, the real one knots)."""
    if "jet_B" not in img:
        return {}
    E = model["images"].shape[0]
    expo = jnp.asarray(img["exposure"])[:, None, None]
    lam_pix = (model["images"] + image_background(img)).reshape(E, len(J.BANDS), -1)
    lam = jnp.einsum("ebp,kp->ebk", lam_pix * jnp.asarray(img["jet_pm"])[:, None, :],
                     jnp.asarray(img["jet_B"])) * expo
    n = jnp.asarray(img["jet_counts"])
    okf = (jnp.asarray(img["bw"])[:, :, None] > 0).astype(lam.dtype)
    N_tot = jnp.sum(n * okf, 0); L_tot = jnp.sum(lam * okf, 0)
    good = N_tot > 25
    L_safe = jnp.where(good, jnp.maximum(L_tot, 1e-30), 1.0)
    r = jnp.where(good, (jnp.log(jnp.maximum(N_tot, 1.0)) - jnp.log(L_safe))
                  / jnp.sqrt(JT.JET_SIGMA ** 2 + 1.0 / jnp.maximum(N_tot, 1.0)), 0.0)
    return {"jet_image": r.ravel()}


#: multi-scale image term (``fine_image_residuals``; 2026-09-27), per fine block
#: size (px): the model error (ln) of a fine block's epoch-summed brightness
#: contrast to its parent 16-px block, and the error (ln variance) of the
#: per-sector amount of that structure -- each calibrated to chi2/N = 1 on the
#: best 128^3 model, 4D-Var(R2) no-warp (R2 itself: 0.56 / 0.43 for the fine
#: term). The 128^3 model has 0.32 of the data's fine-contrast variance at 4 px
#: (corr 0.42), 0.31 at 8 px; ln(V_model / V_data) per sector -1.28 +- 0.75
#: (casa_orlando150/work/ers/integ/calib_fine.json). Other sizes: the fallbacks.
SIGMA_FINE = {4: 0.51, 8: 0.39}
SIGMA_STRUCT = {4: 1.5, 8: 1.45}
SIGMA_FINE_DEFAULT, SIGMA_STRUCT_DEFAULT = 0.5, 1.5
FINE_MIN_COUNTS = 25.0


def fine_contrast(model, img):
    """(c_data, c_model, var_poisson, use): per band and fine block
    (``img['fine_block']`` px), the epoch-summed rate contrast to the parent
    likelihood block, ln(rate_fine / rate_parent), for the data and the model
    (exposure-weighted over the epochs the fine block is valid in; the parent
    rate is over its valid children). The parent amplitude cancels, so the term
    is orthogonal to ``image_residuals``' static one: it measures structure
    between the fine and the block scale (7.9-31.5" at the defaults)."""
    F, b, npix = int(img["fine_block"]), int(img["block"]), int(img["npix"])
    nf, nb, r = npix // F, npix // b, b // F
    expo = jnp.asarray(img["exposure"])[:, None, None, None]
    lam = (model["images"] + image_background(img)) * expo * jnp.asarray(img["pmask"])[:, None]
    lam = lam.reshape(*lam.shape[:-2], nf, F, nf, F).sum((-3, -1))                 # (E, band, nf, nf)
    n = jnp.asarray(img["counts_fine"])
    up = lambda a: jnp.repeat(jnp.repeat(a, r, -2), r, -1)                         # noqa: E731 parent -> children
    ok = (jnp.asarray(img["fmask"]) & up(jnp.asarray(img["bmask"])))[:, None] \
        & (jnp.asarray(img["bw"])[:, :, None, None] > 0)
    okf = ok.astype(lam.dtype)
    N = jnp.sum(n * okf, 0); Lm = jnp.sum(lam * okf, 0); Ex = jnp.sum(expo * okf, 0)   # (band, nf, nf)
    par = lambda a: up(a.reshape(*a.shape[:-2], nb, r, nb, r).sum((-3, -1)))       # noqa: E731
    Np, Lp, Ep = par(N), par(Lm), par(Ex)
    use = (jnp.sum(okf, 0) >= 3) & (N > FINE_MIN_COUNTS) & (Np > r * r * FINE_MIN_COUNTS) & (Lm > 0)
    one = lambda a: jnp.where(use, a, 1.0)                                         # noqa: E731 (safe logs)
    c_d = jnp.where(use, jnp.log(one(N) / one(Ex)) - jnp.log(one(Np) / one(Ep)), 0.0)
    c_m = jnp.where(use, jnp.log(one(Lm) / one(Ex)) - jnp.log(one(Lp) / one(Ep)), 0.0)
    # Poisson variance of ln N - ln N_parent: the child is PART of its parent,
    # cov(ln N, ln Np) = 1 / Np, so 1/N + 1/Np - 2/Np (review 2026-09-27; was + 1/Np;
    # SIGMA_FINE / SIGMA_STRUCT recalibrated with it: unchanged to 1e-4, integ2/calib_fine_v2.json)
    var_p = jnp.where(use, jnp.maximum(1.0 / one(N) - 1.0 / one(Np), 0.0), 0.0)
    return c_d, c_m, var_p, use


def fine_image_residuals(model, img, args):
    """The multi-scale image terms (dict; empty unless ``args.fine_block``):

    * ``image_fine`` -- ``fine_contrast``'s data - model contrast per band and
      fine block, / sqrt(sigma_fine^2 + Poisson), times sqrt(fine_weight);
    * ``image_struct`` (``args.struct_sectors`` > 0) -- per band and azimuthal
      sector about RA0/DEC0, ln of the model's fine-contrast variance over the
      data's (Poisson-bias corrected), / sigma_struct, times sqrt(fine_weight):
      the AMOUNT of 7.9-31.5" structure, which a chaotic knot field can match
      where the pixel-level placement cannot.

    Off by default: a 128^3 model has 4-26x less 5-15" power than the data
    (ers/filaments_RESULT.md), so these terms are for the 512^3 fits."""
    F = int(getattr(args, "fine_block", 0) or 0)
    if not F:
        return {}
    if int(img.get("fine_block", 0)) != F:
        raise ValueError(f"--fine-block {F}: load_image_data(..., fine_block={F}) (img has "
                         f"{img.get('fine_block', 0)})")
    c_d, c_m, var_p, use = fine_contrast(model, img)
    sig = getattr(args, "sigma_fine", None)
    sig = float(SIGMA_FINE.get(F, SIGMA_FINE_DEFAULT) if sig is None else sig)
    w = float(getattr(args, "fine_weight", None) or args.img_weight)
    out = {"image_fine": np.sqrt(w) * jnp.where(use, (c_d - c_m) / jnp.sqrt(sig ** 2 + var_p), 0.0).ravel()}
    n_sec = int(getattr(args, "struct_sectors", 0) or 0)
    if n_sec:
        xy = np.asarray(img["fine_xy"])
        sec = (np.floor((np.degrees(np.arctan2(xy[1], xy[0])) % 360.0) / (360.0 / n_sec))).astype(int) % n_sec
        oh = jnp.asarray(np.eye(n_sec)[sec], c_d.dtype)                           # (nf, nf, n_sec)
        u = use.astype(c_d.dtype)
        cnt = jnp.einsum("kij,ijs->ks", u, oh)
        vd = jnp.einsum("kij,ijs->ks", u * (c_d ** 2 - var_p), oh) / jnp.maximum(cnt, 1.0)
        vm = jnp.einsum("kij,ijs->ks", u * c_m ** 2, oh) / jnp.maximum(cnt, 1.0)
        good = cnt >= 8
        floor = 1e-4
        rs = (jnp.log(jnp.maximum(jnp.where(good, vm, 1.0), floor))
              - jnp.log(jnp.maximum(jnp.where(good, vd, 1.0), floor)))
        s_st = getattr(args, "sigma_struct", None)
        s_st = float(SIGMA_STRUCT.get(F, SIGMA_STRUCT_DEFAULT) if s_st is None else s_st)
        out["image_struct"] = np.sqrt(w) * jnp.where(good, rs / s_st, 0.0).ravel()
    return out


def spectral_use_mask(spec, opts):
    """(E, bin) bins in the spectral likelihood: epochs with a spectrum, n > 20,
    and -- with ``spec_acisi_soft="exclude"`` -- not an ACIS-I soft bin."""
    use = spec["has"][:, None] & (np.asarray(spec["counts"]) > 20)
    if getattr(opts, "spec_acisi_soft", "exclude") == "exclude":
        soft = SPEC_EDGES[1:] <= ACISI_SOFT_KEV + 1e-6
        use = use & ~(np.asarray(spec.get("acis_i", np.zeros(len(use), bool)))[:, None] & soft[None])
    return use


def image_background(img):
    """What the image likelihood adds to the model images (counts/s per pixel):
    the measured background (``--background``, ``attach_stage4``: (E, 6, npix,
    npix)) or the old floor BKG_RATE."""
    return img["bkg_img"] if "bkg_img" in img else BKG_RATE


def spectral_model_counts(model, sp):
    """Predicted counts per spectral bin (E, K): the model x (1 + the in-aperture
    out-of-time fraction) + the particle background (``--background``), times the
    exposure. Without a measured background: model x exposure (as before)."""
    s = model["spectra"]
    if "bkg_mult" in sp:
        s = s * sp["bkg_mult"]
    if "bkg_add" in sp:
        s = s + sp["bkg_add"]
    return s * jnp.asarray(sp["exposure"])[:, None]


def spectral_nuisance_columns(model, sp, opts, lam):
    """The stage-4 per-epoch spectral nuisances as [(name, prior sigma, (E, K)
    d ln lambda / d x_e)] (stop_gradient: fixed sensitivities):

    * gain (``--spec-gain profile``): an energy-scale error g shifts every
      recorded energy by (1 + g), i.e. the model's d(spectrum)/d(uniform line
      shift) (``spectra_dg``, from the D tables) over lambda;
    * soft (``--spec-soft profile``): a calibration factor exp(c) on the model
      in the bins with upper edge <= SOFT_KEV (the ACIS contamination).
    """
    out = []
    expo = jnp.asarray(sp["exposure"])[:, None]
    mult = sp.get("bkg_mult", 1.0)
    lam_safe = jnp.maximum(lam, 1e-30)
    if stage4(opts, "spec_gain") == "profile":
        if "spectra_dg" not in model:
            raise ValueError("--spec-gain profile: the model has no 'spectra_dg' (forward built with the "
                             "gain option off?)")
        out.append(("gain", GAIN_SIGMA, jax.lax.stop_gradient(model["spectra_dg"] * mult * expo / lam_safe)))
    if stage4(opts, "spec_soft") == "profile":
        soft = jnp.asarray((SPEC_EDGES[1:] <= SOFT_KEV + 1e-6).astype(np.float32))
        share = jax.lax.stop_gradient(model["spectra"] * mult * expo / lam_safe)
        out.append(("soft", SOFT_SIGMA, share * soft[None]))
    return out


def profile_linear(R_of, sig):
    """The minimiser of |R_of(x)|^2 + |x / sig|^2 for a LINEAR (affine) R_of,
    in closed form (Gauss-Newton is exact): returns (R_of(x_hat), x_hat / sig).
    Solved in z = x / sig (unit prior), matmuls at HIGHEST precision."""
    sig = jnp.asarray(sig)
    Rz = lambda z: R_of(z * sig)                                             # noqa: E731
    z0 = jnp.zeros(sig.shape, sig.dtype)
    R0 = Rz(z0)
    Jz = jax.jacfwd(Rz)(z0)                                                  # (N, n_x)
    hp = jax.lax.Precision.HIGHEST
    H = jnp.matmul(Jz.T, Jz, precision=hp) + jnp.eye(sig.shape[0], dtype=Jz.dtype)
    z = -jnp.linalg.solve(H, jnp.matmul(Jz.T, R0, precision=hp))
    return Rz(z), z


def _spec_shape(dd, use, w):
    """The per-epoch normalisation profiled out (the spectrum constrains SHAPE)."""
    a = jnp.sum(w * dd, -1) / jnp.maximum(jnp.sum(w, -1), 1e-30)
    return jnp.where(use, dd - a[..., None], 0.0)


def spectrum_setup(model, sp, opts, args):
    """The spectral likelihood's ingredients: lambda, n, use, sig2, d = ln n - ln
    lambda (masked), w, and the nuisance columns."""
    lam = spectral_model_counts(model, sp)
    n = jnp.asarray(sp["counts"])
    use = jnp.asarray(spectral_use_mask(sp, opts))
    sig2 = args.sigma_spec_temporal ** 2 + 1.0 / jnp.maximum(n, 1.0)
    d = jnp.where(use, jnp.log(jnp.maximum(n, 1.0)) - jnp.log(jnp.maximum(lam, 1e-30)), 0.0)
    w = jnp.where(use, 1.0 / sig2, 0.0)
    return SimpleNamespace(lam=lam, n=n, use=use, sig2=sig2, d=d, w=w,
                           cols=spectral_nuisance_columns(model, sp, opts, lam))


def spectrum_terms(model, sp, opts, args, info=None):
    """(spectrum_static, spectrum_temporal) residual vectors: per epoch the
    log residual with its normalisation profiled; static = the epoch-mean shape
    per bin (``sigma_spec_static``), temporal = each epoch's shape relative to
    it (``sigma_spec_temporal`` + Poisson). With stage-4 spectral nuisances
    (gain / soft) they are profiled JOINTLY over all epochs (the static shape
    couples them) in closed form, and their prior residuals x_e / sigma are
    appended to the temporal vector (as the image term's calibration)."""
    S = spectrum_setup(model, sp, opts, args)
    use, sig2, w = S.use, S.sig2, S.w

    def terms(dd):
        r = _spec_shape(dd, use, w)
        nk = jnp.maximum(jnp.sum(use, 0), 1)
        static = jnp.sum(r, 0) / nk
        return (jnp.where(jnp.sum(use, 0) > 0, static / args.sigma_spec_static, 0.0),
                jnp.where(use, (r - static[None]) / jnp.sqrt(sig2), 0.0))
    if not S.cols:
        st, tt = terms(S.d)
        if info is not None:
            info["static"] = st * args.sigma_spec_static
        return st, tt.ravel()
    E, K = S.d.shape
    Xc = jnp.stack([c for _, _, c in S.cols])                                  # (m, E, K)
    m = len(S.cols)

    def R_of(x):
        dd = S.d - jnp.where(use, jnp.sum(x.reshape(m, E)[:, :, None] * Xc, 0), 0.0)
        st, tt = terms(dd)
        return jnp.concatenate([st, tt.ravel()])
    sig = np.repeat([s_ for _, s_, _ in S.cols], E).astype(np.float32 if S.d.dtype == jnp.float32 else np.float64)
    R, z = profile_linear(R_of, sig)
    if info is not None:
        xh = z * jnp.asarray(sig)
        info.update({name: xh[k * E:(k + 1) * E] for k, (name, _, _) in enumerate(S.cols)})
        info["static"] = R[:K] * args.sigma_spec_static
    return R[:K], jnp.concatenate([R[K:], z])


def residual_parts(model, obs, img, theta, args):
    """Dict of residual vectors, one per likelihood term: 'outline', 'motion',
    ['rs'], 'prior' (the hydro Gaussian priors), ['conv_wall', 'wind_nh'],
    ['doppler'], 'image_static', 'image_temporal', ['spectrum_static',
    'spectrum_temporal'], 'prior_extra'. The fix options are ``args.opts``
    (``default_options``)."""
    opts = getattr(args, "opts", None) or default_options()
    legacy_priors = opts.priors == "legacy"
    parts = {}
    th_pd = theta[:len(PD.PARAM_NAMES)]
    parts.update(PD.residual_parts(
        model, dict(obs, use_doppler=False), th_pd, sigma_model_arcsec=args.sigma_model,
        use_rate=False, prior=PD.PRIOR_LEGACY if legacy_priors else PD.PRIOR,
        pm_scale_sigma=opts.pm_scale_sigma if opts.pm_scale_sigma > 0 else None,
        pm_target=opts.pm_target, rs_term=opts.rs_term,
        conv_wall=opts.conv_wall, wind_prior=opts.wind_prior))
    p = dict(zip(PARAM_NAMES, theta))
    if args.doppler:
        dop = obs.get("doppler", PD.DOPPLER)
        e = int(np.argmin(np.abs(np.asarray(obs["years"]) - float(dop["year"]))))
        n_dop = len(dop["v_kms"]); width = 360.0 / n_dop
        if opts.doppler_frame == "sky":
            vm = model["vlos_kms"][e]                 # already in the data's sky sectors
        else:
            centres = (np.arange(n_dop) + 0.5) * width
            vm = PD.periodic_interp(jnp.asarray(centres - 0.5 * width), model["vlos_kms"][e],
                                    jnp.asarray(centres - 0.5 * width) - p["psi"])
        vm = jnp.exp(p["ln_kdop"]) * (vm - vm.mean())
        sig = np.sqrt(dop["v_err_kms"] ** 2 + PD.DOPPLER_SIGMA_SYS ** 2)
        parts["doppler"] = (vm - jnp.asarray(dop["v_kms"])) / jnp.asarray(sig)
    r_s, r_t = image_residuals(model, img, args)
    parts["image_static"] = np.sqrt(args.img_weight) * r_s
    parts["image_temporal"] = np.sqrt(args.img_weight) * r_t
    parts.update(fine_image_residuals(model, img, args))      # multi-scale terms (--fine-block; default off)
    parts.update(jet_image_residuals(model, img, args))       # the jets beyond the rim (--jet-img on; default off)
    if args.spectra:
        parts["spectrum_static"], parts["spectrum_temporal"] = spectrum_terms(model, img["spec"], opts, args)
    ep = EXTRA_PRIOR_LEGACY if legacy_priors else EXTRA_PRIOR
    # b_sx only exists with --sync-trend on, the similarity priors only with the
    # similarity path (the vector is otherwise unchanged)
    sim_on = getattr(opts, "similarity", "off") == "on"
    jet_on = getattr(opts, "jet", "off") == "on"
    parts["prior_extra"] = jnp.array([(p[k] - ep[k][0]) / ep[k][1] for k in ep
                                      if (k != "b_sx" or stage4(opts, "sync_trend") == "on")
                                      and (k not in SIM_NAMES or sim_on)
                                      and (k not in JET_NAMES or jet_on)])
    return parts


def print_ic_diag(label, d, budget0=None):
    b = {k[3:]: v for k, v in d.items() if k.startswith("ic_")}
    print(f"[{label}] IC: {PD.format_budget(b, budget0)}; Y_lm mass renormalisation delta "
          f"{float(d.get('ylm_delta', 0.0)):+.4f}; ballistic date {float(d['t_conv']):.1f} "
          f"(>= {PD.T_CONV_MIN}); wind n_H({PD.WIND_NH_PRIOR[0]:.0f} pc) "
          f"{float(d['n_h_wind']):.3f} cm^-3 (Lee+14 {PD.WIND_NH_PRIOR[1]} +- {PD.WIND_NH_PRIOR[2]})"
          + (f"; similarity map: IC age {float(d['ic_age']):.2f} yr" if "ic_age" in d else "")
          + (f", E / E_Orlando {float(d['e_factor']):.4f}" if "e_factor" in d else "")
          + (f"; jet M_NE {float(d['jet_M_ne']):.4f} M_SW {float(d['jet_M_sw']):.4f} Msun, E_kin "
             f"{float(d['jet_E_kin_ne']) * 1e51:.3g} + {float(d['jet_E_kin_sw']) * 1e51:.3g} erg"
             + (f" (grid gains {float(d['jet_dE_grid']) * 1e51:.3g} erg)" if "jet_dE_grid" in d else "")
             if "jet_M_ne" in d else ""),
          flush=True)


def summarize(model, obs, img, theta, args, label, budget0=None):
    opts = getattr(args, "opts", None) or default_options()
    parts = {k: np.asarray(v) for k, v in residual_parts(model, obs, img, theta, args).items()}
    tot = sum(np.sum(v ** 2) for v in parts.values())
    print(f"[{label}] chi2: " + ", ".join(f"{k} {np.sum(v ** 2):.1f} / {v.size}" for k, v in parts.items())
          + f"; total {tot:.1f}", flush=True)
    hydro = sum(np.sum(parts[k] ** 2) for k in ("outline", "motion", "prior") if k in parts)
    print(f"[{label}] (old grouping: hydro = outline + motion + prior = {hydro:.1f})")
    for k in ("conv_wall", "wind_nh", "rs"):
        if k in parts:
            print(f"[{label}]   {k}: residual {float(parts[k][0]):+.3f}")
    if "motion" in parts and opts.pm_target == "registration" and opts.pm_scale_sigma > 0:
        print(f"[{label}]   PM-scale nuisance: eps/sigma = {float(parts['motion'][-1]):+.3f} "
              f"(1 + eps = {1 + float(parts['motion'][-1]) * opts.pm_scale_sigma:.3f})")
    pd_parts = {k: parts[k] for k in ("outline", "motion", "rs", "prior", "conv_wall", "wind_nh")
                if k in parts}
    PD.summarize(model, dict(obs, use_doppler=False), theta[:len(PD.PARAM_NAMES)], label,
                 args.sigma_model, parts=pd_parts, show_doppler=False, show_budget=False)
    print_ic_diag(label, model, budget0)
    dop = obs.get("doppler", PD.DOPPLER)
    if args.doppler and "v_kms" in dop:
        e = int(np.argmin(np.abs(np.asarray(obs["years"]) - float(dop["year"]))))
        p = dict(zip(PARAM_NAMES, theta))
        vm = np.asarray(model["vlos_kms"][e])
        if opts.doppler_frame != "sky":
            n_dop = len(vm); wd = 360.0 / n_dop
            vm = np.asarray(PD.periodic_interp(jnp.asarray(np.arange(n_dop) * wd), jnp.asarray(vm),
                                               jnp.asarray(np.arange(n_dop) * wd) - float(p["psi"])))
        vm = np.exp(float(p["ln_kdop"])) * (vm - vm.mean())
        vd = np.asarray(dop["v_kms"])
        print(f"[{label}] Doppler 2004 as in the likelihood ({opts.doppler_frame} frame, psi and "
              f"ln_kdop applied): model rms {vm.std():.0f} km/s, data rms {vd.std():.0f}; corr "
              f"{np.corrcoef(vm, vd)[0, 1]:+.2f}")
    mimg = np.asarray(model["images"]) + (np.asarray(img["bkg_img"]) if "bkg_img" in img else 0.0)
    lam = mimg * img["exposure"][:, None, None, None] * img["pmask"][:, None]
    n = img["counts"].sum((-2, -1))
    ratio = lam.sum((-2, -1)) / np.maximum(n, 1)
    print(f"[{label}] model/data counts per band (" + " ".join(f"{a:.1f}-{b:.1f}" for a, b in J.BANDS) + "):")
    sp = img["spec"]
    lam_s = np.asarray(spectral_model_counts(model, sp))
    hs = sp["has"]
    if args.spectra and (stage4(opts, "spec_gain") != "off" or stage4(opts, "spec_soft") != "off"):
        info = {}
        spectrum_terms(model, sp, opts, args, info=info)
        for k, sg_ in (("gain", GAIN_SIGMA), ("soft", SOFT_SIGMA)):
            if k in info:
                x = np.asarray(info[k])
                print(f"[{label}] profiled spectral {k} (%, prior sd {100 * sg_:g}): " + " ".join(
                    f"{lab}:{100 * v:+.2f}" for lab, v, h in zip(obs["epochs"], x, hs) if h), flush=True)
    if hs.any():
        use = spectral_use_mask(sp, opts)
        rat = np.where(use, sp["counts"] / np.maximum(lam_s, 1e-30), np.nan)[hs]
        rat = rat / np.nanmedian(rat, axis=1, keepdims=True)
        mids = 0.5 * (SPEC_EDGES[1:] + SPEC_EDGES[:-1])
        print(f"[{label}] spectrum data/model shape (epoch-mean over the bins in the likelihood, "
              f"normalised), per 0.2 keV bin from {SPEC_EDGES[0]} keV:\n    "
              + " ".join(f"{m:.1f}:{x:.2f}" for m, x in zip(mids, np.nanmean(rat, 0))))
        lr = np.log(rat)
        print(f"[{label}] spectrum shape rms |ln(data/model)| per epoch: " + ", ".join(
            f"{lab} {np.sqrt(np.nanmean(x ** 2)):.3f}" for lab, x in zip(np.asarray(obs["epochs"])[hs], lr))
            + f"; all {np.sqrt(np.nanmean(lr ** 2)):.3f}")
    sf = np.asarray(model["sync_frac"])
    for e, lab in enumerate(obs["epochs"]):
        print(f"    {lab}: " + " ".join(f"{x:5.2f}" for x in ratio[e]) +
              "   sync " + " ".join(f"{x:.2f}" for x in sf[e]) +
              f"   [{', '.join(f'{i[8:]} {w:.2f}' for i, w in img['resp'][e])}]")
    lnh = float(dict(zip(PARAM_NAMES, theta))["ln_nh"])
    print(f"[{label}] N_H grid {img.get('nh_grid')} x 1e22 ({opts.nh_mode} outside); centre "
          f"N_H {np.exp(lnh):.2f}")
    return parts
# =============================================================================
# ============ ↑ Likelihood ↑ =================================================
# =============================================================================


# =============================================================================
# ============ ↓ Epoch exclusion (held-out validation) ↓ ======================
# =============================================================================
# ``--exclude-epochs 2019 2022``: the forward model still evolves through and
# observes EVERY epoch (``--save-model`` keeps all of them, so the held-out
# predictions are saved), but every likelihood term sees only the training
# epochs: the outline means, the images' static / temporal split, the
# spectra's static / temporal split, the Doppler term (dropped if its epoch is
# excluded) and the registration proper motions, which are re-fitted from the
# saved per-epoch shifts without the excluded epochs (``refit_pm``) and
# compared with the model's r_FS slope over the training epochs. The held-out
# epochs are then scored against the training set (``heldout_parts``).
# (``casa_4dvar_data`` re-exports these helpers; they live here so that
# ``casa_xfit`` does not import the module that imports it.)

#: per-epoch keys (leading axis = epoch) of the casa_xfit data dicts
OBS_EPOCH_KEYS = ("years", "r", "sigma", "mask")
IMG_EPOCH_KEYS = ("counts", "exposure", "pmask", "bmask", "bw", "inst_w", "resp_corr", "Cb", "Db", "Sb",
                  "bkg_img", "counts_fine", "fmask", "jet_counts", "jet_pm")
SPEC_EPOCH_KEYS = ("counts", "exposure", "has", "inst_w", "acis_i", "resp_corr", "bkg_add", "bkg_mult")
#: forward-model entries that are NOT per epoch (``state0``'s leading axis is
#: the variable axis, which can coincide with the number of epochs)
MODEL_STATIC_KEYS = ("state0",)


def refit_pm(obs, pm_files, pm_mask, exclude):
    """The registration proper motions of ``PD.load_observations`` re-fitted from
    the saved per-epoch shifts WITHOUT the epochs ``exclude`` (same recipe:
    corr > 0.5, >= 6 epochs per cone, the two windows agreeing to 0.1"/yr, the
    cone mask, PM_SIGMA_SYS). The stored PMs are 2000-2022 slopes, i.e. they
    contain the held-out 2019 / 2022 images (rms change 0.017-0.019"/yr per
    cone, ~0.55 of the total PM sigma)."""
    fa, fb = PD.PM_FILES[pm_files]
    pms = []
    for f in (fa, fb):
        d = np.load(f)
        yrs, sh, good = np.asarray(d["years"]), np.asarray(d["shifts"]), np.asarray(d["corr"]) > 0.5
        keep = np.array([str(e) not in exclude for e in d["epochs"]])
        pm = np.full(sh.shape[1], np.nan); err = np.full(sh.shape[1], np.nan)
        for k in range(sh.shape[1]):
            ok = good[:, k] & keep
            if ok.sum() >= 6:
                c, cov = np.polyfit(yrs[ok], sh[ok, k], 1, cov=True)
                pm[k], err[k] = c[0], np.sqrt(cov[0, 0])
        pms.append((pm, err))
    (a, ea), (b, _) = pms
    stable = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) < 0.1)
    out = dict(obs)
    # + the stage-4 extra cone mask (--pm-mask-extra), recorded by apply_obs_masks
    masked = PD.cone_mask(obs["angles"], pm_mask) | cone_angle_mask(obs["angles"],
                                                                    obs.get("pm_mask_extra_cones", ()))
    out["pm"] = np.where(stable & ~masked, a, np.nan)
    out["pm_sigma"] = np.sqrt(np.nan_to_num(ea, nan=1.0) ** 2 + PD.PM_SIGMA_SYS ** 2)
    out["pm_source"] = f"{obs.get('pm_source', pm_files)}; refitted without {sorted(exclude)}"
    return out


def subset_epochs(obs, img, idx):
    """The data dicts restricted to the epochs ``idx`` (indices into obs['epochs'],
    kept in their order)."""
    idx = np.asarray(idx, int)
    E = len(obs["epochs"])

    def take(a):
        if isinstance(a, list):
            return [a[i] for i in idx]
        if hasattr(a, "shape") and a.ndim >= 1 and a.shape[0] == E:
            return a[jnp.asarray(idx)] if not isinstance(a, np.ndarray) else a[idx]
        raise ValueError(f"not a per-epoch array: {type(a)} {getattr(a, 'shape', None)}")
    o = dict(obs)
    o["epochs"] = [obs["epochs"][i] for i in idx]
    for k in OBS_EPOCH_KEYS:
        o[k] = take(obs[k])
    im = dict(img)
    for k in IMG_EPOCH_KEYS:
        if k in img:
            im[k] = take(img[k])
    im["resp"] = [img["resp"][i] for i in idx]
    sp = dict(img["spec"])
    for k in SPEC_EPOCH_KEYS:
        if k in sp:
            sp[k] = take(sp[k])
    im["spec"] = sp
    return o, im


def subset_model(model, idx, n_epochs):
    """The forward model's per-epoch entries (leading axis ``n_epochs``, except
    ``MODEL_STATIC_KEYS``) restricted to the epochs ``idx``."""
    idx = np.asarray(idx, int)
    return {k: v[idx] if (k not in MODEL_STATIC_KEYS and getattr(v, "ndim", 0) >= 1
                          and v.shape[0] == n_epochs) else v
            for k, v in model.items()}


def heldout_outline_keep(obs, train, e, jump=HELDOUT_JUMP_ARCSEC):
    """(n_cone,) bool: the held-out epoch ``e``'s cones that are NOT detector jumps
    (``HELDOUT_JUMP_ARCSEC`` from the cone's training linear trend; cones with
    < 4 training radii are kept). Data only: the same cut for every model."""
    r = np.asarray(obs["r"], np.float64)
    yrs = np.asarray(obs["years"], np.float64)
    tr = np.asarray(train)
    keep = np.ones(r.shape[1], bool)
    for k in range(r.shape[1]):
        ok = np.isfinite(r[tr, k])
        if ok.sum() >= 4 and np.isfinite(r[e, k]):
            c = np.polyfit(yrs[tr][ok], r[tr, k][ok], 1)
            keep[k] = abs(r[e, k] - np.polyval(c, yrs[e])) <= jump
    return keep


def outline_sigma(obs, train, floor=1.0):
    """Per-cone outline sigma (E, n_cone) from the TRAINING epochs only:
    ``casa_pluto_diff.load_observations``' recipe (scatter of each cone's radii
    about its own linear trend, >= 4 epochs, floored; 5" if undetermined) on
    the epochs ``train``."""
    r = np.asarray(obs["r"], np.float64)[np.asarray(train)]
    yrs = np.asarray(obs["years"], np.float64)[np.asarray(train)]
    resid = np.full_like(r, np.nan)
    for k in range(r.shape[1]):
        ok = np.isfinite(r[:, k])
        if ok.sum() >= 4:
            c = np.polyfit(yrs[ok], r[ok, k], 1)
            resid[ok, k] = r[ok, k] - np.polyval(c, yrs[ok])
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(resid).sum(0)
        sig = np.sqrt(np.nansum(resid ** 2, 0) / np.maximum(cnt, 1))
    sig = np.where(cnt > 0, sig, np.nan)
    sig = np.maximum(np.nan_to_num(sig, nan=5.0), floor)
    return np.broadcast_to(sig, np.asarray(obs["r"]).shape).copy()


def heldout_parts(model, obs, img, train, hold, args):
    """Residual vectors of the held-out epochs ``hold`` against the training
    epochs ``train`` (indices into the all-epoch ``model`` / data):
    'h_image_temporal' (with its profiled calibration), 'h_spectrum',
    'h_outline_rel' (per-cone radius minus the training mean offset, over the
    data's per-cone sigma) and 'h_outline_abs' (over sigma_model). Per held-out
    epoch as well ('h_image_temporal_<label>' ...)."""
    tr, ho = np.asarray(train), np.asarray(hold)
    expo = jnp.asarray(img["exposure"])[:, None, None, None]
    lam = (model["images"] + image_background(img)) * expo * jnp.asarray(img["pmask"])[:, None]
    b, nb = img["block"], img["npix"] // img["block"]
    lam = lam.reshape(*lam.shape[:-2], nb, b, nb, b).sum((-3, -1))
    n = jnp.asarray(img["counts"])
    ok = jnp.asarray(img["bmask"])[:, None] & (jnp.asarray(img["bw"])[:, :, None, None] > 0)
    okf = ok.astype(lam.dtype)
    N_tot = jnp.sum((n * okf)[tr], 0); L_tot = jnp.sum((lam * okf)[tr], 0)
    good = (jnp.sum(okf[tr], 0) >= 3) & (N_tot > 25)
    E_tot = jnp.sum((expo[..., 0, 0][:, :, None, None] * okf)[tr], 0)
    out = {}
    sa = jnp.asarray(CALIB_SIGMA)[None, :]
    rt_all, sp_all, ol_rel, ol_abs = [], [], [], []
    # spectra: the training-mean profiled shape
    sp = img["spec"]
    S = spectrum_setup(model, sp, args.opts, args)
    use_s, sig2_s, d_s, w_s = S.use, S.sig2, S.d, S.w
    if not S.cols:
        a_s = jnp.sum(w_s * d_s, 1) / jnp.maximum(jnp.sum(w_s, 1), 1e-30)
        r_s = jnp.where(use_s, d_s - a_s[:, None], 0.0)
        nk = jnp.maximum(jnp.sum(use_s[tr], 0), 1)
        static_s = jnp.sum(r_s[tr], 0) / nk
    else:
        # the training static shape with the training epochs' nuisances profiled
        # (spectrum_terms on the training subset); each held-out epoch's own
        # gain / soft factor is then profiled against it (a per-epoch calibration
        # is not predictable, as the image term's a_hat)
        info = {}
        sp_tr = {k: (v[jnp.asarray(tr)] if not isinstance(v, np.ndarray) else v[tr])
                 if k in SPEC_EPOCH_KEYS else v for k, v in sp.items()}
        spectrum_terms(subset_model(model, tr, len(obs["epochs"])), sp_tr, args.opts, args, info=info)
        static_s = info["static"]
    # outline: training mean offset per cone; with --heldout-sigma train the
    # per-cone sigma is re-estimated from the TRAINING epochs only (the stored
    # one is the scatter about a 22-yr line through every epoch, held-out
    # ones included)
    m = np.asarray(obs["mask"])
    hs = stage4(args.opts, "heldout_sigma")
    if hs not in ("train", "train-nocut", "data"):
        raise ValueError(f"heldout_sigma {hs!r}")
    sig_o = outline_sigma(obs, tr) if hs.startswith("train") else np.asarray(obs["sigma"])
    d_o = jnp.where(jnp.asarray(m), model["r_fs_arcsec"] - jnp.asarray(np.nan_to_num(obs["r"])), 0.0)
    n_tr = np.maximum(m[tr].sum(0), 1)
    mean_tr = jnp.sum(d_o[tr], 0) / n_tr
    for e in ho:
        lab = obs["epochs"][e]
        n_rel = jnp.log(jnp.maximum(n[e], 1.0) / expo[e] * E_tot / jnp.maximum(N_tot, 1.0))
        use = ok[e] & good & (n[e] > 10)
        l_rel = jnp.log(jnp.where(use, lam[e] / expo[e] * E_tot / jnp.where(good, L_tot, 1.0), 1.0))
        sig2 = args.sigma_temporal ** 2 + 1.0 / jnp.maximum(n[e], 1.0)
        d = jnp.where(use, n_rel - l_rel, 0.0)
        w = jnp.where(use, 1.0 / sig2, 0.0)
        a_hat = jnp.sum(w * d, (-2, -1)) / (jnp.sum(w, (-2, -1)) + 1.0 / sa[0] ** 2)
        r_t = jnp.concatenate([jnp.where(use, (d - a_hat[:, None, None]) / jnp.sqrt(sig2), 0.0).ravel(),
                               a_hat / sa[0]]) * np.sqrt(args.img_weight)
        out[f"h_image_temporal_{lab}"] = r_t
        rt_all.append(r_t)
        if bool(np.asarray(sp["has"])[e]):
            if not S.cols:
                r_sp = jnp.where(use_s[e], (r_s[e] - static_s) / jnp.sqrt(sig2_s[e]), 0.0)
            else:
                def R_e(x, e=e):
                    dd = d_s[e] - jnp.where(use_s[e], sum(x[k] * c[e] for k, (_, _, c) in enumerate(S.cols)), 0.0)
                    r = _spec_shape(dd, use_s[e], w_s[e])
                    return jnp.where(use_s[e], (r - static_s) / jnp.sqrt(sig2_s[e]), 0.0)
                R, z = profile_linear(R_e, np.array([s_ for _, s_, _ in S.cols], np.float32
                                                    if d_s.dtype == jnp.float32 else np.float64))
                r_sp = jnp.concatenate([R, z])
            out[f"h_spectrum_{lab}"] = r_sp
            sp_all.append(r_sp)
        me = m[e]
        if hs == "train":
            me = me & heldout_outline_keep(obs, tr, e)
        rr = ((d_o[e] - mean_tr) / jnp.asarray(sig_o[e]))[me]
        ra = (d_o[e] / args.sigma_model)[me]
        out[f"h_outline_rel_{lab}"] = rr
        out[f"h_outline_abs_{lab}"] = ra
        ol_rel.append(rr); ol_abs.append(ra)
    out["h_image_temporal"] = jnp.concatenate(rt_all)
    if sp_all:
        out["h_spectrum"] = jnp.concatenate(sp_all)
    out["h_outline_rel"] = jnp.concatenate(ol_rel)
    out["h_outline_abs"] = jnp.concatenate(ol_abs)
    return out


#: the held-out terms summed into 'heldout_total' (casa_4dvar's Validator)
HELDOUT_TOTAL_TERMS = ("h_image_temporal", "h_spectrum", "h_outline_rel")


class EpochExclusion:
    """``--exclude-epochs``: the training-epoch likelihood (``parts``) and the
    all-epoch / held-out evaluation (``evaluate``) around the all-epoch forward
    model."""

    def __init__(self, obs, img, exclude, args, opts):
        labels = [str(e) for e in obs["epochs"]]
        exclude = sorted({str(e) for e in exclude})
        unknown = sorted(set(exclude) - set(labels))
        if unknown:
            raise SystemExit(f"--exclude-epochs: unknown epochs {unknown} (have {labels})")
        hold = np.array([lab in exclude for lab in labels])
        self.exclude, self.E = exclude, len(labels)
        self.train, self.hold = np.nonzero(~hold)[0], np.nonzero(hold)[0]
        self.obs_all, self.img_all, self.args_all = obs, img, args
        obs_tr = obs
        if opts.pm_target == "registration" and "pm" in obs:
            n0 = int(np.isfinite(obs["pm"]).sum())
            obs_tr = refit_pm(obs, opts.pm_files, opts.pm_mask, set(exclude))
            dpm = np.asarray(obs_tr["pm"]) - np.asarray(obs["pm"])
            print(f"[xfit] --exclude-epochs {' '.join(exclude)}: proper motions refitted without them: "
                  f"{n0} -> {int(np.isfinite(obs_tr['pm']).sum())} cones, rms change "
                  f"{np.sqrt(np.nanmean(dpm ** 2)):.4f}\"/yr", flush=True)
        else:
            print(f"[xfit] WARNING --exclude-epochs with pm_target={opts.pm_target}: the literature "
                  f"expansion rates cannot be refitted without the excluded epochs", flush=True)
        self.args = SimpleNamespace(**vars(args))
        dop = obs.get("doppler", PD.DOPPLER)
        if args.doppler and "v_kms" in dop:
            e_dop = int(np.argmin(np.abs(np.asarray(obs["years"]) - float(dop["year"]))))
            if hold[e_dop]:
                print(f"[xfit] --exclude-epochs: the Doppler epoch {labels[e_dop]} is excluded; "
                      f"Doppler term dropped", flush=True)
                self.args.doppler = False
        self.obs, self.img = subset_epochs(obs_tr, img, self.train)
        print(f"[xfit] --exclude-epochs: likelihood on {len(self.train)} training epochs "
              f"{', '.join(self.obs['epochs'])}; held out {', '.join(exclude)} (evolved and saved)",
              flush=True)

    def model(self, model):
        return subset_model(model, self.train, self.E)

    def parts(self, model, theta):
        """The likelihood's residual parts on the training epochs."""
        return residual_parts(self.model(model), self.obs, self.img, theta, self.args)

    def evaluate(self, model, theta, label):
        """chi2 per term: 'train' (the objective), 'all' (every epoch with the
        stored all-epoch proper motions, comparable with a fit without
        exclusion) and 'heldout' (``heldout_parts``). Printed and returned."""
        c2 = lambda d: {k: float(jnp.sum(jnp.asarray(v) ** 2)) for k, v in d.items()}  # noqa: E731
        tr = c2(self.parts(model, theta))
        al = c2(residual_parts(model, self.obs_all, self.img_all, theta, self.args_all))
        hp = heldout_parts(model, self.obs_all, self.img_all, self.train, self.hold, self.args_all)
        ho = c2(hp)
        out = dict(train=tr, train_total=sum(tr.values()), all=al, all_total=sum(al.values()),
                   heldout=ho, heldout_total=sum(ho[k] for k in HELDOUT_TOTAL_TERMS if k in ho),
                   heldout_size={k: int(np.size(v)) for k, v in hp.items()})
        print(f"[{label} train] chi2 total {out['train_total']:.1f}: "
              + ", ".join(f"{k} {v:.1f}" for k, v in tr.items()), flush=True)
        print(f"[{label} all epochs, stored PMs] chi2 total {out['all_total']:.1f}: "
              + ", ".join(f"{k} {v:.1f}" for k, v in al.items()), flush=True)
        print(f"[{label} held-out {' '.join(self.exclude)} vs training] total "
              f"({' + '.join(HELDOUT_TOTAL_TERMS)}) {out['heldout_total']:.1f}: "
              + ", ".join(f"{k} {v:.1f}" for k, v in ho.items()), flush=True)
        return out
# =============================================================================
# ============ ↑ Epoch exclusion (held-out validation) ↑ ======================
# =============================================================================


JAC_DUMP = None


def device_batch_runner(fun, n_dev):
    """``thetas (k <= n_dev, P) -> (k, R)``: ONE jitted executable per device,
    each theta committed to its own device and dispatched asynchronously (the
    k runs overlap; the first call compiles once per device).

    The replacement for ``jax.pmap(fun)`` (``--devices-mode jit``, default):
    on 2026-09-27 the pmapped casa_xfit forward returned wrong, nearly
    theta-independent hydro on every device but the first (refit R3's first
    Jacobian: every +-pair on devices 1 / 2 differed by 0 for t_expl, rot_z,
    ln_A, every pair with device 0 by the same ~600 outline-sigma garbage;
    integ/REPORT.md), while jit on one device reproduced the start forward."""
    jf = jax.jit(fun)
    devs = list(jax.local_devices()[:n_dev])

    def run(thetas):
        outs = []
        for b in range(0, len(thetas), len(devs)):         # more thetas than (healthy) devices: rounds
            outs += [jf(jax.device_put(t, d)) for t, d in zip(thetas[b:b + len(devs)], devs)]
        return np.stack([np.asarray(o) for o in outs])

    def health_check(theta, tol=0.05):
        """The central theta on EVERY device; returns a healthy device's residual.

        Devices are compared pairwise (rms in sigma units; device-to-device
        rounding chaos is 0.002-0.007): the reference is the device that agrees
        (rms <= ``tol``) with the most others (ties: the first), and every device
        that disagrees with it is dropped from ``devs`` for the rest of this
        runner. (Review 2026-09-27: comparing with device 0 only would drop the
        HEALTHY devices if device 0 were the corrupt one.) With no agreeing pair
        among >= 2 devices the check cannot tell, keeps only the first device and
        says so."""
        ref = [r.astype(np.float64) for r in run([theta] * len(devs))]
        k = len(ref)
        D = np.array([[float(np.sqrt(np.nanmean((ref[i] - ref[j]) ** 2))) for j in range(k)] for i in range(k)])
        agree = [sum(1 for j in range(k) if j != i and D[i, j] <= tol) for i in range(k)]
        i0 = int(np.argmax(agree))
        if k > 1 and agree[i0] == 0:
            print("[jac] device health: NO two devices agree (rms matrix "
                  + np.array2string(D, precision=3) + f"); keeping device {devs[0].id} only", flush=True)
            i0 = 0
        bad = [d for j, d in enumerate(list(devs)) if j != i0 and not (D[i0, j] <= tol)]
        print(f"[jac] device health (central theta, rms vs device {devs[i0].id}): " + ", ".join(
            f"{d.id}:{e:.3g}" for d, e in zip(devs, D[i0])) + (f"; DROPPING {[d.id for d in bad]}" if bad else ""),
            flush=True)
        for d in bad:
            devs.remove(d)
        return ref[i0]
    run.health_check = health_check
    return run


def jacobian_fd_parallel(fun, theta, steps, free, n_dev, mode="jit"):
    """Central FD Jacobian with the perturbed runs spread over ``n_dev`` GPUs.

    Every perturbed forward pass is an independent 200-yr simulation, so they
    are batched over the devices (each device integrates its own theta; the
    adaptive time loops need no communication): ``mode`` "jit" (default, one
    executable per device, ``device_batch_runner``) or "pmap" (the old
    ``jax.pmap``). Frozen parameters are skipped.
    """
    pmapped = jax.pmap(fun) if mode == "pmap" else None
    runner = device_batch_runner(fun, n_dev) if mode != "pmap" else None
    pf = (lambda ths: pmapped(ths)) if mode == "pmap" else (lambda ths: runner(list(ths)))
    centre = runner.health_check(theta) if runner is not None else None
    idx = [i for i in range(len(theta)) if free[i]]
    thetas = [theta] + [theta + s * jnp.zeros_like(theta).at[i].set(steps[i]) for i in idx for s in (1, -1)]
    if centre is None:
        while len(thetas) % n_dev:
            thetas.append(theta)                                # pad the last batch (pmap)
        out = []
        for b in range(0, len(thetas), n_dev):
            out.append(np.asarray(pf(jnp.stack(thetas[b:b + n_dev]))))
        out = np.concatenate(out)
    else:                   # the health check's (healthy) central run is out[0]; no padding needed
        rest = np.asarray(pf(thetas[1:]))
        out = np.concatenate([centre[None].astype(rest.dtype), rest])
    # NON-FINITE RUNS. Fit s1-fitfix-lm1 (2026-09-25): ONE perturbed run (wa_y)
    # came back NaN in ~5000 rows under pmap, although the same two thetas
    # evaluate finite as single forwards (s1-fitfix-nanprobe) -- a
    # non-deterministic blow-up, not a derivative. Re-run such runs once
    # (same pmapped executable: no recompile) before the LM loop freezes the
    # column.
    redo = [j for j in range(len(idx) * 2 + 1) if not np.all(np.isfinite(out[j]))]
    if redo:
        print(f"[jac] {len(redo)} run(s) non-finite ({[('centre' if j == 0 else PARAM_NAMES[idx[(j - 1) // 2]] + ('+' if j % 2 else '-')) for j in redo]}); re-running once", flush=True)
        for b in range(0, len(redo), n_dev):
            js = redo[b:b + n_dev]
            batch = [thetas[j] for j in js] + [theta] * (n_dev - len(js))
            res = np.asarray(pf(jnp.stack(batch)))
            for q, j in enumerate(js):
                out[j] = res[q]
        still = [j for j in redo if not np.all(np.isfinite(out[j]))]
        print(f"[jac] after the re-run {len(still)} run(s) still non-finite", flush=True)
    val = out[0]
    Jm = np.zeros((val.size, len(theta)))
    dr = np.zeros((val.size, len(idx)))
    for k, i in enumerate(idx):
        dr[:, k] = out[1 + 2 * k] - out[2 + 2 * k]
        Jm[:, i] = dr[:, k] / (2 * steps[i])
    # CHAOTIC ROWS. The forward pass is not bitwise reproducible (GPU atomics)
    # and 22 yr of Rayleigh-Taylor evolution amplifies that: a residual that
    # changes by > 1 sigma under the SMALLEST perturbations is noise, not a
    # derivative, and a few such rows in J^T J shrank every step of fit O to
    # ~1e-6. Rows whose median |Delta r| over the parameters exceeds 1 are
    # dropped from the Jacobian (kept in the residual).
    noisy = np.nanmedian(np.abs(dr), axis=1) > 1.0
    if noisy.any():
        print(f"[jac] {int(noisy.sum())} of {val.size} residual rows are perturbation-noise "
              f"(median |dr| > 1): excluded from J", flush=True)
        Jm[noisy] = 0.0
    if JAC_DUMP:
        np.savez_compressed(JAC_DUMP, J=Jm.astype(np.float32), dr=dr.astype(np.float32), val=val,
                            theta=np.asarray(theta), idx=np.array(idx))
    return val, Jm


def add_fine_arguments(ap):
    """The multi-scale image likelihood flags (``fine_image_residuals``; default off)."""
    g = ap.add_argument_group("multi-scale image likelihood (default off; for 512^3 fits)")
    g.add_argument("--fine-block", type=int, default=0,
                   help="fine block (1.97\" pixels; a divisor of --block, e.g. 4 = 7.9\" or 8 = 15.7\"): add the "
                        "epoch-summed brightness contrast of each fine block to its parent block (image_fine)")
    g.add_argument("--sigma-fine", type=float, default=None,
                   help="model error (ln) of the fine contrast (default SIGMA_FINE[--fine-block]: chi2/N = 1 on "
                        "4D-Var(R2), 0.51 at 4 px, 0.39 at 8 px)")
    g.add_argument("--fine-weight", type=float, default=None, help="weight of the fine terms (default --img-weight)")
    g.add_argument("--struct-sectors", type=int, default=0,
                   help="with --fine-block: per band and azimuthal sector (this many), ln(model / data) of the "
                        "fine-contrast variance (image_struct: the amount of structure, not its placement)")
    g.add_argument("--sigma-struct", type=float, default=None,
                   help="error (ln variance) of image_struct (default SIGMA_STRUCT[--fine-block]: chi2/N = 1 on "
                        "4D-Var(R2), 1.5 at 4 px; tighten (e.g. 0.5) at 512^3)")


def add_fix_arguments(ap):
    """The audit-fix flags (default None = the corrected value, or the legacy
    one under --legacy)."""
    g = ap.add_argument_group("2026-09-25 audit fixes (defaults = corrected; --legacy = before)")
    g.add_argument("--legacy", action="store_true",
                   help="every fix at its pre-2026-09-25 value (the likelihood of fits A-Q2)")
    g.add_argument("--rs-estimator", choices=PD.RS_ESTIMATORS, default=None,
                   help="reverse shock: outer edge of the unshocked ejecta (unshocked, default), "
                        "of the cold ejecta (coldej), or the old artefact (legacy)")
    g.add_argument("--rs-term", action="store_true",
                   help="add r_RS / r_FS = 0.66 +- 0.05 (lit_obs R7) to the likelihood")
    g.add_argument("--centre", choices=("sky", "grid"), default=None,
                   help="outline about RA0/DEC0 with the model at (dw, dn) (sky) or about the "
                        "model centre (grid, legacy)")
    g.add_argument("--recentre-iter", type=int, default=2,
                   help="fixed-point refinements of the re-centring (0 = first order)")
    g.add_argument("--doppler-frame", choices=("sky", "sim"), default=None,
                   help="Doppler sectors in the traced sky frame about RA0/DEC0 (sky) or in the "
                        "sim frame at D_ref rotated by psi (sim, legacy)")
    g.add_argument("--priors", choices=("new", "legacy"), default=None,
                   help="new: (dw, dn) on the expansion centre +- 1.5\", D ln-normal 3.4 +- 0.15 kpc")
    g.add_argument("--conv-wall", choices=("on", "off"), default=None,
                   help="one-sided wall t_expl + 145.5 (1 - 1/s_v) >= 1671.3")
    g.add_argument("--wind-prior", choices=("on", "off"), default=None,
                   help="pre-shock n_H(3 pc) = 0.89 +- 0.30 (Lee+14)")
    g.add_argument("--spec-acisi-soft", choices=("exclude", "include"), default=None,
                   help="ACIS-I spectral bins < 1.5 keV (2022) out of the spectral likelihood")
    g.add_argument("--wind-dipole", choices=("exp", "clip"), default=None)
    g.add_argument("--ylm-mass", choices=("neutral", "free"), default=None,
                   help="renormalise the Y_lm-perturbed ejecta mass to the unperturbed one")
    g.add_argument("--kdop", choices=("fixed0", "free"), default=None,
                   help="ln_kdop frozen at 0 (default) or as given by --theta / --free")
    g.add_argument("--pm-files", choices=tuple(PD.PM_FILES), default=None)
    g.add_argument("--pm-scale-sigma", type=float, default=None, help="0 = no PM-scale nuisance")
    g.add_argument("--pm-mask", default=None, help="none | sw | jet | sw+jet")
    g.add_argument("--pm-target", choices=("registration", "vink22"), default="registration")
    g.add_argument("--nh-mode", choices=("extrap", "clamp", "geo", "auto"), default=None,
                   help="N_H mixing of the node columns: tents in ln N_H + ln-linear extrapolation "
                        "(extrap), tents clamped (clamp), ln(column) linear in N_H throughout (geo); "
                        "auto = geo for --obs v2, extrap for v1")
    g.add_argument("--table-dir", default=str(J.TABLE_DIR),
                   help="emissivity / sync / halo tables (the N_H grid is read from them)")
    o = ap.add_argument_group("observation-model chain (stage 2, 2026-09-25; --legacy = v1 + tables)")
    o.add_argument("--obs", choices=("v1", "v2"), default=None,
                   help="v2 (default): obs_tables v2 -- binned tables on the analysis bins with the NEI "
                        "T_e-history axis, N_H grid 0.5-4e22, halo per N_H node, log-kT, solar CSM, "
                        "exact-window Doppler moments + the v2 Doppler data; v1: the channel tables")
    o.add_argument("--responses", choices=("tables", "ciao", "ciao-arf"), default=None,
                   help="ciao (default): CIAO per-epoch corrections (exposure-map geometry x folded "
                        "ARF x RMF ratio; casa_xfit_responses); ciao-arf: ARF ratio only; tables: none")
    o.add_argument("--csm-solar", choices=("auto", "on", "off"), default=None,
                   help="solar O:Ne:Mg for the CSM part of the tracers (auto: on for v2 + *_solarcsm IC)")
    o.add_argument("--kt-interp", choices=("auto", "linear", "log"), default=None)
    o.add_argument("--no-history", action="store_true",
                   help="v2 tables without the NEI T_e-history axis (rho = 1 slice)")
    o.add_argument("--save-state", default=None, metavar="NPZ",
                   help="the full evolved state at the FIRST epoch (2000), written after the start "
                        "forward and overwritten with the final theta's after --fit (casa_xfit_state)")
    s4 = ap.add_argument_group("stage-4 residual physics (2026-09-26; defaults = new; --stage3 / --legacy = "
                               "the refit R / R' behaviour)")
    s4.add_argument("--stage3", action="store_true",
                    help="every stage-4 option at its old value (the R' likelihood and observation model)")
    s4.add_argument("--background", choices=("measured", "particle", "annulus", "rate"), default=None,
                    help="measured (default): per-epoch, per-band particle background (off-remnant annulus "
                         "minus the out-of-time readout-streak events and the model halo; casa_xfit_bkg) "
                         "plus the streak itself; particle: without the streak term; annulus: the stage-3 "
                         "recipe (no streak correction); rate: the old floor BKG_RATE")
    s4.add_argument("--bkg-file", default=None, help="casa_xfit_bkg output (default casa_xfit_bkg.BKG_FILE)")
    s4.add_argument("--bkg-model", default=None,
                    help="--background annulus: the casa_xfit --save-model npz whose halo is subtracted "
                         "(default work/xfit_Rp_n128.npz)")
    s4.add_argument("--outline-mask", choices=tuple(OUTLINE_MASK_CONES), default=None,
                    help="inner-arc (default): cones 260, 270, 10 out of the outline (detector locks onto an "
                         "inner arc); none")
    s4.add_argument("--pm-mask-extra", default=None,
                    help="extra proper-motion cone mask, cone angles joined by '+' (default 300 = PA 210); none")
    s4.add_argument("--heldout-sigma", choices=("train", "train-nocut", "data"), default=None,
                    help="held-out outline: per-cone sigma from the training epochs only, detector-jump "
                         "cone-epochs (> HELDOUT_JUMP_ARCSEC off the training trend, data only) dropped "
                         "(train, default); without the cut (train-nocut); the stored all-epoch sigma (data)")
    s4.add_argument("--sync-trend", choices=("on", "off"), default=None,
                    help="X-ray synchrotron secular trend b_sx (%%/yr, prior -1.0 +- 0.5) on top of the radio "
                         "anchor (default on)")
    s4.add_argument("--spec-gain", choices=("profile", "off"), default=None,
                    help="per-epoch energy-scale nuisance, prior 0.2 %%, profiled through the line-shift "
                         "derivative (default profile)")
    s4.add_argument("--spec-soft", choices=("profile", "off"), default=None,
                    help="per-epoch calibration factor of the spectral bins <= 1.1 keV, prior 5 %%, profiled "
                         "(default profile)")
    s4.add_argument("--spec-broadening", choices=("thermal", "vlos", "off"), default=None,
                    help="line broadening in the v2 spectra: line-of-sight velocity to second order + the "
                         "ions' thermal spread (thermal, default; +4.3 GB of D / D2 tables), velocity only "
                         "(vlos), none (off)")
    s4.add_argument("--kte", choices=("fixed", "free"), default=None,
                    help=f"post-shock kT_e fixed at {KTE_FIXED_KEV} keV (default; ln_kte frozen) or free")


def resolve_options(args):
    o = default_options(legacy=args.legacy, stage3=getattr(args, "stage3", False))
    for k in FIX_OPTIONS:
        v = getattr(args, k, None)
        if v is not None:
            setattr(o, k, {"on": True, "off": False}.get(v, v))
    o.rs_term = args.rs_term
    o.pm_target = args.pm_target
    o.recentre_iter = args.recentre_iter
    for k in OBS_OPTIONS:
        v = getattr(args, k, None)
        if v is not None:
            setattr(o, k, v)
    for k in STAGE4_OPTIONS:
        v = getattr(args, k, None)
        if v is not None:
            setattr(o, k, v)
    o.history = not getattr(args, "no_history", False)
    o.bkg_file = getattr(args, "bkg_file", None)
    o.bkg_model = getattr(args, "bkg_model", None)
    return resolve_auto(o, args.ic)


def cone_angle_mask(angles, cones):
    """Boolean mask of the cones at the angles ``cones`` (theta = PA + 90)."""
    out = np.zeros(len(angles), bool)
    for a in cones:
        out |= np.abs(((np.asarray(angles) - float(a) + 180.0) % 360.0) - 180.0) < 1e-6
    return out


def parse_cones(spec):
    """'none' | '300' | '300+310' -> tuple of cone angles."""
    return tuple(float(x) for x in str(spec).split("+") if x and x != "none")


def apply_obs_masks(obs, opts):
    """The stage-4 cone masks, in place on ``obs`` (``PD.load_observations``):

    * ``--outline-mask inner-arc``: cones 260, 270, 10 (PA 170 / 180 / 280) out of
      the outline (every epoch; the held-out outline too). There
      casa_real_outline's steepest broadband decline locks onto an inner arc
      (143-146", 138"), and the 2004 hard-band profiles put the forward-shock
      filament at 164-166" / 155", where the model is (stage3/physics REPORT 3:
      ~40 % of the outline chi2);
    * ``--pm-mask-extra 300``: cone 300 (PA 210) out of the proper motions
      (data 0.102 vs model 0.240"/yr: the SW reverse-shock-contaminated
      registration of the masked 290 / 320; 11.0 of V3's 27.4). Recorded in
      ``obs['pm_mask_extra_cones']`` so ``refit_pm`` keeps it.
    Returns obs."""
    oc = OUTLINE_MASK_CONES[stage4(opts, "outline_mask")]
    if oc:
        cm = cone_angle_mask(obs["angles"], oc)
        n0 = int(np.asarray(obs["mask"]).sum())
        obs["mask"] = np.asarray(obs["mask"]) & ~cm[None, :]
        obs["outline_masked_cones"] = oc
        print(f"[xfit] --outline-mask {stage4(opts, 'outline_mask')}: cones {oc} out of the outline "
              f"({n0} -> {int(obs['mask'].sum())} cone-epochs)", flush=True)
    pc = parse_cones(stage4(opts, "pm_mask_extra"))
    if pc:
        obs["pm_mask_extra_cones"] = pc
        if "pm" in obs:
            n0 = int(np.isfinite(obs["pm"]).sum())
            obs["pm"] = np.where(cone_angle_mask(obs["angles"], pc), np.nan, obs["pm"])
            obs["pm_source"] = f"{obs.get('pm_source', '')}; + cones {pc} masked (stage 4)"
            print(f"[xfit] --pm-mask-extra {stage4(opts, 'pm_mask_extra')}: proper motions {n0} -> "
                  f"{int(np.isfinite(obs['pm']).sum())} cones", flush=True)
    return obs


def load_observations(opts):
    """``PD.load_observations`` with the options' PM files / mask, the stage-4
    cone masks applied (``apply_obs_masks``) and, for --obs v2, the v2 Doppler
    data."""
    obs = PD.load_observations(pm_files=opts.pm_files, pm_mask=opts.pm_mask)
    apply_obs_masks(obs, opts)
    if opts.obs == "v2":
        obs["doppler"] = O2.load_doppler_data()
    return obs


def attach_stage4(obs, img, opts, *, bkg_file=None, bkg_model=None):
    """The stage-4 data terms into ``img`` (in place): ``--background`` measured /
    particle / annulus -> ``img['bkg_img']`` (E, 6, npix, npix) counts/s per
    pixel and ``img['spec']['bkg_add']`` / ``['bkg_mult']`` (E, 31)
    (``casa_xfit_bkg.background_terms``); nothing for ``rate``.
    ``bkg_model``: a casa_xfit --save-model npz whose images the ``annulus``
    recipe subtracts (the stage-3 recipe; default the R' model)."""
    import casa_xfit_bkg as XB
    mode = stage4(opts, "background")
    if mode == "rate":
        return None
    labels = [str(e) for e in obs["epochs"]]
    mi = None
    if mode == "annulus":
        path = bkg_model or getattr(opts, "bkg_model", None) or str(PD.WORK / "xfit_Rp_n128.npz")
        M = np.load(path, allow_pickle=True)
        lab = [str(e) for e in M["epochs"]]
        mi = np.stack([np.asarray(M["images"][lab.index(e)]) for e in labels])
    t = XB.background_terms(labels, mode, npix=img["npix"], pix=img["pix"],
                            path=bkg_file or getattr(opts, "bkg_file", None) or XB.BKG_FILE, model_images=mi)
    img["bkg_img"] = jnp.asarray(t["img"], jnp.float32)
    img["spec"]["bkg_add"] = jnp.asarray(np.where(img["spec"]["has"][:, None], t["spec_add"], 0.0), jnp.float32)
    img["spec"]["bkg_mult"] = jnp.asarray(t["spec_mult"], jnp.float32)
    pm = img["pmask"]
    rate = np.asarray(t["img"])
    print(f"[xfit] --background {mode}: mean over the likelihood pixels, counts/s per pixel per band "
          f"(old floor {BKG_RATE:g}); spectrum: added counts/s 0.7-6.9 keV and max OOT factor", flush=True)
    for e, lab in enumerate(labels):
        print(f"    {lab}: " + " ".join(f"{rate[e, b][pm[e]].mean():.2e}" for b in range(rate.shape[1]))
              + f"   spec +{float(np.sum(t['spec_add'][e])):.3f} c/s, x{float(np.max(t['spec_mult'][e])):.4f}",
              flush=True)
    return t


def attach_responses(obs, img, opts):
    """``--responses ciao | ciao-arf``: the per-epoch CIAO correction factors
    (``casa_xfit_responses.corrections``) into ``img["resp_corr"]`` (E, 6,
    npix, npix) and ``img["spec"]["resp_corr"]`` (E, 31)."""
    if opts.responses == "tables":
        return None
    mode = {"ciao": "fold", "ciao-arf": "arf"}[opts.responses]
    corr = XR.corrections([str(e) for e in obs["epochs"]], img["resp"], npix=img["npix"], mode=mode)
    img["resp_corr"] = jnp.asarray(corr["img"], jnp.float32)
    img["spec"]["resp_corr"] = jnp.asarray(corr["spec"], jnp.float32)
    print(f"[xfit] CIAO responses ({mode}): per-epoch band ratio (CIAO / tables) and the mean "
          f"relexp over the likelihood pixels:", flush=True)
    for e, lab in enumerate(obs["epochs"]):
        pm = img["pmask"][e]
        rel = corr["img"][e] / corr["band"][e][:, None, None]
        print(f"    {lab}: " + " ".join(f"{x:.3f}" for x in corr["band"][e]) + "   relexp "
              + " ".join(f"{rel[b][pm].mean():.3f}" for b in range(rel.shape[0]))
              + (f"   spec ratio {corr['spec'][e].min():.3f}-{corr['spec'][e].max():.3f}"
                 if img["spec"]["has"][e] else ""), flush=True)
    return corr


def state_ic(forward, theta):
    """The IC bookkeeping ``XS.save_state`` copies (``ic_age``, the wind / shell
    ``ambient_*`` meta): the IC's own, or -- with the similarity path -- scaled
    by the map's remainder (``casa_rescale.scale_ambient``), plus the TOTAL
    ``similarity_{L,T,M}`` stamp, so the saved state is consistent with the scales
    it was evolved with (a later ``--ic`` / ``ic_similarity_ref`` reads it)."""
    ic = forward.ic
    if not getattr(forward, "sim_on", False):
        # a pre-scaled IC (convert --sim) passes its stamp on; none for Orlando's own
        return ic, {k: np.asarray(ic[k]) for k in ("similarity_L", "similarity_T", "similarity_M",
                                                    "similarity_age_unscaled", "similarity_method") if k in ic}
    import casa_rescale as CR
    p = dict(zip(PARAM_NAMES, np.asarray(theta, np.float64)))
    Lc, Tc, Mc = (float(np.exp(p[k] - forward.sim_ref[k])) for k in SIM_NAMES)
    keep = ("box", "num_cells", "gamma") + XS.IC_META
    out = CR.scale_ambient({k: ic[k] for k in keep if k in ic}, Lc, Tc, Mc)
    out["age"] = float(ic["age"]) * Tc
    stamp = {f"similarity_{c}": float(np.exp(p[f"ln_{c}"])) for c in "LTM"}
    stamp.update(similarity_age_unscaled=float(ic.get("similarity_age_unscaled", ic["age"])),
                 similarity_method=np.array("casa_xfit traced map (casa_pluto_diff.transform_fields sim=) on "
                                            f"an IC with (L, T, M) = ({', '.join(f'{np.exp(v):.4f}' for v in forward.sim_ref.values())})"))
    return out, stamp


def write_outputs(args, model, theta, obs, img, forward, opts, tag, extra=None):
    """--save-model / --save-state for ``theta`` (called after the start forward
    and again after the fit, so the files hold the FINAL theta's model)."""
    opt_json = {k: str(v) for k, v in vars(opts).items()}
    if args.save_model:
        np.savez_compressed(args.save_model, theta=np.asarray(theta), names=np.array(PARAM_NAMES),
                            epochs=np.array(obs["epochs"]), years=obs["years"],
                            counts=img["counts"], bmask=img["bmask"], exposure=img["exposure"],
                            block=args.block, options=json.dumps(opt_json), tag=tag, **(extra or {}),
                            **{k: np.asarray(v) for k, v in model.items() if k != "state0"})
        print(f"[xfit] wrote {args.save_model} ({tag})", flush=True)
    if args.save_state:
        p = dict(zip(PARAM_NAMES, np.asarray(theta, np.float64)))
        e0 = int(np.argmin(np.asarray(obs["years"])))
        ic_s, stamp = state_ic(forward, theta)
        lay = XS.save_state(args.save_state, model["state0"], forward.rv, SCALAR_NAMES, ic=ic_s,
                            theta=np.asarray(theta), names=PARAM_NAMES,
                            epoch_year=float(np.asarray(obs["years"])[e0]),
                            epoch_label=str(obs["epochs"][e0]), t_expl=float(p["t_expl"]), options=opt_json,
                            extra=dict(tag=tag, r_fs_pc=np.asarray(model["r_fs_pc"])[e0],
                                       r_rs_pc=np.asarray(model["r_rs_pc"])[e0], **stamp))
        print(f"[xfit] wrote {args.save_state} ({tag}): state at {obs['epochs'][e0]} "
              f"(age {float(np.asarray(obs['years'])[e0]) - float(p['t_expl']):.1f} yr), "
              f"{len(lay)} variables", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ic", required=True)
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--theta", type=float, nargs="+", default=None,
                    help="padded with prior means (a casa_pluto_diff theta works as is)")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--jvp", action="store_true", help="forward-mode Jacobian (default: FD)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--free", nargs="*", default=[k for k in PARAM_NAMES if k not in SIM_NAMES],
                    choices=PARAM_NAMES, help="default: every parameter except the similarity scales")
    ap.add_argument("--similarity", choices=("auto", "on", "off"), default="auto",
                    help="the traced similarity map (ln_L, ln_T, ln_M; casa_pluto_diff.SIM_NAMES): auto = on "
                         "iff one of them is in --free or differs from the IC's own scales (its similarity_* "
                         "stamp; thetas without them are padded with those), else off = the old forward, "
                         "bitwise. On: ln_sv frozen at 0, their priors (PD.SIM_PRIOR) in prior_extra, the "
                         "ballistic-date wall = t_expl >= 1671.3 (kept; --no-sim-wall drops it)")
    ap.add_argument("--no-sim-wall", action="store_true",
                    help="with the similarity path, DROP the ballistic-date wall (default: kept; with the "
                         "relabelled IC age it is the Thorstensen+01 bound t_expl >= 1671.3, energy/REPORT.md "
                         "step 4; refit R3 ran without it)")
    ap.add_argument("--sim-wall", action="store_true", help="(the default since 2026-09-27; accepted, no effect)")
    ap.add_argument("--jet-img", choices=("on", "off"), default="off",
                    help="the jet image term 'jet_image' (jet_image_residuals): epoch-mean brightness per band "
                         "in 10-deg sky-PA bins of the annulus beyond the rim (casa_jet.JET_PROFILE: 195-215\", "
                         "NE PA 20-110, SW PA 210-300), model error casa_jet.JET_SIGMA")
    ap.add_argument("--jet", choices=("on", "off"), default="off",
                    help="add the Si-rich NE jet + SW counter-jet to the IC (casa_jet; parameters "
                         f"{', '.join(JT.JET_NAMES)}); off (default): the old forward, bitwise, the jet "
                         "parameters at their prior means and frozen")
    ap.add_argument("--block", type=int, default=16, help="likelihood block (1.97\" pixels)")
    # model errors calibrated on fit M's residuals (chi2/N = 1), and the image
    # terms down-weighted by the six bands' shared spatial structure
    ap.add_argument("--sigma-static", type=float, default=0.5,
                    help="model error of the epoch-mean block brightness (ln)")
    ap.add_argument("--sigma-temporal", type=float, default=0.075,
                    help="model error of a block's brightness relative to its mean (ln)")
    ap.add_argument("--img-weight", type=float, default=1.0 / 6.0)
    add_fine_arguments(ap)
    ap.add_argument("--spectra", action="store_true",
                    help="add the integrated spectrum (r < 200\", 0.7-7 keV, 0.2 keV bins) of "
                         "every epoch that has one: static shape + its evolution")
    ap.add_argument("--sigma-spec-static", type=float, default=0.13)
    ap.add_argument("--sigma-spec-temporal", type=float, default=0.046)
    ap.add_argument("--r-max-img", type=float, default=150.0,
                    help="image likelihood radius (arcsec): inside the forward shock; the "
                         "rim is the outline data's job (the observed outer profile has "
                         "emission beyond any model shock, which inflates the remnant)")
    ap.add_argument("--sigma-model", type=float, default=5.0)
    ap.add_argument("--doppler", action="store_true")
    ap.add_argument("--ad-llf-cold", type=float, default=1000.0)
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--devices", type=int, default=1,
                    help="evaluate the FD Jacobian's perturbed forward runs in parallel "
                         "on this many GPUs (pmap; the runs are independent)")
    ap.add_argument("--devices-mode", choices=("jit", "pmap"), default="jit",
                    help="--devices > 1: one jitted executable per device with async dispatch (jit, default) or "
                         "jax.pmap (pmap: returned wrong hydro on all devices but the first on 2026-09-27)")
    ap.add_argument("--state-only", action="store_true",
                    help="with --save-state: evolve to the FIRST epoch only, observe it (for r_fs_pc / "
                         "r_rs_pc) and write the state -- no later epochs, no likelihood, no fit")
    ap.add_argument("--ic-jitter", type=float, default=None,
                    help="--state-only diagnostics: multiply the IC density by 1 + this x N(0, 1)")
    ap.add_argument("--time-reps", type=int, default=1,
                    help="--state-only diagnostics: evaluate this many times (the last is saved)")
    ap.add_argument("--evolve-years", type=float, default=None,
                    help="--state-only diagnostics: evolve the IC only this many years (sharding checks)")
    ap.add_argument("--gpus", type=int, default=1,
                    help="shard ONE forward over this many GPUs of the node (x-slabs; casa_xfit_shard): "
                         "the 512^3 forward / --save-state. Not with --devices / --jvp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jac-dump", default=None, help="save the last FD Jacobian (npz)")
    ap.add_argument("--exclude-epochs", nargs="+", default=None, metavar="EPOCH",
                    help="held-out validation: drop these epochs (labels, e.g. 2019 2022) from EVERY "
                         "likelihood term -- outline, images, spectra, Doppler, and the registration "
                         "proper motions (refitted from the per-epoch shifts without them); the forward "
                         "model still evolves through all epochs and --save-model keeps them all; the "
                         "held-out epochs are scored against the training set (EpochExclusion)")
    ap.add_argument("--positivity", choices=("redistribute", "conservative"), default="redistribute",
                    help="per-stage/per-step positivity mode (default redistribute: bitwise the fitted runs). "
                         "REDISTRIBUTE refills a sub-floor cell with its 3x3x3 neighbour mean WITHOUT debiting "
                         "the donors; next to a dense clump cell that manufactures mass and feeds a single-cell "
                         "density runaway -> NaN at 448^3 (jetdbg 2026-10-03). 'conservative': the "
                         "energy-conserving mode (density floor only)")
    ap.add_argument("--deepvoid-blend", choices=("off", "on"), default="off",
                    help="FOFC-style LLF flux blending in cells within 8x of the density floor (default off)")
    ap.add_argument("--cpu-test", action="store_true",
                    help="PIPELINE TESTS ON CPU ONLY: NATIVE_JAX backend and no FCT flux limiter "
                         "(the XLA:CPU compile of the limiter needs > 200 GB); not physics-grade. "
                         "Run with JAX_PLATFORMS=cpu and XLA_FLAGS=--xla_backend_optimization_level=0")
    add_fix_arguments(ap)
    args = ap.parse_args()
    if args.gpus > 1 and (args.devices > 1 or args.jvp):
        ap.error("--gpus (one sharded forward) excludes --devices / --jvp")
    SH.activate(args.gpus)
    args.opts = opts = resolve_options(args)
    print("[xfit] options: " + ", ".join(f"{k}={v}" for k, v in vars(opts).items()), flush=True)
    global JAC_DUMP
    JAC_DUMP = args.jac_dump
    th = list(args.theta) if args.theta is not None else list(THETA0)
    n_given = len(args.theta) if args.theta is not None else 0
    th = th + [PRIOR[k][0] for k in PARAM_NAMES[len(th):]]
    free = np.array([k in args.free for k in PARAM_NAMES])
    # the similarity scales: missing ones = the IC's own (the identity of the map)
    sim_ref = ic_similarity_ref(args.ic)
    for k in SIM_NAMES:
        if PARAM_NAMES.index(k) >= n_given:
            th[PARAM_NAMES.index(k)] = sim_ref[k]
    sim_free = any(free[PARAM_NAMES.index(k)] for k in SIM_NAMES)
    sim_moved = any(abs(th[PARAM_NAMES.index(k)] - sim_ref[k]) > 1e-7 for k in SIM_NAMES)
    if args.similarity == "off" and (sim_free or sim_moved):
        ap.error("--similarity off with a free or moved similarity scale")
    opts.similarity = "on" if args.similarity == "on" or (args.similarity == "auto" and (sim_free or sim_moved)) \
        else "off"
    if opts.similarity == "on":
        i_sv = PARAM_NAMES.index("ln_sv")
        if th[i_sv] != 0.0 or free[i_sv]:
            print(f"[xfit] similarity path: ln_sv frozen at 0 (was {th[i_sv]:.4g}{', free' if free[i_sv] else ''})",
                  flush=True)
        th[i_sv] = 0.0
        free[i_sv] = False
        if args.no_sim_wall:
            opts.conv_wall = False
        L_, T_, M_ = (float(np.exp(th[PARAM_NAMES.index(k)])) for k in SIM_NAMES)
        print(f"[xfit] similarity path ON: (L, T, M) = ({L_:.4f}, {T_:.4f}, {M_:.4f}) relative to Orlando's "
              f"state (IC's own: {', '.join(f'{np.exp(v):.4f}' for v in sim_ref.values())}); E x "
              f"{M_ * (L_ / T_) ** 2:.4f}; free {[k for k in SIM_NAMES if free[PARAM_NAMES.index(k)]]}; "
              f"ballistic-date wall {'kept' if opts.conv_wall else 'dropped'}", flush=True)
    i_kdop = PARAM_NAMES.index("ln_kdop")
    if opts.kdop == "fixed0":
        if th[i_kdop] != 0.0 or free[i_kdop]:
            print(f"[xfit] ln_kdop frozen at 0 (was {th[i_kdop]:.4g}{', free' if free[i_kdop] else ''})")
        th[i_kdop] = 0.0
        free[i_kdop] = False
    i_kte = PARAM_NAMES.index("ln_kte")
    if stage4(opts, "kte") == "fixed":
        # --kte fixed: the observation model uses KTE_FIXED_KEV; theta carries it too
        # (so the ln_kte prior residual is 0 and saved thetas say what was used)
        if abs(th[i_kte] - np.log(KTE_FIXED_KEV)) > 1e-12 or free[i_kte]:
            print(f"[xfit] --kte fixed: ln_kte = ln {KTE_FIXED_KEV} (was {th[i_kte]:.4g}"
                  f"{', free' if free[i_kte] else ''})", flush=True)
        th[i_kte] = float(np.log(KTE_FIXED_KEV))
        free[i_kte] = False
    i_bsx = PARAM_NAMES.index("b_sx")
    if stage4(opts, "sync_trend") != "on":
        th[i_bsx] = EXTRA_PRIOR["b_sx"][0]
        free[i_bsx] = False
    opts.jet = args.jet
    if opts.jet != "on":
        for k in JET_NAMES:
            th[PARAM_NAMES.index(k)] = EXTRA_PRIOR[k][0]
            free[PARAM_NAMES.index(k)] = False
    else:
        print("[xfit] --jet on: " + ", ".join(f"{k} {th[PARAM_NAMES.index(k)]:.4g}"
                                             f"{' (free)' if free[PARAM_NAMES.index(k)] else ''}"
                                             for k in JET_NAMES), flush=True)

    obs = load_observations(opts)
    if opts.obs == "v2":
        print(f"[xfit] Doppler data {obs['doppler']['source']}", flush=True)
    t_load = time.time()
    opts.jet_img = args.jet_img
    img = load_image_data(obs["epochs"], obs["years"], block=args.block, r_max=args.r_max_img,
                          jet_bins=JT.JET_PROFILE if args.jet_img == "on" else None,
                          table_dir=args.table_dir, obs=opts.obs, history=opts.history,
                          nh_grid=NH_GRID_LEGACY if opts.nh_mode == "clamp" and args.legacy else None,
                          spec_kinds=spec_table_kinds(opts), fine_block=args.fine_block)
    img["spec"] = load_spectrum_data(obs["epochs"], img)
    attach_responses(obs, img, opts)
    attach_stage4(obs, img, opts)
    print(f"[xfit] tables + responses loaded in {time.time() - t_load:.0f} s", flush=True)
    print(f"[xfit] {len(obs['epochs'])} epochs {obs['epochs'][0]}-{obs['epochs'][-1]}; image blocks "
          f"{int(img['bmask'].sum())} x 6 bands at {args.block * img['pix']:.1f}\"; N_H grid "
          f"{img['nh_grid']}; PM {obs.get('pm_source')}", flush=True)
    excl = EpochExclusion(obs, img, args.exclude_epochs, args, opts) if args.exclude_epochs else None
    overrides = None
    if args.cpu_test:
        from astronomix.option_classes.simulation_config import BackendConfig, NATIVE_JAX
        overrides = dict(backend_config=BackendConfig(backend=NATIVE_JAX),
                         positivity_config=fd_positivity(mode=POSITIVITY_REDISTRIBUTE)._replace(
                             preserving_flux=False))
        print("[xfit] --cpu-test: NATIVE_JAX backend, FCT flux limiter OFF (pipeline test only)")
    if args.positivity != "redistribute" or args.deepvoid_blend == "on":
        # opt-in robustness for the 448^3 backgrounds (jetdbg 2026-10-03); the default
        # leaves config_overrides untouched, i.e. every fitted path bitwise unchanged
        from astronomix.option_classes.simulation_config import POSITIVITY_CONSERVATIVE
        mode = POSITIVITY_CONSERVATIVE if args.positivity == "conservative" else POSITIVITY_REDISTRIBUTE
        pc = (overrides or {}).get("positivity_config", fd_positivity(mode=mode))
        pc = pc._replace(per_stage_mode=mode, per_step_mode=mode, deepvoid_blend=args.deepvoid_blend == "on")
        overrides = dict(overrides or {}, positivity_config=pc)
        print(f"[xfit] positivity: {args.positivity}, deepvoid_blend {args.deepvoid_blend}", flush=True)
    forward = make_forward(args.ic, obs, img, ad_llf_cold=args.ad_llf_cold, opts=opts,
                           config_overrides=overrides)
    dtype = jnp.float64 if args.x64 else jnp.float32
    theta = jnp.asarray(th, dtype=dtype)
    ic_diag = forward.lifted.jit(forward.ic_diag)

    if excl is None:
        def resid_fun(t):
            return jnp.concatenate(list(residual_parts(forward(t), obs, img, t, args).values()))
    else:
        def resid_fun(t):
            return jnp.concatenate(list(excl.parts(forward(t), t).values()))
    evals = {}

    def report(model, theta, label):
        """summarize (+ the exclusion's all-epoch / held-out evaluation)."""
        if excl is None:
            summarize(model, obs, img, theta, args, label, forward.budget0)
            return None
        summarize(excl.model(model), excl.obs, excl.img, theta, excl.args, f"{label} [train]",
                  forward.budget0)
        evals[label] = excl.evaluate(model, theta, label)
        return dict(exclude_epochs=np.array(excl.exclude), train_epochs=np.array(excl.obs["epochs"]),
                    eval=json.dumps(evals[label]))

    if args.state_only:
        if not args.save_state or args.fit:
            raise SystemExit("--state-only needs --save-state and excludes --fit")
        fs = forward.lifted.jit(lambda t: forward.first_state(t, args.evolve_years, args.ic_jitter))
        for rep_ in range(args.time_reps - 1):         # timing diagnostics: compile + warm runs
            t0 = time.time()
            jax.block_until_ready(fs(theta))
            print(f"[xfit] --state-only run {rep_}: {time.time() - t0:.1f} s", flush=True)
        t0 = time.time()
        st0, rfs0, rrs0 = jax.block_until_ready(fs(theta))
        print(f"[xfit] --state-only: evolved to {obs['epochs'][int(np.argmin(np.asarray(obs['years'])))]} in "
              f"{time.time() - t0:.1f} s" + (f"; per-device peak GB " + ", ".join(
                  f"{m:.1f}" for _, m, _ in SH.per_device_memory()) if SH.active() else ""), flush=True)
        p = dict(zip(PARAM_NAMES, np.asarray(theta, np.float64)))
        e0 = int(np.argmin(np.asarray(obs["years"])))
        ic_s, stamp = state_ic(forward, theta)
        lay = XS.save_state(args.save_state, st0, forward.rv, SCALAR_NAMES, ic=ic_s,
                            theta=np.asarray(theta), names=PARAM_NAMES,
                            epoch_year=float(np.asarray(obs["years"])[e0]), epoch_label=str(obs["epochs"][e0]),
                            t_expl=float(p["t_expl"]), options={k: str(v) for k, v in vars(opts).items()},
                            extra=dict(tag="state-only", r_fs_pc=np.asarray(rfs0), r_rs_pc=np.asarray(rrs0),
                                       **stamp))
        print(f"[xfit] wrote {args.save_state} (state-only), {len(lay)} variables", flush=True)
        return
    fwd = forward.lifted.jit(forward)
    t0 = time.time()
    model = jax.block_until_ready(fwd(theta))
    print(f"[xfit] forward pass {time.time() - t0:.1f} s", flush=True)
    if SH.active():
        print(f"[xfit] state0 sharding {model['state0'].sharding.spec}; lifted {forward.lifted.nbytes() / 2 ** 30:.2f}"
              " GiB; per-device peak GB " + ", ".join(f"{m:.1f}" for _, m, _ in SH.per_device_memory()), flush=True)
    extra = report(model, theta, "start")
    write_outputs(args, model, theta, obs, img, forward, opts, "start", extra)
    del model
    if not args.fit:
        return
    if args.jvp:
        jac = jax.jit(lambda t: PD.jacobian_jvp(resid_fun, t))
    elif SH.active():
        rfun_s = forward.lifted.jit(resid_fun)

        def jac(t):             # PD.jacobian_fd with the sharded (lifted) residual function
            cols = []
            for i, h in enumerate([FD_STEPS[k] for k in PARAM_NAMES]):
                e = jnp.zeros_like(t).at[i].set(h)
                cols.append((rfun_s(t + e) - rfun_s(t - e)) / (2 * h))
            return rfun_s(t), jnp.stack(cols, axis=-1)
    elif args.devices > 1:
        jac = lambda t: jacobian_fd_parallel(resid_fun, t, [FD_STEPS[k] for k in PARAM_NAMES],  # noqa: E731
                                             free, args.devices, mode=args.devices_mode)
    else:
        jac = lambda t: PD.jacobian_fd(resid_fun, t, [FD_STEPS[k] for k in PARAM_NAMES])  # noqa: E731
    rfun = forward.lifted.jit(resid_fun)
    lam = args.damping
    hist = []

    def write_json(hist, theta, done):
        # after every LM iteration (a job that dies keeps its accepted steps)
        if args.out:
            d = dict(history=hist, theta=[float(x) for x in theta], names=PARAM_NAMES, ic=args.ic, done=done,
                     free=[k for k, f_ in zip(PARAM_NAMES, free) if f_],
                     options={k: str(v) for k, v in vars(opts).items()})
            if excl is not None:
                d.update(exclude_epochs=excl.exclude, train_epochs=list(excl.obs["epochs"]), eval=evals)
            Path(args.out).write_text(json.dumps(d, indent=1))
    for it in range(args.steps):
        t0 = time.time()
        rvec, Jm = jac(theta)
        rvec, Jm = np.asarray(rvec, np.float64), np.asarray(Jm, np.float64)
        if not np.all(np.isfinite(rvec)):
            # a rare non-deterministic blow-up of ONE run (fit L step 4: NaN at a
            # theta that had just evaluated finite): redo the central value
            print(f"[fit {it}] central residuals non-finite; re-evaluating", flush=True)
            rvec = np.asarray(rfun(theta), np.float64)
        Jm[:, ~free] = 0.0
        bad = ~np.all(np.isfinite(Jm), axis=0)
        Jm[:, bad] = 0.0
        chi = float(rvec @ rvec)
        A = Jm.T @ Jm; g = Jm.T @ rvec
        fz = ~free | bad
        A[fz, fz] = 1.0
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
        trial = theta + jnp.asarray(step, dtype=dtype)
        chi_t = float(jnp.sum(rfun(trial) ** 2))
        print(f"[fit {it}] chi2 {chi:.1f} -> {chi_t:.1f} (lambda {lam:.2g}, {time.time() - t0:.0f} s"
              + (f", frozen {[PARAM_NAMES[i] for i in np.nonzero(bad)[0]]}" if bad.any() else "")
              + "); step " + ", ".join(f"{k} {s:+.3g}" for k, s, f_ in zip(PARAM_NAMES, step, free) if f_),
              flush=True)
        print_ic_diag(f"fit {it} trial", ic_diag(trial), forward.budget0)
        hist.append(dict(it=it, chi2=chi, chi2_trial=chi_t, theta=[float(x) for x in theta],
                         step=step.tolist()))
        if chi_t < chi:
            theta = trial; lam = max(lam / 3, 1e-3)
        else:
            lam *= 4
        write_json(hist, theta, done=False)
    model = jax.block_until_ready(fwd(theta))
    extra = report(model, theta, "fit")
    print("[fit] theta = " + " ".join(f"{float(x):.7g}" for x in theta))
    write_json(hist, theta, done=True)
    write_outputs(args, model, theta, obs, img, forward, opts, "fit", extra)


if __name__ == "__main__":
    main()
