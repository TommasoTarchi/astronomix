"""
Differentiable Chandra observations of a 3D remnant state, in JAX.

``casa_observe.py`` is the exact chain (yt -> pyXSIM NEI -> SOXS ACIS), ~1 h
per epoch at 256^3 and not differentiable. This module is its deterministic,
differentiable surrogate, built so that the SAME physics enters:

* the plasma chain of ``_plasma`` (composition -> mu, mu_e, n_e, n_el; the
  single-fluid T; the Ghavamian post-shock T_e with the fixed Coulomb
  relaxation; n_e t from the carried density_time), ported to jnp;
* the sub-grid two-phase split of ``_subgrid`` / ``casa_observe`` (ejecta-only,
  at fixed pressure; three linear components);
* NEI emission from response-folded tables (``casa_jaxobs_tables.py``): APEC
  per-ion spectra weighted by the ``_nei`` ion fractions, TBabs, ARF, RMF --
  the spectrum is linear in each cell's emission measure, so

      rate[ch] = 1e-14 / (4 pi D^2) sum_cells V [n_e n_H C_H(kT)
                 + sum_el n_e n_el / r_sun,el C_el(kT, n_e t)];

* the dust-scattering halo of ``_dusthalo`` as deterministic kernels (image)
  and an aperture-keep table vs projected radius (spectrum), both computed by
  pushing point sources through the same Monte Carlo;
* the view from -y with west = +x, north = +z (``casa_observe.
  fix_projection_parity``), line-of-sight v_y > 0 receding.

Everything is linear in the emission measure, and depends on kT_e and n_e t
through bilinear table weights, so the model is smooth and cheap: the aperture
spectrum is a DEM "splat" into the (kT, n_e t, radius) grid contracted with the
tables; images are per-cell band rates summed along y and splatted to pixels.

Not modelled (vs casa_observe): Poisson noise, instrumental/sky background,
chip layout/dither, the ACIS PSF (optional Gaussian), thermal line broadening,
synchrotron. The Doppler shift is first order (tables D = dC/dbeta).

    ./run.sh casa_jaxobs.py validate STATE.npz --distance 3.05 \\
        --instrument chandra_aciss_cy0 --pyxsim-log obs/xxx.log
"""

# ==== device selection (as a script) ====
import os
import sys
if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =========================================

import argparse
import re
from functools import partial
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import _plasma as P
import _subgrid

TABLE_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs")
#: multi-GPU hook (``casa_xfit_shard.activate`` sets it to that module): the
#: line-of-sight slabs keep the x-split of the state and their column sums the
#: x-split of the sky plane. None (one device): nothing changes.
SHARD = None
ARCSEC_PER_RAD = 206264.806
PC_CM = 3.0856775814913673e18
KPC_CM = 1e3 * PC_CM
BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))
SI_BAND = (1.78, 1.94)
KT_MIN_KEV = 0.09             # casa_observe --kt-min
C_KMS = 2.99792458e5


# =============================================================================
# ============ ↓ Tables ↓ =====================================================
# =============================================================================
def load_tables(instrument="chandra_aciss_cy0", nh=1.2, table_dir=TABLE_DIR):
    """Emissivity tables at column ``nh`` (1e22), stacked over components.

    Returns a dict of jnp arrays: ``C`` and ``D`` of shape (n_comp, n_kt, n_net,
    n_ch) with component 0 = H (broadcast over n_e t), then the elements in
    ``names[1:]``; ``lkt``, ``lnet`` (log10 grids); ``e_ch`` (channel centres,
    keV); ``r_sun`` (solar number ratio per component, 1 for H).
    N_H is interpolated linearly in log N_H between the tabulated columns.
    """
    d = np.load(Path(table_dir) / f"emissivity_{instrument}.npz")
    lnh = np.log(np.asarray(d["nh"]))
    x = np.clip(np.log(nh), lnh[0], lnh[-1])
    j = int(np.clip(np.searchsorted(lnh, x) - 1, 0, len(lnh) - 2))
    w = (x - lnh[j]) / (lnh[j + 1] - lnh[j])
    pick = lambda a: (1 - w) * a[j] + w * a[j + 1]           # noqa: E731
    els = [str(e) for e in d["elements"]]
    nnet = len(d["net"])
    C = [np.repeat(pick(d["C_H"])[:, None, :], nnet, axis=1)]
    D = [np.repeat(pick(d["D_H"])[:, None, :], nnet, axis=1)]
    for el in els:
        C.append(pick(d[f"C_{el}"])); D.append(pick(d[f"D_{el}"]))
    edges = np.asarray(d["ch_edges"])
    return dict(C=jnp.asarray(np.stack(C), jnp.float32), D=jnp.asarray(np.stack(D), jnp.float32),
                lkt=jnp.asarray(np.log10(d["kt"]), jnp.float32),
                lnet=jnp.asarray(np.log10(d["net"]), jnp.float32),
                e_ch=jnp.asarray(0.5 * (edges[1:] + edges[:-1]), jnp.float32),
                ch_edges=edges, names=["H"] + els,
                r_sun=np.array([1.0] + [P.SOLAR_NUMBER_RATIO_TO_H[e] for e in els]))


def load_halo(nh=1.2, table_dir=TABLE_DIR):
    d = np.load(Path(table_dir) / f"dusthalo_nh{nh:g}.npz")
    return {k: np.asarray(d[k]) for k in d.files}


def band_masks(e_ch, bands=BANDS):
    e = np.asarray(e_ch)
    return np.stack([(e >= lo) & (e < hi) for lo, hi in bands]).astype(np.float32)


# ---- v2 tables (casa_jaxobs_tables --bins / --sync-bins / --halo-v2) ---------
#: casa_xfit's spectral analysis bins (0.7-6.9 keV, 0.2 keV) and the v2 N_H grid
SPEC_EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)
NH_GRID_V2 = np.array([0.5, 0.8, 1.2, 1.6, 2.0, 2.6, 3.3, 4.0])
BINNINGS = ("spec", "band", "dop")


def _ln_nh_weights(grid, nh):
    """Host-side tent weights in ln N_H (clipped to the grid): (j, w) with
    value = (1 - w) a[j] + w a[j + 1]."""
    lnh = np.log(np.asarray(grid))
    x = float(np.clip(np.log(nh), lnh[0], lnh[-1]))
    j = int(np.clip(np.searchsorted(lnh, x) - 1, 0, len(lnh) - 2))
    return j, (x - lnh[j]) / (lnh[j + 1] - lnh[j])


def load_binned_stack(instrument="chandra_aciss_cy0", binning="spec", *, history=True,
                      second_order=False, table_dir=TABLE_DIR):
    """The v2 emissivity tables of one binning at EVERY tabulated N_H (host numpy).

    ``binning``: ``"spec"`` (casa_xfit's 31 bins, ``SPEC_EDGES``), ``"band"``
    (the six image bands, exact edges) or ``"dop"`` (the two Doppler moments
    over [1.78, 1.94] keV: counts and sum (E - E0) counts). Returns a dict:

    * ``C``, ``D`` (and ``D2`` if ``second_order``; spec / band only):
      (n_nh, n_comp, n_kt * n_rho, n_net, n_bin), component 0 = H (broadcast
      over rho and n_e t) -- the kT and history axes are COMBINED (index
      ``i * n_rho + k``), so every consumer that indexes ``[c, i, j]`` works;
    * ``nh`` (n_nh,), ``lkt``, ``lnet``, ``rho`` (history grid; ``[1.]`` with
      ``history=False``), ``n_rho``, ``names``, ``r_sun``, ``edges`` (bin
      edges, or (6, 2) bands, or the Doppler window), ``e_ch`` (bin centres;
      E0 twice for ``dop``), ``E0`` (``dop``), ``binning``.

    ``history=False`` keeps only the rho = 1 slice: the legacy ion balance
    (current T_e for the whole history), on the new bins and N_H grid.
    """
    if binning not in BINNINGS:
        raise ValueError(f"binning {binning!r} not in {BINNINGS}")
    d = np.load(Path(table_dir) / f"emissivity_bins_{instrument}.npz")
    els = [str(e) for e in d["elements"]]
    rho = np.asarray(d["rho"], np.float64)
    ks = slice(None) if history else slice(len(rho) - 1, len(rho))
    if not history and abs(rho[-1] - 1.0) > 1e-6:
        raise ValueError("the last rho node must be 1 (the constant-T_e history)")
    rho = rho[ks]
    nnet, nkt, nr = len(d["net"]), len(d["kt"]), len(rho)
    kinds = ("C", "D", "D2") if second_order else ("C", "D")
    if second_order and binning == "dop":
        raise ValueError("no second-order tables for the Doppler moments")
    out = {}
    for k in kinds:
        H = np.asarray(d[f"{k}_H_{binning}"])                          # (n_nh, n_kt, b)
        H = np.broadcast_to(H[:, :, None, None, :], (H.shape[0], nkt, nr, nnet, H.shape[-1]))
        comps = [H] + [np.asarray(d[f"{k}_{el}_{binning}"])[:, :, ks] for el in els]
        A = np.stack(comps, 1)                                         # (n_nh, c, kt, r, n, b)
        out[k] = np.ascontiguousarray(A.reshape(A.shape[0], A.shape[1], nkt * nr, nnet, A.shape[-1]),
                                      np.float32)
    if binning == "spec":
        edges = np.asarray(d["spec_edges"]); e_ch = 0.5 * (edges[1:] + edges[:-1])
    elif binning == "band":
        edges = np.asarray(d["band_edges"]); e_ch = edges.mean(1)
    else:
        edges = np.asarray(d["dop_window"]); e_ch = np.full(2, float(d["dop_E0"]))
    out.update(nh=np.asarray(d["nh"]), lkt=np.log10(np.asarray(d["kt"])), lnet=np.log10(np.asarray(d["net"])),
               rho=rho, n_rho=nr, names=["H"] + els, edges=edges, e_ch=e_ch, binning=binning,
               r_sun=np.array([1.0] + [P.SOLAR_NUMBER_RATIO_TO_H[e] for e in els]),
               instrument=instrument)
    if binning == "dop":
        out["E0"] = float(d["dop_E0"])
    return out


def load_binned_tables(instrument="chandra_aciss_cy0", nh=1.2, binning="spec", *, history=True,
                       second_order=False, table_dir=TABLE_DIR, stack=None):
    """v2 tables at column ``nh`` (1e22; linear in ln N_H between the nodes,
    clipped to 0.5-4), in the ``load_tables`` format -- a drop-in for every
    function here (``aperture_spectrum``, ``band_columns``, ``doppler_sectors``,
    ``band_tables_of``). With the history axis, pass ``plasma``'s output
    unchanged: it carries each cell's ``rho_hist``. ``stack``: a
    ``load_binned_stack`` result to reuse."""
    s = stack if stack is not None else load_binned_stack(instrument, binning, history=history,
                                                          second_order=second_order, table_dir=table_dir)
    j, w = _ln_nh_weights(s["nh"], nh)
    out = {k: jnp.asarray((1 - w) * s[k][j] + w * s[k][j + 1], jnp.float32)
           for k in ("C", "D", "D2") if k in s}
    edges = np.asarray(s["edges"])
    out.update(lkt=jnp.asarray(s["lkt"], jnp.float32), lnet=jnp.asarray(s["lnet"], jnp.float32),
               rho=jnp.asarray(s["rho"], jnp.float32), n_rho=int(s["n_rho"]),
               e_ch=jnp.asarray(s["e_ch"], jnp.float32), ch_edges=edges, names=list(s["names"]),
               r_sun=np.asarray(s["r_sun"]), binning=s["binning"], nh=float(nh))
    if "E0" in s:
        out["E0"] = s["E0"]
    return out


def load_sync_binned(instrument, binning="band", table_dir=TABLE_DIR):
    """v2 synchrotron tables: ``(ln_ecut grid, S (n_nh, n_ecut, n_bin), nh grid)``,
    counts/s per unit ``k w`` (``casa_jaxobs_tables.build_sync_binned``)."""
    d = np.load(Path(table_dir) / f"sync_bins_{instrument}.npz")
    return np.log(d["ecut"]), np.asarray(d[f"S_{binning}"], np.float32), np.asarray(d["nh"])


def load_halo_stack(instrument="chandra_aciss_cy0", table_dir=TABLE_DIR):
    """v2 halo: band kernels (n_nh, n_band, 401, 401) weighted by the node's own
    observed in-band spectrum, and the aperture keep (n_nh, n_e, n_r), at the
    v2 N_H nodes (``casa_jaxobs_tables.build_halo_bands``)."""
    d = np.load(Path(table_dir) / f"dusthalo_v2_{instrument}.npz")
    return {k: np.asarray(d[k]) for k in d.files}


def load_halo_v2(instrument="chandra_aciss_cy0", nh=1.2, table_dir=TABLE_DIR, stack=None):
    """The v2 halo at one column, in the ``load_halo`` format (a drop-in for
    ``project_columns`` / ``aperture_modes``): kernels and keep linear in ln N_H."""
    s = stack if stack is not None else load_halo_stack(instrument, table_dir)
    j, w = _ln_nh_weights(s["nh"], nh)
    out = {k: s[k] for k in ("bands", "kernel_pixel_arcsec", "kernel_half", "e_grid", "r_grid",
                             "aperture_arcsec")}
    out["kernel"] = ((1 - w) * s["kernel"][j] + w * s["kernel"][j + 1]).astype(np.float32)
    out["aperture_keep"] = (1 - w) * s["aperture_keep"][j] + w * s["aperture_keep"][j + 1]
    out["nh"] = float(nh)
    return out


def halo_nh_stacked(stack):
    """The v2 halo with the N_H nodes folded into the band axis, for
    ``project_columns`` on columns laid out as (n_nh * n_band, x, z) (N_H-tent-
    weighted per sky column, as casa_xfit's ``fold``): the image is then summed
    over the N_H nodes, so every column is scattered with the kernel of its own
    column density."""
    out = {k: stack[k] for k in ("kernel_pixel_arcsec", "kernel_half")}
    K = np.asarray(stack["kernel"])
    out["kernel"] = K.reshape(-1, *K.shape[-2:]).astype(np.float32)
    return out
# =============================================================================
# ============ ↑ Tables ↑ =====================================================
# =============================================================================


# =============================================================================
# ============ ↓ The plasma chain (jnp port of _plasma) ↓ =====================
# =============================================================================
def solar_csm_tracers():
    """Solar (angr) mass fraction per tracer group -- ``casa_pluto.
    solar_csm_composition``, recomputed from the same ``_plasma`` table (not
    imported: casa_pluto pulls in the hydro stack)."""
    w = {"H": P.ATOMIC["H"][0]}
    w.update({el: r * P.ATOMIC[el][0] for el, r in P.SOLAR_NUMBER_RATIO_TO_H.items()})
    tot = sum(w.values())
    X = {el: v / tot for el, v in w.items()}
    return {"He": X["He"], "O": X["O"] + X["Ne"] + X["Mg"],
            "Si": X["Si"] + X["S"] + X["Ar"] + X["Ca"], "Fe": X["Fe"]}


#: the solar split of the multi-element tracer groups (number ratios = solar)
SOLAR_GROUP_SPLIT = {"O": P._solar_layer_split(("O", "Ne", "Mg")),
                     "Si": P._solar_layer_split(("Si", "S", "Ar", "Ca")),
                     "Fe": {"Fe": 1.0}, "He": {"He": 1.0}}


def element_fractions(fields, split="xrism_bulk", csm_solar=False):
    """Per-element mass fractions, as ``_plasma.mass_fractions`` +
    ``element_mass_fractions`` (tracer metals clipped collectively to <= 1).

    ``csm_solar``: split each tracer into its circumstellar part and its ejecta
    part before dividing it among elements. The scalars are advected linearly,
    so a cell holds ``C_ej X_ej + (1 - C_ej) X_csm``; with a solar CSM
    (``*_solarcsm`` states, ``casa_pluto.recompose_csm``) the CSM part of tracer
    T is exactly ``(1 - C_ej) X_T,sun`` (capped at the tracer itself), and it is
    split with SOLAR ratios, the rest with the ejecta preset. Without it
    (legacy) the ejecta preset -- Hwang & Laming's O:Ne:Mg = 2.0:0.03:0.03 --
    is applied to the CSM too, which gave the "solar" CSM Ne 0.10x and Mg
    0.27x solar (obs_model section 6).
    """
    X = {el: jnp.asarray(fields[f"C_{el}"]) for el in P.TRACKED_SPECIES if f"C_{el}" in fields}
    tot = sum(X.values())
    # safe denominator in BOTH branches: a 1/max(tot, 1e-30) in the inactive
    # branch has a reciprocal-square derivative outside float32 (Codex review)
    scale = 1.0 / jnp.where(tot > 1.0, tot, 1.0)
    X = {k: v * scale for k, v in X.items()}
    out = {"H": 1.0 - jnp.minimum(tot, 1.0)}
    if csm_solar:
        # where-clips, not jnp.clip / jnp.minimum: pure CSM sits EXACTLY on both
        # kinks (C_ej = 0, and (1 - C_ej) X_sun == C_T), where those split the
        # derivative 0.5 / 0.5 -- the tangent in C_ej was 1/4 of the one-sided
        # derivative (review 2026-09-25). At the tie the CSM part follows C_ej
        # and any extra tracer is ejecta.
        c_ej = jnp.asarray(fields["C_ej"])
        f_csm = 1.0 - jnp.where(c_ej < 0.0, 0.0, jnp.where(c_ej > 1.0, 1.0, c_ej))
        x_sun = solar_csm_tracers()
    for tracer, parts in P.TRACER_SPLIT_PRESETS[split].items():
        if tracer not in X:
            continue
        if csm_solar:
            a = f_csm * x_sun[tracer]
            x_csm = jnp.where(a <= X[tracer], a, X[tracer])
            x_ej = X[tracer] - x_csm
            for el, frac in parts.items():
                out[el] = out.get(el, 0.0) + x_ej * frac
            for el, frac in SOLAR_GROUP_SPLIT[tracer].items():
                out[el] = out.get(el, 0.0) + x_csm * frac
        else:
            for el, frac in parts.items():
                out[el] = out.get(el, 0.0) + X[tracer] * frac
    return out


def moments(X):
    inv_e = inv_i = z2 = 0.0
    for el, x in X.items():
        A, Z = P.ATOMIC[el]
        inv_e = inv_e + x * Z / A
        inv_i = inv_i + x / A
        z2 = z2 + x * Z ** 2 / A ** 2
    return dict(mu=1.0 / jnp.maximum(inv_e + inv_i, 1e-30), mu_e=1.0 / jnp.maximum(inv_e, 1e-30),
                mu_i=1.0 / jnp.maximum(inv_i, 1e-30), z2_a2=z2)


# ---- constants combined in float64 on the host, so no float32 intermediate ----
# ---- ever sees 1e-24 or 1e48 (forward OR backward) --------------------------
#: 3 m_e / (8 sqrt(2 pi) e^4): m_p^2 = 2.8e-48 underflowed to 0 in float32 in
#: the first port, which made tau_eq = 0 and equilibrated every shocked cell
_TEQ_COEF = 3.0 * P.M_E / (8.0 * np.sqrt(2.0 * np.pi) * P.E_ESU ** 4)
_T_SCALE = 1.0e7
#: t_eq = _TEQ_SECONDS theta^1.5 / (lnL n_b z2_a2), theta = (T_e + (m_e/m_p) T_i/mu_i)/1e7
_TEQ_SECONDS = float(_TEQ_COEF * P.M_P * (P.K_B / P.M_E * _T_SCALE) ** 1.5)
_ME_MP = float(P.M_E / P.M_P)
_NB_PER_CODE_RHO = float(P.CODE_DENSITY / P.M_P)                 # nucleons cm^-3
_T_PER_CODE = float(P.CODE_PRESSURE / P.CODE_DENSITY * P.M_P / P.K_B)
_NET_PER_CODE_DT = float(P.CODE_DENSITY * P.CODE_TIME / P.M_P)


def equipartition_time(T_e, T_i, n_b, m):
    """Spitzer electron-ion equilibration time (s); ``n_b = rho / m_p`` (cm^-3)."""
    n_e = n_b / m["mu_e"]
    T_ev = jnp.maximum(T_e, 1.0) / 1.16045e4
    lnL = jnp.clip(24.0 - 0.5 * jnp.log(jnp.maximum(n_e, 1e-30)) + jnp.log(T_ev), 5.0, 40.0)
    theta = (T_e + _ME_MP * T_i / m["mu_i"]) / _T_SCALE
    return _TEQ_SECONDS * jnp.maximum(theta, 1e-12) ** 1.5 / \
        (lnL * jnp.maximum(n_b * m["z2_a2"], 1e-30))


def electron_temperature(T, n_b, t_shock_s, m, kT_e_shock_keV=0.3, n_substeps=48, teq_scale=1.0):
    """Ghavamian post-shock T_e + Coulomb relaxation, the fixed scheme of
    ``_plasma.electron_ion_temperatures`` (log-spaced substeps, midpoint tau)."""
    return _relax_history(T, n_b, t_shock_s, m, kT_e_shock_keV, n_substeps, teq_scale)[0]


def _relax_history(T, n_b, t_shock_s, m, kT_e_shock_keV=0.3, n_substeps=48, teq_scale=1.0):
    """``(T_e, rho_hist)``: the relaxation of ``electron_temperature`` plus the
    history label of the NEI table (``casa_jaxobs_nei``): the time- (= n_e t-,
    at constant density) weighted mean of T_e along the same substeps, over the
    final T_e (trapezoid on the substep edges; 1 for unshocked cells and for
    cells whose electrons started at T)."""
    f_e = m["mu_i"] / (m["mu_i"] + m["mu_e"])
    T_e = jnp.minimum(kT_e_shock_keV * P.KEV_IN_K, T)
    T_i = (T - f_e * T_e) / jnp.maximum(1.0 - f_e, 1e-30)
    T_mean = f_e * T_e + (1.0 - f_e) * T_i
    shocked = t_shock_s > 0.0
    edges = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, n_substeps)])

    def frac(h, tau):
        ok = shocked & (tau > 0.0)
        return -jnp.expm1(-jnp.clip(jnp.where(ok, h, 0.0) / jnp.where(ok, tau, 1.0), 0.0, 50.0))

    @jax.checkpoint
    def step(carry, ab):
        Te, Ti, acc = carry
        h = (ab[1] - ab[0]) * t_shock_s
        fr = frac(0.5 * h, equipartition_time(Te, Ti, n_b, m) * (1.0 - f_e) / teq_scale)
        Te_m, Ti_m = Te + fr * (T_mean - Te), Ti + fr * (T_mean - Ti)
        f = frac(h, equipartition_time(Te_m, Ti_m, n_b, m) * (1.0 - f_e) / teq_scale)
        Te1 = Te + f * (T_mean - Te)
        # the trapezoid in the fraction of the time since the shock (T_scale'd:
        # the accumulator is O(1), not O(1e7), in float32)
        acc = acc + 0.5 * (Te + Te1) / _T_SCALE * (ab[1] - ab[0])
        return (Te1, Ti + f * (T_mean - Ti), acc), None

    ab = jnp.asarray(np.stack([edges[:-1], edges[1:]], 1))
    (T_e1, _, acc), _ = jax.lax.scan(step, (T_e, T_i, jnp.zeros_like(T_e)), ab)
    rho = acc * _T_SCALE / jnp.where(shocked, T_e1, 1.0)
    return jnp.where(shocked, T_e1, T), jnp.where(shocked, rho, 1.0)


def _electron_temperature_v1(T, n_b, t_shock_s, m, kT_e_shock_keV=0.3, n_substeps=48, teq_scale=1.0):
    """The pre-2026-09-25 implementation (kept for the regression test that the
    history accumulator did not change T_e)."""
    f_e = m["mu_i"] / (m["mu_i"] + m["mu_e"])
    T_e = jnp.minimum(kT_e_shock_keV * P.KEV_IN_K, T)
    T_i = (T - f_e * T_e) / jnp.maximum(1.0 - f_e, 1e-30)
    T_mean = f_e * T_e + (1.0 - f_e) * T_i
    shocked = t_shock_s > 0.0
    edges = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, n_substeps)])

    def frac(h, tau):
        ok = shocked & (tau > 0.0)
        return -jnp.expm1(-jnp.clip(jnp.where(ok, h, 0.0) / jnp.where(ok, tau, 1.0), 0.0, 50.0))

    # checkpointed: reverse mode keeps the (T_e, T_i) carry per substep, not
    # every intermediate (~5000 floats per cell: 20 GiB per 1 M-cell slab)
    @jax.checkpoint
    def step(carry, ab):
        Te, Ti = carry
        h = (ab[1] - ab[0]) * t_shock_s
        fr = frac(0.5 * h, equipartition_time(Te, Ti, n_b, m) * (1.0 - f_e) / teq_scale)
        Te_m, Ti_m = Te + fr * (T_mean - Te), Ti + fr * (T_mean - Ti)
        f = frac(h, equipartition_time(Te_m, Ti_m, n_b, m) * (1.0 - f_e) / teq_scale)
        return (Te + f * (T_mean - Te), Ti + f * (T_mean - Ti)), None

    ab = jnp.asarray(np.stack([edges[:-1], edges[1:]], 1))
    (T_e, _), _ = jax.lax.scan(step, (T_e, T_i), ab)
    return jnp.where(shocked, T_e, T)


def plasma(fields, split="xrism_bulk", kT_e_shock_keV=0.3, teq_scale=1.0, fe_scale=1.0,
           csm_solar=False):
    """Cell-wise n_e, per-element number densities, kT_e, n_e t (scaled arithmetic).

    Emission-physics knobs (all traceable): ``kT_e_shock_keV`` the post-shock
    electron temperature (Ghavamian et al. 2007), ``teq_scale`` a multiplier on
    the Coulomb equilibration RATE (> 1: collisionless heating downstream),
    ``fe_scale`` a multiplier on the Fe-group tracer (the nucleosynthesis yield
    the hydro does not care about). ``csm_solar``: solar element ratios for the
    circumstellar part of each tracer (``element_fractions``).

    Also returns ``rho_hist``, the NEI history label (``_relax_history``): v2
    tables with a history axis use it, every other table ignores it.
    """
    if not (isinstance(fe_scale, float) and fe_scale == 1.0):
        fields = dict(fields, C_Fe=jnp.asarray(fields["C_Fe"]) * fe_scale)
    X = element_fractions(fields, split, csm_solar=csm_solar)
    m = moments(X)
    rho_code = jnp.maximum(jnp.asarray(fields["rho"]), 1e-30)
    n_b = rho_code * _NB_PER_CODE_RHO
    T = _T_PER_CODE * m["mu"] * (jnp.asarray(fields["press"]) / rho_code)
    t_s = jnp.asarray(fields["time_since_shock"]) * P.CODE_TIME
    T_e, rho_hist = _relax_history(T, n_b, t_s, m, kT_e_shock_keV, teq_scale=teq_scale)
    net = jnp.asarray(fields["density_time"]) * _NET_PER_CODE_DT / m["mu_e"]
    dens = {el: n_b * (X[el] / P.ATOMIC[el][0]) for el in X}
    shocked = jnp.asarray(fields["shocked_fraction"]) > 0.5
    f_e = m["mu_i"] / (m["mu_i"] + m["mu_e"])
    T_i = (T - f_e * T_e) / jnp.maximum(1.0 - f_e, 1e-30)
    return dict(n_e=n_b / m["mu_e"], dens=dens, kT=T_e / P.KEV_IN_K, net=net, shocked=shocked,
                T_i=T_i, mu_i=m["mu_i"], n_b=n_b, t_shock_s=t_s, rho_hist=rho_hist)


_plasma_impl = plasma          # (name kept for the validation scripts)


def phase_factors(chi, f_mass, net_mode="unchanged"):
    """``_subgrid.phase_factors`` for traced ``chi`` / ``f_mass`` (same formulas;
    concrete floats go through the original, with its range checks)."""
    if isinstance(chi, (int, float)) and isinstance(f_mass, (int, float)):
        return _subgrid.phase_factors(chi, f_mass, net_mode)
    f_vol = f_mass / chi
    rho_u = (1.0 - f_mass) / (1.0 - f_vol)
    net_d = {"density": chi, "unchanged": 1.0, "crossing": jnp.sqrt(chi)}[net_mode]
    return {"dense": (chi, net_d / chi, net_d, f_vol), "diffuse": (rho_u, 1.0, rho_u, 1.0 - f_vol)}


def emitting_components(fields, chi=4.0, f_mass=0.34, population="ejecta",
                        net_mode="unchanged", split="xrism_bulk"):
    """The components casa_observe emits: (fields, em_weight) pairs.

    With the ejecta-only sub-grid split: the unsplit circumstellar part with
    weight (1 - C_ej), and the dense / diffuse phases of the ejecta, each at
    fixed pressure with rho, t_shock, density_time scaled by
    ``_subgrid.phase_factors`` and weight (volume fraction x C_ej).
    """
    if chi is None or chi <= 1.0:
        return [(fields, 1.0)]
    c_ej = jnp.clip(jnp.asarray(fields["C_ej"]), 0.0, 1.0) if population == "ejecta" else 1.0
    comps = [(fields, 1.0 - c_ej)] if population == "ejecta" else []
    for _, (rho_f, t_f, net_f, vol) in phase_factors(chi, f_mass, net_mode).items():
        f = dict(fields)
        f["rho"] = fields["rho"] * rho_f
        f["time_since_shock"] = fields["time_since_shock"] * t_f
        f["density_time"] = fields["density_time"] * net_f
        comps.append((f, vol * c_ej))
    return comps
# =============================================================================
# ============ ↑ The plasma chain ↑ ===========================================
# =============================================================================


# =============================================================================
# ============ ↓ Geometry, table weights, slabs ↓ =============================
# =============================================================================
def sky_geometry(box_pc, n, distance_kpc):
    """Per-(x, z) sky offsets (arcsec; west = +x, north = +z) and projected radius."""
    xs = (np.arange(n) + 0.5) * box_pc / n - 0.5 * box_pc
    to_arcsec = ARCSEC_PER_RAD / (distance_kpc * 1e3)
    WX, NZ = np.meshgrid(xs * to_arcsec, xs * to_arcsec, indexing="ij")
    return WX, NZ, np.hypot(WX, NZ)


def bilinear(lkt_grid, lnet_grid, kT, net):
    """Corner indices (base clipped to n-2) and weights in (log kT, log n_e t).

    Uniform log grids are assumed (the _nei grids are); the base index is clipped
    to n - 2 with the fraction allowed to reach 1, so the upper corner never
    indexes past the table (float32 cannot represent n - 1.000001).
    """
    def axis(grid, v):
        n = grid.shape[0]
        x = (jnp.log10(jnp.maximum(v, 1e-30)) - grid[0]) / (grid[1] - grid[0])
        x = jnp.clip(x, 0.0, float(n - 1))
        i = jnp.minimum(jnp.floor(x).astype(jnp.int32), n - 2)
        return i, x - i
    i, fi = axis(lkt_grid, kT)
    j, fj = axis(lnet_grid, net)
    return i, j, fi, fj


def flux_norm(cell_cm3, distance_kpc):
    """XSPEC norm per unit n_e n_X: 1e-14 V / (4 pi D^2), a host scalar.

    Folded into the per-cell weights BEFORE any sum: n_e n_X V alone is ~1e53
    in cgs, beyond float32 (3.4e38), and the whole model runs in float32.
    """
    return float(1e-14 * cell_cm3 / (4.0 * np.pi * (distance_kpc * KPC_CM) ** 2))


def component_weights(pl, tables, norm, emit_weight):
    """Per-cell XSPEC-norm weights per table component, (n_comp,) + grid."""
    ok = pl["shocked"] & (pl["kT"] >= KT_MIN_KEV)
    base = jnp.where(ok, pl["n_e"] * norm * emit_weight, 0.0)
    return jnp.stack([base * pl["dens"].get(name, 0.0) / rs
                      for name, rs in zip(tables["names"], tables["r_sun"])])


def rho_axis(rho_grid, rho):
    """Base index and fraction on the (non-uniform) NEI-history grid, clipped
    with ``where`` (no 0.5 derivative at the bounds, cf. jnp.clip)."""
    g = jnp.asarray(rho_grid)
    n = g.shape[0]
    x = jnp.where(rho < g[0], g[0], jnp.where(rho > g[-1], g[-1], rho))
    k = jnp.clip(jnp.searchsorted(g, x, side="right") - 1, 0, n - 2).astype(jnp.int32)
    return k, (x - g[k]) / (g[k + 1] - g[k])


def corners(tables, pl):
    """The bilinear (legacy) or trilinear (v2 history axis) corners as stacked
    arrays: indices into the COMBINED kT x rho axis (n_c, ...), n_e t indices
    (n_c, ...) and weights (n_c, ...)."""
    i, j, fi, fj = bilinear(tables["lkt"], tables["lnet"], pl["kT"], pl["net"])
    nr = int(tables.get("n_rho", 1))
    if nr == 1:
        ii = jnp.stack([i, i, i + 1, i + 1])
        jj = jnp.stack([j, j + 1, j, j + 1])
        ww = jnp.stack([(1 - fi) * (1 - fj), (1 - fi) * fj, fi * (1 - fj), fi * fj])
        return ii, jj, ww
    k, fk = rho_axis(tables["rho"], pl.get("rho_hist", jnp.ones_like(pl["kT"])))
    ii, jj, ww = [], [], []
    for di, wi in ((0, 1 - fi), (1, fi)):
        for dk, wk in ((0, 1 - fk), (1, fk)):
            for dj, wj in ((0, 1 - fj), (1, fj)):
                ii.append((i + di) * nr + k + dk); jj.append(j + dj); ww.append(wi * wk * wj)
    return jnp.stack(ii), jnp.stack(jj), jnp.stack(ww)


def kt_pairs(tables, pl):
    """For log-kT interpolation: the lower kT index ``i`` and fraction ``fi``,
    and the remaining corners [(combined offset of i, n_e t index, weight)], so
    the table is combined GEOMETRICALLY between kT nodes i and i + 1."""
    i, j, fi, fj = bilinear(tables["lkt"], tables["lnet"], pl["kT"], pl["net"])
    nr = int(tables.get("n_rho", 1))
    if nr == 1:
        return i, fi, nr, [(i, j, 1 - fj), (i, j + 1, fj)]
    k, fk = rho_axis(tables["rho"], pl.get("rho_hist", jnp.ones_like(pl["kT"])))
    return i, fi, nr, [(i * nr + k + dk, j + dj, wk * wj)
                       for dk, wk in ((0, 1 - fk), (1, fk)) for dj, wj in ((0, 1 - fj), (1, fj))]


def kt_tent(tables, pl):
    """Dense interpolation weights over the combined kT x rho axis, (..., n_kt * n_rho)."""
    i, _, fi, _ = bilinear(tables["lkt"], tables["lnet"], pl["kT"], pl["net"])
    nkt = tables["lkt"].shape[0]
    t = tent(i, fi, nkt)
    nr = int(tables.get("n_rho", 1))
    if nr == 1:
        return t
    k, fk = rho_axis(tables["rho"], pl.get("rho_hist", jnp.ones_like(pl["kT"])))
    return (t[..., :, None] * tent(k, fk, nr)[..., None, :]).reshape(*t.shape[:-1], nkt * nr)


STATE_KEYS = ("rho", "press", "vy", "C_ej", "C_Fe", "C_Si", "C_O", "C_He",
              "shocked_fraction", "time_since_shock", "density_time")


def slab_scan(fields, slab, body, init, cells_per_slab=2 ** 20):
    """Accumulate ``body(carry, slab_fields)`` over line-of-sight (y) slabs.

    Reverse mode through the whole cube stores every intermediate of the plasma
    chain (the 48-step relaxation history alone is ~6 GB at 256^3, ~48 GB at
    512^3 -- Codex review). Scanning over slabs of ``slab`` y-planes with a
    checkpointed body keeps only the small carry (a DEM, an image, the sector
    moments) plus one slab's recomputation.
    """
    n = fields["rho"].shape[1]
    sh = SHARD if (SHARD is not None and SHARD.active()) else None
    if sh is not None:
        # the per-DEVICE scratch is what the budget is for: N x the cells per
        # slab over N x-slabs (else 512^3 on 8 GPUs is 512 one-plane steps)
        cells_per_slab = cells_per_slab * sh.NDEV
    if slab is None:
        # auto: a fixed number of cells per slab, so the per-slab scratch (and
        # the peak memory) does not grow 4x from 256^3 to 512^3 (Codex review)
        slab = max(1, cells_per_slab // (fields["rho"].shape[0] * fields["rho"].shape[2]))
        while n % slab:
            slab -= 1
    if n % slab:
        raise ValueError(f"slab {slab} must divide {n}")
    xs = {k: jnp.moveaxis(jnp.asarray(fields[k]), 1, 0).reshape(n // slab, slab,
                                                               *fields[k].shape[::2])
          for k in STATE_KEYS if k in fields}
    if sh is not None:          # (n_slab, slab, x, z): x split
        xz = (fields["rho"].shape[0], fields["rho"].shape[2])

        def ccols(a):           # only the sky-plane (..., x, z) carries (not e.g. a DEM)
            return sh.ccols(a) if a.ndim >= 2 and tuple(a.shape[-2:]) == xz else a
        xs = {k: sh.constrain(v, sh.P(None, None, "x", None)) for k, v in xs.items()}
        init = jax.tree.map(ccols, init)

    @jax.checkpoint
    def step(carry, sl):
        sl = {k: jnp.moveaxis(v, 0, 1) for k, v in sl.items()}       # (x, slab, z)
        out = body(carry, sl)
        if sh is not None:
            out = jax.tree.map(ccols, out)
        return out, None

    carry, _ = jax.lax.scan(step, init, xs)
    return carry
# =============================================================================
# ============ ↑ Geometry, table weights, slabs ↑ =============================
# =============================================================================


# =============================================================================
# ============ ↓ Observables ↓ ================================================
# =============================================================================
def tent(i, f, n):
    """Dense linear-interpolation weights, (..., n): 1 - f at i, f at i + 1."""
    return jax.nn.one_hot(i, n) * (1.0 - f)[..., None] + jax.nn.one_hot(i + 1, n) * f[..., None]


def aperture_modes(halo, tables, R, aperture_arcsec, rank=3):
    """Low-rank split of the aperture response, keep[ch, r] ~ sum_k U[ch, k] V_k(r).

    Returns ``U`` (n_ch, K) and the per-(x, z) radial mode weights (K, x, z).
    Without a halo: K = 1, U = 1, V = (R < aperture). The Monte-Carlo keep
    table is smooth in r; rank 3 reproduces it to 0.007 (its MC noise level).
    """
    if halo is None:
        return np.ones((len(np.asarray(tables["e_ch"])), 1)), (R < aperture_arcsec)[None].astype(np.float32)
    if abs(float(halo["aperture_arcsec"]) - aperture_arcsec) > 1e-6:
        raise ValueError("halo keep table built for a different aperture")
    Uf, sv, Vt = np.linalg.svd(np.asarray(halo["aperture_keep"]), full_matrices=False)
    e_ch = np.asarray(tables["e_ch"])
    U = np.stack([np.interp(e_ch, halo["e_grid"], Uf[:, k] * sv[k]) for k in range(rank)], 1)
    rg = np.asarray(halo["r_grid"])
    # beyond the tabulated radii (> 220") nothing lands inside a 200" aperture
    V = np.stack([np.where(R <= rg[-1], np.interp(R, rg, Vt[k]), 0.0) for k in range(rank)])
    return U, V.astype(np.float32)


def aperture_spectrum(fields, tables, *, box_pc, distance_kpc, halo=None,
                      aperture_arcsec=200.0, v_los_kms=True, subgrid=dict(chi=4.0, f_mass=0.34),
                      split="xrism_bulk", slab=None, precision=jax.lax.Precision.HIGHEST, plasma_kw=None):
    """Count rate per channel inside the aperture (counts/s), differentiable.

    Every cell's per-component weight is spread over the (kT, n_e t) grid with
    its bilinear tent weights into a DEM, per aperture mode and for the Doppler
    moment (weight x beta), then contracted with the tables:

        rate[h] = sum_{c,t,n,k} (DEM_ck[t, n] C_c[t, n, h] + DEMb_ck D_c) U[h, k]

    The DEM is a dense tensor contraction (matmul), NOT a scatter: a
    segment_sum of ~10^6 cells into the few hot DEM bins serialises on atomics
    (4 M updates into 25 bins: 2.9 s on an A100, vs 0.5 ms into 65 k bins).
    """
    n = fields["rho"].shape[0]
    _, _, R = sky_geometry(box_pc, n, distance_kpc)
    norm = flux_norm((box_pc / n * PC_CM) ** 3, distance_kpc)
    nr = int(tables.get("n_rho", 1))
    nkt, nnet = tables["lkt"].shape[0] * nr, tables["lnet"].shape[0]     # combined kT x rho axis
    U, V = aperture_modes(halo, tables, R, aperture_arcsec)
    K = V.shape[0]
    V = jnp.asarray(V[:, :, None, :])                                   # (K, x, 1, z)
    ncomp = len(tables["names"])

    def body(dem, sl):
        beta = sl["vy"] * 1e3 / C_KMS if v_los_kms else jnp.zeros_like(sl["rho"])
        for fcomp, ew in emitting_components(sl, split=split, **subgrid):
            pl = plasma(fcomp, split, **(plasma_kw or {}))
            w = component_weights(pl, tables, norm, ew)                  # (c, x, s, z)
            _, j, _, fj = bilinear(tables["lkt"], tables["lnet"], pl["kT"], pl["net"])
            q = jnp.stack([w, w * beta])[:, :, None] * V[None, None]    # (2, c, K, x, s, z)
            q = q.reshape(-1, q[0, 0, 0].size)                            # (a, p)
            # two explicit matmul stages; a three-operand einsum let XLA pick a
            # (p, t, n) outer product in the transpose (a 21 GiB buffer)
            qt = (q[:, :, None] * kt_tent(tables, pl).reshape(1, -1, nkt))  # (a, p, t)
            qt = jnp.swapaxes(qt, 1, 2).reshape(-1, q.shape[1])          # (a t, p)
            dem = dem + jnp.dot(qt, tent(j, fj, nnet).reshape(-1, nnet),
                                precision=precision).reshape(dem.shape)
        return dem

    # the (a, p, t) buffer grows with the history axis: fewer cells per slab
    dem = slab_scan(fields, slab, body, jnp.zeros((2, ncomp, K, nkt, nnet)),
                    cells_per_slab=max(2 ** 17 // nr, 2 ** 12))
    U = jnp.asarray(U, jnp.float32)
    return jnp.einsum("cktn,ctnh,hk->h", dem[0], tables["C"], U, precision=precision) + \
        jnp.einsum("cktn,ctnh,hk->h", dem[1], tables["D"], U, precision=precision)


def band_rates(rate, tables, bands=BANDS):
    return jnp.asarray(band_masks(tables["e_ch"], bands)) @ rate


def band_tables_of(tables, bands=BANDS):
    """Band-integrated tables (Cb, Db), each (n_comp, n_kt, n_net, n_band)."""
    b = tables.get("binning")
    if b == "band" and bands == BANDS:
        return tables["C"], tables["D"]                   # v2: already on the exact bands
    if b is not None:
        # v2 spec / dop bins are not channels: a centre mask would silently put
        # whole 0.2 keV bins (or M0 + M1) into bands (review 2026-09-25)
        raise ValueError(f"band_tables_of needs channel (v1) or v2 'band' tables, not binning={b!r}")
    bm = jnp.asarray(band_masks(tables["e_ch"], bands))
    return (jnp.einsum("ctnh,bh->ctnb", tables["C"], bm),
            jnp.einsum("ctnh,bh->ctnb", tables["D"], bm))


KT_INTERP = ("linear", "log")
_TINY = 1e-30
#: k / (m_u c^2) per K: the thermal (beta)^2 per unit T / A
_KT_PER_MC2 = float(P.K_B / (1.66053906660e-24 * (C_KMS * 1e5) ** 2))


def band_columns(fields, tables, *, box_pc, distance_kpc, subgrid=dict(chi=4.0, f_mass=0.34),
                 split="xrism_bulk", v_los_kms=True, slab=None, band_tables=None, plasma_kw=None,
                 kt_interp="linear", thermal_broadening=False):
    """Counts/s per band summed along the line of sight, (n_band, x, z).

    ``band_tables = (Cb, Db)``, each (n_comp, n_kt, n_net, n_band), replaces the
    band integration of ``tables`` -- e.g. traced in N_H, or mixed between two
    ACIS cycles for an epoch between them. A third entry ``D2b`` adds the
    second-order Doppler term, ``Cb + beta Db + beta^2 / 2 D2b`` (v2 tables,
    ``second_order=True``): the line broadening by a spread of line-of-sight
    velocities, which the first-order term cannot represent.

    ``thermal_broadening`` (needs ``D2b``): add the ions' thermal velocity
    spread to the second-order term, beta^2 -> beta^2 + k T_i / (mu_i m_u c^2)
    -- the same velocity dispersion for every species, as mass-proportional
    shock heating gives (before ion-ion equilibration; an upper bound for the
    heavy ions afterwards). APEC broadens only at T_e, negligible here; at the
    reverse shock sigma_v ~ 1000-2000 km/s is 20-45 eV at Fe-K.

    ``kt_interp="log"``: interpolate the table geometrically between kT nodes
    (log C linear in log kT; the Doppler terms scaled by the same factor).
    Linear interpolation of an exp(-E/kT) Boltzmann tail across a 14.6 % kT
    step over-estimates Fe-K by up to ~40 % at low kT (obs_model section 9).
    The bin axis may be anything (bands, spectral bins, Doppler moments, with
    N_H folded in); the tables may carry the v2 NEI-history axis (``corners``).

    The expensive, field-dependent half of the image model (the plasma chain
    per cell); everything after it -- roll, offset, pixelisation, halo, PSF --
    is ``project_columns`` and does not touch the fields again.
    """
    if kt_interp not in KT_INTERP:
        raise ValueError(f"kt_interp {kt_interp!r} not in {KT_INTERP}")
    n = fields["rho"].shape[0]
    norm = flux_norm((box_pc / n * PC_CM) ** 3, distance_kpc)
    if band_tables is None:
        band_tables = band_tables_of(tables)
    Cb, Db = band_tables[0], band_tables[1]
    D2b = band_tables[2] if len(band_tables) > 2 else None
    ncomp = len(tables["names"])
    # the corner lookup takes its grids from ``tables``: band_tables built with a
    # different history setting would be gathered with clamped (silently wrong)
    # indices (review 2026-09-25: 12 % off, no error)
    n_t = tables["lkt"].shape[0] * int(tables.get("n_rho", 1))
    if tuple(Cb.shape[:3]) != (ncomp, n_t, tables["lnet"].shape[0]):
        raise ValueError(f"band_tables {tuple(Cb.shape[:3])} do not match the (comp, kT x rho, n_e t) grid "
                         f"of tables {(ncomp, n_t, tables['lnet'].shape[0])} (history setting?)")
    cidx = jnp.arange(ncomp)[:, None, None, None, None]

    def gather(T, a, b):
        return T[cidx[:, 0], a[None], b[None]]                           # (c, x, s, z, band)

    if thermal_broadening and D2b is None:
        raise ValueError("thermal_broadening needs the second-order tables (band_tables[2])")

    def body(cols, sl):
        beta = (sl["vy"] * 1e3 / C_KMS)[None, ..., None] if v_los_kms else None
        for fcomp, ew in emitting_components(sl, split=split, **subgrid):
            pl = plasma(fcomp, split, **(plasma_kw or {}))
            w = component_weights(pl, tables, norm, ew)                  # (c, x, s, z)
            # second-order weight: beta^2 (+ the thermal spread), per cell
            b2 = None
            if D2b is not None and (v_los_kms or thermal_broadening):
                b2 = beta ** 2 if v_los_kms else 0.0
                if thermal_broadening:
                    b2 = b2 + (_KT_PER_MC2 * jnp.maximum(pl["T_i"], 0.0) / pl["mu_i"])[None, ..., None]
            if kt_interp == "linear":
                ii, jj, ww = corners(tables, pl)                         # (4 or 8, x, s, z)
                # one corner at a time: the gathered table is (c, cells, band)
                # per corner, and with N_H folded into the band axis (30
                # "bands") all four at once was a 26 GiB buffer at 256^3
                for q in range(ii.shape[0]):
                    tab = gather(Cb, ii[q], jj[q])
                    if v_los_kms:
                        tab = tab + beta * gather(Db, ii[q], jj[q])
                    if b2 is not None:
                        tab = tab + 0.5 * b2 * gather(D2b, ii[q], jj[q])
                    cols = cols + jnp.einsum("cxszb,cxsz->bxz", tab, w * ww[q][None])
            else:
                i, fi, nr, rest = kt_pairs(tables, pl)
                f1 = fi[None, ..., None]
                f0 = 1.0 - f1
                for base, jq, wq in rest:
                    a0, a1 = gather(Cb, base, jq), gather(Cb, base + nr, jq)
                    lgeo = f0 * jnp.log(jnp.maximum(a0, _TINY)) + f1 * jnp.log(jnp.maximum(a1, _TINY))
                    geo = jnp.exp(lgeo)
                    tab = geo
                    if v_los_kms or b2 is not None:
                        lin = f0 * a0 + f1 * a1
                        # geo / lin as exp(log geo - log lin): the quotient rule's
                        # 1 / lin^2 underflows to 1/0 in float32 for lin ~ 1e-30 and
                        # made the tangent NaN (0 * inf) in every unshocked cell
                        ok = lin > _TINY
                        sc = jnp.exp(lgeo - jnp.log(jnp.where(ok, lin, 1.0))) * ok
                        dd = 0.0
                        if v_los_kms:
                            dd = beta * (f0 * gather(Db, base, jq) + f1 * gather(Db, base + nr, jq))
                        if b2 is not None:
                            dd = dd + 0.5 * b2 * (f0 * gather(D2b, base, jq) + f1 * gather(D2b, base + nr, jq))
                        tab = tab + dd * sc
                    cols = cols + jnp.einsum("cxszb,cxsz->bxz", tab, w * wq[None])
        return cols

    per = max(2 ** 20 * 6 // Cb.shape[-1], 2 ** 14)
    return slab_scan(fields, slab, body, jnp.zeros((Cb.shape[-1], n, n)),
                     cells_per_slab=per if kt_interp == "linear" else max(per // 3, 2 ** 13))


def project_columns(cols, *, box_pc, distance_kpc, halo=None, npix=256, pix_arcsec=4 * 0.492,
                    psf_sigma_arcsec=0.0, roll_deg=0.0, offset_arcsec=(0.0, 0.0), sky_scale=1.0):
    """Band columns -> counts/s per pixel, (n_band, npix, npix); row = north, col = west.

    Matches the real epoch images' orientation (``read_events``: px increases to
    the WEST, py to the NORTH; ``casa_jaxobs_data``). ``roll_deg`` rotates the
    model on the sky (sim PA + roll = sky PA, PA from west through north) and
    ``offset_arcsec`` = (west, north) moves the explosion centre; both are
    traced, i.e. fit parameters, as is ``sky_scale`` (angular sizes are
    computed at ``distance_kpc`` and multiplied by it: pass D_ref / D for a
    traced distance D; the FLUX scaling (D_ref / D)^2 is the caller's). Rendered on a guard-padded grid so emission
    outside the field can scatter into it, then convolved and cropped.
    """
    n = cols.shape[-1]
    WX, NZ, _ = sky_geometry(box_pc, n, distance_kpc)
    guard = int(halo["kernel_half"]) if halo is not None else 0
    if psf_sigma_arcsec > 0:
        guard += int(np.ceil(4 * psf_sigma_arcsec / pix_arcsec))
    ntot = npix + 2 * guard
    ps = jnp.deg2rad(roll_deg)
    WXj, NZj = jnp.asarray(WX, jnp.float32), jnp.asarray(NZ, jnp.float32)
    wr = sky_scale * (jnp.cos(ps) * WXj - jnp.sin(ps) * NZj) + offset_arcsec[0]
    nr = sky_scale * (jnp.sin(ps) * WXj + jnp.cos(ps) * NZj) + offset_arcsec[1]
    # each cell column is supersampled m x m: a bilinear splat of one point per
    # cell leaves Moire when the cell is not << the pixel
    cell0 = float(WX[1, 0] - WX[0, 0])
    m = max(1, int(np.ceil(1.8 * cell0 / pix_arcsec)))       # margin for sky_scale <= 1.2
    cell = cell0 * sky_scale
    subs = (np.arange(m) + 0.5) / m - 0.5
    flat = cols.reshape(cols.shape[0], -1)
    img = jnp.zeros((cols.shape[0], ntot * ntot))
    for sw in subs:
        for sn in subs:
            col = (wr + sw * cell) / pix_arcsec + 0.5 * ntot - 0.5
            row = (nr + sn * cell) / pix_arcsec + 0.5 * ntot - 0.5
            c0 = jnp.floor(col).astype(jnp.int32); r0 = jnp.floor(row).astype(jnp.int32)
            fc, fr = col - c0, row - r0
            for dc, wc in ((0, 1 - fc), (1, fc)):
                for dr, wr_ in ((0, 1 - fr), (1, fr)):
                    cc, rr = c0 + dc, r0 + dr
                    ok = (cc >= 0) & (cc < ntot) & (rr >= 0) & (rr < ntot)
                    idx = jnp.where(ok, rr * ntot + cc, 0).ravel()
                    wt = (wc * wr_ * ok).ravel() / m ** 2
                    img = img + jax.vmap(lambda a: jax.ops.segment_sum(a * wt, idx, ntot * ntot))(flat)
    img = img.reshape(-1, ntot, ntot)
    if halo is not None:
        if abs(float(halo["kernel_pixel_arcsec"]) - pix_arcsec) > 1e-6:
            raise ValueError("halo kernel pixel != image pixel")
        img = convolve_same(img, jnp.asarray(halo["kernel"]))
    if psf_sigma_arcsec > 0:
        g = int(np.ceil(4 * psf_sigma_arcsec / pix_arcsec))
        x = np.arange(-g, g + 1)
        k = np.exp(-0.5 * (x[:, None] ** 2 + x[None, :] ** 2) / (psf_sigma_arcsec / pix_arcsec) ** 2)
        img = convolve_same(img, jnp.asarray(np.broadcast_to(k / k.sum(), (img.shape[0],) + k.shape)))
    return img[:, guard:guard + npix, guard:guard + npix]


def band_images(fields, tables, *, box_pc, distance_kpc, halo=None, subgrid=dict(chi=4.0, f_mass=0.34),
                split="xrism_bulk", v_los_kms=True, slab=None, **proj):
    """``project_columns(band_columns(...))``: counts/s per pixel per band."""
    cols = band_columns(fields, tables, box_pc=box_pc, distance_kpc=distance_kpc, subgrid=subgrid,
                        split=split, v_los_kms=v_los_kms, slab=slab)
    return project_columns(cols, box_pc=box_pc, distance_kpc=distance_kpc, halo=halo, **proj)


def convolve_same(img, K):
    """Linear (zero-padded) 'same' FFT convolution with odd, centred kernels."""
    npix = img.shape[-1]
    kh = K.shape[-1] // 2
    size = npix + 2 * kh
    f = lambda a: jnp.fft.rfft2(a, s=(size, size))        # noqa: E731
    out = jnp.fft.irfft2(f(img) * f(K), s=(size, size))
    return out[:, kh:kh + npix, kh:kh + npix]


# ---- synchrotron (the chain of _synchrotron.synchrotron_fields, traced) ----
import _synchrotron as SY
_VS2_PER_TI_MU = float(16.0 / 3.0 * P.K_B / P.M_P)                   # v_s^2 = this * T_i / mu_i
_WIDTH_CM = float(SY.ARCSEC_IN_PC * 3.0857e18)
_YR_S = 3.155693e7
_E_RADIO_KEV = SY.RADIO_FREQ_GHZ * 1e9 * 4.135667696e-18


def load_sync_tables(instrument, table_dir=TABLE_DIR, bands=BANDS):
    """Band-integrated synchrotron tables: ``(ln_ecut grid, S[nh, ecut, band])``
    in counts/s per unit ``k w`` (see ``casa_jaxobs_tables.build_sync``)."""
    d = np.load(Path(table_dir) / f"sync_{instrument}.npz")
    e = 0.5 * (d["ch_edges"][1:] + d["ch_edges"][:-1])
    bm = band_masks(e, bands)
    return np.log(d["ecut"]), np.einsum("nkc,bc->nkb", d["S"], bm).astype(np.float32)


def sync_columns(fields, *, lecut, band_table, year, eta=1.0, width_arcsec=2.0,
                 split="xrism_bulk", slab=None, gate_softness=0.15, plasma_kw=None):
    """Non-thermal (synchrotron) counts/s per band along the line of sight, (n_band, x, z).

    As ``_synchrotron.synchrotron_fields``: relative radio weight rho v_s^2 over
    every shocked cell, v_s from the ion temperature, normalised so the summed
    radio emission equals Cas A's 1 GHz flux at ``year`` (secular decline
    included -- so the component is distance-independent); X-rays only from
    cells shocked within the filament-width gate (smoothed, ``gate_softness``
    of the gate), at each cell's loss-limited cutoff E_c = 0.55 keV
    (v_s / 3000 km/s)^2 / eta. ``band_table`` (n_ecut, n_band) is the folded
    response at the epoch's instrument and N_H. The fitted efficiency
    (``--sync-norm`` in casa_observe) is the caller's multiplier.
    """
    lec = jnp.asarray(lecut)
    tab = jnp.asarray(band_table)

    def body(carry, sl):
        cols, wsum = carry
        pl = plasma(sl, split, **(plasma_kw or {}))
        vs2 = _VS2_PER_TI_MU * jnp.maximum(pl["T_i"], 0.0) / pl["mu_i"]            # (cm/s)^2
        w = jnp.where(pl["shocked"], pl["n_b"] * vs2 * 1e-16, 0.0)                  # rho v^2, scaled
        gate = _WIDTH_CM * width_arcsec / jnp.maximum(jnp.sqrt(vs2) / 4.0, 1e3)    # s
        tss = pl["t_shock_s"]
        fresh = jax.nn.sigmoid((gate - tss) / (gate_softness * gate)) * (tss > 0.0)
        ecut = SY.CUTOFF_KEV_AT_3000 * vs2 / 9.0e16 / eta
        x = jnp.clip((jnp.log(jnp.maximum(ecut, 1e-30)) - lec[0]) / (lec[1] - lec[0]),
                     0.0, lec.shape[0] - 1.0)
        i = jnp.minimum(jnp.floor(x).astype(jnp.int32), lec.shape[0] - 2)
        f = x - i
        # log-linear in the band rate (it falls by decades below the cutoff)
        lt = jnp.log(jnp.maximum(tab, 1e-38))
        rate = jnp.exp(lt[i] * (1.0 - f)[..., None] + lt[i + 1] * f[..., None])     # (x, s, z, b)
        cols = cols + jnp.einsum("xszb,xsz->bxz", rate, w * fresh)
        return cols, wsum + jnp.sum(w)

    n = fields["rho"].shape[0]
    cols, wsum = slab_scan(fields, slab, body, (jnp.zeros((tab.shape[-1], n, n)), jnp.zeros(())))
    nu_s_nu = SY.RADIO_FREQ_GHZ * 1e9 * 1e-23 * SY.RADIO_FLUX_JY * \
        (1.0 - SY.RADIO_SECULAR_DECLINE_PER_YR) ** (year - SY.RADIO_EPOCH)
    k = nu_s_nu / (jnp.maximum(wsum, 1e-30) * _E_RADIO_KEV ** (1.0 - SY.RADIO_ALPHA))
    return cols * k


def doppler_moment_tables(tables):
    """(Cm, Dm), each (n_comp, n_kt [* n_rho], n_net, 2): the M0 / M1 moments of a
    v2 ``binning="dop"`` table set as two "bands" for ``band_columns``."""
    if tables.get("binning") != "dop":
        raise ValueError("need load_binned_tables(..., binning='dop')")
    return tables["C"], tables["D"]


def doppler_columns(fields, dop_tables, *, box_pc, distance_kpc, subgrid=dict(chi=4.0, f_mass=0.34),
                    split="xrism_bulk", slab=None, plasma_kw=None, kt_interp="linear", band_tables=None):
    """Per sky column, the two Doppler moments (M0 + beta D0, M1 + beta D1),
    (2, x, z), from the NATIVE-channel moment tables (v2 ``binning="dop"``):
    M0 = counts in [1.78, 1.94] keV, M1 = sum (E - E0) counts -- the data
    statistic's numerator and denominator, exactly (obs_model section 8: the
    v1 statistic took two 117 eV channels spanning 1.752-1.986 keV, 1.38x the
    data's velocity sensitivity). ``band_tables`` overrides the moments (e.g.
    N_H-folded, (c, t, n, n_nh * 2)); the synchrotron continuum, which dilutes
    the data's centroid too, is ``sync_columns`` with ``load_sync_binned(inst,
    "dop")``, added to these columns before ``doppler_from_columns``."""
    return band_columns(fields, dop_tables, box_pc=box_pc, distance_kpc=distance_kpc, subgrid=subgrid,
                        split=split, v_los_kms=True, slab=slab, plasma_kw=plasma_kw, kt_interp=kt_interp,
                        band_tables=band_tables if band_tables is not None else doppler_moment_tables(dop_tables))


def doppler_from_columns(S_xz, *, box_pc, distance_kpc, E0=0.5 * sum(SI_BAND), n_sec=24,
                         annulus=(40.0, 170.0), roll_deg=0.0, offset_arcsec=(0.0, 0.0), sky_scale=1.0,
                         soft_deg=0.0, soft_arcsec=0.0):
    """Sector velocities (km/s) from per-column moments ``S_xz`` (2, x, z), in the
    DATA's frame: every column is placed on the sky as ``project_columns``
    places it (roll, offset of the explosion centre from RA0/DEC0, scale), and
    the sectors / annulus are about RA0/DEC0 -- the centre the data statistic
    uses (``casa_jaxobs_doppler_data``). Sector k spans PA [k, k + 1) x 360/n_sec
    from west through north. Returns ``(v, valid)``, v = -c (M1/M0 - mean)/E0.

    Membership is hard by default; ``soft_deg`` / ``soft_arcsec`` > 0 give
    sigmoid edges in PA / radius, so the statistic is differentiable in roll,
    offset and scale (the hard version is piecewise constant in them).
    """
    n = S_xz.shape[-1]
    WX, NZ, _ = sky_geometry(box_pc, n, distance_kpc)
    ps = jnp.deg2rad(roll_deg)
    WXj, NZj = jnp.asarray(WX, jnp.float32), jnp.asarray(NZ, jnp.float32)
    w = sky_scale * (jnp.cos(ps) * WXj - jnp.sin(ps) * NZj) + offset_arcsec[0]
    nn = sky_scale * (jnp.sin(ps) * WXj + jnp.cos(ps) * NZj) + offset_arcsec[1]
    R = jnp.sqrt(w ** 2 + nn ** 2)
    pa = jnp.rad2deg(jnp.arctan2(nn, w)) % 360.0
    width = 360.0 / n_sec
    lo = jnp.arange(n_sec, dtype=jnp.float32)[:, None, None] * width
    d = (pa[None] - lo) % 360.0                                     # in [0, 360): inside if < width
    if soft_deg > 0:
        # periodic soft box: the distance past each edge, wrapped to (-180, 180]
        a = (d + 180.0) % 360.0 - 180.0
        b = (width - d + 180.0) % 360.0 - 180.0
        m_pa = jax.nn.sigmoid(a / soft_deg) * jax.nn.sigmoid(b / soft_deg)
    else:
        m_pa = (d < width).astype(jnp.float32)
    if soft_arcsec > 0:
        m_r = jax.nn.sigmoid((R - annulus[0]) / soft_arcsec) * jax.nn.sigmoid((annulus[1] - R) / soft_arcsec)
    else:
        m_r = ((R > annulus[0]) & (R < annulus[1])).astype(jnp.float32)
    S = jnp.einsum("kxz,sxz->ks", S_xz, m_pa * m_r[None], precision=jax.lax.Precision.HIGHEST)
    s0, s1 = S[0], S[1]
    valid = s0 > 0.0
    dE = jnp.where(valid, s1 / jnp.where(valid, s0, 1.0), 0.0)
    mean = jnp.sum(jnp.where(valid, dE, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
    return jnp.where(valid, -C_KMS * (dE - mean) / E0, 0.0), valid


def doppler_moment_maps(fields, tables, *, box_pc, distance_kpc, band=SI_BAND,
                        subgrid=dict(chi=4.0, f_mass=0.34), split="xrism_bulk", slab=None,
                        plasma_kw=None, kt_interp="linear"):
    """Per sky column (x, z), the zeroth and first photon-energy moments about E0
    of the Doppler window, (M0 + beta D0, M1 + beta D1): ``doppler_sectors`` up to
    its sector reduction, so a caller can draw the sectors in its own (traced)
    sky frame. Returns ``(S (2, x, z), E0)``.

    ``tables`` decides the statistic:
      * v2 ``load_binned_tables(inst, nh, "dop")``: the native-channel moments in
        exactly [1.78, 1.94] keV (``doppler_columns``; ``band`` must be that
        window) -- the data statistic (obs_model section 8);
      * v1 ``load_tables``: the legacy channel-centre mask on the 117 eV
        channels (1.752-1.986 keV for SI_BAND), bit-for-bit the statistic
        ``doppler_sectors`` always used.
    """
    if tables.get("binning") == "dop":
        win = tuple(float(x) for x in np.asarray(tables["ch_edges"]).ravel()[:2]) if "ch_edges" in tables else band
        if not np.allclose(win, band, atol=1e-6):
            raise ValueError(f"the dop tables are for the window {win}, not {band}")
        S = doppler_columns(fields, tables, box_pc=box_pc, distance_kpc=distance_kpc, subgrid=subgrid,
                            split=split, slab=slab, plasma_kw=plasma_kw, kt_interp=kt_interp)
        return S, float(tables.get("E0", 0.5 * sum(band)))
    if tables.get("binning") is not None:
        # v2 spec / band bins are not channels: the centre mask would take a whole
        # bin (e.g. 1.5-2.1 keV, centre 1.8) as the Doppler window (review 2026-09-25)
        raise ValueError(f"doppler_moment_maps needs v1 channel tables or v2 'dop' tables, "
                         f"not binning={tables['binning']!r}")
    n = fields["rho"].shape[0]
    norm = flux_norm((box_pc / n * PC_CM) ** 3, distance_kpc)
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
        # per-(x, z) moment maps, summed along the slab; the reduction into the
        # sectors is one dense contraction at the end (a scatter of every cell
        # into 24 bins serialises on atomics)
        beta = sl["vy"] * 1e3 / C_KMS
        for fcomp, ew in emitting_components(sl, split=split, **subgrid):
            pl = plasma(fcomp, split, **(plasma_kw or {}))
            w = component_weights(pl, tables, norm, ew)
            ii, jj, ww = corners(tables, pl)
            t = jnp.einsum("cqxszk,cxsz,qxsz->kxsz", MD[cidx, ii[None], jj[None]], w, ww)
            S = S + jnp.stack([t[0] + beta * t[2], t[1] + beta * t[3]]).sum(axis=2)
        return S

    return slab_scan(fields, slab, body, jnp.zeros((2, n, n))), E0


def doppler_sectors(fields, tables, *, box_pc, distance_kpc, band=SI_BAND, n_sec=24,
                    annulus=(40.0, 170.0), subgrid=dict(chi=4.0, f_mass=0.34), split="xrism_bulk",
                    slab=None, plasma_kw=None, moment_tables=None, kt_interp="linear", **geometry):
    """Per sky sector, the centroid-measured line-of-sight velocity (km/s) in
    ``band`` -- the data's statistic (mean photon energy in the band, v = -c dE/E0,
    mean-subtracted over the sectors that have counts), continuum dilution
    included. Returns ``(v, valid)``; moments are taken about E0 (Codex review).

    ``moment_tables`` (v2, ``load_binned_tables(inst, nh, "dop")``): the exact
    native-channel moments in [1.78, 1.94] (``doppler_columns``), and the
    sectors in the data frame with the ``geometry`` of ``doppler_from_columns``
    (roll_deg, offset_arcsec, sky_scale, soft_deg, soft_arcsec). Without it:
    the legacy statistic (channel-centre mask on ``tables``, sim frame).
    """
    if moment_tables is not None:
        S = doppler_columns(fields, moment_tables, box_pc=box_pc, distance_kpc=distance_kpc,
                            subgrid=subgrid, split=split, slab=slab, plasma_kw=plasma_kw,
                            kt_interp=kt_interp)
        return doppler_from_columns(S, box_pc=box_pc, distance_kpc=distance_kpc,
                                    E0=moment_tables.get("E0", 0.5 * sum(band)), n_sec=n_sec,
                                    annulus=annulus, **geometry)
    n = fields["rho"].shape[0]
    WX, NZ, R = sky_geometry(box_pc, n, distance_kpc)
    pa = np.rad2deg(np.arctan2(NZ, WX)) % 360.0
    sec = np.where((R > annulus[0]) & (R < annulus[1]), (pa / (360.0 / n_sec)).astype(int) % n_sec, n_sec)
    in_sec = jnp.asarray((sec[None] == np.arange(n_sec)[:, None, None]).astype(np.float32))
    Sxz, E0 = doppler_moment_maps(fields, tables, box_pc=box_pc, distance_kpc=distance_kpc, band=band,
                                  subgrid=subgrid, split=split, slab=slab, plasma_kw=plasma_kw)
    S = jnp.einsum("kxz,sxz->ks", Sxz, in_sec, precision=jax.lax.Precision.HIGHEST)
    s0, s1 = S[0], S[1]
    valid = s0 > 0.0
    dE = jnp.where(valid, s1 / jnp.where(valid, s0, 1.0), 0.0)
    mean = jnp.sum(jnp.where(valid, dE, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
    return jnp.where(valid, -C_KMS * (dE - mean) / E0, 0.0), valid


def poisson_loglike(model_rate, counts, exposure_s, background_rate=0.0):
    lam = jnp.maximum((model_rate + background_rate) * exposure_s, 1e-30)
    return jnp.sum(counts * jnp.log(lam) - lam)
# =============================================================================
# ============ ↑ Observables ↑ ================================================
# =============================================================================


# =============================================================================
# ============ ↓ Validation against the pyXSIM chain ↓ ========================
# =============================================================================
def parse_pyxsim_log(path):
    txt = Path(path).read_text(errors="ignore")
    m = re.search(r"count rate: synthetic ([\d.]+) vs real ([\d.]+)", txt)
    bands = re.findall(r"^\s+([\d.]+)-([\d.]+) \(.*?\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt, re.M)
    return dict(rate=float(m.group(1)) if m else np.nan, real=float(m.group(2)) if m else np.nan,
                bands=[(float(a), float(b), float(s), float(r)) for a, b, s, r, _ in bands])


def load_fields(path):
    d = np.load(path)
    keys = ("rho", "press", "vx", "vy", "vz", "C_ej", "C_Fe", "C_Si", "C_O", "C_He",
            "shocked_fraction", "time_since_shock", "density_time")
    f = {k: jnp.asarray(np.asarray(d[k], np.float32)) for k in keys}
    return f, float(d["box"]), float(d["age"])


def cmd_validate(args):
    sk = {} if args.slab is None else dict(slab=args.slab)
    P.set_tracer_split(args.split)
    fields, box, age = load_fields(args.state)
    tables = load_tables(args.instrument, args.nh)
    halo = load_halo(args.nh) if args.halo else None
    fn = jax.jit(lambda f: aperture_spectrum(f, tables, box_pc=box, distance_kpc=args.distance,
                                             halo=halo, split=args.split, **sk))
    t0 = time.time(); rate = fn(fields); rate.block_until_ready()
    t1 = time.time(); rate = fn(fields); rate.block_until_ready(); t2 = time.time()
    br = np.asarray(band_rates(rate, tables))
    print(f"[jaxobs] {args.state} (age {age:.0f} yr), {args.instrument}, D {args.distance} kpc, "
          f"halo {'on' if halo is not None else 'off'}: first call {t1 - t0:.1f} s, "
          f"then {t2 - t1:.2f} s")
    print(f"[jaxobs] positivity (C + beta D, rank-3 keep): min channel rate {float(rate.min()):.3e}")
    tot = float(np.asarray(rate)[(np.asarray(tables['e_ch']) > 0.5) & (np.asarray(tables['e_ch']) < 7)].sum())
    ref = parse_pyxsim_log(args.pyxsim_log) if args.pyxsim_log else None
    print(f"[jaxobs] 0.5-7 keV rate in r < 200\": {tot:.1f} counts/s"
          + (f"   (pyXSIM {ref['rate']:.1f}, ratio {tot / ref['rate']:.3f}; real {ref['real']:.1f})" if ref else ""))
    for k, (lo, hi) in enumerate(BANDS):
        line = f"    {lo:.1f}-{hi:.1f} keV  {br[k]:8.2f}"
        if ref and k < len(ref["bands"]):
            s = ref["bands"][k][2]
            line += f"   pyXSIM {s:8.2f}  ratio {br[k] / s:.3f}"
        print(line)
    img_fn = jax.jit(lambda f: band_images(f, tables, box_pc=box, distance_kpc=args.distance,
                                           halo=halo, split=args.split, **sk))
    t0 = time.time(); img = img_fn(fields).block_until_ready(); t1 = time.time()
    img = img_fn(fields).block_until_ready(); t2 = time.time()
    print(f"[jaxobs] min pixel rate {float(img.min()):.3e} (negative = first-order Doppler overshoot)")
    print(f"[jaxobs] band images {img.shape}: compile+run {t1 - t0:.1f} s, then {t2 - t1:.2f} s; "
          f"per-band totals " + " ".join(f"{float(x):.2f}" for x in np.asarray(img).sum((1, 2))))
    dop_fn = jax.jit(lambda f: doppler_sectors(f, tables, box_pc=box, distance_kpc=args.distance,
                                               split=args.split, **sk))
    t0 = time.time(); v, ok = dop_fn(fields); v.block_until_ready(); t1 = time.time()
    print(f"[jaxobs] Si Doppler per sector ({t1 - t0:.1f} s, {int(ok.sum())} valid): "
          + " ".join(f"{x:+.0f}" for x in np.asarray(v)))
    if args.grad:
        # reverse-mode gradients w.r.t. the density field, per observable
        # (peak memory is cumulative, so the order is smallest first)
        obs = dict(
            doppler=lambda f: jnp.sum(doppler_sectors(f, tables, box_pc=box, distance_kpc=args.distance,
                                                      split=args.split, **sk)[0] ** 2),
            images=lambda f: jnp.sum(band_images(f, tables, box_pc=box, distance_kpc=args.distance,
                                                 halo=halo, split=args.split, **sk) ** 2),
            rate=lambda f: jnp.sum(aperture_spectrum(f, tables, box_pc=box, distance_kpc=args.distance,
                                                     halo=halo, split=args.split, **sk)))
        for name, fn_o in obs.items():
            g = jax.jit(jax.grad(lambda rho: fn_o(dict(fields, rho=rho))))
            t0 = time.time(); gr = g(fields["rho"]).block_until_ready(); t1 = time.time()
            gr = g(fields["rho"]).block_until_ready(); t2 = time.time()
            stats = jax.devices()[0].memory_stats() or {}
            print(f"[jaxobs] grad of {name} w.r.t. rho: finite {bool(jnp.all(jnp.isfinite(gr)))}, "
                  f"compile+run {t1 - t0:.1f} s, then {t2 - t1:.2f} s, peak GPU memory so far "
                  f"{stats.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB", flush=True)
            del g, gr
        # directional FD check of the rate part
        dr = jnp.asarray(np.random.default_rng(0).standard_normal(fields["rho"].shape),
                         jnp.float32) * fields["rho"]
        rate_of = jax.jit(lambda rho: jnp.sum(aperture_spectrum(
            dict(fields, rho=rho), tables, box_pc=box, distance_kpc=args.distance,
            halo=halo, split=args.split, **sk)))
        gr_r = jax.jit(jax.grad(rate_of))(fields["rho"])
        _, jv = jax.jvp(rate_of, (fields["rho"],), (dr,))
        line = f"[jaxobs] rate directional derivative: VJP {float(jnp.vdot(gr_r, dr)):.5e}, JVP {float(jv):.5e}"
        for eps in (1e-2, 3e-2):
            fd = (rate_of(fields["rho"] + eps * dr) - rate_of(fields["rho"] - eps * dr)) / (2 * eps)
            line += f", FD(eps {eps:g}) {float(fd):.5e}"
        print(line)


DATA_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs/data")
RA0, DEC0 = 350.8583, 58.8149                  # grid centre (casa_observe, make_epoch_images)
CCO_RADEC = (350.866417, 58.811778)


def poisson_deviance(lam, counts, mask):
    """2 sum [lam - n + n ln(n / lam)] over unmasked pixels (Cash's C, offset-free)."""
    n = counts
    t = lam - n + jnp.where(n > 0, n * jnp.log(jnp.where(n > 0, n, 1.0) / lam), 0.0)
    return 2.0 * jnp.sum(jnp.where(mask, t, 0.0), axis=(-2, -1))


def cmd_fit_image(args):
    """Fit roll, sky offset and per-band amplitudes of a fixed state to a real epoch.

    The first field-level use of the model: the likelihood is the Poisson
    likelihood of every unmasked pixel of the six band images, and the fit runs
    on its exact gradient. The band columns are computed once (the state is
    fixed); only the projection is re-evaluated.
    """
    P.set_tracer_split(args.split)
    fields, box, age = load_fields(args.state)
    tables = load_tables(args.instrument, args.nh)
    halo = load_halo(args.nh)
    dat = np.load(DATA_DIR / f"bands_{args.epoch}.npz")
    counts = jnp.asarray(dat["counts"])
    exposure = float(dat["exposure"])
    npix = counts.shape[-1]
    pix = float(dat["pixel_arcsec"])
    rr = np.hypot(*np.meshgrid((np.arange(npix) - 0.5 * (npix - 1)) * pix,
                               (np.arange(npix) - 0.5 * (npix - 1)) * pix, indexing="ij"))
    # the CCO (Tananbaum 1999; 23:23:27.94 +58:48:42.4) is a point source the
    # hydro model does not contain: masked within 6"
    cw, cn = -(CCO_RADEC[0] - RA0) * np.cos(np.deg2rad(DEC0)) * 3600.0, (CCO_RADEC[1] - DEC0) * 3600.0
    ax = (np.arange(npix) - 0.5 * (npix - 1)) * pix
    NN, WW = np.meshgrid(ax, ax, indexing="ij")                    # rows = north, cols = west
    cco = np.hypot(WW - cw, NN - cn) < 6.0
    mask = jnp.asarray((np.asarray(dat["edge"]) <= 0.02) & (rr < args.r_max) & ~cco)
    t0 = time.time()
    cols = jax.jit(lambda f: band_columns(f, tables, box_pc=box, distance_kpc=args.distance,
                                          split=args.split))(fields).block_until_ready()
    print(f"[fitimg] {Path(args.state).name} (age {age:.0f} yr) vs Chandra {args.epoch} "
          f"({exposure / 1e3:.1f} ks, {', '.join(map(str, dat['detnam']))}), D {args.distance} kpc; "
          f"band columns {time.time() - t0:.1f} s; {int(mask.sum())} pixels in the likelihood", flush=True)
    bkg = args.bkg * exposure

    def model(p):
        img = project_columns(cols, box_pc=box, distance_kpc=args.distance, halo=halo, npix=npix,
                              pix_arcsec=pix, roll_deg=p[0], offset_arcsec=(p[1], p[2]),
                              psf_sigma_arcsec=args.psf)
        return jnp.exp(p[3:])[:, None, None] * img * exposure + bkg

    def loss(p):
        lam = model(p)
        return jnp.sum(jnp.where(mask, lam - counts * jnp.log(lam), 0.0))

    vg = jax.jit(jax.value_and_grad(loss))
    dev = jax.jit(lambda p: poisson_deviance(model(p), counts, mask))
    nb = counts.shape[0]
    # amplitudes at a given geometry are analytic (d loss / d lnA = 0)
    def amp(p):
        lam0 = model(p.at[3:].set(0.0)) - bkg
        return jnp.log(jnp.sum(jnp.where(mask, counts, 0), (1, 2)) /
                       jnp.sum(jnp.where(mask, lam0, 0), (1, 2)))
    amp = jax.jit(amp)
    p = jnp.zeros(3 + nb)
    d0 = np.asarray(dev(p.at[3:].set(amp(p))))
    best = None
    for roll in np.arange(-180.0, 180.0, args.roll_step):
        q = jnp.zeros(3 + nb).at[0].set(roll)
        q = q.at[3:].set(amp(q))
        v = float(vg(q)[0])
        if best is None or v < best[0]:
            best = (v, q)
    p = best[1]
    print(f"[fitimg] roll scan ({args.roll_step:g} deg): best roll {float(p[0]):+.0f} deg; "
          f"deviance at roll 0: " + " ".join(f"{x:.3g}" for x in d0), flush=True)
    # Adam on the geometry (the amplitudes re-solved analytically each step)
    lr = jnp.asarray([0.5, 0.5, 0.5] + [0.0] * nb)
    m1 = jnp.zeros_like(p); m2 = jnp.zeros_like(p)
    for it in range(args.iters):
        v, g = vg(p)
        m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g ** 2
        p = p - lr * (m1 / (1 - 0.9 ** (it + 1))) / (jnp.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-12)
        p = p.at[3:].set(amp(p))
        if it % 20 == 0 or it == args.iters - 1:
            print(f"[fitimg] it {it:3d}: -lnL {float(v):.6e}, roll {float(p[0]):+.2f} deg, offset "
                  f"W {float(p[1]):+.2f}\" N {float(p[2]):+.2f}\", |grad geom| "
                  f"{float(jnp.linalg.norm(g[:3])):.3g}", flush=True)
    d1 = np.asarray(dev(p))
    scales = np.exp(np.asarray(p[3:]))
    print("[fitimg] per band: amplitude (data/model), deviance per pixel (roll 0 -> fit)")
    for k, (lo, hi) in enumerate(BANDS):
        print(f"    {lo:.1f}-{hi:.1f} keV   {scales[k]:.3f}   {d0[k] / float(mask.sum()):8.2f} -> "
              f"{d1[k] / float(mask.sum()):8.2f}")
    out = Path(args.out)
    lam = np.asarray(model(p))
    np.savez_compressed(out.with_suffix(".npz"), params=np.asarray(p), model=lam.astype(np.float32),
                        counts=np.asarray(counts), mask=np.asarray(mask), deviance=d1,
                        deviance_roll0=d0, epoch=args.epoch, state=args.state)
    plot_fit_image(np.asarray(counts), lam, np.asarray(mask), p, args, out)


def plot_fit_image(counts, lam, mask, p, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [(0, "0.5-1.5 keV"), (1, "Si 1.5-2.1"), (5, "Fe-K 6-7")]
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
    for r, (k, lab) in enumerate(show):
        c, m = counts[k], lam[k]
        vmax = np.percentile(c, 99.7)
        for j, (img, t) in enumerate(((c, f"Chandra {args.epoch}"), (m, "model (fit roll/offset/amp.)"))):
            axs[r, j].imshow(np.sqrt(np.maximum(img, 0)), origin="lower", cmap="inferno",
                             vmin=0, vmax=np.sqrt(vmax))
            axs[r, j].set_title(f"{lab}: {t}")
        res = np.where(mask, (c - m) / np.sqrt(np.maximum(m, 1.0)), np.nan)
        im = axs[r, 2].imshow(res, origin="lower", cmap="RdBu_r", vmin=-10, vmax=10)
        axs[r, 2].set_title(f"{lab}: (data - model)/sqrt(model)")
        fig.colorbar(im, ax=axs[r, 2], shrink=0.7)
    for a in axs.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{Path(args.state).name}: roll {float(p[0]):+.1f} deg, offset "
                 f"({float(p[1]):+.1f}\" W, {float(p[2]):+.1f}\" N); columns = west to the right, north up")
    fig.savefig(out.with_suffix(".png"), dpi=90)
    print(f"[fitimg] wrote {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("state")
    v.add_argument("--distance", type=float, required=True)
    v.add_argument("--instrument", default="chandra_aciss_cy0")
    v.add_argument("--nh", type=float, default=1.2)
    v.add_argument("--no-halo", dest="halo", action="store_false")
    v.add_argument("--split", default="xrism_bulk")
    v.add_argument("--pyxsim-log", default=None)
    v.add_argument("--grad", action="store_true")
    v.add_argument("--slab", type=int, default=None,
                   help="y-planes per checkpointed slab (default: each observable's own)")
    v.set_defaults(func=cmd_validate)
    f = sub.add_parser("fit-image")
    f.add_argument("state")
    f.add_argument("--epoch", default="2000")
    f.add_argument("--distance", type=float, required=True)
    f.add_argument("--instrument", default="chandra_aciss_cy0")
    f.add_argument("--nh", type=float, default=1.2)
    f.add_argument("--split", default="xrism_bulk")
    f.add_argument("--r-max", type=float, default=240.0, help="likelihood radius (arcsec)")
    f.add_argument("--bkg", type=float, default=1e-6, help="counts/s/pixel/band floor")
    f.add_argument("--psf", type=float, default=0.5, help="Gaussian PSF sigma (arcsec)")
    f.add_argument("--roll-step", type=float, default=10.0)
    f.add_argument("--iters", type=int, default=120)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fit_image)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
