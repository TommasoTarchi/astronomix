"""
Response-folded NEI emissivity tables for the differentiable observation model.

The pyXSIM/SOXS chain in ``casa_observe.py`` is exact but slow (~1 h per
epoch at 256^3) and not differentiable. Everything it does to a cell's
emission is LINEAR in the cell's emission measure and abundances once
(kT_e, n_e t) are fixed:

    counts/s per channel =
        (1e-14 / (4 pi D^2)) * sum_cells V * [ n_e n_H C_H(kT)
              + sum_el n_e (n_el / r_sun,el) C_el(kT, n_e t) ]

with ``C_el(kT, n_e t)[ch] = RMF . (ARF * TBabs(N_H) * sum_q f_q(kT, n_e t)
v_{el,q}(kT))``: ``v_{el,q}`` the APEC NEI spectrum of ion El^q per unit
norm (soxs ``ApecGenerator(nei=True)``, the generator pyXSIM's
``NEISourceModel`` uses), ``f_q`` the ion fractions of ``_nei`` (the same
table, the same ``El^q`` naming as ``casa_observe.ion_abundance_fields``).
In NEI mode APEC's base spectrum is hydrogen only (He is declared ion by
ion), which is ``C_H``.

This script tabulates ``C_el`` on the ``_nei`` (kT, n_e t) grid, for a small
grid of N_H (so an N_H map can be interpolated later), per instrument, and
also ``D_el = dC_el / d beta``, the first-order Doppler response (beta = v_los
/ c, positive = receding): ``F_obs(E) = F(E) + beta d(E F)/dE``. Channels are
the RMF's, rebinned by ``--rebin``.

Run in the xrayobs env (needs soxs + pyatomdb data)::

    /export/home/lstorcks/xrayobs/bin/python casa_jaxobs_tables.py \\
        --instruments chandra_aciss_cy0 chandra_acisi_cy22
"""

import argparse
import time
from pathlib import Path

import numpy as np

import _nei

OUT_DIR = Path("/export/data/lstorcks/casa_orlando150/jaxobs")
NH_GRID = np.array([0.8, 1.0, 1.2, 1.5, 2.0])          # 1e22 cm^-2
ELEMENTS = _nei.ELEMENTS                                 # He O Ne Mg Si S Ar Ca Fe


def response_matrix(rmf):
    """Dense (n_e, n_ch) redistribution matrix, from unit spectra (exact)."""
    M = np.zeros((rmf.n_e, rmf.n_ch))
    for i in range(rmf.n_e):
        spec = np.zeros(rmf.n_e)
        spec[i] = 1.0
        M[i] = rmf.convolve_spectrum(spec, 1.0, noisy=False, rate=True)
    return M


def build(instrument, rebin, out_dir):
    import soxs
    from soxs.response import AuxiliaryResponseFile, RedistributionMatrixFile
    from soxs.thermal_spectra import ApecGenerator
    from soxs.spectra import get_tbabs_absorb

    reg = soxs.instrument_registry[instrument]
    arf = AuxiliaryResponseFile(reg["arf"])
    rmf = RedistributionMatrixFile(reg["rmf"])
    elo, ehi = np.asarray(rmf.elo), np.asarray(rmf.ehi)
    emid = 0.5 * (elo + ehi)
    de = ehi - elo
    if not np.allclose(de, de[0], rtol=1e-3):
        raise SystemExit("RMF energy grid is not uniform; generalise the APEC grid")
    t0 = time.time()
    M = response_matrix(rmf)                                        # (n_e, n_ch)
    area = np.asarray(arf.interpolate_area(emid).value)             # cm^2
    absorb = np.stack([get_tbabs_absorb(emid, nh) for nh in NH_GRID])  # (n_nh, n_e)
    # channel rebinning
    ch_lo = np.asarray(rmf.ebounds_data["E_MIN"]); ch_hi = np.asarray(rmf.ebounds_data["E_MAX"])
    nb = rmf.n_ch // rebin
    R = np.zeros((rmf.n_ch, nb))
    for b in range(nb):
        R[b * rebin:(b + 1) * rebin, b] = 1.0
    ch_edges = np.append(ch_lo[::rebin][:nb], ch_hi[rebin * nb - 1])
    # fold photons -> channels for every N_H: (n_nh, n_e, nb)
    fold = (absorb * area)[:, :, None] * (M @ R)[None]
    print(f"[tables] {instrument}: response {rmf.n_e} x {rmf.n_ch} -> {nb} channels "
          f"({time.time() - t0:.0f} s)")

    kt, net, frac = _nei.load_table()
    nkt, nnet = len(kt), len(net)
    elo0, ehi0 = float(elo[0]), float(ehi[-1])

    def photon_spec(gen, kT, names, onehot):
        s = gen.get_nei_spectrum(kT, {n: (1.0 if n == onehot else 0.0) for n in names}, 0.0, 1.0)
        return np.asarray(s.flux.value) * de       # photons/s/cm^2 per bin at norm 1

    out = {"kt": kt, "net": net, "nh": NH_GRID, "ch_edges": ch_edges,
           "elements": np.array(list(ELEMENTS)), "instrument": np.array(instrument)}
    # hydrogen continuum: an NEI generator with ONE declared ion (He^2, fully
    # ionized) at zero abundance leaves exactly the H base spectrum
    gH = ApecGenerator(elo0, ehi0, len(elo), binscale="linear", nei=True,
                       var_elem=["He^2"], abund_table="angr")
    CH = np.zeros((len(NH_GRID), nkt, nb)); DH = np.zeros_like(CH)
    for i, T in enumerate(kt):
        F = gH.get_nei_spectrum(float(T), {"He^2": 0.0}, 0.0, 1.0)
        F = np.asarray(F.flux.value) * de
        dEF = np.gradient(emid * F / de, emid) * de            # d(E F)/dE per bin
        CH[:, i] = np.einsum("e,nec->nc", F, fold)
        DH[:, i] = np.einsum("e,nec->nc", dEF, fold)
    out["C_H"], out["D_H"] = CH.astype(np.float32), DH.astype(np.float32)
    print(f"[tables] H continuum ({time.time() - t0:.0f} s)")

    for el, Z in ELEMENTS.items():
        names = [f"{el}^{q}" for q in range(Z + 1)]
        gen = ApecGenerator(elo0, ehi0, len(elo), binscale="linear", nei=True,
                            var_elem=names, abund_table="angr")
        C = np.zeros((len(NH_GRID), nkt, nnet, nb)); D = np.zeros_like(C)
        f = frac[el]                                          # (nkt, nnet, Z+1)
        for i, T in enumerate(kt):
            # EVERY call returns the generator's base spectrum too -- H, plus He
            # for every element but He (He is undeclared there) -- so the one-hot
            # basis vectors must have it subtracted, or each element table adds
            # the H+He continuum again, weighted by that element's abundance
            # (3x too bright; caught cell by cell against an all-ions spectrum)
            base = photon_spec(gen, float(T), names, None)
            basis = np.stack([photon_spec(gen, float(T), names, n) - base for n in names])
            F = f[i] @ basis                                  # (nnet, n_e)
            dEF = np.gradient(emid[None] * F / de, emid, axis=1) * de
            C[:, i] = np.einsum("te,nec->ntc", F, fold)
            D[:, i] = np.einsum("te,nec->ntc", dEF, fold)
        out[f"C_{el}"], out[f"D_{el}"] = C.astype(np.float32), D.astype(np.float32)
        print(f"[tables] {el} ({time.time() - t0:.0f} s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"emissivity_{instrument}.npz"
    np.savez_compressed(path, **out)
    print(f"[tables] wrote {path}")


def build_sync(instrument, rebin, out_dir, n_ecut=64):
    """Response-folded synchrotron shape tables, per unit ``_synchrotron`` weight.

    ``_synchrotron.spectral_shape`` is the loss-limited ENERGY flux density
    ``E^-alpha exp(-sqrt(E / E_cut))`` (arbitrary units, 1 at 1 keV without
    cutoff); a cell contributes ``k w`` times it in erg/cm^2/s/keV. Folded
    here: photons = shape / E (keV -> erg), x ARF x TBabs(N_H), through the
    RMF, per rebinned channel -> ``S[nh, ecut, ch]`` in counts/s per unit
    ``k w`` (erg/cm^2/s/keV at 1 keV).
    """
    import soxs
    from soxs.response import AuxiliaryResponseFile, RedistributionMatrixFile
    from soxs.spectra import get_tbabs_absorb
    import _synchrotron as SY

    reg = soxs.instrument_registry[instrument]
    arf = AuxiliaryResponseFile(reg["arf"])
    rmf = RedistributionMatrixFile(reg["rmf"])
    elo, ehi = np.asarray(rmf.elo), np.asarray(rmf.ehi)
    emid = 0.5 * (elo + ehi); de = ehi - elo
    M = response_matrix(rmf)
    area = np.asarray(arf.interpolate_area(emid).value)
    absorb = np.stack([get_tbabs_absorb(emid, nh) for nh in NH_GRID])
    nb = rmf.n_ch // rebin
    R = np.zeros((rmf.n_ch, nb))
    for b in range(nb):
        R[b * rebin:(b + 1) * rebin, b] = 1.0
    ch_lo = np.asarray(rmf.ebounds_data["E_MIN"]); ch_hi = np.asarray(rmf.ebounds_data["E_MAX"])
    ch_edges = np.append(ch_lo[::rebin][:nb], ch_hi[rebin * nb - 1])
    fold = (absorb * area)[:, :, None] * (M @ R)[None]          # (n_nh, n_e, nb)
    ecut = np.geomspace(0.01, 100.0, n_ecut)
    kev_erg = 1.602176634e-9
    photons = np.stack([SY.spectral_shape(emid, Ec) / (emid * kev_erg) * de for Ec in ecut])  # (n_ecut, n_e)
    S = np.einsum("ke,nec->nkc", photons, fold)
    path = Path(out_dir) / f"sync_{instrument}.npz"
    np.savez_compressed(path, ecut=ecut, nh=NH_GRID, ch_edges=ch_edges, S=S.astype(np.float32),
                        instrument=np.array(instrument))
    print(f"[sync] wrote {path}")


#: bands of the image model and of casa_observe's comparison table (keV)
BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))


def build_halo(out_dir, nh=1.2, n_photons=400_000, aperture=200.0, seed=3):
    """Deterministic dust-halo response from the same Monte Carlo casa_observe uses.

    ``casa_observe.apply_dust_halo`` gives every photon Poisson(tau_sca(E)) Mie
    deflections at random positions along the sightline (``_dusthalo``). The
    differentiable model needs that as KERNELS, so a point source is pushed
    through ``scatter_sky_positions`` itself:

    * ``kernel[b]``: the 2D sky kernel for image band b (energies uniform in the
      band), on a (2 * half + 1)^2 grid of ``kpix`` arcsec pixels, summing to 1
      including the unscattered core (photons scattered beyond the grid are lost,
      as they are in the finite real image);
    * ``aperture_keep[e, r]``: the fraction of photons of energy ``e`` from a
      point at projected radius ``r`` that land inside ``aperture`` arcsec.
    """
    from _dusthalo import scatter_sky_positions
    RA0, DEC0 = 350.8583, 58.8149
    rng = np.random.default_rng(seed)
    kpix, half = 4 * 0.492, 200               # kernel grid: +-393"
    edges = (np.arange(2 * half + 2) - half - 0.5) * kpix
    kernels = []
    for lo, hi in BANDS:
        e = rng.uniform(lo, hi, n_photons)
        ra = np.full(n_photons, RA0); dec = np.full(n_photons, DEC0)
        ra2, dec2, _ = scatter_sky_positions(ra, dec, e, nh=nh, seed=int(rng.integers(1e9)),
                                             verbose=False)
        dx = -(ra2 - RA0) * np.cos(np.deg2rad(DEC0)) * 3600.0     # west positive
        dy = (dec2 - DEC0) * 3600.0
        H, _, _ = np.histogram2d(dy, dx, bins=[edges, edges])
        kernels.append(H / n_photons)
        print(f"[halo] band {lo}-{hi} keV: unscattered {H[half, half] / n_photons:.3f} "
              f"(central pixel), kept on grid {H.sum() / n_photons:.3f}")
    e_grid = np.geomspace(0.3, 11.0, 64)
    r_grid = np.arange(0.0, 230.0, 10.0)
    keep = np.zeros((len(e_grid), len(r_grid)))
    m = 40_000
    for i, e0 in enumerate(e_grid):
        for j, r0 in enumerate(r_grid):
            ra = np.full(m, RA0 - r0 / 3600.0 / np.cos(np.deg2rad(DEC0)))  # r0 to the west
            dec = np.full(m, DEC0)
            ra2, dec2, _ = scatter_sky_positions(ra, dec, np.full(m, e0), nh=nh,
                                                 seed=int(rng.integers(1e9)), verbose=False)
            dx = -(ra2 - RA0) * np.cos(np.deg2rad(DEC0)) * 3600.0
            dy = (dec2 - DEC0) * 3600.0
            keep[i, j] = np.mean(np.hypot(dx, dy) < aperture)
    path = Path(out_dir) / f"dusthalo_nh{nh:g}.npz"
    np.savez_compressed(path, bands=np.array(BANDS), kernel=np.array(kernels, dtype=np.float32),
                        kernel_pixel_arcsec=kpix, kernel_half=half, e_grid=e_grid,
                        r_grid=r_grid, aperture_keep=keep, aperture_arcsec=aperture, nh=nh)
    print(f"[halo] aperture keep at 1 keV: r=0 {np.interp(1.0, e_grid, keep[:, 0]):.3f}, "
          f"r=150\" {np.interp(1.0, e_grid, keep[:, 15]):.3f}; wrote {path}")


# =============================================================================
# ============ ↓ v2: tables on the ANALYSIS bins (audit 2026-09-25) ↓ =========
# =============================================================================
# The v1 tables above fold onto 8x-rebinned (117 eV) channels; casa_xfit then
# spreads those uniformly onto its 0.2 keV bins, and the line energies sit on
# the coarse channel edges (Si He-a 1.86 vs 1.8688): a 5-20 % line-shaped
# artefact (obs_model section 4). v2 folds each photon spectrum through the
# NATIVE 14.6 eV channels and bins those by their exact overlap with the
# analysis bins -- which is what binning the events by energy does, up to the
# distribution of event energies inside one 14.6 eV channel. Three binnings,
# one file per instrument (``emissivity_bins_{instrument}.npz``):
#
# * ``spec``: casa_xfit's 0.2 keV spectral bins, SPEC_EDGES (0.7-6.9 keV, 31);
# * ``band``: the six image bands with exact edges (casa_jaxobs_data.BANDS);
# * ``dop``:  two MOMENTS over the Doppler window [1.78, 1.94] keV, M0 = counts
#   and M1 = sum (E - E0) counts, E0 = 1.86: the data statistic of
#   casa_jaxobs_doppler_data (mean event energy in the window) is M1 / M0.
#
# Plus: the extended N_H grid NH_GRID_V2 (0.5-4e22; v1 clamped Q2's map at
# 0.8 and 2.0), the history-aware NEI axis ``rho`` (casa_jaxobs_nei; rho = 1 is
# the v1 ion balance), and the second-order Doppler response D2 (spec, band),
# N_obs = C + beta D + beta^2 / 2 D2, so a spread of line-of-sight velocities
# broadens the lines (Fe-K: 1500-2500 km/s is 33-55 eV against ACIS' ~65 eV).
NH_GRID_V2 = np.array([0.5, 0.8, 1.2, 1.6, 2.0, 2.6, 3.3, 4.0])     # 1e22 cm^-2
SPEC_EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)               # == casa_xfit.SPEC_EDGES
DOPPLER_WINDOW = (1.78, 1.94)                                       # doppler_si_2004 band_kev
V2_INSTRUMENTS = ("chandra_aciss_cy0", "chandra_aciss_cy10", "chandra_aciss_cy22", "chandra_acisi_cy22")


def binned_path(instrument, out_dir=OUT_DIR):
    return Path(out_dir) / f"emissivity_bins_{instrument}.npz"


def channel_overlap(ch_lo, ch_hi, lo, hi):
    """(n_ch, n_bin): the fraction of each native channel's counts inside each
    bin [lo, hi), for counts uniform within a channel."""
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    ov = np.minimum(hi[None], ch_hi[:, None]) - np.maximum(lo[None], ch_lo[:, None])
    return np.clip(ov, 0.0, None) / (ch_hi - ch_lo)[:, None]


def binning_matrices(ch_lo, ch_hi):
    """Native channel -> analysis bin maps: {'spec': (n_ch, 31), 'band': (n_ch, 6),
    'dop': (n_ch, 2) = [overlap, overlap x (mean energy of the overlap - E0)]}."""
    lo, hi = DOPPLER_WINDOW
    e0 = 0.5 * (lo + hi)
    a, b = np.maximum(ch_lo, lo), np.minimum(ch_hi, hi)
    w0 = np.clip(b - a, 0.0, None) / (ch_hi - ch_lo)
    w1 = np.where(w0 > 0, w0 * (0.5 * (a + b) - e0), 0.0)
    bands = np.array(BANDS)
    return {"spec": channel_overlap(ch_lo, ch_hi, SPEC_EDGES[:-1], SPEC_EDGES[1:]),
            "band": channel_overlap(ch_lo, ch_hi, bands[:, 0], bands[:, 1]),
            "dop": np.stack([w0, w1], 1)}


def _response(instrument):
    """(elo, ehi, dense RMF (n_e, n_ch), ARF area (n_e,), channel lo/hi) for a soxs instrument."""
    import soxs
    from soxs.response import AuxiliaryResponseFile, RedistributionMatrixFile
    reg = soxs.instrument_registry[instrument]
    arf = AuxiliaryResponseFile(reg["arf"])
    rmf = RedistributionMatrixFile(reg["rmf"])
    elo, ehi = np.asarray(rmf.elo), np.asarray(rmf.ehi)
    M = response_matrix(rmf)
    area = np.asarray(arf.interpolate_area(0.5 * (elo + ehi)).value)
    ch_lo = np.asarray(rmf.ebounds_data["E_MIN"], np.float64)
    ch_hi = np.asarray(rmf.ebounds_data["E_MAX"], np.float64)
    return elo, ehi, M, area, ch_lo, ch_hi, dict(rmf=str(reg["rmf"]), arf=str(reg["arf"]))


def basis_path(elo, ehi, out_dir=OUT_DIR):
    return Path(out_dir) / f"apec_nei_basis_{elo[0]:.4f}-{ehi[-1]:.4f}-n{len(elo)}.npz"


def load_basis(elo, ehi, out_dir=OUT_DIR, build_missing=True):
    """APEC NEI per-ion photon spectra (per unit norm, photons/s/cm^2 per energy
    bin) on an RMF energy grid: {'H': (n_kt, n_e), el: (n_kt, Z + 1, n_e)}.

    Cached per grid; a cached grid that CONTAINS this one (same edges, e.g.
    cy10's 0.3-9.3 keV inside cy0's 0.3-11) is sliced instead of rebuilt.
    """
    for p in sorted(Path(out_dir).glob("apec_nei_basis_*.npz")):
        d = np.load(p)
        g_lo = np.asarray(d["elo"])
        k = int(np.argmin(np.abs(g_lo - elo[0])))
        if k + len(elo) <= len(g_lo) and np.allclose(g_lo[k:k + len(elo)], elo, rtol=0, atol=1e-7) \
                and np.allclose(np.asarray(d["ehi"])[k:k + len(elo)], ehi, rtol=0, atol=1e-7):
            return {key: np.asarray(d[key])[..., k:k + len(elo)] for key in ["H"] + list(ELEMENTS)}, \
                np.asarray(d["kt"])
    if not build_missing:
        raise FileNotFoundError(f"no APEC basis for the grid {elo[0]}-{ehi[-1]} ({len(elo)})")
    return build_basis(elo, ehi, out_dir)


def build_basis(elo, ehi, out_dir=OUT_DIR):
    """The expensive half of ``build``: one APEC NEI spectrum per ion per kT (same
    generator settings and base-spectrum subtraction as v1), cached."""
    from soxs.thermal_spectra import ApecGenerator
    de = ehi - elo
    if not np.allclose(de, de[0], rtol=1e-3):
        raise SystemExit("RMF energy grid is not uniform; generalise the APEC grid")
    kt = _nei.KT_GRID
    t0 = time.time()
    out = {"kt": kt, "elo": elo, "ehi": ehi}
    gH = ApecGenerator(float(elo[0]), float(ehi[-1]), len(elo), binscale="linear", nei=True,
                       var_elem=["He^2"], abund_table="angr")
    out["H"] = np.stack([np.asarray(gH.get_nei_spectrum(float(T), {"He^2": 0.0}, 0.0, 1.0).flux.value) * de
                         for T in kt])
    for el, Z in ELEMENTS.items():
        names = [f"{el}^{q}" for q in range(Z + 1)]
        gen = ApecGenerator(float(elo[0]), float(ehi[-1]), len(elo), binscale="linear", nei=True,
                            var_elem=names, abund_table="angr")
        B = np.zeros((len(kt), Z + 1, len(elo)))
        for i, T in enumerate(kt):
            spec = lambda one: np.asarray(gen.get_nei_spectrum(   # noqa: E731
                float(T), {n: (1.0 if n == one else 0.0) for n in names}, 0.0, 1.0).flux.value) * de
            base = spec(None)
            B[i] = np.stack([spec(n) - base for n in names])
        out[el] = B
        print(f"[basis] {el} ({time.time() - t0:.0f} s)", flush=True)
    path = basis_path(elo, ehi, out_dir)
    np.savez_compressed(path, **out)
    print(f"[basis] wrote {path}")
    return {k: out[k] for k in ["H"] + list(ELEMENTS)}, kt


def _doppler_derivatives(F, emid, de):
    """First and second beta-derivatives of the observed photons per bin,
    N_obs(E) = N(E / (1 - beta)) / (1 - beta):  D = d(E N)/dE, D2 = d^2(E^2 N)/dE^2
    (N per keV; returned per bin, like F)."""
    n = F / de
    D = np.gradient(emid * n, emid, axis=-1) * de
    D2 = np.gradient(np.gradient(emid ** 2 * n, emid, axis=-1), emid, axis=-1) * de
    return D, D2


def build_binned(instrument, out_dir=OUT_DIR, nei_history=None, nh_grid=NH_GRID_V2):
    """``emissivity_bins_{instrument}.npz``: C, D (and D2 for spec/band) per
    binning, with the N_H axis first and the NEI-history axis rho.

    Keys: ``C_H_{b}`` (n_nh, n_kt, n_bin); ``C_{el}_{b}`` (n_nh, n_kt, n_rho,
    n_net, n_bin); same for D / D2; grids ``kt, net, rho, nh``; ``spec_edges``,
    ``band_edges`` (6, 2), ``dop_window``, ``dop_E0``.
    """
    import casa_jaxobs_nei as NH
    from soxs.spectra import get_tbabs_absorb
    path = binned_path(instrument, out_dir)
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    t0 = time.time()
    elo, ehi, M, area, ch_lo, ch_hi, files = _response(instrument)
    emid, de = 0.5 * (elo + ehi), ehi - elo
    absorb = np.stack([get_tbabs_absorb(emid, nh) for nh in nh_grid])            # (n_nh, n_e)
    bm = binning_matrices(ch_lo, ch_hi)
    fold = {b: (absorb * area)[:, :, None] * (M @ m)[None] for b, m in bm.items()}  # (n_nh, n_e, n_b)
    basis, kt_b = load_basis(elo, ehi, out_dir)
    kt, rho, net, frac = NH.load_table(nei_history or NH.TABLE_PATH)
    if not np.allclose(kt, kt_b):
        raise SystemExit("basis and NEI-history kT grids differ")
    print(f"[bins] {instrument}: response {M.shape}, basis + tables loaded ({time.time() - t0:.0f} s)",
          flush=True)
    out = {"kt": kt, "net": net, "rho": rho, "nh": np.asarray(nh_grid), "elements": np.array(list(ELEMENTS)),
           "instrument": np.array(instrument), "spec_edges": SPEC_EDGES, "band_edges": np.array(BANDS),
           "dop_window": np.array(DOPPLER_WINDOW), "dop_E0": 0.5 * sum(DOPPLER_WINDOW),
           "rmf": np.array(files["rmf"]), "arf": np.array(files["arf"]),
           "nei_history": np.array(str(nei_history or NH.TABLE_PATH))}
    second = ("spec", "band")
    F = basis["H"]                                                                # (n_kt, n_e)
    D, D2 = _doppler_derivatives(F, emid, de)
    for b, fo in fold.items():
        out[f"C_H_{b}"] = np.einsum("te,neb->ntb", F, fo).astype(np.float32)
        out[f"D_H_{b}"] = np.einsum("te,neb->ntb", D, fo).astype(np.float32)
        if b in second:
            out[f"D2_H_{b}"] = np.einsum("te,neb->ntb", D2, fo).astype(np.float32)
    nr, nn = len(rho), len(net)
    for el, Z in ELEMENTS.items():
        acc = {f"{k}_{b}": np.zeros((len(nh_grid), len(kt), nr, nn, fold[b].shape[-1]), np.float32)
               for b in fold for k in (("C", "D", "D2") if b in second else ("C", "D"))}
        for i in range(len(kt)):
            F = frac[el][i].reshape(nr * nn, Z + 1) @ basis[el][i]                 # (r n, n_e)
            D, D2 = _doppler_derivatives(F, emid, de)
            for b, fo in fold.items():
                for k, A in (("C", F), ("D", D), ("D2", D2)):
                    if k == "D2" and b not in second:
                        continue
                    acc[f"{k}_{b}"][:, i] = np.einsum("pe,neb->npb", A, fo).reshape(
                        len(nh_grid), nr, nn, -1)
        for key, v in acc.items():
            k, b = key.split("_")
            out[f"{k}_{el}_{b}"] = v
        print(f"[bins] {instrument}: {el} ({time.time() - t0:.0f} s)", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"[bins] wrote {path} ({path.stat().st_size / 2**20:.0f} MiB, {time.time() - t0:.0f} s)")


def build_sync_binned(instrument, out_dir=OUT_DIR, nh_grid=NH_GRID_V2, n_ecut=64):
    """``sync_bins_{instrument}.npz``: ``S_{spec,band,dop}`` (n_nh, n_ecut, n_bin),
    the synchrotron counterpart of ``build_sync`` on the v2 binnings."""
    from soxs.spectra import get_tbabs_absorb
    import _synchrotron as SY
    path = Path(out_dir) / f"sync_bins_{instrument}.npz"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    elo, ehi, M, area, ch_lo, ch_hi, _ = _response(instrument)
    emid, de = 0.5 * (elo + ehi), ehi - elo
    absorb = np.stack([get_tbabs_absorb(emid, nh) for nh in nh_grid])
    ecut = np.geomspace(0.01, 100.0, n_ecut)
    kev_erg = 1.602176634e-9
    photons = np.stack([SY.spectral_shape(emid, Ec) / (emid * kev_erg) * de for Ec in ecut])
    out = dict(ecut=ecut, nh=np.asarray(nh_grid), instrument=np.array(instrument), spec_edges=SPEC_EDGES,
               band_edges=np.array(BANDS), dop_window=np.array(DOPPLER_WINDOW))
    for b, m in binning_matrices(ch_lo, ch_hi).items():
        fo = (absorb * area)[:, :, None] * (M @ m)[None]
        out[f"S_{b}"] = np.einsum("ke,neb->nkb", photons, fo).astype(np.float32)
    np.savez_compressed(path, **out)
    print(f"[sync] wrote {path}")


#: which observed epoch spectrum stands for each response node's in-band
#: spectrum (the halo kernels are weighted by it): the node's own epoch
HALO_EPOCH_OF = {"chandra_aciss_cy0": "2000", "chandra_aciss_cy10": "2010",
                 "chandra_aciss_cy22": "2019", "chandra_acisi_cy22": "2022"}
HALO_NH_REF = 1.43       # Q2's mean column: the observed spectra are "unabsorbed" with it
EPOCH_SPEC_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")


def build_halo_mono(out_dir=OUT_DIR, nh_grid=NH_GRID_V2, e_grid=None, n_photons=400_000,
                    aperture=200.0, seed=11, scratch=None):
    """Monochromatic halo kernels and aperture keep per N_H node (``_dusthalo`` MC).

    The v1 kernels drew photon energies UNIFORMLY within each band, at N_H = 1.2
    only (obs_model section 9: the 0.5-1.5 keV core 0.49 unscattered where the
    observed counts give 0.61-0.65). Here every (N_H, E) point gets its own
    point-source MC (8-fold symmetrised: the kernel is isotropic), so a band
    kernel for ANY in-band spectrum is a weighted sum (``build_halo_bands``).
    The aperture keep uses the same photons (a point at radius r: the fraction
    of |(r, 0) + offset| < aperture). Writes ``dusthalo_v2_keep.npz`` and keeps
    the (large) monochromatic kernels in ``scratch``.
    """
    from _dusthalo import scatter_sky_positions
    RA0, DEC0 = 350.8583, 58.8149
    e_grid = np.geomspace(0.3, 11.0, 40) if e_grid is None else np.asarray(e_grid)
    rng = np.random.default_rng(seed)
    kpix, half = 4 * 0.492, 200
    edges = (np.arange(2 * half + 2) - half - 0.5) * kpix
    r_grid = np.arange(0.0, 240.0, 10.0)
    K = np.zeros((len(nh_grid), len(e_grid), 2 * half + 1, 2 * half + 1), np.float32)
    keep = np.zeros((len(nh_grid), len(e_grid), len(r_grid)))
    unsc = np.zeros((len(nh_grid), len(e_grid)))
    t0 = time.time()
    for a, nh in enumerate(nh_grid):
        for i, e0 in enumerate(e_grid):
            ra = np.full(n_photons, RA0); dec = np.full(n_photons, DEC0)
            ra2, dec2, _ = scatter_sky_positions(ra, dec, np.full(n_photons, e0), nh=float(nh),
                                                 seed=int(rng.integers(1e9)), verbose=False)
            dx = -(ra2 - RA0) * np.cos(np.deg2rad(DEC0)) * 3600.0
            dy = (dec2 - DEC0) * 3600.0
            unsc[a, i] = np.mean((dx == 0) & (dy == 0))
            H = 0.0
            for sx, sy, swap in ((1, 1, False), (-1, 1, False), (1, -1, False), (-1, -1, False),
                                 (1, 1, True), (-1, 1, True), (1, -1, True), (-1, -1, True)):
                u, v = (sy * dy, sx * dx) if not swap else (sx * dx, sy * dy)
                H = H + np.histogram2d(u, v, bins=[edges, edges])[0]
            K[a, i] = H / (8.0 * n_photons)
            for j, r0 in enumerate(r_grid):
                keep[a, i, j] = np.mean(np.hypot(dx + r0, dy) < aperture)
        print(f"[halo2] N_H {nh:g}: 1 keV unscattered {np.interp(1.0, e_grid, unsc[a]):.3f}, "
              f"keep(r=0) {np.interp(1.0, e_grid, keep[a, :, 0]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    path = Path(out_dir) / "dusthalo_v2_keep.npz"
    np.savez_compressed(path, nh=np.asarray(nh_grid), e_grid=e_grid, r_grid=r_grid, aperture_keep=keep,
                        aperture_arcsec=aperture, unscattered=unsc, kernel_pixel_arcsec=kpix,
                        kernel_half=half, n_photons=n_photons)
    print(f"[halo2] wrote {path}")
    if scratch:
        mono = Path(scratch) / "dusthalo_v2_mono_kernels.npy"
        np.save(mono, K)
        print(f"[halo2] monochromatic kernels -> {mono}")
    return K, e_grid


def in_band_weights(e_grid, instrument, nh, *, nh_ref=HALO_NH_REF):
    """(n_band, n_e): tent weights on ``e_grid`` (log E) of the in-band count
    spectrum of the node's observed epoch, re-absorbed from ``nh_ref`` to ``nh``
    (the ARF cancels: counts(nh) ~ counts_obs exp(-sigma (nh - nh_ref)))."""
    from soxs.spectra import get_tbabs_absorb
    d = np.load(EPOCH_SPEC_DIR / f"epoch_{HALO_EPOCH_OF[instrument]}_spectrum.npz")
    eb = np.asarray(d["ebins"]); c = np.asarray(d["counts"], np.float64)
    em = 0.5 * (eb[1:] + eb[:-1])
    c = c * get_tbabs_absorb(em, nh) / get_tbabs_absorb(em, nh_ref)
    le = np.log(e_grid)
    x = np.clip(np.interp(np.log(em), le, np.arange(len(le))), 0, len(le) - 1)
    i0 = np.minimum(x.astype(int), len(le) - 2); f = x - i0
    W = np.zeros((len(BANDS), len(e_grid)))
    for b, (lo, hi) in enumerate(BANDS):
        m = (em >= lo) & (em < hi)
        np.add.at(W[b], i0[m], c[m] * (1 - f[m]))
        np.add.at(W[b], i0[m] + 1, c[m] * f[m])
        W[b] /= max(W[b].sum(), 1e-300)
    return W


def build_halo_bands(K, e_grid, out_dir=OUT_DIR, nh_grid=NH_GRID_V2, instruments=V2_INSTRUMENTS):
    """``dusthalo_v2_{instrument}.npz``: band kernels (n_nh, n_band, 401, 401),
    weighted by that node's observed in-band spectrum at each N_H."""
    keep = np.load(Path(out_dir) / "dusthalo_v2_keep.npz")
    for inst in instruments:
        ker = np.zeros((len(nh_grid), len(BANDS)) + K.shape[-2:], np.float32)
        W = np.stack([in_band_weights(e_grid, inst, nh) for nh in nh_grid])        # (n_nh, band, e)
        for a in range(len(nh_grid)):
            ker[a] = np.einsum("be,exy->bxy", W[a], K[a])
        half = int(keep["kernel_half"])
        core = ker[:, :, half, half]
        path = Path(out_dir) / f"dusthalo_v2_{inst}.npz"
        np.savez_compressed(path, kernel=ker, nh=np.asarray(nh_grid), bands=np.array(BANDS),
                            kernel_pixel_arcsec=float(keep["kernel_pixel_arcsec"]), kernel_half=half,
                            e_grid=e_grid, band_weights=W, epoch=HALO_EPOCH_OF[inst],
                            nh_ref=HALO_NH_REF,
                            **{k: np.asarray(keep[k]) for k in ("r_grid", "aperture_keep", "aperture_arcsec")})
        print(f"[halo2] {inst} (spectrum of {HALO_EPOCH_OF[inst]}): central pixel at N_H "
              + ", ".join(f"{nh:g}: {core[a, 0]:.3f}" for a, nh in enumerate(nh_grid)) + " (0.5-1.5 keV)")
        print(f"[halo2] wrote {path}")


def check_binned(instrument="chandra_aciss_cy0", out_dir=OUT_DIR, kT=1.5, net=2e11, nh=1.2,
                 elements=("Si", "S", "Fe", "O")):
    """Independent check of one v2 table entry (rho = 1): an APEC NEI spectrum with
    ALL of the element's ions at their ``_nei`` fractions in ONE generator call,
    through soxs' own ``convolve_spectrum``, the ARF and TBabs, binned by exact
    native-channel overlap, against ``C_{el}_spec`` / ``C_{el}_band``; plus the
    v1 route (117 eV channels spread uniformly onto the bins) for comparison."""
    import soxs
    from soxs.response import AuxiliaryResponseFile, RedistributionMatrixFile
    from soxs.spectra import get_tbabs_absorb
    from soxs.thermal_spectra import ApecGenerator
    d = np.load(binned_path(instrument, out_dir))
    kt, netg, rho, nhg = d["kt"], d["net"], d["rho"], d["nh"]
    i = int(np.argmin(np.abs(kt - kT))); j = int(np.argmin(np.abs(netg - net)))
    a = int(np.argmin(np.abs(nhg - nh)))
    reg = soxs.instrument_registry[instrument]
    arf = AuxiliaryResponseFile(reg["arf"]); rmf = RedistributionMatrixFile(reg["rmf"])
    elo, ehi = np.asarray(rmf.elo), np.asarray(rmf.ehi); emid, de = 0.5 * (elo + ehi), ehi - elo
    ch_lo = np.asarray(rmf.ebounds_data["E_MIN"]); ch_hi = np.asarray(rmf.ebounds_data["E_MAX"])
    bm = binning_matrices(ch_lo, ch_hi)
    kt0, net0, frac = _nei.load_table()
    v1 = np.load(Path(out_dir) / f"emissivity_{instrument}.npz")
    v1_edges = np.asarray(v1["ch_edges"])
    lo1, hi1 = v1_edges[:-1], v1_edges[1:]
    O1 = channel_overlap(lo1, hi1, SPEC_EDGES[:-1], SPEC_EDGES[1:])           # (n_ch1, 31)
    a1 = int(np.argmin(np.abs(np.asarray(v1["nh"]) - nhg[a])))
    mids = 0.5 * (SPEC_EDGES[1:] + SPEC_EDGES[:-1])
    for el in elements:
        Z = ELEMENTS[el]
        names = [f"{el}^{q}" for q in range(Z + 1)]
        gen = ApecGenerator(float(elo[0]), float(ehi[-1]), len(elo), binscale="linear", nei=True,
                            var_elem=names, abund_table="angr")
        f = frac[el][i, j]
        tot = np.asarray(gen.get_nei_spectrum(float(kt[i]), {n: float(x) for n, x in zip(names, f)},
                                              0.0, 1.0).flux.value) * de
        base = np.asarray(gen.get_nei_spectrum(float(kt[i]), {n: 0.0 for n in names}, 0.0, 1.0).flux.value) * de
        ph = (tot - base) * np.asarray(arf.interpolate_area(emid).value) * get_tbabs_absorb(emid, float(nhg[a]))
        ch = rmf.convolve_spectrum(ph, 1.0, noisy=False, rate=True)
        ref = {b: ch @ m for b, m in bm.items()}
        for b in ("spec", "band", "dop"):
            tab = np.asarray(d[f"C_{el}_{b}"][a, i, -1, j], np.float64)
            err = np.max(np.abs(tab - ref[b]) / np.maximum(np.abs(ref[b]), 1e-3 * np.abs(ref[b]).max()))
            print(f"[check] {instrument} {el} kT {kt[i]:.2f} n_e t {netg[j]:.1e} N_H {nhg[a]:g}: "
                  f"{b:4s} table vs direct soxs fold, max rel err {err:.2e}")
        # the second-order Doppler table against APEC's own velocity broadening
        # (sigma_v): N + sigma_beta^2 / 2 D2 vs the broadened spectrum
        for sv in (1000.0, 2000.0):
            tb = np.asarray(gen.get_nei_spectrum(float(kt[i]), {n: float(x) for n, x in zip(names, f)},
                                                 0.0, 1.0, velocity=sv).flux.value) * de
            bb = np.asarray(gen.get_nei_spectrum(float(kt[i]), {n: 0.0 for n in names}, 0.0, 1.0,
                                                 velocity=sv).flux.value) * de
            phb = (tb - bb) * np.asarray(arf.interpolate_area(emid).value) * get_tbabs_absorb(emid, float(nhg[a]))
            refb = rmf.convolve_spectrum(phb, 1.0, noisy=False, rate=True) @ bm["spec"]
            sb2 = (sv / 2.99792458e5) ** 2
            second = np.asarray(d[f"C_{el}_spec"][a, i, -1, j], np.float64) + \
                0.5 * sb2 * np.asarray(d[f"D2_{el}_spec"][a, i, -1, j], np.float64)
            m = ref["spec"] > 0.05 * ref["spec"].max()
            e_broad = np.max(np.abs(refb[m] / ref["spec"][m] - 1))
            e_model = np.max(np.abs(second[m] / refb[m] - 1))
            print(f"[check]   sigma_v {sv:.0f} km/s: broadening moves bins by up to {e_broad:.3f}; "
                  f"C + s^2/2 D2 matches the broadened spectrum to {e_model:.4f}")
        old = np.asarray(v1[f"C_{el}"][a1, i, j], np.float64) @ O1              # (n_ch1,) @ (n_ch1, 31)
        r = ref["spec"] / np.maximum(old, 1e-30)
        r = r / np.median(r[ref["spec"] > 1e-3 * ref["spec"].max()])
        sel = [k for k, m in enumerate(mids) if round(m, 1) in (0.8, 1.2, 1.4, 1.8, 2.0, 2.4, 2.6, 6.4, 6.6, 6.8)]
        print(f"[check]   exact / v1-route (median-norm.): " + " ".join(f"{mids[k]:.1f}:{r[k]:.3f}" for k in sel))


def check_kt_interp(instrument="chandra_aciss_cy0", out_dir=OUT_DIR, kt_range=(0.25, 8.0),
                    nets=(3e10, 1e11, 3e11, 1e12), elements=("Fe", "Si", "S", "O")):
    """Interpolation error in kT: the exact band rate at the geometric MIDPOINT
    between two kT nodes (APEC at that kT, ion fractions from the eigen solution
    at that kT) against linear and log-linear interpolation of the node values
    (rho = 1, N_H 1.2). Returns {el: (band, linear errors, log errors)}."""
    import casa_jaxobs_nei as NH
    from soxs.thermal_spectra import ApecGenerator
    from soxs.spectra import get_tbabs_absorb
    d = np.load(binned_path(instrument, out_dir))
    kt, netg, nhg = d["kt"], d["net"], d["nh"]
    a = int(np.argmin(np.abs(nhg - 1.2)))
    elo, ehi, M, area, ch_lo, ch_hi, _ = _response(instrument)
    emid, de = 0.5 * (elo + ehi), ehi - elo
    fold = (area * get_tbabs_absorb(emid, float(nhg[a])))[:, None] * (M @ binning_matrices(ch_lo, ch_hi)["band"])
    iks = [i for i in range(len(kt) - 1) if kt_range[0] <= kt[i] and kt[i + 1] <= kt_range[1]]
    out = {}
    for el in elements:
        Z = ELEMENTS[el]
        names = [f"{el}^{q}" for q in range(Z + 1)]
        gen = ApecGenerator(float(elo[0]), float(ehi[-1]), len(elo), binscale="linear", nei=True,
                            var_elem=names, abund_table="angr")
        err_lin, err_log = [], []
        for i in iks:
            km = float(np.sqrt(kt[i] * kt[i + 1]))
            for tau in nets:
                j = int(np.argmin(np.abs(netg - tau)))
                pop = NH.ionise_along(el, [km], [1.0], [netg[j]])[0]
                spec = lambda ab: np.asarray(gen.get_nei_spectrum(km, ab, 0.0, 1.0).flux.value) * de  # noqa: E731
                ph = spec({n: float(x) for n, x in zip(names, pop)}) - spec({n: 0.0 for n in names})
                exact = ph @ fold
                c0 = np.asarray(d[f"C_{el}_band"][a, i, -1, j], np.float64)
                c1 = np.asarray(d[f"C_{el}_band"][a, i + 1, -1, j], np.float64)
                lin = 0.5 * (c0 + c1)
                lg = np.sqrt(np.maximum(c0, 1e-300) * np.maximum(c1, 1e-300))
                ok = exact > 1e-6 * exact.max()
                err_lin.append(np.where(ok, lin / exact - 1, np.nan))
                err_log.append(np.where(ok, lg / exact - 1, np.nan))
        el_, eg_ = np.array(err_lin), np.array(err_log)
        out[el] = (el_, eg_)
        for b, (lo, hi) in enumerate(BANDS):
            x, y = np.abs(el_[:, b]), np.abs(eg_[:, b])
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            if len(x):
                print(f"[kt-interp] {el:2s} {lo:.1f}-{hi:.1f} keV: |error| linear median {np.median(x):.3f} / "
                      f"95th {np.percentile(x, 95):.3f}; log median {np.median(y):.3f} / 95th {np.percentile(y, 95):.3f}"
                      f"  ({len(x)} midpoints, kT {kt_range[0]}-{kt_range[1]} keV)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instruments", nargs="+",
                    default=["chandra_aciss_cy0", "chandra_acisi_cy22"])
    ap.add_argument("--rebin", type=int, default=8, help="RMF channels per output channel")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--halo", action="store_true", help="build only the dust-halo kernels")
    ap.add_argument("--sync", action="store_true", help="build only the synchrotron tables")
    # ---- v2 (analysis-bin tables; nothing above changes without these flags) ----
    ap.add_argument("--basis", action="store_true", help="v2: build only the APEC per-ion basis")
    ap.add_argument("--bins", action="store_true", help="v2: emissivity_bins_{instrument}.npz")
    ap.add_argument("--sync-bins", action="store_true", help="v2: sync_bins_{instrument}.npz")
    ap.add_argument("--halo-v2", action="store_true", help="v2: dusthalo_v2_{keep,<instrument>}.npz")
    ap.add_argument("--nei-history", default=None, help="v2: casa_jaxobs_nei table (default its TABLE_PATH)")
    ap.add_argument("--scratch", default=None, help="v2 halo: where the monochromatic kernels go")
    ap.add_argument("--check", action="store_true", help="v2: check one table entry against soxs directly")
    args = ap.parse_args()
    if args.check:
        for inst in args.instruments:
            check_binned(inst, Path(args.out))
            check_kt_interp(inst, Path(args.out))
        return
    if args.basis:
        for inst in args.instruments:
            elo, ehi, *_ = _response(inst)
            load_basis(elo, ehi, Path(args.out))
        return
    if args.bins:
        for inst in args.instruments:
            build_binned(inst, Path(args.out), args.nei_history)
        return
    if args.sync_bins:
        for inst in args.instruments:
            build_sync_binned(inst, Path(args.out))
        return
    if args.halo_v2:
        K, e_grid = build_halo_mono(Path(args.out), scratch=args.scratch)
        build_halo_bands(K, e_grid, Path(args.out), instruments=args.instruments)
        return
    if args.halo:
        build_halo(Path(args.out))
        return
    if args.sync:
        for inst in args.instruments:
            build_sync(inst, args.rebin, Path(args.out))
        return
    for inst in args.instruments:
        build(inst, args.rebin, Path(args.out))


if __name__ == "__main__":
    main()
