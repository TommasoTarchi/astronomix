"""
The v2 observation tables for casa_xfit (``--obs v2``), wired as the
obs_tables worker's ``INTEGRATION.md`` (2026-09-25) specifies.

Host-side loaders only (numpy / jnp constants); the forward-model pieces that
use them live in ``casa_xfit.make_forward`` (``obs == "v2"`` branch):

* emissivity tables on the exact analysis binnings (``emissivity_bins_*``:
  the six image bands, the 31 spectral bins 0.7-6.9 keV, the two Doppler
  moments over exactly [1.78, 1.94] keV), with the NEI T_e-history axis
  (kT x rho combined, ``history=True``) -- no channel rebinning;
* the v2 N_H grid 0.5-4.0e22 (read from the files);
* synchrotron tables on the same binnings (``sync_bins_*``);
* the v2 dust halo, per N_H node and band, weighted by the observed in-band
  spectrum (``dusthalo_v2_*``), and its aperture keep per N_H node;
* the exact-window Doppler data statistic (``data/doppler_si_2004_v2.npz``).

Per-instrument stacks are kept on the device (4 instruments) and mixed per
epoch inside the scan with the epoch's instrument weights (as the v1 spectra
already were): stacking 15 epochs of band tables would cost 1.3 GB.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import casa_jaxobs as J

INSTRUMENTS = ("chandra_aciss_cy0", "chandra_aciss_cy10", "chandra_aciss_cy22", "chandra_acisi_cy22")
DOPPLER_V2_FILE = J.DATA_DIR / "doppler_si_2004_v2.npz"
#: keV added to (E - E0) in the stored first Doppler moment (``load_tables``)
DOP_SHIFT = 0.1


def nh_grid(table_dir=J.TABLE_DIR, instruments=INSTRUMENTS):
    """The v2 N_H grid (1e22), checked to be the same in every v2 file."""
    table_dir = Path(table_dir)
    grids = {}
    for i in instruments:
        grids[f"emissivity_bins_{i}"] = np.load(table_dir / f"emissivity_bins_{i}.npz")["nh"]
        grids[f"sync_bins_{i}"] = np.load(table_dir / f"sync_bins_{i}.npz")["nh"]
        grids[f"dusthalo_v2_{i}"] = np.load(table_dir / f"dusthalo_v2_{i}.npz")["nh"]
    g0 = np.asarray(next(iter(grids.values())), np.float64)
    for k, g in grids.items():
        if np.shape(g) != g0.shape or not np.allclose(g, g0):
            raise ValueError(f"v2 N_H grids differ: {k} {g} vs {g0}")
    return g0


def _stack(binning, table_dir, history, kinds=("C", "D")):
    out = {k: [] for k in kinds}
    meta = None
    for i in INSTRUMENTS:
        st = J.load_binned_stack(i, binning, history=history, second_order="D2" in kinds, table_dir=table_dir)
        for k in kinds:
            out[k].append(st[k])
        if meta is None:
            meta = st
        elif not (np.allclose(st["lkt"], meta["lkt"]) and np.allclose(st["lnet"], meta["lnet"])
                  and np.allclose(st["rho"], meta["rho"]) and np.allclose(st["nh"], meta["nh"])):
            raise ValueError(f"v2 {binning} tables of {i} are on a different grid")
    return {k: np.stack(v) for k, v in out.items()}, meta


def _sync(binning, table_dir):
    S, lec0 = [], None
    for i in INSTRUMENTS:
        lec, s, _ = J.load_sync_binned(i, binning, table_dir=table_dir)
        if lec0 is not None and not np.allclose(lec, lec0):
            raise ValueError("synchrotron cutoff grids differ")
        lec0 = lec
        S.append(s)
    return lec0, np.stack(S)


def load_tables(table_dir=J.TABLE_DIR, *, history=True, spectra=True, doppler=True, spec_kinds=("C",)):
    """Every v2 table casa_xfit needs, as device arrays (per instrument, in
    ``INSTRUMENTS`` order).

    Returns a dict:
      ``nh`` (8,), ``lng`` ln nh;
      ``tables0`` / ``tables_dop0``: v2 dicts (grids for the corner lookup);
      ``Cb``, ``Db`` (4, 8, 10, 384, 57, 6), ``Sb`` (4, 8, 64, 6), ``lecut``;
      ``halo_K`` (4, 8, 6, 401, 401) and ``halo_meta`` (kernel_half, pixel);
      ``Cs`` (4, 8, 10, 384, 57, 31), ``Ss`` (4, 8, 64, 31), ``keep`` (8, 31, 24),
      ``r_grid`` (24,); with ``spec_kinds`` containing "D" / "D2" also the first /
      second-order Doppler spectral tables ``Ds`` / ``D2s`` (same shape as ``Cs``,
      2.2 GB each: the stage-4 gain derivative and line broadening);
      ``Cd``, ``Dd`` (4, 8, 10, 384, 57, 2), ``Sd`` (4, 8, 64, 2), ``E0``.
    """
    table_dir = Path(table_dir)
    nh = nh_grid(table_dir)
    out = dict(nh=nh, lng=np.log(nh), history=history)
    out["tables0"] = J.load_binned_tables(INSTRUMENTS[0], 1.2, "band", history=history, table_dir=table_dir)
    band, meta = _stack("band", table_dir, history)
    lec, Sb = _sync("band", table_dir)
    out.update(Cb=jnp.asarray(band["C"]), Db=jnp.asarray(band["D"]), Sb=jnp.asarray(Sb), lecut=lec)
    # halo: per instrument, per N_H node and band
    K, h0 = [], None
    for i in INSTRUMENTS:
        h = J.load_halo_stack(i, table_dir)
        if h0 is None:
            h0 = h
        elif (int(h["kernel_half"]) != int(h0["kernel_half"])
              or abs(float(h["kernel_pixel_arcsec"]) - float(h0["kernel_pixel_arcsec"])) > 1e-9):
            raise ValueError("halo kernels differ in geometry between instruments")
        K.append(np.asarray(h["kernel"], np.float32))
    out["halo_K"] = jnp.asarray(np.stack(K))
    out["halo_meta"] = dict(kernel_half=int(h0["kernel_half"]),
                            kernel_pixel_arcsec=float(h0["kernel_pixel_arcsec"]))
    if spectra:
        kinds = tuple(k for k in ("C", "D", "D2") if k in tuple(spec_kinds) + ("C",))
        if "D2" in kinds and "D" not in kinds:
            kinds = ("C", "D", "D2")
        spec, _ = _stack("spec", table_dir, history, kinds=kinds)
        _, Ss = _sync("spec", table_dir)
        mids = 0.5 * (J.SPEC_EDGES[1:] + J.SPEC_EDGES[:-1])
        keep = np.stack([np.stack([np.interp(mids, h0["e_grid"], h0["aperture_keep"][a, :, k])
                                   for k in range(len(h0["r_grid"]))], 1)
                         for a in range(len(nh))])                                  # (8, 31, 24)
        out.update(Cs=jnp.asarray(spec["C"]), Ss=jnp.asarray(Ss), keep=jnp.asarray(keep, jnp.float32),
                   r_grid=jnp.asarray(h0["r_grid"], jnp.float32), spec_kinds=kinds)
        if "D" in kinds:
            out["Ds"] = jnp.asarray(spec["D"])
        if "D2" in kinds:
            out["D2s"] = jnp.asarray(spec["D2"])
    if doppler:
        dop, dmeta = _stack("dop", table_dir, history)
        _, Sd = _sync("dop", table_dir)
        # The first moment M1 = sum (E - E0) counts is SIGNED (negative in 77-100 %
        # of the table entries: the continuum falls across the window), but the
        # log-kT interpolation of band_columns and the log-linear cutoff
        # interpolation of sync_columns clamp their table to > 0 -- a negative M1
        # would become ~0. The tables therefore carry the POSITIVE moment
        # M1' = M1 + DOP_SHIFT M0 = sum (E - E0 + DOP_SHIFT) counts (DOP_SHIFT >
        # E0 - 1.78 keV), and the forward model subtracts DOP_SHIFT M0 after the
        # (linear) N_H mixing -- exact at the table nodes.
        for T in (dop["C"], dop["D"], Sd):
            T[..., 1] += DOP_SHIFT * T[..., 0]
        out["tables_dop0"] = J.load_binned_tables(INSTRUMENTS[0], 1.2, "dop", history=history,
                                                  table_dir=table_dir)
        if not (np.min(dop["C"][..., 1]) >= 0.0 and np.min(Sd[..., 1]) >= 0.0):
            raise ValueError("the shifted first Doppler moment is not >= 0: raise DOP_SHIFT")
        if DOP_SHIFT <= float(dmeta["E0"]) - float(np.asarray(dmeta["edges"]).ravel()[0]):
            raise ValueError("DOP_SHIFT must exceed E0 - (window low edge)")
        out.update(Cd=jnp.asarray(dop["C"]), Dd=jnp.asarray(dop["D"]), Sd=jnp.asarray(Sd),
                   E0=float(dmeta["E0"]), dop_shift=DOP_SHIFT, dop_window=tuple(float(x) for x in np.asarray(dmeta["edges"]).ravel()[:2]))
    return out


def inst_weights(resp):
    """(4,) weights of the table instruments for an epoch's ``resp`` list."""
    w = np.zeros(len(INSTRUMENTS))
    for inst, ww in resp:
        w[INSTRUMENTS.index(inst)] += ww
    return w


def load_doppler_data(path=DOPPLER_V2_FILE):
    """The exact-window data Doppler statistic (centre RA0/DEC0, CCO masked), in
    ``casa_pluto_diff.DOPPLER``'s format."""
    d = np.load(path)
    out = {"n": len(d["v_kms"]), **{k: np.asarray(d[k]) for k in d.files}}
    out["annulus_arcsec"] = np.asarray(out["annulus_arcsec"], float)
    out["source"] = str(path)
    return out
