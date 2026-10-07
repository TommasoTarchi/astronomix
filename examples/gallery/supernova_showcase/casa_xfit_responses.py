"""
Per-epoch CIAO responses for casa_xfit (``--responses ciao``), as correction
factors on the instrument tables the observation model already folds through.

The emissivity / synchrotron tables (``casa_jaxobs_tables``) are folded
through the soxs aimpoint responses (``aciss_aimpt_cy{0,10,22}``,
``acisi_aimpt_cy22``), mixed per epoch linearly in time
(``casa_xfit.epoch_responses``). The CIAO products
(``/export/data/lstorcks/chandra_casa/responses``, CIAO 4.18 / CALDB 4.12.4,
contamination N0016) give each epoch's real response. The cheapest faithful
route keeps the tables and multiplies the predicted counts:

* images -- per pixel and band, ``relexp_band`` (the exposure-map geometry:
  vignetting, chip gaps, bad columns, dither, relative to the epoch's r < 200"
  weighted ARF) times the per-band area ratio
  <A_ciao>_b / <A_model>_b, both photon-weighted with the epoch's own
  r < 200" spectrum (the weighting ``combine_epochs`` used for
  ``band_area_w``), so per-pixel band area = relexp x band_area_w exactly;
* spectra -- per 0.2 keV analysis bin, the same photon-weighted ARF ratio on
  the bin (the weighted ARF already averages the vignetting over r < 200").

What this leaves out (quantified by ``validate``): the RMF difference between
the CIAO epoch RMF and the soxs RMF (the ratio is taken in photon energy, the
counts are in channels), and the dependence of the in-band / in-bin ratio on
the spectral shape (the model's spectrum is not the data's).

    ./run.sh casa_xfit_responses.py validate     # CPU, numpy + astropy only
"""
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

RESP_DIR = Path("/export/data/lstorcks/chandra_casa/responses")
SOXS_DIR = Path("/export/data/lstorcks/soxs_data")
SPEC_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")
CACHE_DIR = Path("/export/data/lstorcks/casa_orlando150/work/stage2/integrate")
BANDS = ((0.5, 1.5), (1.5, 2.1), (2.1, 2.8), (2.8, 4.2), (4.2, 6.0), (6.0, 7.0))
SPEC_EDGES = np.round(np.arange(0.7, 7.0001, 0.2), 3)
R_AP_ARCSEC, PIX_NATIVE = 200.0, 0.492
#: the soxs response behind each table instrument (casa_jaxobs_tables folds
#: through ``soxs.instrument_registry[inst]``'s arf / rmf)
SOXS_BASE = {"chandra_aciss_cy0": "aciss_aimpt_cy0", "chandra_aciss_cy10": "aciss_aimpt_cy10",
             "chandra_aciss_cy22": "aciss_aimpt_cy22", "chandra_acisi_cy22": "acisi_aimpt_cy22"}


# =============================================================================
# ============ ↓ OGIP readers ↓ ===============================================
# =============================================================================
def read_arf(path):
    """(e_lo, e_hi, area cm^2)."""
    with fits.open(path) as f:
        d = f["SPECRESP"].data
        return (np.asarray(d["ENERG_LO"], float), np.asarray(d["ENERG_HI"], float),
                np.asarray(d["SPECRESP"], float))


def read_rmf(path):
    """Dense RMF (e_lo, e_hi, M[n_e, n_ch], ch_lo, ch_hi), rows = channel
    probabilities (no ARF)."""
    with fits.open(path) as f:
        mext = [h for h in f if h.name in ("MATRIX", "SPECRESP MATRIX")][0]
        eb = f["EBOUNDS"].data
        ch_lo, ch_hi = np.asarray(eb["E_MIN"], float), np.asarray(eb["E_MAX"], float)
        tlmin = mext.header.get(f"TLMIN{mext.columns.names.index('F_CHAN') + 1}", 1)
        d = mext.data
        e_lo, e_hi = np.asarray(d["ENERG_LO"], float), np.asarray(d["ENERG_HI"], float)
        M = np.zeros((len(e_lo), len(ch_lo)))
        for i in range(len(e_lo)):
            fc = np.atleast_1d(d["F_CHAN"][i]).astype(int)
            nc = np.atleast_1d(d["N_CHAN"][i]).astype(int)
            m = np.asarray(d["MATRIX"][i], float)
            k = 0
            for f0, n in zip(fc, nc):
                if n > 0:
                    M[i, f0 - tlmin:f0 - tlmin + n] = m[k:k + n]
                    k += n
    return e_lo, e_hi, M, ch_lo, ch_hi
# =============================================================================
# ============ ↑ OGIP readers ↑ ===============================================
# =============================================================================


# =============================================================================
# ============ ↓ Photon weights (as combine_epochs) ↓ =========================
# =============================================================================
def _counts_from_events(obsids):
    """0.3-10 keV, 50 eV counts spectrum inside r < 200" from the obsids'
    r265 event files (combine_epochs.counts_spectrum_from_events)."""
    eb = np.round(np.arange(0.3, 10.0001, 0.05), 3)
    tot = np.zeros(len(eb) - 1)
    for o in obsids:
        info = json.loads((RESP_DIR / str(o) / "prep.json").read_text())
        with fits.open(info["evt"], memmap=True) as f:
            ev = f["EVENTS"].data
            x, y, e = np.asarray(ev["x"]), np.asarray(ev["y"]), np.asarray(ev["energy"]) * 1e-3
        x0, y0 = info["sky_centre"]
        m = np.hypot(x - x0, y - y0) * PIX_NATIVE < R_AP_ARCSEC
        tot += np.histogram(e[m], eb)[0]
    return eb, tot


def photon_weights(label, obsids, cache=True):
    """The epoch's r < 200" photon spectrum on the first obsid's ARF grid,
    unfolded through that ARF (no RMF) -- exactly the ``w`` of
    ``combine_epochs`` (so the band areas reproduce ``band_area_w``)."""
    cpath = CACHE_DIR / "ciao_photon_weights.npz"
    key = f"w_{label}"
    if cache and cpath.exists():
        d = np.load(cpath)
        if key in d.files:
            return np.asarray(d[f"e_{label}"]), np.asarray(d[key])
    lo, hi, aw0 = read_arf(RESP_DIR / str(obsids[0]) / f"{obsids[0]}_r200.arf")
    e_arf = 0.5 * (lo + hi)
    p = SPEC_DIR / f"epoch_{label}_spectrum.npz"
    if p.exists():
        d = np.load(p)
        eb, c = np.asarray(d["ebins"], float), np.asarray(d["counts"], float)
    else:
        eb, c = _counts_from_events(obsids)
    em = 0.5 * (eb[1:] + eb[:-1])
    a = np.interp(em, e_arf, aw0)
    s = np.where(a > 1.0, c / np.maximum(a, 1.0) / (eb[1:] - eb[:-1]), 0.0)
    w = np.interp(e_arf, em, s, left=0.0, right=0.0)
    if cache:
        old = dict(np.load(cpath)) if cpath.exists() else {}
        old.update({key: w, f"e_{label}": e_arf})
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, **old)
    return e_arf, w
# =============================================================================
# ============ ↑ Photon weights (as combine_epochs) ↑ =========================
# =============================================================================


# =============================================================================
# ============ ↓ Correction factors ↓ =========================================
# =============================================================================
S_CYCLE_YEAR = {0: 1999.0, 10: 2009.0, 22: 2021.0}


def epoch_responses(detnam, year):
    """casa_xfit.epoch_responses (no jax import): [(table instrument, weight)]."""
    chips = [int(c) for n in detnam for c in str(n).replace("ACIS-", "") if c.isdigit()]
    if chips and max(chips) <= 3:
        return [("chandra_acisi_cy22", 1.0)]
    cy = sorted(S_CYCLE_YEAR, key=S_CYCLE_YEAR.get)
    yrs = np.array([S_CYCLE_YEAR[c] for c in cy])
    y = float(np.clip(year, yrs[0], yrs[-1]))
    k = int(np.clip(np.searchsorted(yrs, y) - 1, 0, len(yrs) - 2))
    w = (y - yrs[k]) / (yrs[k + 1] - yrs[k])
    return [(f"chandra_aciss_cy{cy[k]}", 1.0 - w), (f"chandra_aciss_cy{cy[k + 1]}", w)]


def model_arf(resp, e):
    """The ARF behind the epoch's table mix, sum_i w_i A_soxs,i(E), at ``e``."""
    out = 0.0
    for inst, w in resp:
        lo, hi, a = read_arf(SOXS_DIR / f"{SOXS_BASE[inst]}.arf")
        out = out + w * np.interp(e, 0.5 * (lo + hi), a)
    return out


def _bin_ratio(e, w, a_num, a_den, edges):
    """Photon-weighted sum_bin w a_num / sum_bin w a_den per [edges) bin."""
    out = []
    for b0, b1 in edges:
        m = (e >= b0) & (e < b1)
        den = (w[m] * a_den[m]).sum()
        out.append((w[m] * a_num[m]).sum() / den if den > 0 else 1.0)
    return np.array(out)



def _fold(e, w, a, rmf):
    """Counts per channel for photon spectrum w (per keV, on e) through ARF a
    (on e) and a dense RMF; returns (counts, channel mid energies)."""
    el, eh, M, cl, ch = rmf
    er = 0.5 * (el + eh)
    s = np.interp(er, e, w, left=0.0, right=0.0) * np.interp(er, e, a) * (eh - el)
    return s @ M, 0.5 * (cl + ch)


def _channel_bins(counts, ch_mid, edges):
    return np.array([counts[(ch_mid >= b0) & (ch_mid < b1)].sum() for b0, b1 in edges])


_RMF_CACHE = {}


def _rmf(path):
    if str(path) not in _RMF_CACHE:
        _RMF_CACHE[str(path)] = read_rmf(path)
    return _RMF_CACHE[str(path)]


def folded_counts(e, w, resp, a_c, rmf_c, edges):
    """(CIAO, table) counts per [edges) channel bin for photon spectrum w: the
    epoch's ARF x RMF, and sum_i w_i A_i R_i of the table instruments (the
    tables of an epoch are mixed after folding)."""
    cc, chm = _fold(e, w, a_c, rmf_c)
    mod = 0.0
    for inst, wi in resp:
        l2, h2, a2 = read_arf(SOXS_DIR / f"{SOXS_BASE[inst]}.arf")
        cm, chm2 = _fold(e, w, np.interp(e, 0.5 * (l2 + h2), a2), _rmf(SOXS_DIR / f"{SOXS_BASE[inst]}.rmf"))
        mod = mod + wi * _channel_bins(cm, chm2, edges)
    return _channel_bins(cc, chm, edges), mod


SPEC_BINS = list(zip(SPEC_EDGES[:-1], SPEC_EDGES[1:]))
#: correction modes: "fold" = the epoch's photon spectrum folded through the
#: CIAO ARF x RMF over the same folded through the tables' soxs ARF x RMF, per
#: channel band / bin (removes the static <= 8 % RMF pattern the plain ARF
#: ratio leaves at the 2.0 / 2.6 / 3.2 keV bins; ``validate``); "arf" = the
#: photon-weighted ARF ratio (RMFs assumed equal)
MODES = ("fold", "arf")


def epoch_correction(label, resp, mode="fold"):
    """Correction factors for one epoch: dict(img (6, 256, 256), band_ratio (6,),
    spec_ratio (31,), and diagnostics)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    em = np.load(RESP_DIR / "epochs" / f"epoch_{label}_expmaps.npz")
    obsids = [int(o) for o in np.asarray(em["obsids"]).ravel()]
    e_arf, w = photon_weights(label, obsids)
    lo, hi, a_c = read_arf(RESP_DIR / "epochs" / f"epoch_{label}_r200.arf")
    e = 0.5 * (lo + hi)
    if not np.allclose(e, e_arf):
        w = np.interp(e, e_arf, w, left=0.0, right=0.0)
    a_m = model_arf(resp, e)
    band_arf = _bin_ratio(e, w, a_c, a_m, BANDS)
    band_area = _bin_ratio(e, w, a_c, np.ones_like(e), BANDS)          # <A_ciao>_b
    spec_arf = _bin_ratio(e, w, a_c, a_m, SPEC_BINS)
    rmf_c = _rmf(RESP_DIR / "epochs" / f"epoch_{label}_r200.rmf")
    cb, mb = folded_counts(e, w, resp, a_c, rmf_c, BANDS)
    cs, ms = folded_counts(e, w, resp, a_c, rmf_c, SPEC_BINS)
    band_fold = cb / mb
    spec_fold = np.where(ms > 0, cs / np.where(ms > 0, ms, 1.0), 1.0)
    band_ratio, spec_ratio = (band_fold, spec_fold) if mode == "fold" else (band_arf, spec_arf)
    rel = np.asarray(em["relexp_band"], np.float64)
    # counts-weighted mean relexp over r < 200" is ~1 by construction; this is the plain mean
    ax = (np.arange(rel.shape[-1]) - 0.5 * (rel.shape[-1] - 1)) * 4 * PIX_NATIVE
    NN, WW = np.meshgrid(ax, ax, indexing="ij")
    ap = np.hypot(NN, WW) < R_AP_ARCSEC
    return dict(img=(rel * band_ratio[:, None, None]).astype(np.float32), band_ratio=band_ratio,
                spec_ratio=spec_ratio, band_arf=band_arf, band_fold=band_fold, spec_arf=spec_arf,
                spec_fold=spec_fold, band_area_w=np.asarray(em["band_area_w"], float),
                band_area_check=band_area / np.asarray(em["band_area_w"], float),
                relexp_mean_r200=np.array([rel[b][ap].mean() for b in range(len(BANDS))]),
                livetime=float(em["livetime"]), obsids=obsids, e=e, w=w, a_c=a_c, mode=mode)


def corrections(epochs, resps, npix=256, mode="fold"):
    """Stacked corrections for casa_xfit: img (E, 6, npix, npix), spec (E, 31),
    band (E, 6), and the per-epoch dicts."""
    out = [epoch_correction(lab, r, mode) for lab, r in zip(epochs, resps)]
    img = np.stack([o["img"] for o in out])
    if img.shape[-1] != npix:
        raise ValueError(f"expmap grid {img.shape[-1]} != image grid {npix}")
    return dict(img=img, spec=np.stack([o["spec_ratio"] for o in out]),
                band=np.stack([o["band_ratio"] for o in out]), per_epoch=out, mode=mode)
# =============================================================================
# ============ ↑ Correction factors ↑ =========================================
# =============================================================================


# =============================================================================
# ============ ↓ Validation ↓ =================================================
# =============================================================================
def _sharpen(e, w, p=1.5):
    """w^p renormalised per 0.2 keV bin: the same broad shape, more line
    contrast (the ARF-unfolded data spectrum is RMF-smeared; the model's
    photon spectrum has sharp lines)."""
    out = np.zeros_like(w)
    for b0, b1 in [(0.0, 0.7)] + SPEC_BINS + [(6.9, 20.0)]:
        m = (e >= b0) & (e < b1)
        if w[m].sum() > 0:
            out[m] = w[m] ** p * w[m].sum() / (w[m] ** p).sum()
    return out


def validate(epochs=None):
    """For every fit epoch: (1) the ARF band areas reproduce ``band_area_w``;
    (2) the corrected table response vs the full CIAO response (epoch ARF x
    RMF): fold test spectra -- the epoch's own, tilted by E^-+1, and a
    line-sharpened one -- through both, in the six bands and the 31 analysis
    bins, for both correction modes (the correction always built from the
    epoch's own spectrum)."""
    man = json.loads((RESP_DIR / "manifest.json").read_text())
    by = {}
    for m in man["obsids"]:
        by.setdefault(m["epoch"], []).append(m)
    obs = np.load("/export/data/lstorcks/casa_orlando150/work/observed_outlines.npz")
    years = dict(zip([str(x) for x in obs["epochs"]], np.asarray(obs["years"], float)))
    epochs = epochs or [str(x) for x in obs["epochs"]]
    report = {}
    for lab in epochs:
        detnam = [m["detnam"] for m in by[lab]]
        resp = epoch_responses(detnam, years.get(lab, float(lab) + 0.4))
        c = epoch_correction(lab, resp, "fold")
        e, w0, a_c = c["e"], c["w"], c["a_c"]
        rmf_c = _rmf(RESP_DIR / "epochs" / f"epoch_{lab}_r200.rmf")
        use = np.ones(len(SPEC_BINS), bool)
        if resp[0][0].startswith("chandra_acisi"):
            use = SPEC_EDGES[1:] > 1.5 + 1e-6          # out of the likelihood at 2022
        tests = {"own": w0, "tilt-1": w0 * (e / 2.0) ** -1, "tilt+1": w0 * (e / 2.0), "sharp": _sharpen(e, w0)}
        rows = {}
        for name, w in tests.items():
            cb, mb = folded_counts(e, w, resp, a_c, rmf_c, BANDS)
            cs, ms = folded_counts(e, w, resp, a_c, rmf_c, SPEC_BINS)
            cs = np.where(cs > 0, cs, np.nan)
            row = dict(band_uncorr=(mb / cb).tolist(), spec_uncorr=(ms / cs).tolist())
            for mode in MODES:
                rb = mb * c[f"band_{mode}"] / cb
                rs = ms * c[f"spec_{mode}"] / cs
                row[f"band_{mode}"] = rb.tolist()
                row[f"spec_{mode}"] = rs.tolist()
                row[f"band_{mode}_maxdev"] = float(np.max(np.abs(rb - 1)))
                row[f"spec_{mode}_maxdev"] = float(np.nanmax(np.abs(rs[use] - 1)))
                row[f"spec_{mode}_rmsdev"] = float(np.sqrt(np.nanmean((rs[use] - 1) ** 2)))
            rows[name] = row
        report[lab] = dict(resp=resp, band_arf=c["band_arf"].tolist(), band_fold=c["band_fold"].tolist(),
                           spec_arf=c["spec_arf"].tolist(), spec_fold=c["spec_fold"].tolist(),
                           band_area_check=c["band_area_check"].tolist(),
                           relexp_mean_r200=c["relexp_mean_r200"].tolist(), tests=rows)
        print(f"[resp] {lab} [{', '.join(f'{i[8:]} {x:.2f}' for i, x in resp)}]: band ratio fold "
              + " ".join(f"{x:.3f}" for x in c["band_fold"]) + " (arf " + " ".join(f"{x:.3f}" for x in c["band_arf"])
              + f") | ARF band-area check max|dev| {np.max(np.abs(c['band_area_check'] - 1)):.1e}"
              + " | mean relexp r<200 " + " ".join(f"{x:.3f}" for x in c["relexp_mean_r200"]), flush=True)
        for mode in MODES:
            print(f"        {mode:4s} corrected/CIAO max|dev|: bands " + ", ".join(
                f"{k} {v[f'band_{mode}_maxdev']:.3f}" for k, v in rows.items()) + " | spec bins " + ", ".join(
                f"{k} {v[f'spec_{mode}_maxdev']:.3f} (rms {v[f'spec_{mode}_rmsdev']:.3f})" for k, v in rows.items()),
                flush=True)
    return report
# =============================================================================
# ============ ↑ Validation ↑ =================================================
# =============================================================================


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        rep = validate(sys.argv[2:] or None)
        out = CACHE_DIR / "ciao_validation.json"
        out.write_text(json.dumps(rep, indent=1))
        print(f"[resp] wrote {out}")
    else:
        print(__doc__)
