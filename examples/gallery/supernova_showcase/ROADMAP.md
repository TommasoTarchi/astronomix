# Roadmap: from the calibrated Cas A model to field-level inference

*Written 2026-09-02 after the audit recorded in `CALIBRATION.md` Result 26.
`vision.md` states the goal; `janka_mails.md` states the strongest objections
to it. This document answers those objections with a concrete design, lists
every field and parameter the inference would carry, the physics and data each
stage needs, and the order in which to build it. `OVERVIEW.md` remains the map
of what exists; this is the map of what is missing.*

---

## 0. The vision, restated so it can be tested

> Sample initial fields (at ~150 yr) and the physical parameters governing the
> remnant phase, such that the subsequent evolution reproduces ~30 years of
> Chandra observations, with a proper Bayesian treatment of the ambiguities.

Three words in that sentence need definitions before any of it is well posed,
and Janka's mail is right that the project has not supplied them:

1. **"Initial state."** Not the star, not the explosion — the *remnant-phase
   hydrodynamic state* at a chosen epoch t₀: (ρ, v, p, composition, shock
   history) on a grid. From t₀ onward the physics is adiabatic hydrodynamics
   plus known microphysics (Coulomb equilibration, NEI, absorption), which is
   exactly the regime this pipeline has verified (t_cool = 2.8 Myr; Result 7).
   That is a defensible object. The explosion is *not* inferred; it enters
   only as a prior on what remnant-phase states are plausible.
2. **"Ambiguity."** The known degeneracies are: E–n_w (broken by n_post),
   δ–M_ej (unbroken; Result 19), T_e prescription vs emission weighting
   (Result 5.5), projection along the line of sight (broken only by Doppler
   data), and the chaotic sensitivity of small-scale structure to the seed
   (JVP vs FD 0.2–37 % over 200 yr). A Bayesian treatment must carry these
   explicitly, and the design below picks a t₀ that minimises the last one.
3. **"Reproduces the observations."** Through a forward model that includes
   what the telescope sees — composition, NEI, T_e, dust scattering, response,
   *per-epoch* response, non-thermal emission — so the likelihood compares
   like with like. Janka's second mail is a statement about this forward model,
   and §4 lists where it is still wrong.

**Design decision (recommended): two tiers with different t₀.**

| tier | t₀ | what is inferred | why this t₀ |
|---|---|---|---|
| **A. parametric** | 150 yr (Route-B mapping) | ~15 scalar physics parameters (§2) | the 1D→3D chain is calibrated and cheap; 200 yr of evolution lets the parameters act; small-scale chaos is *marginalised*, not fitted |
| **B. field-level** | **the first Chandra epoch (2000, age ≈ 319 yr)** | the 3D state itself, in a latent parameterisation | the state is nearly observable; 23 yr is 7 % of the age, so gradients are meaningful and the posterior is well defined; the *dynamics* Chandra sees (proper motions, flux decline) live exactly on this baseline |

Tier A's posterior predictive at 319 yr is the **prior** for tier B. That is
the honest answer to "what does the initial state mean": it is the remnant's
state at the first observation, constrained by 23 years of subsequent data,
with a physics-based prior from the calibrated explosion-into-wind model.
Nothing about the star is claimed.

---

## 1. What the audit changed (read before planning on any recorded number)

Full record: `CALIBRATION.md` Result 26. The items that affect the plan:

* **Age vs epoch.** Every model is scored at **350 yr** against data taken at
  **319–323 yr** (explosion 1681 ± 19, Fesen et al. 2006). At v_FS ≈ 5250 km/s
  that is 0.145 pc = 6 % of r_FS, most of the r_FS tolerance. The 1D fiducial
  at 323 yr gives r_FS 2.40 / r_RS 1.65 against 2.55 / 1.74 at 350. For a
  single-epoch fit this is a bias inside 1σ; for *multi-epoch dynamics* it is
  fatal, so **the age becomes an inferred parameter with the 1681 ± 19 prior**
  and every epoch is compared at its own model time. Measured (Result 27): the
  same configuration at 319 yr against the 2000 epoch has rate 0.99 instead of
  0.84, IME bands 1.33–1.51 instead of ~1.1, coherence 0.92 instead of 0.84,
  and a projected outline ten times rounder than the data — every score moves.
* **Response drift.** The 2004 comparison used the launch (cycle-0) ACIS-S
  response because 2004 is equidistant from the two available cycles. The
  contaminant had removed tens of percent of the effective area below 1 keV by
  2004, so the synthetic soft band is overstated by about that much — the same
  size as the 21 % soft deficit that Result 25 attributes to the dust grains.
  **Per-epoch ARF/RMF are mandatory for anything multi-epoch.**
* **N_H is a fixed constant (1.2 × 10²²)** while it varies 1–2 × 10²² across
  the remnant (Hwang & Laming 2012) and the soft band goes as exp(−τ). It must
  be a spatial input (their map) or an inferred field, not a constant, before
  any soft-band residual is attributed to the remnant or the dust.
* **The sub-grid two-phase split acts on every cell, wind included**, while
  the XRISM contrast it is calibrated to is an ejecta quantity and 79 % of the
  continuum emission measure is shocked wind. `f_mass = 0.2` is therefore
  mostly boosting the wind. An `ejecta`-only variant must be scored before χ,
  f_mass are carried into an inference as "physics parameters".
* **Synchrotron:** two unit errors (×5 in the band integral, ×2.5 in the
  cutoff). Corrected, the radio-anchored X-ray flux is 10–25× too *faint* at
  the observed filament width, not "within a factor of a few". The non-thermal
  normalisation is a **fitted** efficiency until a B-field model exists.
* **The synthetic chip gap changes the morphology scores.** Re-observing the
  A1 = 0.5 photons with the remnant on one chip moves the 2.5″ / 4.4″ amplitude
  ratios from 1.15 / 0.76 to 1.74 / 1.04 and the coherence from 0.81 to 0.85.
  Every recorded morphology number was measured with the gap; absolute values
  must be re-derived, rankings re-checked.
* **The A1 ladder survives**, with the estimator fixed (sub-cell crossing,
  anisotropic reference): 0.123 / 0.173 / 0.230 / 0.300 / 0.460 pc for
  A1 = 0 / 0.3 / 0.4 / 0.5 / 0.7, and A1 = 0.5 matches the shell on spread,
  PA standard deviation *and* m = 1 amplitude. But the observed "0.2–0.4 pc"
  is **unsourced**; measure r_FS(PA) on the 2004 image with the same estimator
  before using it as a target. (Done, Result 27: on the image detector the data
  give std 0.21–0.24 pc and m = 1 ≈ 0.2 pc at PA ≈ 200°; A1 = 0.5 gives half
  that, so the dipole needs A1 ≈ 0.7–0.8 or a shell component, at the observed
  PA.)
* **1D unshocked mass** was biased low by ~0.08 M☉ (whole interior wind
  subtracted). Corrected, the δ = 1 fiducial gives 0.47 M☉ (1D) and the 3D
  tracer 0.42–0.46 — consistent with each other, 1σ above the 0.35 ± 0.10
  target. The fitted δ ≈ 1.38 of Result 19 must be re-fit.

---

## 2. Fields and parameters the inference carries

### 2.1 Fields (the state at t₀)

| field | carried today | needed for | note |
|---|---|---|---|
| ρ, v (3), p | yes | dynamics, EM, T | float32 + dual energy in 3D |
| C_ej, C_Fe, C_Si, C_O, C_He | yes | μ, μ_e, abundances, NEI | H is the remainder; S/Ar/Ca/Ne/Mg are split from Si/O by `TRACER_SPLIT` — the largest composition assumption left |
| entropy_initial, shocked_fraction, time_since_shock, density_time | yes (solver-managed) | n_e t, T_e relaxation | exact for adiabatic parcels |
| T_e | derived (`--te-model`) | spectrum | prescription bracketed (§5.5); becomes a parameter, not a field |
| **sub-grid (χ, f_mass) per cell** | global constants | spectrum normalisation and shape | should be a *field* (ejecta-only first) — the clumping the grid cannot resolve varies with composition and shock age |
| **B** | none (MHD optional, null on dynamics) | synchrotron | a compressed-ambient B is 50× too weak; needs an amplification model or a fitted per-cell efficiency |
| **N_H (sky map)** | constant | absorption, halo | input from Hwang & Laming 2012 or inferred |

### 2.2 Parameters (tier A) and their present status

| parameter | present value | status | breaks its degeneracy with |
|---|---|---|---|
| E | 2.09 × 10⁵¹ | fitted (1D) | n_w via n_post |
| M_ej | 3.0 | fitted (1D), **unresolved with δ** | needs shocked *element* masses (composition in the loop) |
| n_w | 0.928 | fitted (1D) | — |
| δ (inner slope) | 1.0 | production; fit biased (Result 26) | re-fit |
| envelope slope, core fraction, r₀ | 9, 0.5, 0.05 pc | fixed | — |
| A1, A2, axis (θ, φ) | 0.5, 0, (30°, 50°) | A1 set by PA spread; axis **arbitrary** | r_FS(PA) *shape* and RS velocity vs PA fix the axis |
| CSM shell | off | removed (Results 22–25) | keep as a discrete model alternative; the Green Monster is a shell-like object |
| pistons (Table 4) | on | fixed at Orlando's values | Fe-K morphology / Doppler |
| clump σ, k-band, seed | 5, N/6 | fixed; seed chaotic | marginalise, do not fit |
| plume field | off | measured null | — |
| age (explosion date) | 350 yr / **not a parameter** | **must become one**, prior 1681 ± 19 | proper motions |
| distance | 3.4 kpc fixed | should be a parameter, prior 3.4 (+0.3/−0.1) | angular sizes × velocities |
| T_e model, kT_e,shock / β | ghavamian 0.3 | bracketed; fiducial at optimum | — |
| χ, f_mass, net_mode | 4, 0.2, unchanged | χ from XRISM; f_mass fitted to the rate | ejecta-only test pending |
| TRACER_SPLIT | xrism_bulk / hwang_laming | choice | per-pixel line ratios |
| N_H | 1.2 × 10²² | fixed | map |
| dust a_max, geometry | MRN, uniform | fixed; robust | — |
| synchrotron η, width, efficiency | 1, 2″, — | **uncalibrated** | radio + hard-band + rim profile |
| per-epoch response | cy0 for 2004 | **wrong by 5 yr** | real ARF/RMF |
| aimpoint / chip layout | SOXS default (gap at −98″) | artefact | one-chip pointing |

Fifteen or so continuous parameters plus a few discrete choices. That is a
size at which forward-mode JVPs (validated in `casa_diff.py`), Laplace
approximations and simulation-based inference are all affordable at 128³–256³.

### 2.3 Latent parameterisation for tier B (the field)

A 256³ × 15 state is not a sensible inference space; the seed dependence is
chaotic and the data (a 1024² image per epoch, a few spectra) cannot fix 10⁸
numbers. Candidate latent spaces, all generative from what already exists:

1. **Route-B ensemble + smooth corrections**: tier-A parameters × a small set
   of large-scale modes (low-order spherical harmonics of the ejecta density and
   velocity at t₀, the same for the ambient). Order 10²–10³ numbers.
2. **The clump field's Fourier amplitudes below a cutoff k** (the seed made
   explicit), with the phases above k marginalised. Order 10³–10⁴.
3. **A learned prior** (normalising flow / diffusion trained on the ensemble),
   which is what "sample initial fields" literally asks for — later.

Recommendation: start with (1); it is what the data can constrain and it makes
"multiple shooting" meaningful (segment continuity is enforced on the same
smooth modes).

---

## 3. Data: what exists, what is missing, what each constrains

| data | status | constrains |
|---|---|---|
| Chandra ACIS images, 19 epochs 2000–2023 (`epoch_images/`) | exist, common 0.492″ grid | morphology; **proper motions** (FS ≈ 0.3″/yr → ~7″ over the baseline, RS motion, knot motions); flux decline |
| real spectra per epoch (r < 200″) | 2004 only cached | band ratios per epoch, needs per-obsid response |
| per-obsid ARF/RMF | **missing** (CIAO `specextract` on the evt2 already on disk) | the soft band at every epoch |
| N_H map | missing: Hwang & Laming 2012 publish their ~6000-region N_H only as a figure (checked 2026-09-03; VizieR has no table) — obtain by asking the authors, or fit N_H per region from the evt2 with the model's own spectrum as template | soft band per region |
| XRISM/Resolve maps (Vink et al. 2026) | numbers in `casa_xrism.py`; no data product | kT_e, n_e t, σ_v, Doppler per 30″ pixel — the **line-of-sight** constraint |
| Chandra Doppler maps (DeLaney et al. 2010), optical/IR 3D kinematics (Milisavljevic & Fesen 2013; JWST) | not used | 3D structure; breaks projection ambiguity |
| radio (VLA) flux and morphology | one number (2720 Jy) | synchrotron rim, B |
| published proper motions (Patnaude & Fesen 2009 FS; Sato et al. 2018 / Vink et al. 2022 RS; Fesen et al. 2025 west RS stationary) | referenced, not used as targets | dynamics; the A1 axis; shell vs dipole |
| Orlando et al. 3D remnant states | **not requested yet** (draft addressed to the explosion model) | prior samples for tier B; structural realism |

The cheapest high-value additions are the first three rows: they use data
already on disk.

---

## 4. Physics and software the vision needs, and what each costs

1. **Per-epoch instrument** (days). Custom SOXS instruments from the real
   obsid ARF/RMF (`soxs.add_instrument_to_registry`), one-chip aimpoint, and the
   epoch's exposure. Removes two systematics the audit found.
2. **Multi-epoch scoring** (days). `casa_observe.py --compare` for every epoch
   at the *model time of that epoch*; a proper-motion measurement (the
   phase-correlation already in the timelapse pipeline, applied to model pairs
   and data pairs identically); FS/RS radius vs PA vs epoch.
3. **Differentiable emission model** (weeks). Tabulate APEC/NEI band
   emissivities on (kT_e, n_e t, per-element abundance) grids; `jnp.interp`
   them; project along y with the dust-halo kernel and the PSF as fixed
   convolutions; Poisson likelihood on the 1024² image and the band spectra.
   Validate against pyXSIM on the fiducial (rate to ~10 %, bands to ~5 %). This
   is the piece without which nothing is "field-level".
4. **Adjoint through 3D hydro over 23 yr** (weeks). Forward mode is enough
   for tier A. Tier B needs reverse mode with checkpointing through
   O(10²–10³) steps at 128³–256³ — `diffhydro` (arXiv:2512.13403) did exactly
   this scale. Memory rule: 566 B/cell/device forward; checkpoint every step
   at 256³ is ~1 GB/state.
5. **Non-thermal component done properly** (weeks, and still a fit). Tabulated
   curved spectrum instead of `PowerLawSourceModel` over the full band; a B
   field from compression × an amplification factor that is *fitted* to the
   radio and labelled; per-cell cutoff from the local shock speed (already in
   `_synchrotron.py`).
6. **Sub-grid clumping as a field, ejecta-only** (days to test, weeks to
   calibrate against XRISM per pixel).
7. **Composition beyond four tracers** (days per scalar, ~1 %/step each): S,
   Ar, Ca, Ne, Mg as their own scalars remove the largest remaining
   assumption in the spectral model.
8. **Bayesian machinery** (weeks). Tier A: Laplace/Fisher from JVPs, then SBI
   or ensemble MCMC over the 15 parameters using 128³ runs (minutes each).
   Tier B: MAP in the latent space by gradient descent, Laplace around it,
   multiple shooting with continuity penalties between 2000 / 2007 / 2012 /
   2019 / 2023 segments.
9. **Physics not worth building here**: an in-house explosion (degenerate
   EOS wall, §7 of `OVERVIEW.md`); thermal conduction (15 days/run); MHD for
   the dynamics (measured null).

---

## 5. Staged plan with falsifiable checkpoints

**Stage 0 — close the audit. DONE 2026-09-02/03 except N_H (Result 27):**
recalibrated at 319 yr (E 2.43e51, δ 0.82, M_ej held at 3.0), A1 = 0.75 matches
the observed outline without the halo, ejecta-only split with f_mass 0.34 and
`xrism_bulk` gives rms 0.065 dex at rate 1.03, the FS expansion and the 22-yr
brightness decline match the 2000→2022 data. The synchrotron rim is now the
gating item for halo-on image comparisons. Earlier status of this stage:
the observed outline is measured (std 0.21–0.24 pc, m = 1 0.17–0.22 pc at
PA ≈ 200°; the A1 = 0.5 model has half that on the same detector), the FS
expansion is measured (0.28–0.35″/yr), the response cycle is bracketed (true
2004 soft ratio ~0.70), δ is re-fit (1.30). Still open: the age-matched run,
the ejecta-only sub-grid, an N_H map.

Re-observe the A1 = 0.5 fiducial with a one-chip aimpoint and the 2000 epoch
(exact cycle-0 match) at the 2000 model age; re-derive the spectral and
morphology scores; run the ejecta-only sub-grid variant; re-fit δ with the
corrected unshocked mass; measure r_FS(PA) on the real image to source the
PA-spread target. *Check:* the Result 25 verdict either survives at a
matched epoch/response or it does not — record which.

**Stage 1 — multi-epoch data products (1–2 weeks).**
Per-obsid responses; per-epoch real spectra keyed on their file set (now
enforced); proper-motion maps between epoch pairs; FS/RS radius vs PA per epoch
from the data. *Check:* the fiducial's FS expansion rate over 2000→2023 against
the measured one, and the RS motion sign per PA (west stationary). A
spherically symmetric-in-wind dipole model that gets the east–west RS
asymmetry wrong is the first thing this will show — and it decides between the
dipole and a shell-like Green Monster component *dynamically* rather than by
outline alone.

**Stage 2 — differentiable observables (3–4 weeks).**
Band-emissivity tables; differentiable projection; validation against pyXSIM.
*Check:* fiducial count rate and six bands within 10 % / 5 % of the pyXSIM
result; gradient of the rate with respect to f_mass matches a finite
difference.

**Stage 3 — tier A: parametric posterior at 150 yr (3–4 weeks).**
128³ ensemble over the parameters of §2.2 including age and distance; SBI or
ensemble MCMC with the 2000-epoch image + spectrum and the 2000→2019 proper
motions as data. *Check:* posterior widths on E, n_w, δ, A1, age; report the
δ–M_ej degeneracy honestly; posterior predictive of the 2023 epoch, which is
*not* used in the fit.

**Stage 4 — tier B: field-level MAP and uncertainty at t₀ = 2000 (2–3
months).** Latent space (1) of §2.3; reverse-mode through 23 yr at 128³ then
256³; multiple shooting across the five best epochs. *Check:* held-out epoch
prediction beats tier A's posterior predictive; the inferred large-scale modes
agree with the XRISM/Doppler line-of-sight structure they were not fitted to.

**Stage 5 — collaboration and data (in parallel, one email each).**
Rewrite `GARCHING_REQUEST.md` to ask Orlando for *remnant-stage* 3D states
(Janka pointed there explicitly); ask for the XRISM per-pixel products; obtain
the Hwang & Laming N_H map. Orlando's states are the natural external prior
samples for tier B and a direct test of whether our Route-B ensemble spans
them.

---

## 6. Decisions only the author can make

1. **Age convention.** Adopt "model time = epoch − 1681" everywhere now (a
   one-line change in each script plus re-running the 1D calibration at each
   epoch's age), or keep 350 yr for the recorded results and switch at Stage 1.
   Recommendation: switch now; the re-runs are cheap and every later number
   depends on it.
2. **Whether to adopt the A1 = 0.5 configuration as fiducial** before Stage 0's
   matched-epoch, one-chip re-observation. Recommendation: not yet; the audit
   found two soft-band systematics the size of its remaining residual.
3. **Whether to send the Orlando request**, and in what form (data request vs
   collaboration). Janka's reply invites the former.
4. **Whether the 1024³ rung is worth the queue time** given that the structure
   statistics now clear at 256³ with the dipole. Recommendation: no, not until
   tier A exists; resolution is not the binding constraint any more.
