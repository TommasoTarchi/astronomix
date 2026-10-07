# Orlando's 146-yr Cas A state in astronomix

*2026-09-22/23. S. Orlando's 3D MHD snapshot of the W15-IIb-sh model (Orlando
et al. 2022) at 145.5 yr, converted, evolved to the Chandra epochs, observed,
and fitted to 22 years of forward-shock outlines. Scripts: `casa_pluto.py`
(read/analyse/convert), `casa_orlando.py --from-state` (evolve),
`casa_observe.py --distance` (observe), `casa_pluto_diff.py` (fit). Data and
all outputs: `/export/data/lstorcks/casa_orlando150/` (`work/`, gitignored).*

## 1. The delivery

PLUTO 4.3, MHD + GLM, HLLC, RK3, tabulated cooling, 2048³ static grid reduced
to **512³ float32 `.flt`** files (one per variable, x fastest), box ±1.331 pc,
t = 0.1486 code = **145.5 yr** (unit time 979 yr). `grid.out` lists
(cell centre, centre + fine dx), not edges.

| quantity | value |
|---|---|
| mass in box | 7.97 M☉ (ejecta 3.27, of which 1.76 shocked; shocked CSM 2.57) |
| energy | kinetic 1.22e51 + thermal 0.30e51 = 1.51e51 erg |
| r_FS / r_CD / r_RS (400 rays) | 1.21 ± 0.02 / 0.89 ± 0.07 / 0.80 ± 0.07 pc |
| wind | n = 0.797 (r / 2.5 pc)⁻² cm⁻³ (μ = 1.289), smooth to 3e-4 |
| shell (Orlando+22 Eq. 1) | n_sh 19.8, r_sh 1.502, σ 0.022, H 0.708 pc, θ 51.9°, φ 65.2° (our convention); fit residual 4e-4; **1.95 M☉, not yet reached** (1.29 inside the PLUTO box) |
| B | toroidal wind field, 66 µG at 1.5 pc (∝ 1/r); post-shock 330 µG, β ≈ 350 → dynamically negligible |

**Tracers (inferred from the data; no header was delivered — confirm with
Orlando):** tr1 shock time, tr2 shell marker, tr4 ejecta; tr9–19 element
fractions, identified as C, Fe-group?, H?, He, Mg?, Ne, Si, O, S/Ar?, Ca, Fe
(`casa_pluto.TRACER_KEY`, with the evidence). Folded into the pipeline's four
scalars as Fe = tr19+tr10, Si = tr15+tr17+tr18, O = tr16+tr14+tr13+tr9,
He = tr12 (H the remainder). Ejecta Fe-group 0.15 M☉ (0.011 shocked at 146 yr).

## 2. Conversion (`casa_pluto.py convert`)

Separable conservative overlap remap onto a 7 pc astronomix box: mass,
momentum, internal energy and every scalar's mass conserved (1e-5 = float32);
sub-cell kinetic energy dropped, not heated (−0.46 % at 512³, −0.9 % at 256³).
Outside PLUTO's box the fitted wind + cell-averaged shell fill the grid
(exterior wind T capped at 1e5 K). Orlando's shock time becomes the library's
`shocked_fraction` / `time_since_shock` / `density_time`.

## 3. Forward evolution as delivered (146 → 360 yr)

Mass conserved to 2e-5 and energy to 1e-4 over 215 yr (256³); 512³ agrees to
<1 % in r_FS (512³ has a 0.07 % mass / 0.25 % energy start-up transient in the
first ~15 yr, flat afterwards).

| age | r_FS (tracer, sky-plane cones) | r_RS (casa_orlando) |
|---|---|---|
| 319 yr (2000 if 1681) | 2.12 pc (256³), 2.125 (512³) | 1.36 |
| 341 yr (2022) | 2.23 / 2.24 | 1.42 |

Synthetic Chandra (NEI, halo, ejecta-only sub-grid χ 4 / f_mass 0.34,
xrism_bulk, epoch's ACIS array + cycle) at 3.4 kpc:

| epoch | rate | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 2.8–4.2 | 4.2–6 | 6–7 |
|---|---|---|---|---|---|---|---|
| 2000 (319 yr) | 0.90 | 0.83 | 0.82 | 1.03 | 1.17 | 1.17 | 2.36 |
| 2022 (341 yr) | 0.91 | 1.08 | 0.75 | 0.89 | 1.02 | 0.99 | 1.82 |

The count-rate ratio is the same at both epochs, so **the model's 22-yr X-ray
fading matches the data**; Fe-K is ~2× too bright. But the remnant is
**19 % too small** at 3.4 kpc (outline ~127″ vs 159″ in 2000), while its
2000→2022 expansion (0.31″/yr) matches: it is under-decelerated,
m = V t / R ≈ 0.80 against the data's ≈ 0.6–0.7.

Orientation needs no rotation: pyXSIM along +y puts image right = +x (west),
up = +z, so image PA = simulation PA; the model's m = 1 lopsidedness points at
197° vs the data's 202°, and the fit returns ψ = 4 ± 4°.

Open imaging item: faint axis-aligned "cross" lines along x = 0 and y = 0 in
the unshocked ejecta and a warm spot at the centre, most likely inherited from
the explosion model's own grid (the raw B field shows the same axis), and the
probable origin of the vertical stripe in the synthetic image. Not filtered —
ask Orlando.

## 4. The differentiable fit (`casa_pluto_diff.py`)

Eight smooth, traced transformations of the 146-yr state: interior speed-up
ln s_v (v → s v, p → s² p), unshocked wind density ln f_w and slope ds_w,
shell density ln f_sh and radius shift d_rsh, explosion year, distance, and
rotation about the line of sight. Data: `casa_real_outline` r_FS per 10° cone
on the 15 full-coverage Chandra epochs 2000–2022 (513 radii). Likelihood per
cone = epoch-mean residual against a 5″ model-shape error + deviations from
it against the measured per-cone noise (so the 22-yr motion is not drowned by
static shape mismatch). Model r_FS = power mean of a T > 1e7 K sigmoid in
sky-plane cones, windowed to the outer shell; agrees with the tracer radius
to 1 %. 128³, ~50 s per forward pass; Levenberg–Marquardt on a
finite-difference Jacobian (see §5 for why not JVP).

| | fit A (all free) | fit B (explosion year held at 1681) |
|---|---|---|
| ln s_v (energy) | 0.000 ± 0.037 | 0.045 ± 0.033 |
| ln f_w (wind density) | 0.30 ± 0.16 | 0.28 ± 0.19 |
| ds_w (wind slope) | −0.05 ± 0.41 | −0.27 ± 0.36 |
| ln f_sh / d_rsh [pc] | −0.18 ± 0.23 / 0.25 ± 0.09 | −0.38 ± 0.20 / 0.48 ± 0.04 |
| explosion year | **1617 ± 8** | 1681 (fixed) |
| distance | **3.15 ± 0.06 kpc** | **2.87 ± 0.03 kpc** |
| ψ | 4.0 ± 4.5° | −3.0 ± 4.1° |
| χ² shape / motion / prior | 118 / 547 / 14 (34 / 510 data) | 129 / 563 / 11 |

Both fits reproduce the mean radius at every epoch and the mean expansion
(0.302 vs 0.294″/yr) and the broad outline shape (`figures/pluto146_outline_fit.png`).
**The result is a degeneracy line, not a point:** the 146-yr W15-IIb-sh state
fits 22 years of outlines only if it is ~65 yr older than the kinematic age
(1681 ± 19, Fesen+06) or ~15 % closer than 3.4 kpc — i.e. its blast wave is
under-decelerated at the known age and distance, and no wind density, wind
slope, or shell change within the priors removes that. Neither end is
excluded by the outline data alone (Δχ² ≈ 20 for ~500 radii with an
approximate noise model); an independent distance or age measurement
decides it, as would reverse-shock motions (not yet a target).

**The X-rays break the degeneracy.** Both best fits were re-run at 256³ with
full composition (`casa_pluto_diff.py --write-ic` → `casa_orlando.py
--from-state`) and observed at their own distance (`casa_observe.py
--distance`, same fiducial plasma model as §3):

| | epoch | rate | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 2.8–4.2 | 4.2–6 | 6–7 |
|---|---|---|---|---|---|---|---|---|
| fit A (1617, 3.15 kpc) | 2000 (383 yr) | **1.19** | 1.05 | 1.10 | 1.39 | 1.57 | 1.51 | 3.03 |
| fit A | 2022 (405 yr) | **1.23** | 1.39 | 1.01 | 1.22 | 1.42 | 1.36 | 2.45 |
| fit B (1681, 2.87 kpc) | 2000 (319 yr) | 2.48 | 2.52 | 2.38 | 2.66 | 2.25 | 1.72 | 2.39 |
| fit B | 2022 (341 yr) | 2.70 | 2.90 | 2.34 | 2.82 | 3.01 | 2.74 | 4.42 |

Fit B is 2.5–2.7× too bright in every band (denser wind: EM × 1.7; closer:
flux × 1.4); switching the fitted sub-grid clumping off (f_mass → 0, ~× 0.55)
still leaves it ~35 % high. Fit A is within the f_mass range (0.34 → ~0.25
brings the rate to 1) and keeps the 22-yr fading to 3 %. Its image has the
observed size, but it is still a smooth ring with a bright western rim and
eastern knots, not Chandra's filamentary web, and it shows the axis stripe.
So **the combined outline + X-ray evidence prefers the older, farther end:
Orlando's state needs ~65 yr more evolution than the 1681 kinematic date**
(or equivalently a slower early expansion than it has at 146 yr), unless
the data expansion rate is nearer 0.345″/yr (first caveat below). The rate
is not yet in the likelihood; a differentiable emission proxy (ROADMAP
Stage 2) would put it there.

### 4b. Expansion by registration, and the X-ray rate in the likelihood

**The data expansion rate, re-measured** (`casa_expansion.py`). Instead of an
edge detector per epoch, each epoch's cone profile is registered on the
2004 (143 ks) epoch: the radial shift maximising the correlation of the
profiles' log-derivatives, after re-centring every epoch on the CCO (removes
up to 1.3" of pointing offset, 2022 included). The epoch-mean shift is
monotonic and clean (−1.2" in 2000 → +5.5" in 2022):

| window | mean over cones | median | epoch-mean slope (ACIS-S only) |
|---|---|---|---|
| fixed 130–210" | 0.301"/yr | 0.333 | 0.301 (0.298) |
| ±12" around each cone's 2004 outline | 0.276 | 0.324 | 0.283 (0.278) |
| ±8" | 0.235 | 0.307 | 0.229 (0.223) |

Per-cone errors are ~0.01"/yr; narrow windows lock onto wrong features in a
few cones (−0.5"/yr), which is why their means fall. **The robust answer is
0.30"/yr (mean) / 0.31–0.33"/yr (median)** — the 0.345"/yr of the outline
detector was its skew, so the age tension of §4 stands. A real feature in
every variant: **the west (PA 290–320°) moves only 0.04–0.12"/yr**, at the
shell's dense side — either the shell interaction Orlando's model was built
for, or bright shell/ejecta structure masking the blast wave there.

**New likelihood**: outline shape (epoch-mean radius per cone, 5" model
error) + registration proper motion per cone (30 cones where the fixed and
±12" windows agree to 0.1"/yr; σ = error ⊕ 0.03"/yr) + the 0.5–7 keV rate in
2000 and 2022 through a hot-gas EM / D² proxy (calibrated on the six full
pyXSIM runs: 580 / 275 counts s⁻¹ per unit, scatter ±5 % / ±13 %; σ_ln = 0.15,
which also absorbs the sub-grid f_mass freedom) + priors.

| | **fit C** (all free) | fit C-1681 (year fixed) |
|---|---|---|
| explosion year | **1620 ± 7.5** | 1681 |
| distance | **3.05 ± 0.07 kpc** | 2.84 ± 0.04 kpc |
| ln s_v / ln f_w / ds_w | −0.04 ± 0.06 / 0.48 ± 0.20 / −0.56 ± 0.44 | −0.02 / 0.34 / −0.25 |
| shell ln f_sh / d_rsh | 0.10 ± 0.27 / 0.08 ± 0.09 pc (≈ Orlando's) | −0.86 ± 0.14 / 0.53 ± 0.03 (gutted) |
| ψ | 1.9 ± 4.6° | −11.9 ± 5.2° |
| rate proxy 2000 / 2022 vs Chandra | 0.92 / 0.93 | 1.02 / **1.33** |
| registration PM mean (data 0.312"/yr) | 0.308 | 0.315 |
| per-cone PM correlation with data | 0.36 | 0.22 |
| χ² shape / motion / rate / prior | 119 / 223 / 0.6 / 15.5 = **358** | 145 / 231 / 3.6 / 13.5 = 393 |

With the X-ray rate in the likelihood the young solution survives only by
weakening the shell to 42 % and moving it out to 2.03 pc — and then fades
too slowly (the 2000→2022 rate falls to 54 % against Chandra's 42 %). The old
solution keeps Orlando's shell essentially as published, fits the rate and its
fading at both epochs, the mean proper motion to 1 %, and the proper-motion
pattern best. Δχ² = 35; with χ²_ν ≈ 5.5 (the per-cone shape and proper-motion
pattern are only partly reproduced) that rescales to a ~2.5σ preference.
**Result: Orlando's W15-IIb-sh state, evolved in astronomix, fits 22 years of
Chandra outlines, proper motions and X-ray fading best for an explosion in
~1620 at ~3.05 kpc — earlier than the 1681 ± 19 kinematic age, which the model
can only reach at 2.84 kpc with a much weaker shell and the wrong fading.**
Read as a statement about the model rather than about Cas A: at 146 yr its
blast wave is expanding too fast for its age (under-decelerated), by the
equivalent of ~60 yr.

**Full forward-model check of fit C** (256³ with composition, pyXSIM/SOXS at
3.05 kpc, same plasma model as §3; `plC_n256_age{380,402}yr.npz`):

| epoch | rate | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 2.8–4.2 | 4.2–6 | 6–7 |
|---|---|---|---|---|---|---|---|
| 2000 (380 yr) | 1.14 | 1.09 | 1.03 | 1.27 | 1.36 | 1.23 | 2.26 |
| 2022 (402 yr) | 1.11 | 1.35 | 0.92 | 1.09 | 1.22 | 1.11 | 1.85 |

The fading matches to 3 % and the level is within the sub-grid f_mass range
(0.34 → ~0.28 gives 1.0). For reference, Orlando's state as delivered at
**512³** (3.4 kpc, 2000): rate 1.04, bands 1.00 / 0.94 / 1.15 / 1.25 / 1.18 /
2.23 (0.90 at 256³).

**Resolution systematic in the X-ray rate:** at fixed parameters the EM
proxy is 324 (128³) vs 388 counts/s (256³) for fit A, and the full rate rises
another ~15 % from 256³ to 512³ (0.90 → 1.04 as delivered). The absolute rate
is therefore not converged (+15–20 % per doubling, about the σ_ln = 0.15 of
the rate term); the fading ratio between two epochs at one resolution is, and
it is the fading that the 1681 solution gets wrong. Fe-K is 2× too bright at
every resolution and in every model.

Caveats that matter for the numbers:
* χ²_ν ≈ 5.5: the model reproduces the mean size, expansion and fading but
  not the detailed per-cone outline (rms 10") nor the west's slow rim;
* the rate proxy carries a ±5–13 % calibration scatter and the full forward
  model's f_mass freedom (σ_ln = 0.15 covers both); a proper differentiable
  emission model (ROADMAP Stage 2) would sharpen the rate term;
* the reverse shock mean radius is not a discriminator: both solutions sit
  within 1σ of Gotthelf et al. (2001)'s 95.8 ± 9.7" (tracer r_RS 101–108");
  per-PA reverse-shock motions would be, and need their own image analysis;
* 128³ for the fits (r_FS agrees with 256³/512³ to ~3 %).

## 5. Forward-mode derivatives through the 3D solver

Out of the box every JVP was NaN. Three library fixes (primal unchanged):
1. `_flux_blending.py`: the Zalesak ratio `min(1, Q/P)` → guarded division
   (JAX's min-JVP multiplies an overflowed float32 tangent by 0 → NaN in 13 %
   of the box), and `stop_gradient` on the positivity/cold-crush blend weight.
2. `SimulationConfig.weno_ad_frozen_weights`: freeze the WENO nonlinear
   weights (d α/d IS ~ 2/ε³ ~ 1e21 in cold gas), the characteristic
   eigensystem (L carries 1/c² terms), the WENO splitting speed and the LLF
   dissipation speed — the standard WENO linearisation.
3. The smooth r_FS estimator is windowed to the outer shell.

With these, JVPs are finite over the whole 200 yr and agree with FD for
parameters that act outside the mixing layer (explosion year 1.70 vs 1.63,
ψ 0.2045 vs 0.2045, shell density 46.9 vs 47.8). Parameters that act through
the interior (s_v, f_w) are still off by 3–10³×: after ~250 yr the pointwise
tangent grows exponentially (e-fold ~10–15 yr) in the Rayleigh–Taylor mixing
layer. That is chaos, not a bug, and freezing switches cannot remove it.
Pointwise JVPs are therefore reliable only over windows shorter than a few
e-folds — the multiple-shooting design of ROADMAP tier B — while the 200-yr
tier-A fit uses finite differences (which see the ensemble-scale response).

**The 22-yr window test** (fit A's own 2000 state at 256³ → 2022,
`plA_state2000_n256.npz`, `--check-grad`):

| parameter | JVP | FD |
|---|---|---|
| wind density ln f_w | −9.46 | −9.64 (2 %) |
| distance ln D | 1519 | 1504 |
| rotation ψ | 0.6297 | 0.6299 |
| shell (already crossed) | 1.0e-3 / −1.1e-2 | 1.5e-3 / −0.9e-2 |
| wind slope ds_w | 0.52 | 1.06 |
| interior speed-up ln s_v | 2.3e5 | 73 — still fails |

So on a window the tangent is healthy for everything acting ahead of the
blast wave, which is what tier B's ambient/large-scale modes are; perturbing
the already-turbulent interior directly (s_v at 383 yr) is still chaotic
within 22 yr and needs either shorter segments or an averaged (e.g.
ensemble or low-pass) tangent. (t_expl is meaningless in this window by
construction.)

### 5a. The last amplifier: cold dense knots (2026-09-23/24)

With frozen WENO weights/eigensystem the tangent still grew ×10⁵ in ~15 yr
from a 2000 state. `casa_pluto_jvp_probe.py` located it: always in **cold,
dense ejecta knots** (T 10⁴–10⁶ K) just inside the reverse shock, not in the
hot mixing layer. Bisected and excluded: every positivity switch
(redistribute, cold-crush blend, vacuum-rest, nan-safe), dual energy, and
float32 (float64 grows identically). So the amplifier is the frozen-weight
linear high-order scheme itself at the sharp edges of cold knots — the
stencils were chosen for the primal's smoothness, not for the tangent's
stability.

* **Rejected: a tangent low-pass.** A conservative smoothing of the tangent
  between 1-yr segments keeps the magnitude sane but gives wrong gradients
  (ln f_w flips sign against FD) — it mis-propagates shock displacements.
* **Adopted: `SimulationConfig.ad_tangent_llf_cold_factor`.** On faces whose
  colder side is below factor × the 10⁴ K floor, the DERIVATIVE of the flux is
  taken through the monotone LLF flux (straight-through: value = the primal
  flux exactly, derivative = the LLF-blended flux's). Factor 100 (10⁶ K) is not
  enough (the knots sit at ~10⁶ K); **factor 1000 (10⁷ K) makes the tangent
  bounded**: over 14 yr max |∂ρ/∂θ| 0.2 → 2.2 and the 99.9th percentile grows
  linearly, located behind the forward shock where a shock-displacement
  sensitivity belongs. JVP-vs-FD validation:

  *22-yr window* (fit A's 2000 state, 256³; FD at h = 0.02):

  | | ln s_v | ln f_w | ds_w | ln D | ψ |
  |---|---|---|---|---|---|
  | JVP before | 2.3e5 | −9.46 | 0.52 | 1519 | 0.6297 |
  | **JVP now** | **55.6** | **−10.0** | **1.14** | 1519 | 0.6297 |
  | FD | 73.0 | −9.64 | 1.06 | 1504 | 0.6298 |

  *Full 146 yr → 2022* (128³, at the prior, where the gradients are large):

  | | ln s_v | ln f_w | ln f_sh | d_rsh | t_expl | ln D | ds_w | dip x / y / z |
  |---|---|---|---|---|---|---|---|---|
  | JVP | −5017 | 811 | 473 | 616 | 26.7 | 10096 | 327 | 319 / −59 / −395 |
  | FD | −4281 | 635 | 349 | 554 | 24.1 | 10085 | 249 | 318 / −55 / −372 |

  All finite (before: NaN or 1e13), all the right sign, 0–36 % apart; the
  JVP runs 10–35 % high for parameters acting through the interior, as
  expected of a frozen-switch / cold-LLF linearisation against an FD secant
  through a chaotic flow. The forward pass is bitwise reproducible (checked),
  so near the optimum, where the gradients are O(1–10), FD differences are the
  chaotic secant response, not noise. Good enough for Gauss–Newton: fit G is
  the first fit driven entirely by forward-mode JVPs.

## 5b. Do we have the physics for correct observations? (audit, 2026-09-23)

Where the emission comes from decides which physics matters. In the 512³
state at 319 yr (and in fit C at 380 yr, same split):

| component | share of thermal EM | ⟨kT_e⟩_EM | ⟨log n_e t⟩_EM | mass |
|---|---|---|---|---|
| shocked ejecta | 0.16 (0.12) | 3.2 keV | 11.6 | 2.4 M☉ |
| shocked wind | 0.20 (0.28) | 3.1 keV | 11.0 | 4.1 M☉ |
| **dense shocked shell** | **0.64 (0.60)** | 2.5 keV | 11.6 | 2.1 M☉ |

**75 % of the Fe emission comes from circumstellar gas, not ejecta.** Orlando's
ejecta carry only 0.043 M☉ of shocked Fe at 319 yr (Hwang & Laming: 0.14) and
0.24 M☉ of shocked O (2.0), because the W15-IIb ejecta are He-dominated (1.5
M☉ He; 0.56 M☉ O in total). So the circumstellar composition is not a detail —
and it was wrong:

1. **CSM metallicity (ERROR, fixed for these states).** `_common.CSM_COMPOSITION`
   = He 0.28, O 0.01, Si 0.005, Fe 0.005 by mass, labelled "cosmic" — the total
   Z = 0.02 is cosmic, the split is not: Si-group 4× and Fe 2.7× solar.
   `casa_pluto.py recompose` swaps the CSM part to Anders & Grevesse (the table
   the X-ray model normalises to) exactly, since the scalars mix linearly.
   Every spectral score of the calibrated track in `CALIBRATION.md` used the
   old values (their wind carries ~79 % of the continuum EM) and should be
   re-scored. Result of the fix: see the table below.
2. **Synchrotron: absent from every observation above.** Cas A's forward-shock
   rim is non-thermal and ~half of its 4–6 keV continuum is synchrotron
   (Helder & Vink 2008). Our thermal-only 4.2–6 keV band is already 1.1–1.2,
   i.e. the thermal continuum is too hot/strong — consistent with ⟨kT_e⟩ =
   2.5–3.2 keV against ~2 keV observed. Orlando's B field (66 µG toroidal wind
   field → ~330 µG shocked) exists at 146 yr but is not evolved (hydro run);
   `casa_observe --synchrotron` uses a compression-based B with a FITTED
   efficiency. Test run below.
3. **Electron–ion relaxation (BUG, fixed in `_plasma.py`).** The Coulomb
   relaxation took 16 equal substeps with τ_eq frozen at each substep's start.
   τ_eq ∝ T_e^{3/2} is shortest right behind the shock, so the first substep
   jumped the electrons to equipartition: the `minimal` start (no
   collisionless heating) came out at T_e/T = 0.96, hotter than the 0.3 keV
   Ghavamian start (0.36), which a one-variable relaxation cannot do. Now
   log-spaced substeps with a midpoint predictor–corrector (48 steps, within
   0.3 % of a 2048-step reference). On fit C, EM-weighted kT_e 2.58 → **2.30
   keV** (−11 %) with Ghavamian; `minimal` 6.4 → 2.29 keV — the two now agree,
   i.e. at these n_e t the initial electron heating is forgotten and T_e is
   set by Coulomb collisions in the dense gas. A monotonicity + convergence
   assertion was added to `_plasma._assert_physics`. Every T_e-dependent
   number produced before (all spectra in `CALIBRATION.md` included) was
   biased hot by ~10 %; the remaining excess over ~2 keV is physical to this
   model (the dense shell's high n_e t), not a T_e prescription choice.
4. **NEI assumptions.** One shock, current T_e used for the whole history
   (over-ionises, stated in `_nei.py`). Weakest for the shell: shocked by the
   blast, then by reflected shocks. n_e t is ~5× the observed EM-weighted value.
5. **Resolution.** The X-ray rate rises 15–20 % per doubling (128³ → 256³ →
   512³) — the compressed shell and clumps are not converged; the sub-grid
   clumping model (f_mass) was calibrated to the rate of a different model.
6. **Absorption.** N_H constant 1.2e22 (true 1–2e22 across the face) — the
   <0.7 keV deficit in every spectrum.
7. **Instrument.** Epoch-matched ACIS array + cycle (cy0 for 2000, ACIS-I cy22
   for 2022); per-obsid ARF/RMF still missing for the intermediate epochs, which
   is why only 2000/2022 rates are fit targets.
8. **Negligible, checked:** radiative cooling (t_cool ~ Myr), B dynamics
   (β ≈ 350 behind the shock), light-travel time across the remnant (±8 yr,
   zero on the sky-plane rim), thermal line broadening at CCD resolution.
9. **Not modelled:** cosmic-ray back-reaction on the shock (γ_eff < 5/3 raises
   compression and lowers T_i), dust destruction / IR cooling, the tracer key
   (inferred), and the axis artefact from the delivered state.

**Effect of the two fixes on fit C** (256³, 3.05 kpc, same plasma model):

| run | epoch | rate | 0.5–1.5 | Si | S | Ar/Ca | 4.2–6 | Fe-K |
|---|---|---|---|---|---|---|---|---|
| metal-rich CSM (old) | 2000 | 1.14 | 1.09 | 1.03 | 1.27 | 1.36 | 1.23 | 2.26 |
| + synchrotron (norm 1) | 2000 | 1.15 | 1.10 | 1.04 | 1.28 | 1.38 | 1.25 | 2.27 |
| **solar CSM** | 2000 | 0.86 | 0.91 | **0.73** | 0.85 | 1.00 | 1.04 | **1.48** |
| **solar CSM** | 2022 | 0.81 | 1.09 | **0.65** | 0.71 | 0.89 | 0.92 | 1.16 |
| **solar CSM + T_e fix** | 2000 | 0.86 | 0.94 | 0.73 | 0.83 | 0.93 | 0.90 | **1.18** |
| **solar CSM + T_e fix** | 2022 | 0.79 | 1.11 | 0.65 | 0.69 | 0.83 | 0.81 | **0.95** |

* The solar CSM removes most of the Fe-K excess (2.26 → 1.48 / 1.16) and the
  Ar/Ca excess — they were circumstellar metals that are not there.
* It EXPOSES the ejecta-line deficit: Si 0.65–0.73, S 0.71–0.85. Part of the
  earlier "match" was wind silicon at 4× solar. Orlando's state has 0.012 M☉
  of shocked Si at 319 yr (Hwang & Laming 0.08) and the ejecta give only
  12–16 % of the thermal EM; the real remnant's lines say it is
  ejecta-dominated. This is a property of the delivered explosion model
  (He-rich W15-IIb ejecta, reverse shock only 0.8 pc in at 146 yr), not of
  our observation chain.
* Synchrotron at the radio-anchored efficiency adds 1–2 %; matching the
  observed 4.2–6 keV non-thermal flux (~8.7e-11 erg/cm²/s, roughly half the
  band) needs an efficiency ×18. Adding that on top of a thermal continuum
  already at 0.92–1.04 means **the thermal electrons are too hot**, which
  is the same statement as ⟨kT_e⟩ 2.5–3.2 keV vs ~2 keV.
* With the T_e fix the hard band drops (continuum 1.04 → 0.90 / 0.81, Fe-K
  1.48 → 1.18 / 0.95): Fe-K, the band that was 2.3× off, is now within 20 %.
  The continuum now leaves room for a synchrotron share, though if the
  non-thermal 4–6 keV fraction is really ~half (Helder & Vink 2008) the
  thermal part is still ~1.6–1.8× too strong.
* The 2022/2000 fading ratio moves from 0.97 to 0.94 (0.92 with the T_e fix)
  of the observed —
  robust to the composition fix, as expected, so it is the rate information
  the fit should use (`--rate-mode fading`, fits D below).

**Fits D — the fading ratio as the rate constraint** (`--rate-mode fading`:
solar-CSM calibration, absolute rate at σ_ln = 0.3, 2022/2000 ratio at 0.05):

| | fit D (all free) | fit D-1681 |
|---|---|---|
| explosion / distance | **1620 ± 6 / 3.04 ± 0.06 kpc** | 1681 / 2.82 ± 0.04 kpc |
| wind ×, slope | 1.63, −0.50 ± 0.42 | 1.40, −0.28 |
| shell ×, Δr | 1.20, +0.08 pc (≈ Orlando) | 0.42, +0.51 pc |
| rate 2022/2000 (Chandra 0.416) | **0.404** | 0.509 (4σ off) |
| χ² shape / motion / rate / prior | 119 / 221 / 2.0 / 15.5 = **357** | 143 / 234 / 16.9 / 14.0 = 408 |

Δχ² = 51 (≈ 3σ after rescaling by χ²_ν ≈ 5.3): with the most robust X-ray
observable in the likelihood the preference for an OLDER remnant than the
1681 kinematic date strengthens. The fading is the discriminator — a younger
remnant in a lighter, closer configuration fades too slowly.

## 5c. Orientation: the synthetic images were reflected (fixed)

The side-by-side images looked differently oriented, and they were. A toy
test settles the geometry exactly: through `project_photons(..., "y", ...)`
a blob at simulation **+x lands 21" NORTH, a blob at +z lands 21" WEST** —
pyXSIM uses yt's axis-aligned image plane (z, x) for a string normal. With
west = +z and north = +x the image is a proper view only for an observer at
+y, while pyXSIM's Doppler factor E (1 − v_y/c) is for an observer at −y.
**Every synthetic image before 2026-09-24 was the view from −y reflected about
the NW–SE diagonal (PA → 90° − PA), with correct Doppler signs.**
`casa_observe.fix_projection_parity` (default on; `--legacy-parity` for the old
behaviour) swaps x ↔ z and v_x ↔ v_z before projecting, which gives west = +x,
north = +z — Orlando's convention and `casa_pluto_diff`'s. (A first attempt,
reflecting x, flipped the image north–south; it was caught by comparing the
new image with the old: correlation 0.998 with the old flipped up–down.) The
outline and Doppler terms of the fits were never affected — they work in
simulation coordinates with west = +x.

**Measured Doppler pattern** (2004 events, Si/S/Fe He-α centroids per PA,
km/s, + = receding, errors 7–36):

| | W | NW | N | NE | E | SE | S | SW |
|---|---|---|---|---|---|---|---|---|
| Si | +235 | +397 | +199 | +261 | −750 | −1050 | +250 | +458 |
| Fe-K | +179 | +647 | +984 | +349 | −1667 | −1087 | +158 | +437 |

— E/SE approaching, N/NW receding, as in the literature.

**Orlando's model in the correct orientation** (view from −y, roll 0),
correlation with the data over 24 PA sectors: broadband brightness +0.28;
Si Doppler −0.39 / −0.46 (256³ fit C / 512³ as delivered); Fe −0.24 / −0.42.
The brightness asymmetry is roughly right way round, but the **Si-rich ejecta
move the wrong way on the E–W axis**: the model's SE Si ejecta recede (+500)
where Cas A's approach (−1050). Weighting matters: the velocity of ALL hot
ejecta (dominated by the He envelope) correlates +0.31 with the Si data, the
Si-weighted velocity −0.16. No roll about the line of sight makes it good
(best mean correlation +0.23). This is a property of the delivered explosion
model — its Si-rich plumes are not where Cas A's are — and it is the
strongest single discrepancy found.

### Fit F — Si-weighted Doppler with a velocity dipole (a dead end)

Si-weighted, the starting Doppler correlation is −0.16 (χ² 1228). A global
ejecta velocity dipole reaches χ² 587 only with dip_y = −0.96 (3.2σ; a 96 %
line-of-sight speed asymmetry) at the cost of the outline (shape χ² 156): a
dipole is the wrong knob — the discrepancy is WHERE the Si-rich ejecta are.

### Fit G — the first fit driven by forward-mode JVPs

Same likelihood as fit F2, Jacobian from `jax.jvp` through the full 146 yr →
2022 evolution at 128³ with `--ad-llf-cold 1000` (no finite differences):
χ² 1585 → 1060 in 5 Levenberg–Marquardt steps (F2 with FD: → 1006), landing in
the same basin — explosion 1627 ± 6 (F2 1635), dip_y −0.76 (−0.96), ψ 15° (19°).
Cost: ~1100 s per step after a ~2000 s first compile, ≈ 2× the FD Jacobian at
12 parameters; the advantage is exactness (and memory-flat scaling with the
number of steps), not speed at this parameter count.

### The explosion's orientation is a free parameter

Nothing Orlando delivered constrains the explosion model's axes relative to
the sky or the CSM. Rigidly rotating fit C's 380-yr state over 3000 random
orientations (Si-weighted Doppler + broadband brightness, 24 sectors):

| orientation | Si Doppler corr | brightness corr |
|---|---|---|
| Orlando's | −0.16 | +0.24 |
| best Doppler | +0.92 | −0.40 |
| **best combined** (141° from Orlando's) | **+0.88** | **+0.47** |

61 % of random orientations beat Orlando's on the Doppler pattern. This is an
upper bound (the CSM rotated too). As a differentiable parameter —
`rot_x/y/z`, a rotation vector applied to the interior (r < 1.25 pc) at 146 yr
by trilinear resampling, velocities rotated, wind and shell fixed
(`rotate_interior`) — the best-combined orientation gives Doppler +0.52 at the
start of fit H (identity reproduces the unrotated model exactly). Caveat: this
leans on the Si tracer identification (tr15), which is inferred.

### Fit H — explosion orientation free (best fit so far)

15 parameters (FD Jacobian, 128³), started from the best-combined orientation:

| | F2 (dipole only) | **H (orientation + dipole)** |
|---|---|---|
| χ² total | 1006 | **537** |
| shape / motion / rate / Doppler / prior | 156 / 222 / 17 / 587 / 24 | **108** / 230 / 8.3 / **164** / 27 |
| Doppler corr (Si-weighted), model rms | +0.27, 275 km/s | **+0.83, 530 km/s** (data 545) |
| explosion orientation (rot. vector) | Orlando's | **(−110 ± 2, −23 ± 3, −88 ± 4)°** |
| explosion / distance | 1635 / 2.94 | 1636 ± 6 / 2.90 ± 0.05 kpc |
| wind ×, shell × | 1.32, 0.54 | 2.14, 0.49 |
| dipole d | (0.17, −0.96, 0.16) | (−0.68, −0.57, 0.45), \|d\| ≈ 1.0 |

Rotating the explosion 140° from Orlando's orientation reproduces the
measured Si Doppler pattern (+0.83) AND gives the best outline of any fit. The
orientation is sharply determined. But the solution leans on an unphysically
large velocity dipole (|d| ≈ 1: one side twice as fast), a closer distance and
a 2× denser wind, with the X-ray rate proxy at 0.54 — so the orientation is
real, the remaining freedom is being absorbed by the crude dipole. Fit H0
(dipole off) separates the two.

**Fit H, full forward model** (256³ with composition, solar CSM, corrected
T_e and orientation, pyXSIM at 2.90 kpc; `figures/plH256_{2000,2022}_*`,
diagnostics `figures/pluto146_fit_diagnostics.png`):

| epoch | rate | 0.5–1.5 | Si | S | Ar/Ca | 4.2–6 | Fe-K |
|---|---|---|---|---|---|---|---|
| 2000 (364 yr) | 0.55 | 0.58 | 0.48 | 0.54 | 0.61 | 0.62 | 0.71 |
| 2022 (386 yr) | 0.56 | 0.74 | 0.47 | 0.49 | 0.60 | 0.61 | 0.65 |

The image is the most Cas A-like so far — the brightest, filamentary ejecta in
the E/SE as in Chandra — and the fading is right (0.55 → 0.56), but the
remnant is ~1.8× too faint in every band (the loose rate term let the fit
trade brightness for kinematics) and Si remains the weakest line.

### Fit H0 — orientation free, no dipole

χ² 1720 → 768: Doppler 220 / 24, correlation **+0.80** (H: +0.83), orientation
(−112, −20, …)° — the same as H. So **the rotation alone explains the Doppler
pattern**, robustly. But shape 222 and motion 286 (H: 108 / 230): the dipole in
H was repairing the OUTLINE and proper motions that the rotated explosion
spoils when it meets the CSM, which is still in Orlando's orientation. The
physical candidate for that job is an asymmetric circumstellar medium (a wind
density dipole, the A1 ≈ 0.75 ingredient of our own calibrated track), not a
|d| ≈ 1 ejecta velocity dipole — the next parameter to add.

### Fit I — orientation free + wind density dipole, no ejecta dipole (best)

FD-Jacobian LM, 6 steps, χ² 780 → **463.5** (H: 537): shape 104 / 34, motion
239 / 30, X-ray rate 3.2 / 3, Doppler 94 / 24 with correlation **+0.88** (the
best yet), and the outline's m = 1 lopsidedness 11.2″ vs 11.2″ observed.
Explosion 1632 ± 7, D = 3.02 ± 0.06 kpc, wind ×2.7, orientation (−107, −38,
−81)° (H: −110, −23, −88). The wind dipole has |a| = 0.83 toward sim
(+0.62, +0.48, −0.27), i.e. mostly **west** on the sky — the denser CSM where
the rim is slowest (0.04–0.12″/yr at PA 290–320) — and consistent with the
A1 ≈ 0.75 of the calibrated track. So the ejecta dipole of fit H was doing
the CSM's job: with the physical ingredient available, it is not needed.
Remaining: the per-cone proper motions still do not correlate (0.18), the
reverse shock is at 33″ (Gotthelf+01: ~96″), and the rate proxy is at 0.71.

## 6. The differentiable observation model (`casa_jaxobs.py`)

The pyXSIM/SOXS chain is exact but slow (~1 h per epoch at 256³) and not
differentiable. Everything it does to a cell is linear in the emission measure
and abundances once (kT_e, n_e t) are fixed, so it factorises into
**response-folded NEI tables** (`casa_jaxobs_tables.py`, xrayobs env): per
element, C_el(kT, n_e t, channel) = RMF · (ARF × TBabs(N_H) × Σ_q f_q v_el,q),
on the `_nei` grid (48 kT × 57 n_e t), 5 N_H columns, 128 channels, for ACIS-S
cycle 0 and ACIS-I cycle 22, plus the first-order Doppler response
D = dC/dβ. (Bug caught on the way: soxs' per-element NEI generators add the
H+He base spectrum to every call, so the ion basis vectors must have it
subtracted — the first tables were 3× too bright.) The dust halo is a set of
per-band sky kernels and an aperture-keep table (energy × source radius) from
the same Monte Carlo `casa_observe` uses.

The model (all jnp, float32, host-combined float64 constants):

* the plasma chain of `_plasma` — composition, moments, Ghavamian T_e +
  log-spaced predictor-corrector Coulomb relaxation — and the ejecta-only
  sub-grid split of `_subgrid`, per cell;
* `aperture_spectrum`: DEM over (kT, n_e t) per component, aperture mode and
  Doppler moment, contracted with the tables; the halo keep enters as a rank-3
  SVD in source radius (error 0.007 = its MC noise);
* `band_columns` → `project_columns`: LOS-summed band emission, then roll,
  sky offset (both traced), 2×2-supersampled splat, halo and PSF by linear FFT
  convolution on a guard-padded grid; rows = north, columns = west as
  `read_events`;
* `doppler_sectors`: the data's statistic (Si-band photon-energy centroid per
  sector, continuum dilution included);
* everything accumulates in a **checkpointed `lax.scan` over line-of-sight
  slabs** with a fixed number of cells per slab, so reverse mode stores one
  slab's plasma history rather than the cube's.

Two performance lessons: an unrolled per-component/corner loop of ~500 scatters
took XLA > 15 min to compile (vectorised: 3 s); and **scatter-adds into few,
hot bins serialise on atomics** — 4 M updates into 25 bins take 2.9 s on an
A100 vs 0.5 ms into 65 k bins — so the DEM is two dense tent-matrix matmuls
and the Doppler sectors are per-pixel moment maps contracted at the end.

**Validation at 256³** (`casa_jaxobs.py validate`, plC state at 2000, one A100):

| band (keV) | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 2.8–4.2 | 4.2–6 | 6–7 |
|---|---|---|---|---|---|---|
| JAX / pyXSIM (post-halo) | 1.005 | 0.973 | 0.999 | 0.997 | 0.993 | 1.029 |

Forward: spectrum 2.2 s, band images 0.34 s, Doppler < 1 s (vs ~1 h).
Reverse-mode gradient w.r.t. the density field: 0.6 s (Doppler), 0.9 s
(images), 2.8 s (spectrum), **peak 7.3 GiB** (22 GiB before the 48-substep
T_e relaxation was checkpointed: its tape is ~5000 floats per cell; with a
fixed number of cells per slab, 512³ should need ~15 GiB); VJP = JVP = 2.15396 vs central FD 2.164 (0.45 %,
the remainder is the hard shock/kT masks and float32). Regression tests:
`test_casa_jaxobs.py` (CPU, 32³: plasma vs numpy, DEM vs direct sum, image
vs aperture conservation, VJP = JVP, roll/offset gradients).

**Real data on the model grid** (`casa_jaxobs_data.py`): all 18 epochs'
events binned into the six bands at 4 × 0.492″, with a dithered-chip-edge
fraction per pixel for masking (no exposure maps). 2006 and 2020 are CCO
subarrays and are not usable for the remnant.

**First pixel-level fit** (`casa_jaxobs.py fit-image`, fit-H 256³ state vs
Chandra 2000, 44.6 k pixels × 6 bands, Poisson likelihood, exact gradients):
roll −18°, explosion centre 5″ E / 3″ S of the grid centre (the Thorstensen
expansion centre is ~14″ E / 4″ S — same direction), amplitudes 1.5–2.1× (the
known faintness). The per-pixel deviance stays ~100 in the soft bands: the
model's knots are not Cas A's, and at the pixel level that dominates any
global parameter. This is the quantitative statement of what field-level
modelling has to fix (`figures/jaxobs_fitimg_H2000.png`).

Open items from Codex's second review (ranked): first-order `C + βD` and the
truncated keep are not positivity-guaranteed (diagnostic added; min rates are
reported by `validate`); the halo is applied after RMF folding and per band
(small, documented); for real data — per-ObsID ARF/RMF and exposure maps,
pile-up of bright knots and the CCO (masked within 6″), registration with
uncertainties, particle + sky background, synchrotron, and one coherent count
likelihood (spectrum, images and Doppler from the same events are not
independent).

## 6b. Every epoch, in X-rays (`casa_xfit.py`, 2026-09-24)

The forward model now carries the whole state the X-rays need — hydro, the
five composition scalars (solar CSM, `casa_pluto.py recompose` applied to the
146-yr IC) and the library's shock history (`track_shock_history`) — from
146 yr through all 15 usable epochs, and observes every epoch with
`casa_jaxobs` in six bands against the real counts (`casa_jaxobs_data.py`).
Per epoch the response is the epoch's detector with the ACIS-S tables
**interpolated in time** between cycles 0, 10 and 22 (the filter contaminant
grows smoothly; nearest-cycle is the known 2004 error). New physics in the
model: **synchrotron** (`sync_columns`: the `_synchrotron` chain traced —
ρv_s² weights normalised to Cas A's 1 GHz flux at the epoch, loss-limited
cutoff, smoothed filament-width gate; matches numpy to 0.14 %), and an
**N_H map** (a gradient on the sky; the columns are computed at all five
tabulated N_H and mixed per pixel). Parameters: the 18 of `casa_pluto_diff`
+ sky offset, ln A, ln N_H, ∇N_H (2), ln η_sync, and the emission physics
(post-shock kT_e, Coulomb-rate multiplier, Fe yield scale, clump mass
fraction). One forward pass (22 yr of evolution, 15 epochs observed) is
85–100 s at 128³; the FD Jacobian's perturbed runs go through `jax.pmap`
(`--devices N`), so an LM step with 20+ parameters is **4–6 min on 3 H100s**.

**Fit I, observed** (`figures/xfit_I_n128.png`): with no flux fudge
(ln A = 0) at 3.02 kpc the band counts are already 0.9–1.07 (soft), 0.75–0.84
(Si), 0.86–0.94 (S), 0.98–1.08, 0.96–1.11, 1.45–1.69 (Fe-K) of the data, and
the 20-yr light curves follow the data — the soft-band decline, mostly
contamination, exactly.

**The likelihood has to know that the knots are wrong the same way every
year.** Comparing each epoch's image independently (fit K) counted the one
static structure mismatch 15 times: image χ² −45 % while every kinematic
term degraded (outline 104 → 328, motions 239 → 374, Doppler correlation
+0.70 → +0.17, expansion 0.339 vs 0.294″/yr). The image term is now split
into a **static** part (epoch-mean brightness per 31″ block, σ = 0.25 in ln)
and a **temporal** part (each epoch relative to the block's own mean,
σ = 0.05) — expansion, fading and ionisation history, with knot placement
cancelling to first order. (Also: the variance must use (λ + n)/2, not λ —
blocks the model leaves dark otherwise give χ² ~ n².)

**Fit L** (23 free, 6 steps): static χ² 20.1k → 9.5k, temporal 58.3k →
54.8k, kinematics moderately worse (outline 224, motions 291, Doppler +0.43);
explosion 1618, D 2.89 kpc; the N_H gradient comes out **+0.85 per 100″ to
the west and −0.57 to the north — higher toward the W/SW, as the
spectroscopic N_H maps have it**, measured here from broadband images alone.
What no hydro parameter moves: the model is spectrally too hard (soft 0.7–0.85,
Si 0.7–0.78, 4.2–6 keV 1.1–1.2, Fe-K 1.6–1.8) and its hard bands fade too
slowly (−14 % vs −26 % in 4.2–6 keV over 2000–2019). That is the electron
temperature / ionisation physics — hence fit M with the emission parameters
free.

**Fit M** (+ post-shock kT_e, Coulomb-rate multiplier, Fe yield, clump mass
fraction): the spectrum comes right — soft 0.81–0.97, Si 0.91–1.03, S
1.1–1.25, 2.8–4.2 keV 1.25–1.35, 4.2–6 keV 1.1–1.2, **Fe-K 0.99–1.12** (was
1.6–1.8) — with **kT_e,0 = 0.29 keV** (Ghavamian's 0.3 survives), the
equilibration at **0.47× Spitzer**, Fe × 0.66, f_mass 0.19. But it paid in
kinematics again (outline 365, Doppler +0.17, explosion 1600). Three
systematics were doing it, all fixed in the likelihood:
(1) the soft band's model/data dips 0.95 → 0.81 → 0.91 across the decade —
the linear-in-time contamination interpolation, not the remnant — so a
**per-(epoch, band) calibration amplitude is profiled analytically** out of
the temporal term (prior 10 % soft, 3 % elsewhere); (2) the images want
emission outside the model's shock (the known r⁻⁴…⁻⁷ outer profile) and
inflate the remnant, so the **image likelihood stops at 150″** and the rim is
the outline data's; (3) the model errors are **calibrated to χ²/N = 1** on
fit M (σ_static 0.5, σ_temporal 0.075) and the image terms weighted 1/6 (the
six bands see the same regions).

**Fit N — the first fit in which the X-ray images and the kinematics agree**
(from fit I's hydro + fit M's emission parameters; 26 free; 8 steps, 45 min):

| | fit I | **fit N** |
|---|---|---|
| outline χ² / 34 | 104 | **95** |
| proper motions χ² / 30 | 239 | 234 |
| Doppler χ² / 24 (corr) | 458 (+0.70, measured-centroid statistic) | **115 (+0.87)** |
| image static / temporal χ² | 232 / 1096 | 181 / 1035 (N = 1536 / 23130) |
| r_FS 2000 / 2022 (data 158.9 / 165.8″) | 159.7 / 166.0 | 159.3 / 165.5 |
| expansion (data 0.294″/yr) | 0.296 | 0.292 |

Explosion **1634**, D **3.08 kpc**, synchrotron ~50 % of the 4–7 keV
continuum (Helder & Vink 2008 measure about half), N_H gradient toward the
W/SW. The regional light curves' relative fading now matches in every band
(e.g. 2.8–4.2 keV −20 % model vs −23 % data, Fe-K −23 vs −28 % over
2000–2019); absolute band levels are only loosely held (σ_static 0.5 on
31″ blocks) — soft ~0.8, Fe-K ~0.7 — which the integrated spectrum per
epoch should constrain next. `figures/xfit_N_n128.png`.

Running: **fit P** (fit N at 256³, 3 H100s, ~2.5 h per LM step: resolution
matters — at 256³ the reverse shock moves out from 34″ to ~50″ and the hard
bands brighten) and **fit O** (128³ + 15 field modes: ejecta log-density
perturbations on real spherical harmonics l = 1–3 at 146 yr, the first
non-global degrees of freedom).

**Fit P (fit N at 256³, 3 steps):** χ² 1832 → 1707; outline 98, motions
236, **Doppler correlation +0.90** (best), r_FS 158.0″ (data 158.9″),
expansion 0.295″/yr (0.294), explosion 1632, 3.08 kpc — the 128³ solution
transfers. The reverse shock is at 42″ (128³: 34″; observed ~96″): the
largest physics residual left.

**Fit O → O2 (field modes).** O (15 Y_lm modes, FD step 0.05) took a step of
~1e-6 in every parameter; O2 (l ≤ 2, 8 modes, FD step 0.1, noisy-row guard —
which found none) worked: χ² 1703 → **1505** — Doppler 114 → 63, motions
234 → 212, image static 181 → 116, temporal 1035 → 962, outline 97.
Explosion 1637, 3.09 kpc. The modes are large (ejecta density ×e^±1: the
l = 2 axisymmetric mode −0.97, the x-dipole −0.86), and the emission
parameters moved with them (kT_e,0 0.29 → 0.58 keV, f_mass 0.13) — a
degeneracy between density structure and emissivity that the spectra should
break. Fit Q2 = O2 + spectra (σ calibrated to χ²/N = 1 at the start:
0.13 static, 0.046 temporal; uncalibrated, the spectral evolution term was
2.5k of 4.2k — fit M's trap again).

**Fit Q2 = O2 + the 9 epoch spectra** (σ calibrated: 0.13 / 0.046): χ² 2031 →
1972 (spectrum shape 45 / 31, evolution 467 / 465); kinematics held — outline
98, motions 208 (per-cone correlation +0.33, the best), Doppler 58 (+0.89);
explosion 1635, 3.13 kpc. The spectra move the emission physics decisively
to **cold electrons: kT_e,0 = 0.11 keV** (O2: 0.58), equilibration 0.36×
Spitzer, f_mass 0.15, Fe × 0.89. The remaining shape residual (data/model,
normalised): 0.8 keV 0.67, **Si 1.8 keV 1.50**, S 2.4 keV 1.26, continuum
3–5.5 keV 0.9–1.0, Fe-K 1.2–1.36 — lines weak against the continuum, Si
weakest, as in every model since §3. Reverse shock 31″.

**What is missing (the residuals no fitted parameter removes).** (1) The
reverse shock: r_RS/r_FS ≈ 0.2–0.27 vs Cas A's ~0.6. The fits need a ~3×
denser wind to decelerate the forward shock at a ~1635 age, and that drives
the reverse shock in. Candidates: a cosmic-ray-modified forward shock
(energy loss / effective γ < 5/3 in the shocked CSM: decelerates the forward
shock without a denser wind, and produces the thin FS–CD gap, Warren+05);
CSM structure beyond a dipole (the W reverse shock moves inward in the
observer frame, Sato+18); cooling of dense ejecta knots; resolution (34″ at
128³ → 42–44″ at 256³); and the estimator itself (Gotthelf+01 use the inner
edge of the Si emissivity, we a pressure/entropy detector — to check on the
synthetic images). (2) Weak lines: abundances beyond Fe are fixed by the
tracer map (Si = tr15 unconfirmed); electron energy loss to ionisation in
pure-metal ejecta (Hamilton & Sarazin 84 — the cold-electron fit is its
effective form); one n_e t per cell (no sub-cell/parcel-history spread); an
over-strong continuum (synchrotron at 27× the radio anchor); ACIS
contamination and pile-up at the soft end and in bright knots.

**Next (new session): the full-state fit** — 4D-Var in the observer frame
at 2000 with this fit evolved from 146 yr as the background (not the 146-yr
state itself: 220 yr of RT chaos make its adjoint useless), a 22-yr window,
control fields on a half-resolution grid preconditioned by B^−1/2, the full
Chandra event-cube Poisson likelihood with resolved Doppler maps (Doppler
velocity ~ depth for freely expanding ejecta breaks the line-of-sight
degeneracy), and hold-out epochs. First: verify reverse mode through the
solver and measure adjoint growth. See the memory handoff.

**Integrated spectra** (`--spectra`): the 9 epochs
with a Chandra spectrum (r < 200″) enter as 0.2 keV bins over 0.7–7 keV,
modelled channel by channel inside the aperture — per-pixel N_H and halo
aperture-keep at the traced sky radius, synchrotron included, evaluated only
at spectral epochs (`lax.cond`) — with the per-epoch normalisation profiled
out, so they constrain spectral SHAPE (line ratios: T_e, ionisation age,
abundances) as a static + temporal pair like the images.

## 7. Next

* **field-level likelihood in the fit** — done at the smooth-parameter level
  (§6b); next: a low-dimensional field basis (e.g. ejecta density/velocity
  perturbation modes at 146 yr) on top of the globals;
* 256³ fits (memory-fixed; pmap FD over 3–8 GPUs);
* the reverse shock at 33″ vs ~96″ and the per-cone proper motions — the two
  residuals fit I leaves;
* confirm the Si tracer (tr15) with Orlando — the orientation result rests on it;
* re-score CALIBRATION.md after the solar-CSM, T_e and parity fixes.

## 8. Audit 2026-09-25 (read `AUDIT_2026_09_25.md`)

An eight-auditor review of this track found:
* **The reverse-shock "residual" (31–42″) was an estimator artefact.** The corrected estimator puts Q2 at 92″ (0.58 r_FS).
* **Cosmic-ray back-reaction is not the key missing factor.** This is bounded by the γ-ray budget and confirmed by an in-house 1D two-fluid test (Δm < 0.02).
* **The fitted wind dipole mostly compensates an explosion-centre inconsistency.** The outline and image terms used different centres, and the data outline is round about the Thorstensen+01 expansion centre.
* **The fitted explosion date ≈ 1635 is the initial condition's own clock.** Cas A's explosion date is bounded below by the knot convergence date, 1671.3.
* **A list of observation-model and response artefacts**, now fixed (see that file).
* **2000-epoch reconstruction.** `casa_4dvar.py`, a strong-constraint 4D-Var run from the refit's 2000 state, predicts the held-out 2019/2022 data (never fitted) substantially better than its background: s3 χ² 415 → 337 (R′) and 426 → 354 (R2). The production state is `work/stage4/run/4dv_R2_nowarp/state2000_4dvar_stage3.npz`. Details and limits are in `AUDIT_2026_09_25.md` §5c–6.
