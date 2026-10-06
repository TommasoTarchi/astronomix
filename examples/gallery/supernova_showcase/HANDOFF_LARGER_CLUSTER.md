# HANDOFF: Cas A differentiable reconstruction on a larger GPU cluster (2026-10-06)

**Read this first, then `AUDIT_2026_09_25.md` §5c–§8.**

> **Status on `main` (consolidated 2026-10-06).** The port of §2.2 is done in the
> repository: the solver is the positivity-preserving WENO of `weno-stability`,
> every `casa_*` script and the reverse-mode pytests use it (no `--positivity`,
> `--deepvoid-blend`, `--ad-llf-cold` flags any more; `casa_4dvar --tangent` is
> `exact | approx | auto`), the `asym_field` fix is kept, and **P1** (finite
> float32 derivatives of the admissibility scaling) is in the library
> (`_weno_positivity._floored_ratio`). Still open: **P2** (PP-native cold-face
> tangent, only if gate G2 needs it), **P3** (separate frozen-theta switch), the
> gates G0–G2 on a GPU, and the callers outside the repository (`W/ers/**/*.sh`,
> `W/run_fit{C,H}.sh`, `W/run_n512.sh` still pass `--positivity`; the `W/stage*`
> harnesses still point at `~/jf1uids`). The `run.sh` wrapper now imports the
> checkout it lives in. Flags and options mentioned below for the pre-PP solver
> are historical.

The three review reports behind this hand-off are in `/export/data/lstorcks/casa_orlando150/work/final_review/`:
- `results.md`: state of the science, costs, data inventory;
- `stability_port.md`: porting onto `weno-stability`, with the probes in `stability_port_probe/`;
- `design.md`: the t0 question and the Bayesian stage.

Abbreviations:
- **W** = `/export/data/lstorcks/casa_orlando150/work`
- **S** = `examples/gallery/supernova_showcase` (in the repo)
- **PP** = positivity-preserving WENO (`weno_positivity_preserving=True`)

## 0. Most important finding

**Nothing we validated was made with the solver that will run next, and on the `weno-stability` branch the Cas A 4D-Var does not run at all.**
- The scripts fail at import: `casa_xfit` imports `POSITIVITY_REDISTRIBUTE`, and every `casa_4dvar*.py` imports `casa_xfit`.
- Half of the production "approx" tangent was deleted: `ad_tangent_llf_cold_factor`, in commit `5f23a0a`.
- Exact-mode gradients through the PP limiter are all NaN in float32. The probe: 1D, 64 cells, 20 steps.
- Every fit, state, checkpoint and tangent-growth number was made with REDISTRIBUTE or CONSERVATIVE positivity plus the Zalesak FCT.

**So the first GPU-hours on the new cluster go to:**
1. the port;
2. a 128³ re-baseline;
3. a tangent-growth gate.

Spend nothing at 448³ until those pass.

**Scientific headline.**
- The large-scale state is identifiable and validated out of sample: held-out χ² −17 to −19 %.
- The fine structure lies beyond the predictability horizon of pixel-level fitting. Exact tangents grow ×10 per ≲ 0.15 yr at 128³.
- So "fine structures right" needs statistical likelihood terms, short windows, or a weak-constraint/ensemble formulation, not more L-BFGS iterations.

---

## 1. Final review: the findings that matter

V = validated, P = preliminary, O = open.

| # | Finding | Decisive number | Status | Source |
|---|---|---|---|---|
| 1 | 4D-Var at 2000 predicts the held-out epochs | 2019+2022 held-out χ² (s3 scorer): R′ 415.1 → **336.7**, R2 425.6 → **353.5** (s4 scorer, = casa_4dvar's own `eval_2018.json`: R2 299.3 → 227.5). Data-only baselines 2506 / 1398 / 1212.7. Longer windows help: 421 → 393 → 363 → 337. Limits: 2018-map persistence beats every model at 2019 (14.6 vs 26.1); per-cone linear extrapolation beats every model on the outline. | **V** | AUDIT §5c–5d; `W/stage4/run/final/SUMMARY.md` |
| 2 | Predictability horizon of the tangent-linear model (TLM) | Exact TLM matches finite differences to 0.3–0.6 % over 2–5 yr (64³), then grows ×8.4e3 by 11 yr and ×1.1e13 by 22 yr. At 128³: ×10 per ≲ 0.15 yr. Approx TLM: bounded (0.74 over 22 yr) but 20–25 % biased. | **V** (64/128³); P (512³: 0.01–0.04 yr per e-fold, extrapolated) | `W/stage1/tlm/growth/SUMMARY.md`; `W/ers/filaments_RESULT.md` |
| 3 | The W15 starting state is under-energetic | 1.514e51 erg in the state. Required E/M_ej is 0.60–0.74 B/M☉ against 0.46, i.e. **2.0–2.3e51 erg**. | **V** | `W/ers/energy_review_RESULT.md`; AUDIT §7 |
| 4 | Exact similarity rescaling (L, T, M) fixes the age label | Fitted explosion date 1635–1642 before rescaling; 1671.1 (R3w), 1671.7 (R3v) and 1674.8 (R4b) after, against the knot bound 1671.3 ± 0.9 | **V** | `S/casa_rescale.py`; `W/xfit_R3w.json`, `xfit_R3v.json`, `xfit_R4b.json` |
| 5 | Missing filaments come from resolution plus fit design | 5–15″ contrast 0.0145 / 0.036 / 0.054 at 128/256/512³ against 0.126 in the data; needs ~1250³ (filament density) to ~1400–2100³ (5–15″ power). Production 4D-Var images are 26× below the data at 5–15″ (in log terms ≈60 % resolution, ≈40 % fit trajectory ×0.6 plus v2 observation chain ×0.4). | V (forward); P (256³ fine design); **O**: the v2 observation chain loses 2.4× more 5–15″ contrast than v1 | `W/ers/filaments_RESULT.md`; AUDIT §7 |
| 6 | REDISTRIBUTE positivity manufactures mass | At 448³, rho 15.8 → 9000 in 5 yr (+0.030 M☉), then NaN. At 128³ the switch moves χ² by only +2.6 (+0.3 %), but **all 128³ fits used it**. | **V** | `W/ers/jetdbg/REPORT.md` |
| 7 | NE jet + SW counter-jet | Held-out jet-image χ² 532.5 → **93.0** over 97 bins. R4b held-out 287.5 under the jet masks. The model's jet bulge is 2–3× the observed ~15″. | **P** | `W/ers/jet/REPORT.md`, `W/ers/jet_review/REPORT.md` |
| 8 | Cosmic-ray back-reaction is not key | γ-ray bound W_p 0.5–2.8e50 erg; in-house 1D Δm −0.0006 to −0.004 | **V** | AUDIT §1.2, §5 |

**Also validated:**
- the reverse-shock estimator fix: the tracer agrees to 0.003–0.005 r_FS;
- the centre bug fix: an 11–12″ m=1 term becomes 2.7″ about the true centre;
- the s0 shock-label sanitisation: it was all NaN at 448³;
- the device L-BFGS: matches scipy step for step at 128³, host RAM 10.5 GB flat.

**Open:**
- the expansion-rate systematic: the in-house rate is ~10 % below Vink+22. R3v (Vink rates) ties R3w on held-out, 310.0 vs 310.7, and has no 4D-Var.
- the outline detector in some cones;
- the v2 fine-contrast loss;
- R4's finite-difference Jacobian NaNs;
- multi-GPU runs of the device L-BFGS on real GPUs (verified only on fake CPU devices);
- the 512³ gradient: never measured.

**Rejected (keep as negatives):**
- the reduced-rank weak constraint: its continuity term catches ~8 %;
- the forward-shock warp: held-out 336.7 → 357.5;
- a 512³ run with the current design.

**Production state at 2000 (128³, 4D-Var R2, no warp):** `W/stage4/run/4dv_R2_nowarp/state2000_4dvar_stage3.npz`

| Quantity | Model | Reference |
|---|---|---|
| r_RS | 95.2″ | 95.8 ± 9.7 (Gotthelf) |
| Expansion | 0.200 %/yr | 0.218 (Vink) |
| Doppler correlation | 0.983 | – |
| Ejecta | 3.02 M☉ | – |
| E_box | 2.24e51 erg | – |
| D | 3.20 kpc | – |

**448³ run (stopped).** `W/ers/n448/run_4dv_R4b_jet/` reached J 931.7 → 767.5 in 4 evaluations on 4×H200 (scipy optimiser, `--positivity conservative`), at 68.6–70.6 GB/GPU and ~18 min per gradient for the stage-0 window 2000–2004.5 (Taylor-run gradient 1084 s, `W/ers/s0fix/taylor448_s0fix.json`). It was killed by the compgpu12 crash on 2026-10-05; the node's fabric manager has failed (`CUDA_ERROR_SYSTEM_NOT_READY`). Its checkpoint is valid **only with the old solver**. No GPU jobs are running.
- The stale watcher `W/ers/r4b/watch_restart_jax.sh` (PID 3947020 on compgpu11) was killed by PID on 2026-10-06. Its one restart (pq `1791230887600-n448-4dv-jet-jax`) died at `cuInit`. No watchers or jobs of ours remain.

---

## 2. Code state and the port onto `weno-stability`

### 2.1 Where things are

| Tree | State |
|---|---|
| `~/jf1uids`, branch `casa-orlando-calibrated` @ `7fafdf1` + **~120 uncommitted paths** | the whole Cas A pipeline (`S/casa_*.py`: 53 scripts, incl. `casa_4dvar{,_lbfgs,_shard,_robust,_control,_data,_warp,_wc}.py`, `casa_xfit*.py`, `casa_jaxobs*.py`, `casa_pluto*.py`, `casa_rescale.py`, `casa_jet.py`, `casa_cr_1d.py`, `casa_anim_compare.py`, `run.sh`, tests) plus the library work in `astronomix/`. **Irreplaceable and uncommitted: back it up first.** |
| `~/jf1uids-weno-stability`, branch `weno-stability` @ `b8eeef8` | the user's PP-WENO and the phase-out (see below); has only 13 of the `casa_*.py` scripts |

**The library half of the port is already done.**
- `weno-stability`'s first commit, `285eab1`, is a byte-identical snapshot of our uncommitted `astronomix/`. Today `git diff 285eab1 -- astronomix` lists only the 3 untracked files (`_modules/_resistivity/{__init__,_resistivity}.py`, `test_setups/reference_solutions/cr_shock_tube.py`) as deletions; they are `cmp`-identical to the snapshot (re-verified 2026-10-06). After commit C_a below, `git diff 285eab1 HEAD -- astronomix` is empty.
- So the reverse-mode AD, remat/chunks, `_weno_omega_weights_ad`, s0 sanitisation, `sharded_roll`, `_native_tangent_sharded`, the CR module and the shock finder are all on the branch.
- Later commits changed only the positivity layer.

**What changed on the branch:**

| | Mechanisms |
|---|---|
| **Removed** | per-stage floors and `_enforce_positivity*.py`; REDISTRIBUTE and CONSERVATIVE; `vacuum_rest`; `nan_safe`; `default_positivity_protection`; prot; the deep-void blend; Zalesak `preserving_flux`; the HLLC fallback; the gravity backstop; `ad_tangent_llf_cold_factor`; `positivity_max_velocity` (the velocity cap of 50) |
| **Kept** | per-step HARD_FLOOR (finite-volume); `clamp_in_estimates` (fixed: no longer writes the carried state); `coldcrush_blend`; dual energy |
| **Behaviour** | `weno_admissible_face_state` is on by default, so the forward changes even with PP off. Under PP, `weno_ad_frozen_weights` also freezes θ and the common splitting speed. |

### 2.2 Ordered port (from `stability_port.md` §4)

**Never** merge or rebase a commit that contains the `astronomix/` changes parented at `7fafdf1`. A simulated 3-way merge silently resurrects the deleted cold-LLF block without any conflict.

1. **Back up and commit on `casa-wip`, with astronomix last and never ported** (in `~/jf1uids`):
   ```bash
   cd ~/jf1uids && mkdir -p /export/data/lstorcks/backup        # the backup dir does not exist yet
   tar czf /export/data/lstorcks/backup/jf1uids_wt_2026-10-06.tgz --ignore-failed-read \
       $(git status --porcelain | awk '{print $NF}')               # one staged deletion (.swp) is not on disk
   git switch -c casa-wip
   git add -A examples/gallery/supernova_showcase pytests/cosmic_rays pytests/differentiability PROGRESS.md .gitignore
   git rm --cached examples/gallery/supernova_showcase/GARCHING_REQUEST.md.save 2>/dev/null   # leave the editor backup out
   git commit -m "Cas A pipeline WIP (pre PP-WENO)"                                    # C_e
   git add -A examples/scripts/forward/mhd/turbulence && git commit -m "dynamo WIP"   # C_d
   git add -A astronomix && git commit -m "solver WIP (identical to 285eab1)"         # C_a
   git diff --stat 285eab1 HEAD -- astronomix                                         # must print nothing
   ```
2. **Port:**
   ```bash
   git worktree add ~/jf1uids-casa-pp -b casa-pp weno-stability
   cd ~/jf1uids-casa-pp && git cherry-pick C_e C_d
   ```
   - `supernova_showcase/` merges without conflicts (simulated).
   - The one conflict is `examples/scripts/forward/mhd/turbulence/dynamo_convergence.py`: delete `vacuum_protection=False`, keep `forcing_modes` and `ou_unit_rms_each_step`.
   - Check that the `asym_field` NameError fix in `casa_orlando.py` survives. It is at l.1477–1482 in the calibrated tree; the bug is at l.1227 on the branch.
3. **Library patches.** Each is tangent-only, so the primal stays bitwise; each gets a pytest.
   - **P1 (blocker for exact/semi tangents): an f32-safe derivative of `_weno_positivity._admissible_fraction`.**
     - Use double-`where` guarded divisors (the `0·inf` pattern already fixed once in the deleted Zalesak limiter), or a `custom_jvp` that is non-zero only where 0 < θ < 1.
     - Acceptance: `W/final_review/stability_port_probe/pp_ad_{c,d}.py` gives a finite f32 gradient matching x64. Reference values: x64 2.85581 (1.54377 with dual energy); f32 with θ stop-gradiented 2.7448 against finite differences 2.7552.
   - **P2 (conditional; decided by gate G2): a tangent-only cold-face linearisation.** The preferred form is a PP-native `ad_tangent_cold_theta_factor`:
     - θ_t = 0 on faces whose colder side has T < factor·T_floor;
     - value = `stop_gradient(F(θ)) + F(θ_t) − stop_gradient(F(θ_t))`.

     The two reports disagree on whether P2 is needed:
     - The design ablation (`W/stage1/tlm/runs/`, 64³ x64, old positivity) **measured** that freezing the eigensystem alone keeps the TLM bounded (11 and 22 yr at 64³; 5 yr at 128³). Frozen weights alone explode (JVP 2.7 vs ≈ 0.01). On the branch, `weno_ad_frozen_weights` freezes the eigensystem too.
     - **Caveat:** these windows are very short in steps. `tlm_ablate.py` records `steps_cfl` = 2 / 7 / 14 for 2 / 11 / 22 yr at 64³ (dt ≈ 1.6 yr) and 7 for 5 yr at 128³. Boundedness over the thousands of steps of a production window (≈19k at 512³) is untested, so G2 must include ≥ 128³ long windows.
     - The deleted docstring claims ~5× growth per year at cold knots without cold-LLF.

     The measurement outranks the docstring. Run G2 with frozen-only first and implement P2 only if it fails. P2 is ~20 lines, so having it as an opt-in is cheap either way.
   - **P3 (optional):** split `weno_ad_frozen_theta` from `weno_ad_frozen_weights`, so that `semi` is well defined.
   - **P4: migrate the scripts.**
     - `casa_xfit` l.119, 857–860, 2639, 2667–2679, 2766–2779: drop `--positivity`, `--deepvoid-blend` and `--ad-llf-cold`; use `weno_positivity_preserving=True`.
     - `casa_4dvar` l.171–174: the `TANGENTS` table. Make "approx" = `{weno_ad_frozen_weights: True}` (+ cold θ if P2), and "exact" = frozen off.
     - `casa_4dvar` l.230–239 and l.1109.
     - `casa_pluto_diff` l.122, 1140–1143, 1533.
     - `casa_pluto_jvp_probe`, `casa_xfit_compare`, `casa_cr_1d` (`nan_safe`/`vacuum_rest`).
     - `pytests/differentiability/test_fd_reverse_mode.py`: new config, re-recorded sha256.
     - The 12 `work/ers/**/*.sh` scripts (13 occurrences: 9 conservative, 4 redistribute) with `--positivity`.
     - The work-dir harnesses `W/stage1/lib_ad/adj_vjp_check.py` (`--ad-llf-cold`, `--no-ppflux`) and `W/stage1/tlm/tlm_ablate.py` (`--positivity`, `--llf-cold`, `--ppflux`).
     - **Path coupling (silent old-code trap).** About 55 work-dir `.py` harnesses hard-code `/export/home/lstorcks/jf1uids`:
       - `adj_vjp_check.py` inserts `~/jf1uids` at `sys.path[0]` (it overrides `ASTRO_ROOT`) and imports `_common` from the frozen `W/stage1/baseline/showcase` (override: `ADJ_SHOW_DIR`);
       - `tlm_ablate.py` defaults to `ASTRO_ROOT=W/astro_snapshot_2026_09_25` and `TLM_SHOW=W/stage1/baseline/showcase`;
       - `W/stage3/validation/eval_model.py` (called by `run_4dv_R2.sh`) and `W/stage4/run/score.py` hard-code `SHOW=~/jf1uids/.../supernova_showcase`.

       Point all of them at the `casa-pp` tree, or they run the old scripts/solver while looking healthy.
   - Then run `pip install .` in the astx env, or always use `run.sh` with `ASTRO_ROOT=~/jf1uids-casa-pp`. The site-packages wheel is stale.
4. **Gate G0 (CPU, `JAX_PLATFORMS=cpu`):**
   - the branch suite (25), `pytests/cosmic_rays` (28);
   - `test_ad_tangent_safety.py`, `test_entropy_label_sanitize.py`, the migrated `test_fd_reverse_mode.py`;
   - `S/test_casa_4dvar_lbfgs.py` (15) and `S/test_casa_jaxobs.py`;
   - `casa_jet.py test`;
   - `examples/scripts/validation/weno_stability/pp_gradient_check.py`, plus a **new** f32 + dual-energy + passive-scalar + remat variant: VJP = JVP, dot-product test, finite differences. The branch's own check (x64, 12³, 3 steps, no dual energy, no frozen weights) cannot see the f32 NaN.

### 2.3 What the PP change does to the physics (to measure, not assume)

- **Mass.** The flux form conserves mass to round-off, so the REDISTRIBUTE runaway mechanism is gone. Provably PP for C_cfl ≤ 0.75 with SSPRK4; Cas A runs at 0.3, so there is no time-step penalty.
  - Validated for Cas A only at 128³ (energy within 0.01 %) and in 64³ smoke runs. **Never with the jet and clumps at 448³.**
- **No velocity cap any more.** The 448³ IC has a hot near-vacuum pocket on the NE jet axis (cell (178,220,250), r = 0.72 pc) that reaches |v| = 88.2–88.8 code units (1000 km/s each, i.e. ≈ 89,000 km/s) at 0.5 yr with the jet on *and* off; jetdbg judged it harmless under the old recipe (the NaN was the REDISTRIBUTE clump runaway). The 256³ R2 state has a 13,400 km/s cell at ρ 1e-5 that sets dt. Watch the dt history.
- **Frozen-mode tangent bias** (x64 probe, against finite differences): −12 % without dual energy, **+61 % with dual energy**. f32 frozen equals x64 frozen to 2e-5.
- **Fine structure.**
  - Contact (entropy-mode) error grows by (|v|+c)/|v| = 1 + 1/M_lab: ≈ 1.75× just behind the forward shock (M_lab ≈ 1.3, `stability_port.md`), ≈ 1.1–1.5× in the shocked shell at M_lab 2–10 (`design.md`), more near stagnation and the centre. The two reviews quote different regimes; measure it.
  - θ admissibility uses the total-energy pressure, so in cold, kinetic-dominated f32 ejecta θ < 1 (first order) can fire inside knots.
  - The deleted FCT had the same check, so this is not a regression by construction, but it must be measured: θ < 1 face fraction, x32 vs x64, and 5–15″ power at 256³ PP vs the old recipe.
- **Possible upside (untested):** the smooth admissible face state may remove the vacuum-edge gradient spikes that stalled late 4D-Var stages.
- **Multi-GPU PP has never run on the branch.** A 2-GPU sharded `turb.py` hung after compile.

---

## 3. Data inventory to transfer

Classes:
- **IRR**: irreplaceable.
- **EXP**: regenerable but expensive, and not bitwise under PP anyway.
- **DER**: cheap to derive.
- **DL**: re-downloadable.

Sizes are `du -sh`.

| Item | Path | Size | Class | Needed? |
|---|---|---|---|---|
| Repo + uncommitted work (or the `casa-wip` branch / bundle) | `~/jf1uids` (showcase dir 387 M) | <0.5 G | **IRR** | yes, first |
| Project record | `S/{AUDIT_2026_09_25,ROADMAP,PLUTO150,CALIBRATION}.md`; `W/{audit_2026_09_25,stage1..4,ers,final_review}/**/*.md`; memory `handoff_casa_audit_2026_09_25.md` | small | IRR | yes |
| Raw Orlando PLUTO delivery | `/export/data/lstorcks/casa_orlando150/*.flt` (27 files), `flt.out`, `grid.out`, `pluto.ini`, `definitions.h`, `pluto.0.log`, i.e. everything there except `work/` and `jaxobs/`. **`flt.out` is required:** `casa_pluto.py` reads it (`_read_flt_out`). | **6.9 G** | **IRR** | yes |
| Rescaled ICs (R3w/R4b family) | `W/ers/integ/ic/pluto146sim_L1.42_T1.21_M1.00_n128_solarcsm.npz` (16 M, the R4b refit IC); `W/ers/energy/ic/*_n{128,256,512}.npz` (1.3 G) | 1.3 G | DER | yes (128³), else regenerate |
| 448³ IC recipe | `W/ers/n448/make_ic_R4b.sh`, `theta_{R3w,R4b}_bg_n448.txt` (ICs 3.7 G, ~6 min CPU each) | small | DER | the recipe only |
| Fit parameters, Jacobians | `W/xfit_*.json` (126 K), `W/xfit_*_jac.npz` (15 M; `xfit_R4b_jac.npz` = 1.7 M is the 448³ `--jac`) | 15 M | EXP | **yes** |
| Fit model products | `W/xfit_*_n128.npz` (15 files; `xfit_Rp_n128.npz` is the default for `casa_xfit_bkg`) | 288 M | EXP | yes |
| 128³ states at 2000 + 4D-Var analyses | `W/state2000_*_n128.npz` (0.6 G); `W/stage4/run/` (916 M) incl. production `4dv_R2_nowarp/` (359 M); `W/stage3/oos/run_Rp/` (337 M) | ~1.9 G | EXP | yes (old-solver references for the A/B) |
| Tangent-harness state | `W/plH_n256_age364yr_solarcsm.npz` | 0.4 G | EXP | yes (for G2) |
| 448³ jet background (old solver) | `W/state2000_R4b_n448.npz` | 3.3 G | EXP | optional (A/B reference only) |
| 448³ checkpoint | `W/ers/n448/run_4dv_R4b_jet/{ckpt.npz,evals.jsonl}` | 0.85 G | EXP, old solver only | optional (warm start only) |
| X-ray model tables v1 + v2 | `/export/data/lstorcks/casa_orlando150/jaxobs/` | 2.5 G | DER (slow) | yes |
| Data bands, measured background, Doppler v2 | `…/jaxobs/data/` | 23 M | DER | yes |
| Proper motions, outlines, old Doppler | `W/expansion*.npz`, `W/observed_outlines.npz`, `W/doppler_si_2004.npz` | <1 M | DER | yes |
| Likelihood caches | `W/stage2/integrate/ciao_photon_weights.npz`, `W/stage4/xfit/streak_cache/`, `W/stage3/physics/background.json` | <1 M | DER | yes |
| Chandra events (22 ObsIDs) | `/export/data/lstorcks/chandra_casa/evt2/` | 7.1 G | DL | yes (or re-download) |
| Epoch images and spectra | `…/chandra_casa/epoch_images/` | 24 M | DER | yes |
| CIAO responses (CIAO 4.18, CALDB 4.12.4) | `…/chandra_casa/responses/`; runtime needs only `epochs/`, `manifest.json`, `*/prep.json`, `*/*_r200.arf` | 9.6 G (minimal ~0.3 G) | DER (hours; pin CALDB) | minimal set |
| **Do not copy** | `W/ers/n448/diag/state2000_R4b_n448_nojet.npz` (stale shock history), `*_s0nan*`, `*_ABORTED_nan*`, `state2000_R2_n448crop.npz`, the `.nfs0000000000075d5700012bd5` handle | ~10.5 G on disk (~13 G apparent) | drop | no |

**Totals.**
- Minimal set: **≈ 30 GB**, or ≈ 40 GB with the full responses.
- The whole work directory is 145 G, mostly regenerable diagnostics.

**Environments and dependencies.**
- **`astx` mamba env (jax/jaxlib 0.10.2).** Recreate it, don't copy it (4.6 G).
  - Always set `PYTHONPATH=<repo>` via `run.sh`.
  - `run.sh` hard-codes `ASTX=/export/home/lstorcks/.local/share/mamba/envs/astx` and the default `ASTRO_ROOT`. Edit both on the new cluster.
- **Other environments and data packages:**
  - `xrayobs` venv (576 M): pyXSIM/soxs;
  - `soxs_data` (622 M, read at runtime);
  - `atomdb` (294 M, `$ATOMDB`);
  - `/export/data/lstorcks/supernova_showcase/nei_ion_fractions.npz` (read by `_nei.py`; it is **not** in `S/`), and `dust_halo_mrn_amax*.npz` from the same directory (read by `_dusthalo.py`, `CASA_DUST_TABLE_DIR`, for rebuilding the halo tables). Copy only these files: that directory is 280 G;
  - `pylib_ffmpeg` (38 M, animations only);
  - the CIAO env (6.3 G): only needed to regenerate responses.
- **`autocvd`** (0.2.1 in astx; install it in the new env): called at import time by `casa_xfit.py`, `casa_4dvar.py`, `casa_xfit_shard.py`, `casa_pluto_diff.py` and ~20 other `S/` scripts, plus the work-dir harnesses, always gated on `CUDA_VISIBLE_DEVICES` being unset. Under a scheduler that sets it, nothing changes. On interactive nodes, autocvd must pick the GPUs; never hand-set `CUDA_VISIBLE_DEVICES`.
- **Hard-coded `/export/data/lstorcks/...` paths** appear in 33 `S/` files, among them:
  - `casa_4dvar`, `casa_pluto_diff` (`WORK`);
  - `casa_jaxobs*`, `casa_xfit*` (incl. `casa_xfit_bkg`, `casa_xfit_responses`);
  - `casa_expansion`, `casa_anim_compare`, `_nei`, `_dusthalo`;
  - `test_casa_4dvar_lbfgs`, `test_casa_jaxobs`.

  Mirror the tree, or symlink `/export/data/lstorcks` on the new cluster.
- **Hard-coded `/export/home/lstorcks/jf1uids`** in ~55 work-dir harnesses and in `run.sh`'s default `ASTRO_ROOT` (see P4, path coupling).

---

## 4. Run plan on the larger cluster

### 4.0 Hardware and the common job prologue

**Hardware.**
- **448³** needs ≥ 80 GB/GPU and 4–8 GPUs on one NVLink node: ~70 GB/GPU on 4×H200 at `--ckpt 4 --remat-chunks 8`. A100-40 does not fit, and 11 GB cards OOM even at 32³ (a single 3.45 GiB allocation).
- **512³+** wants 8×H200 (or 8×H100-80).

**Job prologue** (template; translate `pq sub` to the local scheduler):
```bash
ulimit -c 0                                    # no core dumps of multi-GPU processes
export NCCL_NVLS_ENABLE=0                      # required for the sharded runs (2026-08-03)
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.75   # H200 setting
export CASA_4DVAR_MAX_HOST_GB=300              # hard host-RSS cap (casa_4dvar_robust)
export ASTRO_ROOT=$HOME/jf1uids-casa-pp        # run.sh refuses a root without astronomix/
REPO=$ASTRO_ROOT; S=$REPO/examples/gallery/supernova_showcase; W=/export/data/lstorcks/casa_orlando150/work
mkdir -p $W/pp                                 # output dir of the templates below
cd $S
# guard: a failed cuInit makes JAX fall back to CPU SILENTLY (compgpu12, 2026-10-05).
# autocvd-gated and without preallocation, so on an interactive node it does not grab every GPU.
XLA_PYTHON_CLIENT_PREALLOCATE=false ./run.sh -c "import os
if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    from autocvd import autocvd; autocvd(num_gpus=1)
import jax; d=jax.devices(); assert d[0].platform=='gpu', d; print(d)"
```
Optionally, `export JAX_PLATFORMS=cuda` in GPU jobs: JAX then fails at start-up instead of falling back to CPU. This is untested with these scripts, and their CPU paths check for `JAX_PLATFORMS=cpu`.

**A/B runs with both trees:** the prologue exports `ASTRO_ROOT=casa-pp`, and `run.sh` honours it. For the old-solver leg, set it explicitly: `cd ~/jf1uids/examples/gallery/supernova_showcase && ASTRO_ROOT=$HOME/jf1uids ./run.sh ...`. On the old tree, `casa_xfit`/`casa_4dvar` default to `--positivity redistribute`, which is bitwise the fitted 128³ runs; the 448³ runs used `conservative`.

**Code-side recipe settings:**
- the mesh stays `AxisType.Auto`;
- `COORD_PRECISION = HIGHEST` (the TF32 bug).

**Node hygiene:** refuse any node that has corrupted outputs. compgpu14 did it twice; the run scripts already refuse it.

### 4.1 Phases and gates

| Phase | What | GPUs | Cost | Gate to pass |
|---|---|---|---|---|
| A | Back up, transfer, envs, port P1/P4 (+P2 if G2 fails), CPU tests | 0 | – | **G0** (§2.2 step 4) |
| B1 | 128³ forward A/B, old vs PP, at the R4b and R2 θ | 1 | in P0 | **G1**: Δχ² train/held-out small (precedent: conservative vs redistribute +2.6); mass to round-off; energy to 0.01 %; finite dt history; θ < 1 fraction by region; morphology statistics (4.4″, χ, coherence) |
| B2 | Tangent growth (64³ x64) at 2/5/11/22 yr, + 42 yr if a pre-2000 start is pursued, **and at 128³ for 5/11/22 yr** (the 64³ windows are only 2–14 CFL steps); Taylor tests (exact/semi/approx) at 128³ | 1 | in P0 | **G2**: approx (frozen-only) adjoint growth ≤ 1 at all windows (old 0.97/0.92/0.84/0.74). JVPs comparable to the old approx values (E-only vs the old library mode already differed by up to ~35 % per direction at 11 yr) and consistent with large-h FD/secants; Taylor along −g ≈ 1 |
| B3 | 128³ refit R4b (or R3v, to test the expansion rate), then a 128³ 4D-Var with held-out scoring | 1 | in P0 (128³ gradient 0.022 GPU-h for the full window) | **G3**, like for like with the same scorer (`W/stage4/run/score_all.sh`, s3 primary): a PP 4D-Var(R2) on a PP-regenerated R2 background should reach the old production held-out **353.5** (s3; s4 227.5) or better; 336.7 is the R′ number. No old 128³ 4D-Var exists for R4b/R3v, so there require a held-out gain over the own PP background comparable to the old 17–19 %. The ±2 jitter band is the design reviewer's estimate, not a measurement. Else stop and diagnose |
| C1 | Sharding on PP: 128³ on 2 and 4 GPUs vs 1 | 2–4 | in P0 | **G4**: forward bitwise or within a stated tolerance; gradient cosine ≥ 0.9999999 (old: 0.99999999 at 8 GPUs); **no post-compile hang** |
| C2 | Regenerate the 448³ jet background with PP; one 448³ gradient | 4 | bg ~1.5 h (old: 5674 s on 4 A100, conservative + jet); gradient ~40 min incl. compile | **G5**: ≤ ~75 GB/GPU; host RSS flat; Taylor along −g in the old 1.005–1.014 range; mass and dt sane through the jet |
| D | 448³ 4D-Var, fresh start | 4 (8 if available) | **220–410 GPU-h** (≈ 80 h wall on 4×H200) | stage-by-stage held-out score |
| E | 512³+ (only if the fine-scale design is redone, §4.3) | 8 | estimate: 300–450 s/gradient, 60–95 h; never measured | memory and Taylor first |

Design's P0 budget for phases A–C: **40–60 GPU-h**.

### 4.2 Command templates (on `casa-pp`, `--positivity` removed)

**B1: 128³ A/B forward.**
- Pattern: `W/ers/s0fix/run_128_ab.sh`. Run each tree's scripts with that tree's library: old = `~/jf1uids` scripts plus old library, new = `~/jf1uids-casa-pp`.
- Without `--fit`, `casa_xfit` evaluates the forward model and the likelihood.
```bash
TH=$(python -c "import json;print(*json.load(open('$W/xfit_R4b.json'))['theta'])")
./run.sh casa_xfit.py --ic $W/ers/integ/ic/pluto146sim_L1.42_T1.21_M1.00_n128_solarcsm.npz --theta $TH \
  --doppler --spectra --obs v2 --responses ciao --exclude-epochs 2019 2022 --jet on --jet-img on \
  --outline-mask inner-arc+jet --pm-mask-extra 300+140+150+340+350+0 \
  --save-state $W/pp/state2000_R4b_n128_pp.npz --save-model $W/pp/model_R4b_n128_pp.npz
```

**B2: tangent growth.** Uses the harness in `W/stage1/lib_ad/` (the one behind `W/stage1/tlm/growth/SUMMARY.md`, which also used `--remat axis`), migrated per P4, including its path coupling: without that it imports `~/jf1uids` and the stage-1 `_common`. `--years` takes one float, so loop:
```bash
for y in 2 5 11 22; do for m in approx exact; do        # + 42 for the pre-2000 question; + --n 128
  ./run.sh $W/stage1/lib_ad/adj_vjp_check.py --state $W/plH_n256_age364yr_solarcsm.npz --n 64 --x64 --remat axis \
    --years $y $([ $m = exact ] && echo --no-frozen) --out $W/pp/growth_g64_${y}yr_${m}.json
done; done
```
The state is the fit-H state at 2000, so "42 yr" probes 2000→2042, only a proxy for a 1978→2020 window. A state dumped at 1978 from a background run is the proper test.

**B3: refit, then 128³ 4D-Var.**
- Refit: `W/ers/jet/run_R4.sh`, with `--fit --steps 6`, `--free` from `xfit_R4b.json`, and new output names.
- 4D-Var: the production-R2 settings (`W/stage4/run/run_4dv_R2.sh OUTDIR`) on the new 2000 state. That script hard-codes `--state $W/state2000_R2_n128.npz --jac $W/xfit_R2_jac.npz` in `COMMON`; edit both. For a jet background, add `--outline-mask inner-arc+jet --pm-mask-extra 300+140+150+340+350+0` as in the 448³ script. Its second step, `W/stage3/validation/eval_model.py`, hard-codes the old showcase path (P4).

**C2: 448³ jet background.**
- Template: `W/ers/n448/run_bg_R4b_g4.sh`; the IC comes from `make_ic_R4b.sh`.
- **Before regenerating, decide the box and resolution:**
  - At 448³ in 6.125 pc, r_FS(2000) = 163.0″, ≈ 2 % above the data's 159–160″. The suspect is the 1.034 resolution correction `C_L`.
  - The jet cones reach 2.90–3.03 pc against a half-width of 3.06 pc; 3e-3 M☉ wraps around; the jet-image ring at 195–215″ lies outside the box.
  - Recommended: a **≥ 7 pc box**. 512³ in 7.0 pc keeps the 448³ cell size (7/512 = 6.125/448 = 0.0137 pc).
```bash
./run.sh casa_xfit.py --ic <IC_n448_or_n512_solarcsm.npz> --theta $(cat <theta_bg.txt>) --doppler --spectra \
  --obs v2 --responses ciao --exclude-epochs 2019 2022 --jet on --outline-mask inner-arc+jet \
  --pm-mask-extra 300+140+150+340+350+0 --gpus 4 --similarity on --state-only --save-state $W/pp/state2000_R4b_nXXX.npz
```

**D: 448³ 4D-Var.**
- Template: `W/ers/n448/run_4dv_R4b_jet_g4.sh` minus `--positivity` and `--resume`.
- Optionally warm-start the control from the old checkpoint with an **empty** L-BFGS memory: `--init-z ckpt.npz`. J changes with the solver, so it is never a resume. This only works on the **unchanged** 448³ / 6.125 pc grid: z has n = 112,394,264 components, and a 7 pc box or 512³ changes the control size.
```bash
./run.sh casa_4dvar.py --gpus 4 --state $W/pp/state2000_R4b_n448.npz --jac $W/xfit_R4b_jac.npz \
  --pm-train-only --holdout 2019 2022 --run --windows 2004.5 2009.9 2014.5 2018.5 --iters 15 25 25 35 \
  --ckpt 4 --remat-chunks 8 --fine-ctrl-ell 0.5 --fine-ctrl-sigma 0.3 --fine-block 4 --struct-sectors 12 \
  --spec-broadening off --outline-mask inner-arc+jet --pm-mask-extra 300+140+150+340+350+0 \
  --tangent approx --optimizer jax --maxcor 10 --out-dir $W/pp/run_4dv_R4b_n448
```

On the **first evaluation**, check:
- GPU memory ≤ ~75 GB/GPU;
- flat `host_rss_GB` in `evals.jsonl`;
- a finite J and |g|.

Fallback optimiser: `--optimizer scipy`. Its maxcor is auto-capped at 6 at n = 1.12e8.

**Sanity check (`--taylor`):** `W/ers/s0fix/run_taylor448.sh` with `--windows 2004.5 --taylor --tangent approx --dirs 1`.

### 4.3 What a fine-structure (512³+) 4D-Var needs before it is worth running (AUDIT §7)

- **A finer control.** The 14″ control smoothing passes only 3e-8 of the amplitude at 15″.
- **A multi-scale likelihood:** 8–16″ blocks, per-sector structure statistics.
- **A fix for the v2 observation chain's 2.4× fine-contrast loss.**
- **Fine-scale terms limited to what is predictable:**
  - feature proper motions (~10″ cross-correlation);
  - pixel brightness only between epochs ≤ 1–2 yr apart.

### 4.4 Should the evolution start ≥ 20 yr before the first observation (t0 ≈ 1978–1980)?

**Recommendation:**
- **Yes, but only as a coarse, archival-data-anchored extension, never as a single strong-constraint control at 1980.**
- **Pilot it at 128³ before any 448³ use.**

**The lead's position, which this hand-off adopts:** a pre-2000 start is worthwhile only with **either** (a) archival constraints **or** (b) a weak-constraint/model-error formulation at the first epoch. Without either, the post-2000 data constrain the earlier state only through chaotic dynamics.

**Why.**
1. **Information.**
   - Without pre-2000 data, the 1980 state is seen only through M_{1980→2000}.
   - Fine scales (≲ 10 cells ≈ 8.5″ at 448³) are unidentifiable: the exact TLM grows ×10 per ≲ 0.15 yr at 128³ and ×1.1e13 over 22 yr at 64³.
   - Coarse modes pass through the contractive approximate TLM (0.97 / 0.92 / 0.84 / 0.74 at 2 / 5 / 11 / 22 yr). Information at 1980 is therefore *damped*, not amplified.
   - Over 22 yr the approximate JVPs already differ from finite differences by factors of 0.5–2.7 depending on direction. A 40-yr window will be worse.
2. **Little physics gained for free.**
   - In the production 2000 state, only **0.9 %** of the hot emission measure was shocked after 1980 (4.8 % of the shocked mass). Re-computed by the verifier from `time_since_shock / shocked_fraction`: the 4.8 % (and 2.3 % / 10 % for 10 / 40 yr) reproduce exactly. The EM fraction depends on the "hot" cut: 1.8 % for ρ²·f_sh without a cut, ≤ 0.02 % above the median shocked temperature. So the robust statement is "≲ 2 % of the emission measure".
   - The background run from the rescaled start (R4b: ≈ 176 yr after 1674.8, i.e. ~1851) already supplies the ionisation history.
   - Coarse increments have sound-crossing times of ~300 yr, so a 20-yr pre-roll does not balance them either.
3. **Cost.**
   - ×2.2 per gradient for t0 = 1978 (1978–2018.5 vs 2000–2018.5).
   - At 448³, 590–1090 GPU-h if every stage carries the pre-roll, against 220–410 GPU-h without.
4. **What it does buy:**
   - archival terms: Einstein HRI 1979 and ROSAT HRI 1995/96, ≈ 4″, with a 1979→1996 expansion of **0.200 ± 0.006 %/yr** (Vink+98). Radio knots 1978–1990 (Anderson & Rudnick 95) serve as validation only.
   - a backward hindcast test;
   - a shock-history product r_FS, r_RS(t, PA) with posterior bands.
   - **Limit:** the archival expansion pins the deceleration only to below 1σ (predicted −2.5 % against 3 % errors). It does not resolve the age/energy/wind tension; it is a 3 % check of the absolute expansion rate.

**Design (from `design.md` §1.3; it is formulation (b), with (a) as a hindcast).**
- **Two control blocks, separated by scale:**
  - **at t0 = 1978.0:** the globals, the CSM Y_lm and coarse ejecta modes only (32³ control, ≥ 30″);
  - **at 2000:** the existing fine increment (≤ 30″, tapered). This is the model-error/increment at the first epoch that absorbs what the dynamics cannot carry.
  - The scale split makes the two spaces nearly orthogonal and removes the degeneracy that broke the stage-3 weak constraint.
  - Expected benefit (the design reviewer's interpretation): the stage-2 4D-Var on the 2000–2004 window improved training but worsened held-out image-temporal 326 → 359 (`W/stage2/4dvar_RESULT.md`). The design reads this as fitting the first epochs with a non-dynamical increment. The source report itself says only that four years of data do not constrain 2019–2022, and longer windows then fixed it (421 → 337).
- **Alternating optimisation:**
  - the fine block reuses the cached 2000 state;
  - the coarse block runs the full 1978–2018.5 adjoint every 3–5 outer iterations;
  - about +60 % cost at 448³ instead of ×2.7.
- **Pilot at 128³:** ≈ 6 GPU-h per schedule, **20–30 GPU-h** including a twin experiment.
  - Hold out Einstein/ROSAT as a hindcast.
  - **Pre-registered adoption rule for 448³:**
    - (a) the 2019/2022 held-out χ² is not worse than the t0 = 2000 MAP by more than its jitter (±2);
    - (b) the Einstein→ROSAT per-sector expansion hindcast beats the background trajectory's by > 2σ.

    If (b) fails, keep t0 = 2000 and use the archive only as validation.
- **Prerequisites:**
  - G2 passed at **≥ 42 yr**, at ≥ 128³ and ideally from a 1978 state: approx TLM bounded and P2 decided;
  - P1;
  - PP forward robustness over 1978–2000 with the jet and the near-vacuum edge (the 448³ jet NaN was fixed by `conservative`, which no longer exists);
  - an Einstein/ROSAT HRI response (a broad 0.1–2.4 keV band table) in `casa_jaxobs`, plus attitude/plate-scale nuisances;
  - a new `casa_xfit --state-year` option. `--state-only --evolve-years` currently stamps `epoch_year` = the first data epoch (2000), and `casa_4dvar` reads t0 from `epoch_year`: a mislabel trap;
  - a two-block control in `casa_4dvar` (new code; `--wc` is the closest existing path);
  - downloading the Einstein/ROSAT HRI data, which are not on disk.
- **Better value per effort, available now with t0 = 2000:**
  - the Fesen+25 optical reverse-shock velocities, using their HST 1999–2022 part as an r_RS(t, PA) term at 15 locations;
  - extending the window *forward* (Chandra 2023, XRISM 2024).
- **1950s starts:** only at 128³–256³, globals plus l ≤ 4, with the full Fesen+25 1951–2022 baseline. Not at 448³.

---

## 5. Second stage: gradient-based Bayesian modelling

### 5.1 Constraints

- ~1.1e8 controls at 448³ (n = 112,394,264), 46–56 refit parameters (R2: 46 with 34 free; R4b: 56 with 42 free), 14 4D-Var globals.
- The data-informed rank is expected to be O(10²–10³).
- Gradients are surrogates:
  - globals are exact (Taylor 0.97–1.04);
  - state directions range 0.47–1.25;
  - J is rough at small scales: ±0.1 GPU non-determinism at 18 yr, plus spike gradients at reverse-shock gate cells.
- A 448³ full-window gradient costs **2.3–4 GPU-h**. Field-level HMC at 448³ (1e4–1e5 gradients) is therefore out.

**Two facts shape the method.**
- **HMC with biased gradients stays exact:** the accept/reject step uses the true J, so the bias only lowers acceptance.
- **The rough small-scale J must leave the target:** replace it with ensemble-calibrated statistics and an R_chaos(scale, lead time) representativeness covariance.

### 5.2 Staged plan (`design.md` §2.3)

| Stage | Content | GPU-h |
|---|---|---|
| B0 | Chaos/representativeness ensembles at 128/256/448³ (perturb < 10 cells at 1e-3 at 2000, run to 2022) → R_chaos, Q, horizons, the J noise floor | 35–55 |
| P1 (alt.) | Multi-incremental 4D-Var: outer loop nonlinear at 448³, inner Gauss-Newton CG at 224–256³. Gives the MAP plus the Lanczos Hessian eigenpairs | 80–150 (vs 220–410 for plain L-BFGS) |
| B1 | Low-rank Laplace (Gauss-Newton HVPs at 256³, r = 200) + marginal of the globals. jvp-of-vjp at 448³ would need > 140 GB/GPU | 10–115 |
| B2 | RML/EDA: 24 perturbed 4D-Vars at 256³ + IEnKS cross-check of the approximate gradients at 128³ | 135 |
| B3 | NUTS/MCLMC on the globals plus the likelihood-informed subspace at 128³ (mass matrix from B1), plus a fitted 128³ → 448³ discrepancy δ(θ) from ~50 448³ forwards | 270–520 |
| B4 | Summary-statistic likelihood for fine structure (per-sector band power at < 5 / 5–15 / 15–40″, filament statistics, topology χ), with an emulator and a ≥ 900³ forward check | 120–190 |
| B5 | Forcing-form stochastic weak constraint at 256³ | 30 |
| **Total** | core | **≈ 0.75–1.3k**; 1.1–1.8k with the L-BFGS P1 and the 448³ pre-roll; ≈ 2.5–3k with the optional 448³ ensembles |

**Order:** B0 before B1–B3, and B1 before B3. Without B0 the posterior is overconfident and the rough J breaks the samplers.

**Pre-registered checks:**
- Laplace vs RML marginal standard deviations of the globals agree within 30 %; otherwise B3 is mandatory.
- Stop every optimiser at the chaos floor.
- The held-out 2019/2022 data (and the hindcast, if adopted) fall inside the 68/95 % predictive bands at about the nominal rates.
- The discrepancy residual is < 1 σ_chaos.

---

## 6. Known pitfalls

### Cluster and operations
- **Node health.**
  - A failed fabric manager or `cuInit` (`CUDA_ERROR_SYSTEM_NOT_READY`) makes JAX **fall back to CPU silently**: assert the GPU platform in every job.
  - compgpu14 wrote corrupted outputs twice; refuse such nodes.
  - Single-GPU H200 runs are not bitwise reproducible; A100 and 4×H200 runs are.
- **Shared-node etiquette (compgpu12 rules, keep them anywhere shared):**
  - our jobs ≤ 4 H200 per node in total;
  - never stack two multi-GPU jobs;
  - `ulimit -c 0`;
  - host-RSS logging and the 300 GB cap.
  - Resubmitting after an incident needs the user's OK.
  - The ≤ 4-GPU rule conflicts with the 8-GPU phases (D on 8, E, 512³+): run those only on nodes the scheduler allocates exclusively to us, or after asking the user.
- **pq** (if it is still used): `pq stat/sub/log/cancel`.
  - Never run `pq agent --help`: it blocks.
  - pq sets `CUDA_VISIBLE_DEVICES`; scripts gate autocvd on it.
- **Killing processes:** never `pkill -f` with a pattern that appears in your own command line (it kills the caller). Kill by PID.
- **Storage:** home quota was ~99 %. Big outputs go to the data disk.
- **Compile time:** the first 448³ evaluation took 40 min including compile. Budget ~25 min of compile per stage.
- **Untested gates:** the 224³ memory proxies hung for > 6 h. The 512³ gradient gate (`grad512_h200.sh`) never ran.

### Numerics
- **Positivity.**
  - REDISTRIBUTE created mass; it is gone on PP, but never mix old and PP results in one comparison.
  - PP has no velocity cap. Watch near-vacuum cells (an ≈ 89,000 km/s pocket on the NE axis exists in the 448³ IC, jet on or off) and the dt history.
  - The PP θ uses the total-energy pressure, so first order can fire in cold ejecta. Measure the θ < 1 fraction.
- **Kept mechanisms:** `coldcrush_blend` stays on. It is unvalidated without it at ≥ 512³ with cooling.
- **Shock label:** s0 must be finite (`sanitize_entropy_label`; the loader re-seeds bad labels). A NaN s0 freezes the shock history.
- **Box size:** 448³ in 6.125 pc is marginal for the jet. Use ≥ 7 pc.
- **The `--cpu-test` path is not physics-grade:** native backend, no FCT. On the old library the XLA:CPU compile of the FCT limiter needed > 200 GB.

### AD
- **Exact tangents:** usable only for windows of ≤ 4–5 yr. In f32 they are NaN through PP until P1 lands.
- **Frozen-mode (approx) tangents:** biased (+61 % with dual energy on the probe). Always line-search on the true J.
- **Late-stage stalls:** come from gradient spikes at vacuum-edge cells next to the reverse shock. These are the true derivative; the spike guard is a mitigation only.
- **Sharding.**
  - Use `sharded_roll` (`shard_map` + ppermute). GSPMD's periodic roll gave 906 all-to-alls per 256³ gradient and a wrong gradient at 2 GPUs.
  - Clear JIT caches between alternating sharded and unsharded calls (`UnspecifiedValue.addressable_devices_indices_map`).
- **Optimiser:** scipy L-BFGS-B overflows its 32-bit workspace at n = 1.12e8 with maxcor 10. Use `--optimizer jax`; the scipy path auto-caps maxcor at 6.
- **Checkpoints:** `ckpt.npz` and every J value are tied to the old solver. Never `--resume` across the solver change.

### Code
- **Stale wheel:** run `pip install .` after library edits, or always go through `run.sh` with the right `ASTRO_ROOT`.
- **Merge hazard:** never merge or rebase the old `astronomix/` into `weno-stability` (silent resurrection of the cold-LLF block).

---

## 7. Checklist

- [ ] Kill the stale watcher `watch_restart_jax.sh` (PID 3947020 on compgpu11) by PID.
- [ ] Back up the uncommitted tree (tar), then commit C_e / C_d / C_a on `casa-wip`; `git diff 285eab1 HEAD -- astronomix` is empty.
- [ ] Transfer the ≈ 30–40 GB set (§3); mirror or symlink `/export/data/lstorcks`; recreate astx (jax 0.10.2), xrayobs, soxs_data, atomdb; edit `run.sh` paths.
- [ ] `casa-pp` worktree off `weno-stability`; cherry-pick C_e C_d; resolve `dynamo_convergence.py`; keep the `asym_field` fix.
- [ ] P1 (f32-safe θ derivative) + P4 (script migration, incl. the ~55 work-dir harnesses that hard-code `~/jf1uids` or the stage-1 baseline showcase); `pip install .` or `ASTRO_ROOT`.
- [ ] G0 CPU suite green, including the new f32 + dual-energy + scalar + remat gradient check.
- [ ] G1: 128³ A/B forward (Δχ², mass, energy, dt, θ < 1 fraction, morphology).
- [ ] G2: tangent growth 2/5/11/22 (+42) yr at 64³ **and 128³**, and Taylor tests → decide P2.
- [ ] G3: 128³ refit + 4D-Var; held-out like for like (PP 4D-Var(R2) vs 353.5 s3 / 227.5 s4, same scorer).
- [ ] G4: 2/4-GPU sharding (cosine ≥ 0.9999999, no hang).
- [ ] Decide the box (≥ 7 pc) and r_FS calibration (`C_L`); regenerate the 448³/512³ background with PP.
- [ ] G5: one 448³ gradient (≤ ~75 GB/GPU, flat host RSS, Taylor ≈ 1.005–1.014).
- [ ] D: 448³ 4D-Var fresh (`--optimizer jax`); check the first evaluation.
- [ ] In parallel at 128³: the t0 = 1978 two-block pilot + the Fesen+25 RS term (adoption rule §4.4).
- [ ] Then the Bayesian stage in the order B0 → (P1 multi-incremental) → B1 → B2 → B3, with B4/B5 for fine structure and model error.
