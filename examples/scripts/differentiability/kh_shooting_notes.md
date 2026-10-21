# KH shooting study: pre-registered search for a multiple-shooting demonstration

Written 2026-10-10, before any of the runs below. The KH section stays in
the paper only if it can show an honest multiple-shooting (MS) advantage;
otherwise it is dropped. Code: `kh_recon.py`. Earlier findings that motivate
this search: with a terminal velocity observation, single-shooting
Gauss-Newton (SS-GN) recovers 16/16 reporting runs at T = 60 t_g,
15/16 at 80, 10/16 at 100 and 4/16 at 120 (failures: wrong local minima);
constrained Gauss-Newton multiple shooting (MS-GN) rescued 0/4 SS-GN
failures at T = 100.

## Common rules

* Problem as before unless a candidate says otherwise: 2D KH, N = 256,
  M = 8 segments, control p in R^{5x2} (kx = 2..6), inviscid FD WENO-Z,
  Pallas primal, noiseless twin; Gauss-Newton with frozen WENO weights
  (`--frozen-weno-weights`).
* Seeds: tuning = truths {1, 2} x inits {0, 1}; reporting = truths
  {3, 4, 5, 6} x inits {0, 1, 2, 3} (16 runs). Extra single-shooting starts
  for multi-start parity use inits i + 4k (k = 1, 2, ...) of the same truth.
* Equal hyperparameter search. Gauss-Newton: none for either method; both
  variants use the same Levenberg-Marquardt settings (initial damping 1e-2,
  /3 on success, x4 on failure, at most 60 iterations, stop when ten damped
  steps fail). The single MS-specific setting, the continuity weight
  rho_c = 1 of the line-search merit, is fixed here, not tuned.
* Compute: counted in Jacobian evaluations (ten forward-mode tangents over the
  full horizon = 1 unit; for MS the condensed sensitivities are the same ten
  tangents through all segments) plus optimizer wall time on one GH200.
* Headline metrics (feasible): the misfit of the recovered control alone,
  integrated over the full horizon with the method's own observation operator,
  relative to the misfit of the zero control (J/J(0)); the IC error
  ||u0(p) - u0(p*)|| / ||u0(p*)||. Success = IC error < 0.1. Medians, IQRs,
  success counts.
* Multi-start parity: on the reporting runs SS gets K starts per run, with
  K = ceil(median MS optimizer time / median SS optimizer time) measured on the
  tuning runs at the decision horizon; the start with the lowest feasible
  misfit (no truth used) is SS's result for that run.
* Decision rule: MS is demonstrated if, on the 16 reporting runs at the
  decision horizon, it beats SS with multi-start parity in >= 13/16 paired
  comparisons on BOTH feasible metrics (`kh_recon.py decide`).
* Budget: all pilots <= 40 GPU-h. Anything larger is reported first.

## Candidate 1: multi-epoch observations (withdrawn before any run)

Withdrawn by the author on 2026-10-10: with observations at intermediate
times the problem becomes nearly trivial, so it would not demonstrate
multiple shooting. All candidates below observe the velocity field at T only,
as in the paper.

## Candidates 2 and 2b: Adam, single vs multiple shooting (N = 128)

Pre-registered 2026-10-10, after timing only (N = 128, T = 60: per run and
step 1.8-3.6 s single shooting, 1.2-1.5 s multiple shooting, batch 32 to 8).

* Candidate 2b: the paper's problem (kx = 2..6, 10 controls).
  Candidate 2: higher-dimensional control kx = 2..16 (30 controls,
  `--wavenumber-max 16`), truths drawn the same way. Modes with
  k delta > 1 (k >= 8) are linearly stable and weakly observable at T, so the
  IC-error threshold may be out of reach for both methods; the paired
  comparisons still decide.
* N = 128, M = 8, decision horizon T = 60 t_g (the paper's long horizon),
  terminal velocity observation, exact WENO weights (the library default and
  the original campaign's choice), Pallas primal.
* Single shooting: Adam, cosine decay to 5 %, 1980 steps. Multiple shooting:
  Adam on the augmented Lagrangian, 180 x 11 steps, rho0 = 1, x1.6 growth
  (the paper's settings). Equal search: learning rate in {3e-4, 1e-3, 3e-3}
  for both; nothing else tuned.
* Tuning runs (budget-limited): truths {1, 2} x init {0}, all three learning
  rates in one batch per method. Selection per method: lowest median feasible
  J/J(0).
* Go/no-go on the tuning runs: the selected multiple-shooting configuration
  beats the selected single-shooting one on both feasible metrics in 2/2
  runs. Only then the 16 reporting runs (with multi-start parity) are costed
  and proposed; they exceed the pilot budget.
* Pilot cost estimate: about 20 GPU-h per candidate (14 single, 5.5 multiple
  shooting); both candidates about 40 GPU-h with the timing.

## Candidate 3: Gauss-Newton multiple shooting with other M (not run)

At T = 100, GN-MS (M = 8, rho_c = 1) rescued 0/4 single-shooting failures.
Other segment counts (M = 4, 16) cost about 1 GPU-h per run; with the Adam
pilots filling the budget this is not run now, and is proposed only if the
author wants it.

## Amendment (2026-10-10, budget; written before any final pilot result)

Measured single-shooting cost at N = 128 is about 16-18 h per batch of 6
(estimate was 14 h), which puts both candidates at about 48 GPU-h, over the
40 GPU-h pilot budget. Added stopping rule, applied when the multiple-shooting
pilots finish (about 6 h, before single shooting): a candidate whose
multiple-shooting runs recover no tuning run at any learning rate
(IC error >= 0.1 on all six) cannot be a multiple-shooting demonstration, so its
single-shooting chain is cancelled and the candidate is reported as not
demonstrated. A candidate with at least one multiple-shooting recovery
continues to the pre-registered go/no-go; if both continue and the budget
does not cover both, candidate 2b is completed first.

## Candidate 4: multiple shooting with time-reversed nodes (2026-10-10, new session)

Written after the probe below and before any optimisation run of this
candidate. Status of candidates 2/2b: closed (no MS recovery); candidate 3
not run.

### Motivation (probe, no optimisation)

Bock's argument for multiple shooting is that the lifted nodes can carry
knowledge about the state trajectory that single shooting can only enter
through the initial control. With terminal-only data such knowledge can come
from the observation itself: the inviscid Euler equations are invariant under
(rho, v, p)(t) -> (rho, -v, p)(T - t), so the observed terminal velocity (with
the base density and the incompressible Poisson pressure for the unobserved
fields), reversed, integrated forward and reversed again, estimates the state
at earlier times (`reversal_node_estimates`). Probe (N = 256, M = 8, truths 1
and 2, `/e/scratch/astronomix/revision/kh_ms/tools/reversal_probe.py`, job
2263897): relative error of the estimated velocity perturbation at the nodes,
t = 0 ... T in steps of T/8:

| T | node errors (truth 1) | node errors (truth 2) |
|---|---|---|
| 20 | 0.53 0.42 0.40 0.37 0.26 0.17 0.11 0.09 | 0.59 0.47 0.44 0.36 0.23 0.14 0.07 0.08 |
| 60 | 8.7 5.9 2.4 0.76 0.27 0.10 0.03 0.01 | 7.1 5.0 2.0 0.64 0.22 0.09 0.04 0.02 |
| 100 | 11 9.7 3.1 0.66 0.16 0.08 0.06 0.02 | 15 6.6 1.3 0.31 0.15 0.07 0.03 0.02 |
| 120 | 15 8.7 2.3 0.47 0.15 0.10 0.05 0.02 | 16 6.8 2.0 0.67 0.22 0.07 0.04 0.02 |

(The cold trajectory, i.e. no information, has errors ~1 at every node.) The
reversal recovers the late, nonlinear half of the trajectory to 2-20 % and
fails in the linear phase (t < ~40 t_g), where the modes that decay forward
grow backward. The projection of the t = 0 estimate onto the control, the only
way single shooting can use the reversal, has IC errors 4-9 (seed amplitude
4-9x too large: the basin of the single-shooting failures).

### Methods (all Levenberg-Marquardt, frozen WENO weights, the settings of the GN runs above)

* (a) SS-GN from the cold start (`gauss_newton`): existing reporting runs
  (truths 3-6 x inits 0-3) plus extra starts for multi-start parity.
* (b) SS-GN from the reversal control (`gauss_newton --control-start reversal`).
* (c) MS-GN, M = 8, continuity weight 1, interior nodes at t >= T/2 from the
  reversal, earlier nodes on the cold control's trajectory
  (`gauss_newton_multiple --node-start reversal --reversal-fraction 0.5`).
  Nothing of (c) is tuned: M, the weight and the fraction are fixed here.
* (d) ablation, MS-GN with all nodes on the cold trajectory: the existing four
  T = 100 runs (0/4 rescued); not rerun.

### Tuning runs (go/no-go only)

Truths {1, 2} x inits {0, 1} at T = 100 and 120 for (a) and (c), truths
{1, 2} for (b). Go: (c) succeeds (IC error < 0.1) in >= 6 of its 8 tuning
runs. The tuning runs also fix the multi-start parity K = ceil(median
optimizer time of (c) / median optimizer time of (a)) per horizon.

### Decision rule (reporting runs: truths 3-6 x inits 0-3)

The paired-wins rule of the earlier candidates (>= 13/16 paired wins on both
feasible metrics) is not used here: when both methods converge, both end at
round-off-level misfits (J/J(0) ~ 1e-9, IC error ~ 1e-5) and the paired
comparison is a coin flip, so the rule cannot be met even by a method that
never fails. It is still reported. Decision rule for candidate 4, per decision
horizon: MS (c) is demonstrated if (i) it succeeds in >= 13/16 reporting runs
and (ii) it has at least 4 more successes than SS (a) with multi-start parity
(K starts per run, inits i + 4k, the start with the lowest feasible misfit is
SS's result; no truth used in the selection). Decision horizons: T = 120 and
the longest horizon of {160} at which the tuning runs of (c) pass the go
criterion; T = 100 is reported. (b) is reported at every horizon.

Addendum (same day, before any result of (c)): ablation (e), two-stage
single shooting: SS-GN fits the cold control to the reversed velocity at the
first reversal node (t = T/2, over that sub-horizon), and SS-GN on the
terminal data continues from the fit (`gauss_newton --control-start
reversal_node`). It asks whether the gain of (c) needs the lifted problem or
only the node information; it is a fixed-node special case of multiple
shooting, and is reported next to (c) without entering the decision rule.
Tuning runs as for (a), at T = 100, 120 and 160.

Amendment (tuning stage, after the first tuning result of (c), before any
reporting run): the first (c) run (T = 160, truth 2, init 0) stalled after two
accepted steps (merit 2128 -> 1709, IC error 1.35 -> 1.47). Cause: the run
starts with a data misfit of ~0 (the reversed nodes reproduce the
observation) and one large defect at the junction t = T/2; Bock's condensed
step closes every defect with a full linear Newton step, which overwrites the
reversed node with the linear prediction of the cold trajectory, i.e.
discards the node information at once. Added (c'): the same with Bock's
relaxation strategy for large start defects (matching conditions
r_j - (1 - kappa) r_j^0 = 0, kappa from 0.1 in steps of 0.1 per accepted step
to 1; `--relaxation-initial 0.1 --relaxation-increment 0.1`), fixed here, not
tuned further. (c) and (c') both finish their tuning runs; the go criterion
applies to (c'), and whichever of (c), (c') passes it with more tuning
successes is the reporting configuration (tie: (c')).

### Outcome of candidate 4 (tuning runs, truths 1-2 x inits 0-1; IC error < 0.1 = success)

| T | SS cold (a) | SS reversal control (b) | two-stage (e) | MS (c) | MS relaxed (c') |
|---|---|---|---|---|---|
| 100 | 3/4 | 0/2 | 4/4 | 2/4 | 3/3 (t2 i1 pending) |
| 120 | 0/3 (t2 i1 pending) | 0/1 | 3/4 | 0/3 (t2 i1 pending) | 0/2 (t2 pending) |
| 160 | 1/2 (2 pending) | 0/2 | 1/1 (3 pending) | 0/4 | 0/3 (t1 i1 pending) |

Go criterion (>= 6/8 for (c') or (c)) failed for both: **candidate 4 is not a
multiple-shooting demonstration**; its reporting runs are not run. The
multiple-shooting runs that fail end in the same wrong basin as single
shooting (feasible J/J(0) ~ 1e-2, IC error 1-2), also with the relaxation,
which closes the junction defect (842 -> 0.8 at T = 120) but lets the control
converge to the wrong minimum. The reversal information helps only through
(e): fitting the control to the reversed state at t = T/2 first (IC error
0.3-1.4 after that stage) and then to the observation. (e) is also cheaper
than (a) (total optimizer time 0.4-0.85 ks vs 0.5-1.8 ks at T = 100-120).

## Candidate 5: two-stage reversal-informed single shooting vs single shooting (pre-registered here, before any reporting run of (e))

Not a multiple-shooting method; registered so that the author can decide
whether the KH section is rewritten around long-horizon reconstruction or
dropped. Methods: (e) and (a) exactly as above (frozen WENO weights,
Levenberg-Marquardt settings of the GN runs, reversal fraction 0.5, nothing
tuned). Reporting runs: truths 3-6 x inits 0-3 at T = 100, 120 and 160 for (e);
(a) has them at T = 100 and 120 (job campaign of 2026-10-09) and gets them at
T = 160. Multi-start parity: K = ceil(median total optimizer time of (e) /
median of (a)) on the tuning runs per horizon = 1 at T = 100 and 120 ((e) is
cheaper). Decision (per horizon, decision horizon T = 120; 100 and 160
reported): (e) is better if it succeeds in >= 13/16 runs and in >= 4 more
runs than (a).

### Final tuning counts of candidate 4 (all 4 runs per cell done)

| T | (a) SS cold | (b) SS reversal control | (e) two-stage | (c) MS | (c') MS relaxed |
|---|---|---|---|---|---|
| 100 | 3/4 | 0/2 | 4/4 | 2/4 | 4/4 |
| 120 | 0/4 | 0/2 | 3/4 | 0/4 | 0/4 |
| 160 | 1/4 | 0/2 | 2/4 | 0/4 | 0/4 |

Go criterion (>= 6 of the 8 runs at T = 100, 120): (c) 2/8, (c') 4/8, both fail.
Candidate 4 closed: no multiple-shooting demonstration.

### Outcome of candidate 5 (reporting runs, truths 3-6 x inits 0-3)

| T | (e) two-stage | (a) SS cold (K = 1) | median optimizer time (e) / (a) |
|---|---|---|---|
| 100 | 16/16 | 10/16 | 643 s / 932 s |
| 120 | 16/16 | 4/16 | 764 s / 1607 s |
| 160 | 4/16 | 2/16 | 2306 s / 2777 s |

Decision horizon T = 120: (e) succeeds in 16/16 (>= 13) and in 12 more runs
than (a) (>= 4): **(e) is better than single shooting** by the pre-registered
rule; also at T = 100. Stage 1 lands on the same control from every cold
start of a truth (IC error after stage 1 identical to 3 digits across inits),
i.e. the reversed node removes the init dependence. Both fail at T = 160
(the node at t = 80, whose sub-horizon fit itself falls into wrong minima).

## Candidate 6: multiple shooting with anchored reversal nodes (2026-10-11, before any run)

Why (c)/(c') do not reproduce (e): in (e) the reversed node is a fixed target
and only the control moves (stage 1); in the lifted problem the junction
defect can be removed by moving either side, and Bock's condensed step moves
the node (an exact linear Newton step on the constraint), which discards the
reversal information. Bock's relaxation schedules *how much* defect is closed,
not *which side* closes it. The schedule that contains (e) is one on the nodes:
an anchor term w/2 sum_j ||v(s_j) - v(b_j)||^2 on the reversal nodes, w large
first (nodes held: stage 1, solved by the lifted GN) and decayed to 0 (nodes
released: the plain lifted problem). Settings fixed here: w0 = 10 (data units),
x0.7 per accepted step, continuity weight 1, M = 8, reversal fraction 0.5;
(f) early nodes lifted, (f') early nodes re-simulated from the control after
every accepted step (`--early-restore`). Tuning: truths 1, 2 x inits 0, 1 at
T = 120 and 160; go criterion for a reporting set as for candidate 4
(>= 6/8 at T = 120 and 160 together, and at least the tuning count of (e)).

### Outcome of candidate 6 (2026-10-11)

All eight tuning runs ((f) and (f'), T = 120 and 160, truths 1-2 x inits 0-1)
accepted no step (IC error and misfit unchanged; 10 damped trials rejected).
Not a bug but the structure: Bock's condensed step satisfies every continuity
condition exactly in its linear model, so the junction node always moves to
the linear prediction of the cold control's trajectory; with an anchor on that
node the junction mismatch only moves from the defect term (weight 1/2) to the
anchor term (weight w/2), and every trial raises the merit. No schedule of a
node anchor (nor of Bock's relaxation, candidate 4) can make the exact-
constraint method "hold the node and move the control": the two-stage method
(e) is multiple shooting with the junction continuity as a *penalty* and the
late nodes frozen in its first phase (a block-coordinate solution of the
penalised lifted problem), which the condensed Gauss-Newton method cannot
represent. Candidate 6 closed; no reporting runs.

## Candidate 7: penalty multiple shooting with a continuity-weight schedule (2026-10-12, before any run)

Author request (2026-10-12): run the lifted formulation that is still missing,
and try longer horizons with it. Candidates 4 and 6 showed that Bock's
condensed step, which satisfies the linearised matching conditions exactly,
always closes the junction defect by moving the reversed node; the two-stage
method (e) is multiple shooting with continuity as a *penalty*, solved
block-wise. The lifted formulation that contains (e) is penalty multiple
shooting: merit 1/2 ||data residual||^2 + w/2 sum_j ||Phi_h(s_j) - s_{j+1}||^2,
with the weight w raised from small to large over the iterations (a schedule
on the slack of the matching conditions, the author's suggestion of
2026-10-11). With w small a defect can be closed from either side, so the
early chain and the control can move towards the reversed node; as w grows the
lifted trajectory is tightened onto a feasible one.

Method (g), `kh_recon.py run --method gauss_newton_penalty`: Gauss-Newton
(Levenberg-Marquardt) on the penalty merit. Each node moves in the subspace of
Bock's condensed step, node j by T_j a_j + c_j beta_j (T_j: the control
tangents carried along the current nodes; c_j: the carried defects), but every
node has its own coefficients (a_0 = the control step). Bock's step is the
member a_j = a_0, beta_j = 1, so the method tends to Bock's as w -> infinity,
and one iteration costs Bock's forward-mode sweep (11 tangents per segment).
Damping, line search and stopping rule as `gauss_newton_multiple`. Phase 1
(lifted) ends with the first accepted step at w_max; phase 2 is single-shooting
Gauss-Newton from its control (at most 60 iterations, the settings of (a)),
exactly as stage 2 of (e). Nodes start as in (c): t >= T/2 from the time
reversal, earlier ones on the cold control's trajectory.

Settings, fixed here and not tuned: w_0 = 1e-2, x3 per accepted step,
w_max = 1e4 (14 accepted lifted steps); M = 8; reversal fraction 0.5; LM
settings of the GN runs; frozen WENO weights; N = 256. Reasoning: at the start
the data misfit is ~0 (reversed nodes) and the junction defect is O(1)
relative to the state perturbation, so w_0 = 1e-2 starts loose; w_max = 1e4
(10^4 x the continuity weight of (c)) hands phase 2 a nearly continuous
trajectory. One amendment of w_0 is allowed before any reporting run, only if
the lifted phase fails as an algorithm (no accepted step, or the junction
defect not reduced 10x by the end of phase 1); it is recorded here.

Implementation checks (not results; job khpen_smoke, N = 64, T = 40, M = 4):
at w = 1e8 the first step must give a_j = a_0 and beta_j = 1 (Bock's step), and
the merit change along a short step must match the Gauss-Newton model.

Tuning runs (go/no-go): truths {1, 2} x inits {0, 1} at T = 120 and 160.
Go: (g) succeeds (IC error < 0.1) in >= 6 of the 8 and in at least as many as
(e) on the same runs (e: 3/4 at T = 120, 2/4 at T = 160, so >= 5).

Reporting runs (only on go; ~55 GPU-h, reported to the author before launch):
truths 3-6 x inits 0-3 at T = 120 and 160. Decision horizon T = 160, where (e)
has 4/16 and (a) 2/16: (g) is a multiple-shooting demonstration if it succeeds
in >= 13/16 and in >= 4 more runs than (e) and than (a) with multi-start
parity (K from the tuning runs as before). T = 120 is reported.

Longer horizon (exploratory, author request; reported, no decision): T = 200,
tuning seeds, for (a), (e) and (g).

Budget of this candidate before the report: tuning at T = 120/160 (8 runs) and
T = 200 (12 runs), about 35 GPU-h.

### Implementation checks of candidate 7 (2026-10-12/13, before any tuning run)

Round 1 (job khpen_smoke 2294509, N = 64, T = 40, M = 4): the model check
passes (merit change / Gauss-Newton model 0.9994-0.9996 at w = 1e8,
0.990-0.992 at w = 1e-2), so the linearisation is assembled correctly. The
coefficient check as worded above fails at face value (|a_j - a_0| / |a_0| up
to 2.2, beta_j 0.92-0.96 at w = 1e8). Reason: the Levenberg-Marquardt damping
(1e-2 x diag) acts on every node coefficient, whereas Bock's method damps the
control step only, and the control tangents T_j of the later nodes are nearly
dependent, so the damping moves the coefficients along directions the merit
hardly sees. Without damping the only member of the step subspace with zero
linearised defects is Bock's (a_j = a_0, beta_j = 1), so the check is redone
undamped and compared with Bock's condensed step itself, in control space and
node by node in state space (`KH_PENALTY_CHECK=1` now prints both).
Round 2: job khpen_smoke2 (2303365; damping 1e-12 and 1e-2 at w = 1e8, and
w = 1e-2). The tuning runs (`penalty_tuning.sh`, jobs 2303440-59) wait for it
and are cancelled if the undamped step differs from Bock's by more than the
conditioning of the problem allows.

Round 2 result (khpen_smoke2): undamped (1e-12) at w = 1e8 the step is Bock's:
|a_j - a_0| / |a_0| <= 2.7e-9, beta_j = 1, control step within 1.8e-3 of Bock's
condensed least-squares step, node steps within 1-2e-3. With the default damping
the step differs (control 0.97, nodes 0.5-1.7), as explained above. Check passed;
the tuning runs are valid.

### Outcome of candidate 7 (2026-10-13, tuning runs, IC error < 0.1 = success)

| T | (a) SS-GN cold | (e) two-stage | (g) penalty MS | median optimizer time (a) / (e) / (g) |
|---|---|---|---|---|
| 120 | 0/4 | 3/4 | 1/4 | 1366 / 385 / 873 s |
| 160 | 1/4 | 2/4 | 1/4 | 2676 / 1424 / 3404 s |
| 200 (exploratory) | 0/4 | 0/4 | 0/4 | 3509 / 1534 / 4413 s |

Go criterion (>= 6 of 8 at T = 120 and 160, and >= (e)'s 5): (g) 2/8 -> **no go**;
no reporting runs. The lifted phase worked as an algorithm in every decision-horizon
run (all 14 continuation steps accepted, w_max = 1e4 reached, junction defect reduced
94x to 6e5x), so the allowed w_0 amendment does not apply. It ends on the wrong
control: IC error after the lifted phase 0.49-2.5 (the two successes, truth 1 init 0
at T = 120 and 160, are rescued by the polish from 0.49 and 1.02). Both successes of
(g) are runs that (e) also solves. At T = 200 the lifted phase no longer closes the
defect (5-9x in three of four runs) and every method fails, (e) included (its
stage-1 fit to the node at t = 100 falls into wrong minima, as at T = 160).
Candidate 7 closed: no multiple-shooting demonstration. Outputs:
`revision_data/kh_shooting/reversal/tuning/gauss_newton_penalty__*`, logs `runs/khpen_*`.

## Diagnostic for a candidate 8: does the anchored late chain sharpen the reversal target? (2026-10-14, before any run)

Idea (with the author, 2026-10-13): multiple shooting needs something that pins
the nodes; with terminal-only data that can only be the time reversal, so the
nodes should be uncertain pseudo-observations with a trust schedule (anchored
penalty multiple shooting: anchors lambda_j on the reversed velocity, continuity
weight w; lambda large -> small). Its limits are (e) (one node, lambda = inf,
solved in sequence) and candidate 7 (lambda = 0). The one mechanism by which it
could beat (e): (e) fails from T = 160 on because the reversed target at T/2 is
inaccurate (the stage-1 horizon of 80 alone is easy: SS-GN 15/16 at T = 80); a
lifted late chain that is dynamically consistent with the more accurate later
nodes and with the observation might correct that target before the control is
fitted to it. Candidate 7 could not, because its nodes moved only along the
control tangents of the cold (linear-phase) trajectory; here the late nodes
move in full state space (L-BFGS, reverse mode, Pallas adjoint, frozen WENO
weights), with the control out of the loop.

Script `kh_node_refinement.py`: late nodes t = 80..140 (M = 8, T = 160, N = 256)
start at the reversal; stages lambda = 1, 0.1, 0.01, 0 (w = 1), 40 L-BFGS
iterations each; after each stage the velocity error of every late node against
the truth relative to the true perturbation there; then (e) with the raw target
and with each stage's target, init 0. Seeds: tuning truths 1, 2 and fresh truths
7-10 (the reporting truths 3-6 stay untouched).

Decision, fixed now: the mechanism exists if some stage lowers the t = T/2 node
error by >= 20 % relative to the raw reversal in >= 4 of the 6 truths AND the
two-stage method with that stage's target succeeds on at least one truth where
the raw target fails, losing none that the raw target solves. Only then is a
candidate 8 pre-registered (schedule fixed on these six truths, decision on the
reporting truths 3-6 at T = 160 against (e)'s 4/16). Otherwise the hybrid cannot
beat (e) and the idea is closed.

### Outcome of the diagnostic (2026-10-14): mechanism absent, idea closed

Implementation check (N = 128, T = 160): the Pallas-adjoint gradient of the late
chain matches the native-tangent gradient to ~1e-3 along three random directions
(the remaining gap to central differences is the frozen-weight linearisation);
the Pallas adjoint is ~5x faster (new opt-in `KHParams.pallas_adjoint`).

N = 256, T = 160, 4 stages x 40 L-BFGS iterations (jobs 2313744-46, outputs
`/e/scratch/astronomix/revision/kh_ms/node_refinement/n256/`):

| truth | t = T/2 node error raw -> best stage | its misfit at T after forward integration | continuity reduced |
|---|---|---|---|
| 1 | 0.279 -> 0.280 | 0.027 | 3.3x |
| 2 | 0.371 -> 0.371 | 0.038 | 4.6x |
| 7 | 0.382 -> 0.382 | 0.031 | 5.8x |
| 8 | 0.482 -> 0.482 | 0.050 | 4.0x |
| 9 | 0.272 -> 0.273 | 0.046 | 6.1x |
| 10 | 0.382 -> 0.382 | 0.032 | 7.4x |

(node errors: velocity error relative to the true perturbation at that time; the
misfit column has the same normalisation at T). The anchored chain becomes
dynamically consistent and fits the observation better, but the T/2 target does
not move towards the truth in any truth (0/6 with >= 20 %; criterion >= 4/6), so
by the rule above no downstream runs and no candidate 8. N = 128 (truths 1, 2)
gave the same picture.

Why (the structural reason, now measured): time reversal is accurate in the
directions that grow forward (they contract backward) and wrong in the
directions that decay forward (they grow backward, fed by numerical dissipation
and the missing terminal density and pressure). Forward consistency and the
terminal datum see only the former. The reversed T/2 state is 27-48 % off, but
integrated to T it misses y by 3-5 %: ~90 % of the target error is invisible
from T, and no lifting of the late chain can recover it. With terminal-only data,
multiple shooting therefore has nothing to add to the reversal-informed
two-stage method; it needs information at the nodes about the forward-decaying
directions, i.e. intermediate data.

## Where the reversal error lives (2026-10-15, analysis, `kh_reversal_error.py`)

N = 256, truths 1, 2, 7-10, node t = T/2 (outputs
`/e/scratch/astronomix/revision/kh_ms/reversal_error/T{120,160}.json`). At
T = 160 the velocity error is large-scale (|k| <= 8: 78-92 % of its energy;
error/perturbation 0.24-0.36 at |k| <= 4, 0.29-0.58 at 4-8, ~1 above 32), the
density error 1.3-1.8x the density perturbation. Each band of the error alone
is visible at T (0.14-0.41 of the perturbation at T for |k| <= 8), the whole
error only at 0.03-0.05: a coherent near-null direction of the forward map,
not dissipated small scales. Corrects the earlier reading ("small scales lost
to numerical dissipation"): coarse node data at |k| <= 8 would constrain it.

Pilot (exploratory, no decision): two-stage method with the T/2 target
low-passed truth (|k| <= 8) + reversal (|k| > 8), T = 160 and 200, truths 1, 2,
7-10, init 0 (`kh_coarse_target.py`, jobs 2333883-2333910).

### Outcome of the coarse-target pilot (2026-10-15/16, jobs 2333883-2333910)

Two-stage method, init 0, T/2 target = reversal (raw) or low-passed truth
|k| <= 8 + reversal |k| > 8 (coarse); entries: target error -> IC error after
stage 1 -> final IC error (success < 0.1).

| T | truth | raw | coarse |
|---|---|---|---|
| 160 | 1 | 0.280 -> 1.44 -> 0.0006 | 0.114 -> 0.23 -> 0.0016 |
| 160 | 2 | 0.371 -> 4.05 -> 1.00 | 0.130 -> 1.58 -> 0.0003 |
| 160 | 7 | 0.382 -> 2.19 -> 0.91 | 0.140 -> 7.41 -> 3.93 |
| 160 | 8 | 0.482 -> 3.35 -> 3.51 | 0.224 -> 0.54 -> 0.0041 |
| 160 | 9 | 0.272 -> 2.38 -> 0.98 | 0.109 -> 5.26 -> 5.06 |
| 160 | 10 | 0.382 -> 2.17 -> 1.98 | 0.108 -> 0.40 -> 0.0005 |
| 200 | 1 | 0.373 -> 1.28 -> 1.29 | 0.098 -> 0.22 -> 0.134 |
| 200 | 2 | 0.236 -> 1.21 -> 1.76 | 0.102 -> 1.21 -> 1.51 |
| 200 | 7 | 0.273 -> 1.57 -> 1.58 | 0.105 -> 8.14 -> 3.70 |
| 200 | 8 | 0.410 -> 1.42 -> 1.12 | 0.139 -> 3.82 -> 2.32 |
| 200 | 9 | 0.581 -> 3.87 -> 4.00 | 0.107 -> 10.5 -> 3.68 |
| 200 | 10 | 0.295 -> 1.79 -> 2.09 | 0.094 -> 8.86 -> 8.38 |

Coarse late information cuts the target error ~3x and lifts the two-stage method
from 1/6 to 4/6 at T = 160; at T = 200 both 0/6. With targets of the same
quality (0.09-0.14) stage 1 itself falls into wrong minima at T = 200 (stage-1
IC errors up to 10): stage 1 is single shooting over T/2 = 100 t_g without data
inside, the length at which single shooting fails even with exact data
(SS-GN 10/16 at T = 100). The coarse data remove the target bottleneck and
expose the next one, the unpinned front half -- where multiple shooting has
nothing to pin its nodes either.

Next (exploratory): nested N-stage, the coarse-corrected T/2 state reversed
again to T/4 as the pin of a shorter first stage (fits T/4 -> T/2 -> T);
`kh_coarse_target.py --nested`.

### Outcome of the nested N-stage pilot (2026-10-16, jobs 2344801-2344818)

Three stages (fits to T/4, T/2, T), init 0; the T/4 pin is the T/2 state
reversed once more (raw3: the plain reversal, terminal-only; coarse3: the
coarse-corrected T/2 state). Pin errors (T/4, T/2) relative to the true
perturbation, IC errors after the stages -> final (success < 0.1):

| T | truth | pins raw / coarse | raw3 | coarse3 |
|---|---|---|---|---|
| 160 | 1 | 1.97, 0.28 / 0.69, 0.11 | 10.6, 0.92 -> 0.001 | 4.09, 0.23 -> 0.001 |
| 160 | 2 | 2.29, 0.37 / 1.41, 0.13 | 8.26, 6.68 -> 4.11 | 3.57, 2.33 -> 2.65 |
| 160 | 7 | 3.23, 0.38 / 2.39, 0.14 | 8.04, 6.14 -> 3.78 | 6.63, 6.18 -> 4.15 |
| 160 | 8 | 1.35, 0.48 / 1.19, 0.22 | 8.26, 5.94 -> 1.83 | 8.16, 0.54 -> 0.009 |
| 160 | 9 | 1.52, 0.27 / 1.19, 0.11 | 3.62, 3.36 -> 3.62 | 4.13, 0.23 -> 0.001 |
| 160 | 10 | 1.76, 0.38 / 1.01, 0.11 | 8.86, 7.83 -> 3.54 | 4.59, 0.15 -> 0.001 |
| 200 | 1 | 1.48, 0.37 / 0.67, 0.10 | 5.67, 4.20 -> 3.96 | 2.61, 0.23 -> 0.005 |
| 200 | 2 | 1.99, 0.24 / 1.70, 0.10 | 4.55, 7.22 -> 6.20 | 5.09, 10.7 -> 10.4 |
| 200 | 7 | 1.48, 0.27 / 1.33, 0.11 | 6.20, 4.92 -> 3.43 | 6.72, 4.78 -> 4.09 |
| 200 | 8 | 1.01, 0.41 / 0.82, 0.14 | 6.85, 4.23 -> 5.24 | 6.09, 5.71 -> 4.74 |
| 200 | 9 | 2.03, 0.58 / 1.01, 0.11 | 6.11, 4.73 -> 4.81 | 3.38, 0.92 -> 1.09 |
| 200 | 10 | 1.42, 0.30 / 1.13, 0.09 | 7.39, 7.50 -> 5.20 | 5.99, 4.61 -> 4.33 |

Success counts (raw2 / raw3 / coarse2 / coarse3): T = 160 1/6, 1/6, 4/6, 4/6
(coarse2 and coarse3 on different truths); T = 200 0/6, 0/6, 0/6, 1/6.

* No reliable T/4 pin exists: reversed once more, even from the corrected T/2
  state, the T/4 pin is off by 0.7-2.4 of the true perturbation (raw 1.0-3.2);
  T/4 = 40-50 t_g is the end of the linear phase, where the reversal fails.
  The first stage then lands at IC errors 2.6-10.6, worse than the cold start
  (~1); what the T/2 stage recovers afterwards behaves like a restart.
* Terminal-only N = 3 does not help (same counts as two stages) and moves the
  intermediate control away from the truth.
* coarse3 vs coarse2: no systematic gain (swaps truths at T = 160; one success
  at T = 200, the truth that coarse2 nearly solved).

Conclusion of the pin experiments: a pin helps only if it is reliable (T/2
pins at 0.1-0.4 work, T/4 pins at 0.7-3 do not), and the pins must be spaced
within the reach of single shooting (~80-100 t_g). Time reversal (with or
without coarse late data) supplies reliable pins only in the late nonlinear
phase, so a front half longer than that reach (T = 200) stays unpinned, for
continuation and for multiple shooting alike; only information about the
early phase would pin it.
