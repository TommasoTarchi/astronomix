# WENO stability without the patchwork

Branch `weno-stability`, worktree `/export/home/lstorcks/jf1uids-weno-stability`.

**Goal:** replace the post-hoc stabilisers that the high-Mach turbulence and
self-gravity runs need by a principled change inside the WENO scheme. The
stabilisers in question: vacuum `prot`, per-stage/per-step hard floors,
`vacuum_rest`, deep-void / cold-crush LLF blending, the Zalesak flux limiter,
and a 0.02 density floor.

All measurements below use **no** positivity machinery
(`minimum_density = 1e-10`, no `prot`, no floors, no blending) unless stated.

## In one paragraph

The bare scheme has two structural defects, both inside WENO:

1. The characteristic basis is evaluated at an interface "state" that is not
   the state of any gas.
2. The split-flux reconstruction is not positivity-preserving: next to a
   low-density cell it is *anti-diffusive* and drains it.

The fix is two options (native kernel and all three Pallas kernels):

* `weno_admissible_face_state`: average the pressure, not the enthalpy.
* `weno_positivity_preserving`: Zhang & Shu (2012) split-flux scaling inside
  the reconstruction. Implies the face state.

With them, Mach-10 driven turbulence completes without any other stabiliser:
isothermal MHD at 64³, 128³ and 256³; adiabatic MHD; isothermal hydro at the
provable CFL. So does the cold Evrard collapse, down to 32³. Measured
side-by-side, the current production recipe survives or aborts at 128³
depending on run-to-run variation.

A third failure, specific to self-gravity, is **not** a WENO defect. The
energy-conserving gravity coupling digs internal-energy holes wherever the
grid does not resolve the pressure scale height. The WENO fixes make those
runs complete and conserve energy, but only resolution or a different
coupling removes the holes.

## Defect 1: the characteristic basis is evaluated at a non-state

`_eigenvector_building_blocks` averaged `rho` and `m`, so `v = <m>/<rho>` is
mass-weighted, while the specific enthalpy `h` was averaged *unweighted*. It
then set `c^2 = (gamma-1)(<h> - v^2/2)`. For two cells of equal temperature
this gives the values below (`face_c2_demo.py`; true `c^2 = 1`):

| velocity jump | density ratio 1 | 10 | 100 | 10^4 |
|---|---|---|---|---|
| 3 c  | 1.75 | 0.02 | **-0.44** | **-0.50** |
| 10 c | 9.3 | **-9.9** | **-15** | **-16** |
| 100 c | 834 | **-1087** | **-1600** | **-1665** |

The values also change in a boosted frame, so the average is not Galilean
invariant. When `c^2 < 0` the clamp sets `inv_c2 = 0`. That **removes the
acoustic upwind correction** and leaves the undissipated central flux at
exactly the strongest jumps.

A cold dense slab rammed at Mach ~800 into tenuous gas (`riemann1d.py`,
`cold_ram`) blows up within ~2 steps. With `c^2 = gamma <p>/<rho>` (enthalpy
rebuilt from it) it gives the exact solution. The change is free elsewhere:

* identical L1 on Sod, Einfeldt 123, near-vacuum, Toro-3 and LeBlanc;
* still 5th order;
* ~+4 % runtime.

Alone it does not stabilise the turbulence: adiabatic beta = 0.1 MHD fails
because of pressure positivity.

## Defect 2: the split-flux reconstruction drains one-cell minima

Bare Mach-10 driven turbulence (HOW-MHD ISM case, `turb.py`) dies at
t/t_c ~ 0.6-0.85, isothermal hydro and MHD alike. The dying cell (`turb_diag.py`)
is a *single* density minimum (rho ~ 0.02 among 0.3-3) whose velocity is
unrelated to its neighbours'. Restarting that state without forcing
(`turb_restart.py`) reproduces the blow-up in ~15 steps: the cell's density
falls 2-3x **per step**, while the local divergence allows ~4 %.

Mechanism:

1. The Lax-Friedrichs speed is set by the low-density cell's large fast speed.
2. That makes the dense neighbours' split fluxes `(F - alpha q)/2` large.
3. The fifth-order extrapolation from the dense side then reverses the split
   flux that should flow into the minimum.

| change on the restart | outcome |
|---|---|
| float64 | NaN at the same time |
| CFL 1.5 -> 0.75 | NaN (later) |
| one splitting speed, no limiter | NaN (later) |
| existing Zalesak limiter (`preserving_flux`) | **density goes negative** (-4.9e3), NaN at the same time |
| LLF in cells with rho < 0.1 (probe) | stable |
| **positivity-preserving reconstruction** | **stable** |

The existing limiter fails for two reasons:

* it checks admissibility per axis with no dimensional factor;
* it bounds the *update*, so at small time steps a gradual drain continues.

### The fix: positivity-preserving split-flux reconstruction

Let `alpha` be the stencil's spectral radius. Splitting field `s` with speed
`alpha_s <= alpha` makes each split flux a scaled vector:

    f^+- = +-(alpha/2) w^+-,   w^+- = q +- F/alpha + sum_s (alpha_s/alpha - 1) R_s L_s q.

The WENO face value `w_hat` is replaced by `w + theta (w_hat - w)`. The
scaling is the largest `theta` in [0, 1] for which both of these are
admissible:

* the face value `w + theta d`;
* its mirror `(q +- F/alpha) - theta d` about the unshifted state.

The forward-Euler update of a cell is then a convex combination of admissible
states for `lambda (alpha_left + alpha_right) <= 1`, i.e. C_cfl <= 1/2 in the
code's sum-of-speeds CFL, or 0.75 with SSPRK4 (Zhang & Shu 2012). The cell's
own flux cancels between its two faces, so face-local speeds are fine. The
density bound is closed form. The pressure is concave along the segment, so
the chord's root is admissible: one closed-form step, no iteration.

`theta < 1` is a first-order candidate entering the WENO combination with a
weight set by admissibility instead of smoothness. In smooth flow `theta = 1`.

**Which splitting speeds.** Fields that carry mass use the common `alpha`.
That makes the splitting in the frozen basis monotone for *every* cell on the
stencil, however badly the face basis represents it (a void with a large fast
speed next to dense gas). Fields that carry no mass keep their own speed:
hydrodynamic shear waves and isothermal-MHD Alfven waves. Their correction
leaves the density alone and changes the pressure by
`(gamma-1) kappa rho dv^2 [1 - kappa rho/(2 rho_w)] >= 0`, so vortical modes
keep the default dissipation.

*Explored and rejected* (commits 25e1312, 96a191f): moving every field's speed
toward `alpha` only as far as needed to keep the split states admissible.
This leaves smooth flow exactly at the default errors, including low-Mach
contacts. But Mach-10 beta-0.1 isothermal MHD then NaNs at t/t_c ~ 1.05 at
128³ and 256³, at CFL 1.5, 1.0 and 0.75 alike, with rho >= 0.02 just before.
Admissibility of the split states is not the monotonicity that the common
speed provides. The rigorous per-field bound (Gershgorin on `L A_m R` over
the stencil) needs a Jacobian product per cell and field.

## Results

### Driven Mach-10 turbulence, beta = 0.1, CFL 1.5, no floors / prot / blending

| run | bare | PP-WENO |
|---|---|---|
| isothermal MHD 64³ | NaN t/t_c = 0.83 | **complete (5 t_c)** |
| isothermal MHD 128³ (native, Pallas x2) | NaN ~0.85 | **complete**, 133 s on an A100 |
| isothermal MHD 256³ (Pallas) | — | **complete** (x2), min rho 9e-4, 33 min |
| adiabatic MHD 64³ / 128³ | NaN 0.50 / 0.60 (face fix alone 0.36) | **complete** |
| isothermal hydro 128³ | NaN / dt collapse ~0.67 | CFL 1.5: NaN 3.4; **CFL 0.75: complete** |

The production recipe (prot + hard floors + `vacuum_rest` + rho_min 0.02),
same setup:

* 128³ native: aborted at 1.55;
* 128³ Pallas: one run completed (max|v| spiked to 12), an identical second
  run aborted at 4.6.

PP-WENO completed every MHD run and resolves physical voids an order of
magnitude below the recipe's floor. Isothermal hydro at Mach 10 evacuates
cells to 1e-9-1e-10, which is genuine for isothermal double rarefactions
(rho ~ exp(-dv/2c)). Their velocity is meaningless, so robustness there needs
the provable CFL.

Statistics (`turb_stats.py`, t > 2 t_c, 256³ PP-WENO as the reference):

| | PP-WENO 256³ | PP-WENO 128³ | recipe 128³ |
|---|---|---|---|
| sigma(ln rho) | 1.19 | 1.16 | 1.32 |
| min rho | 2.2e-3 | 2.4e-3 | 0.02 (floor) |

PP-WENO's density-PDF width converges from below. The recipe's 1.32 lies
*above* the 256³ value, which points at its floor wall and velocity spikes
rather than at physics. Near the grid scale (k >~ 100, the last factor ~3
below Nyquist) PP-WENO at 128³ has less density and kinetic power than the
recipe, and 256³ has more than both: the common splitting speed dissipates
slow magnetosonic modes more. Figure:
`out/stats_f_iso256_vs_f_iso128_vs_pstats_recipe128.png`.

### Cold Evrard collapse (e0 = 0.05, fourth-order conservative gravity, fp32)

| N | bare | face state | PP-WENO |
|---|---|---|---|
| 32 | NaN t = 0.10 (fp64 too) | NaN 0.22 | **complete**, dE/E 1.2e-5 (fp64: 8e-8) |
| 64 | NaN t = 0.25 | — | **complete**, dE/E 3e-5 (min p -0.015, coupling) |
| 128 | complete, dE/E 7e-5 | — | complete, dE/E 7e-5 |

## Defect 3 (not WENO): the conservative gravity coupling

Even completed runs keep cells with `p < 0`:

* 32³: min p = -0.05 at r ~ 0.45 late in the infall;
* 64³: -0.015;
* 128³: -6e-4.

These cells' *total* energy is below their kinetic energy. The coupling charges
the work `F_rho (phi_face - phi_cell)` done on mass *in transit* to the
receiving cell. That mass's momentum is never decelerated, so the work comes
out of internal energy, at first order already. The transported gas can
afford the climb across half a cell only if

    e / rho  >~  |g| dx / 2,

i.e. **the grid must resolve the pressure scale height**. For cold Evrard
(e0 = 0.05, g ~ 1 at the edge) the half-cell climb is:

* 0.0625 at 32³ (criterion violated);
* 0.031 at 64³ (marginal);
* 0.016 at 128³ (satisfied).

This is why the conservative schemes were known to NaN below 128³, and why
warm Evrard (e0 = 0.2) is fine at 32³. Late in the collapse g ~ 5 at r ~ 0.45,
which violates the criterion again at 64³, exactly where the holes are.

The remedies are coupling-level, not WENO-level:

* KE-only source `rho v.g`: positive, but energy error 41 % at 64³ and 20 % at 128³.
* Conservative + dual energy: positive, but dE/E ~ 5 % at 32³, because the
  primitive state carried between steps resets E from g.
* Charge the work to the kinetic energy along g (solve for the momentum
  increment), falling back to internal energy only when that budget is
  exhausted. This changes the momentum source at unresolved edges, so it is a
  physics decision.

## Costs

* Accuracy (smooth advected waves, x64, `smooth1d.py` / `smooth2d.py`): all
  5th order.
  * Shear (vortical) waves: unchanged.
  * Entropy waves (contacts) carry mass and use the common speed, so their
    error grows by (|v|+c)/|v|: 2.3x at Mach 0.77, 14x at Mach 0.08. That is
    ~1.2x / 1.7x in linear resolution. Nothing at high Mach.
* Runtime:
  * native, one x-flux, 64³, A100: ~1.2x for isothermal MHD, ~2.3x for ideal
    MHD (shared GPU, noisy);
  * Pallas, end to end: 128³ isothermal MHD in 133-136 s, *faster* than the
    recipe's 179 s, because no velocity spikes collapse dt.

## Side findings

* 1D finite-difference runs with periodic boundaries converge at **first
  order**: the 1D config keeps too few ghost cells, while 2D/3D switch to
  `PERIODIC_ROLL`. `smooth1d.py` forces the roll.
* Isothermal hydro in 1D does not run (`momentum_index.x` on an int).
* `_lsrk4_with_ct` never calls the flux blending (survey).

## Open

* Default switches: `weno_admissible_face_state` is a bug fix worth making
  the default (identical smooth-flow results). PP-WENO trades low-Mach
  contact accuracy for robustness, so it should be on for supersonic and
  self-gravity runs.
* A cheap, rigorous per-field monotonicity bound would remove the low-Mach
  contact cost.
* The gravity coupling (above).
* The fused Pallas WENO+divergence kernel carries neither option and is
  skipped when they are on.

## Reproduce

All scripts take `WENO_VARIANT=baseline|face|pp`. GPU jobs go through
`stability_lab/run.sh` and `pq`. Large outputs go to
`/export/data/lstorcks/weno_stability`.

    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python stability_lab/riemann1d.py
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python stability_lab/smooth2d.py shear 0.1
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python stability_lab/evrard.py --n 32
    PYTHONPATH=. JAX_PLATFORMS=cpu python stability_lab/pallas_equivalence_mhd.py
    WENO_VARIANT=pp pq sub -t a100 -n 1 --name t128 -- stability_lab/run.sh stability_lab/turb.py --N 128 --backend pallas --tag pp128
    python -m pytest pytests/hydrodynamics/positivity_preserving_weno.py
