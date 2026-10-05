# WENO stability without the patchwork

Branch `weno-stability`, worktree `/export/home/lstorcks/jf1uids-weno-stability`.
Goal: replace the post-hoc stabilisers needed by the high-Mach turbulence and
self-gravity runs (vacuum `prot`, per-stage/per-step hard floors,
`vacuum_rest`, deep-void / cold-crush LLF blending, the Zalesak flux limiter,
large density floors) by a principled change inside the WENO scheme.

Everything below was measured with **no** positivity machinery switched on
(`minimum_density = 1e-10`, no `prot`, no floors, no blending) unless stated.

## Result in one paragraph

The bare scheme has two independent structural defects, both inside WENO.
(1) The characteristic basis is evaluated at an interface "state" that is not
the state of any gas. (2) The split-flux reconstruction is not
positivity-preserving: next to a low-density cell it is *anti-diffusive* and
drains it. The fix is two options of the native kernel:
`weno_admissible_face_state` (average the pressure, not the enthalpy) and
`weno_positivity_preserving` (Zhang & Shu 2012 split-flux scaling inside the
reconstruction; implies the face state). With them, Mach-10 driven turbulence
(isothermal and adiabatic MHD, beta = 0.1, CFL 1.5) and the cold Evrard
collapse run to completion without any other stabiliser. A third failure,
specific to self-gravity, is **not** a WENO defect: the energy-conserving
gravity coupling digs internal-energy holes wherever the grid does not resolve
the pressure scale height. The WENO fixes make those runs survive down to 32³
and conserve energy, but only resolution or a different coupling removes the
holes (criterion below).

## Defect 1 — the characteristic basis is evaluated at a non-state

`_eigenvector_building_blocks` averaged `rho`, `m` (so `v = <m>/<rho>` is
mass-weighted) and the specific enthalpy `h` *unweighted*, then set
`c^2 = (gamma-1)(<h> - v^2/2)`. For two cells of equal temperature this gives
(`stability_lab/face_c2_demo.py`, true `c^2 = 1`):

| velocity jump | density ratio 1 | 10 | 100 | 10^4 |
|---|---|---|---|---|
| 3 c  | 1.75 | 0.02 | **-0.44** | **-0.50** |
| 10 c | 9.3 | **-9.9** | **-15** | **-16** |
| 100 c | 834 | **-1087** | **-1600** | **-1665** |

and different numbers in a boosted frame (not Galilean invariant). When `c^2 < 0`
the clamp sets `inv_c2 = 0`, which **removes the acoustic upwind correction**
and leaves the undissipated central flux at exactly the strongest jumps. A cold
dense slab rammed at Mach ~800 into tenuous gas (`riemann1d.py`, `cold_ram`)
blows up within ~2 steps; with the pressure averaged instead
(`c^2 = gamma <p>/<rho>`, enthalpy rebuilt from it) it gives the exact
solution. It is free elsewhere: identical L1 on Sod, Einfeldt 123, near-vacuum,
Toro-3 and LeBlanc, still 5th order on smooth waves, +4 % runtime.

## Defect 2 — the split-flux reconstruction drains one-cell minima

Bare Mach-10 driven turbulence (HOW-MHD ISM case, `turb.py`) dies at
t/t_c ~ 0.6-0.85 at 64³ and 128³, **isothermal hydro and MHD alike**. Forensics
on the last finite state (`turb_diag.py`): the dying cell is a *single* cell,
a density minimum in all three directions (rho ~ 0.02 among 0.3-3), with a
velocity unrelated to its neighbours. Restarting that state without forcing
(`turb_restart.py`) reproduces the blow-up in ~15 steps: the cell's density
falls by 2-3x **per step** while the local divergence allows ~4 %.

Mechanism: the Lax-Friedrichs speed `alpha` is set by the low-density cell's
large fast speed, so the dense neighbours' split fluxes `(F - alpha q)/2` are
O(alpha rho_dense). The fifth-order extrapolation from the dense side then
reverses the sign of the split flux that should flow *into* the minimum, i.e.
mass is pumped out of it. Diagnostic experiments on the restart:

| change | outcome |
|---|---|
| float64 | NaN at the same time |
| CFL 1.5 -> 0.75 | NaN (later) |
| one splitting speed for all fields | NaN (later) |
| existing Zalesak PP limiter (`preserving_flux`) | **density goes negative** (-4.9e3), NaN at the same time |
| LLF in cells with rho < 0.1 (probe) | stable |
| **positivity-preserving reconstruction** | **stable**, min rho holds at 0.08 |

The existing limiter fails because it checks admissibility per axis with no
dimensional factor and bounds the *update*, which at a small time step lets a
gradual drain continue; the reconstruction-level bound does not.

### The fix: positivity-preserving split-flux reconstruction

With ONE splitting speed `alpha >= |v_n| + c` for every field that carries
mass, each split flux is a scaled admissible state:
`f^+- = +-(alpha/2) w^+-`, `w^+- = q +- F/alpha`. The WENO face value `w_hat` of
each is replaced by `w + theta (w_hat - w)` about its upwind cell's `w`, with
the largest `theta in [0, 1]` keeping both `w_hat` and the mirror `2w - w_hat`
admissible. The density bound is closed form. The pressure is concave along the
segment, so the chord's root is admissible: one closed-form step, no iteration.
`theta < 1` is a first-order candidate entering the WENO combination with a
weight set by admissibility instead of smoothness. In smooth flow `theta = 1`
and the scheme is unchanged apart from the splitting speed. Zhang & Shu's proof
gives positivity of each forward-Euler stage for `C_cfl <= 1/2` (sum-of-speeds
CFL); in practice it is robust at the production CFL 1.5.

Fields that carry no mass keep their own splitting speed: hydrodynamic shear
waves and isothermal-MHD Alfven waves. Their correction moves `w` along
`(0, 0, 1, v_t)`, which leaves the density alone and changes the pressure by
`(gamma-1) kappa rho dv^2 [1 - kappa rho / (2 rho_w)] >= 0`, so the proof
survives.

## Results

### Driven Mach-10 turbulence, beta = 0.1, CFL 1.5, no floors / prot / blending

| run | bare | PP-WENO |
|---|---|---|
| isothermal MHD 64³ | NaN t/t_c = 0.83 | **complete (5 t_c)**, min rho 7e-3 |
| isothermal MHD 128³ | NaN t/t_c ~ 0.85 | see `out/turb/` (running at time of writing) |
| isothermal hydro 64³ / 128³ | NaN 0.66 / dt-collapse 0.67 | 128³: running, past 1.4 t_c, min rho 1.5e-4 |
| adiabatic MHD 64³ | NaN 0.50 (face fix alone: 0.36) | **complete**, min rho 0.06 |

The production recipe needs `rho_min = 0.02`. PP-WENO resolves physical voids
an order of magnitude deeper without any floor.

### Cold Evrard collapse (e0 = 0.05, fourth-order conservative gravity, fp32)

| N | bare | face state | PP-WENO |
|---|---|---|---|
| 32 | NaN t = 0.10 (fp64 too) | NaN 0.22 | **complete**, dE/E 1.1e-5 (fp64: 8e-8) |
| 64 | NaN t = 0.25 | — | **complete**, dE/E 3.1e-5 |
| 128 | complete, dE/E 7e-5 | — | complete, dE/E 6.8e-5 |

## Defect 3 (not WENO) — the conservative gravity coupling

Even when the run completes, the conservative energy source leaves cells with
`p < 0` (32³: min p = -0.05 at r ~ 0.45 late in the infall; 64³: -0.015;
128³: -6e-4). It is not a reconstruction effect: the cells' *total* energy
drops below their kinetic energy. The coupling charges the work
`F_rho (phi_face - phi_cell)` done on mass *in transit* to the receiving
cell's total energy. That mass's momentum is never decelerated, so the work is
paid from internal energy, at first order already. The transported gas can
afford the climb across half a cell only if

    e / rho  >~  |g| dx / 2,

i.e. **the grid must resolve the pressure scale height**. Cold Evrard
(e0 = 0.05, g ~ 1 at the edge): half-cell climb 0.0625 at 32³ (violated),
0.031 at 64³ (marginal), 0.016 at 128³ (satisfied). This is why the
conservative schemes were known to NaN below 128³, and why warm Evrard
(e0 = 0.2) is fine at 32³. Late in the collapse g ~ 5 at r ~ 0.45, which
violates the criterion again at 64³, exactly where the holes are.

The options are coupling-level, not WENO-level:

* KE-only source `rho v.g`: always positive, but energy error 20 % at 128³ and 41 % at 64³.
* conservative + dual energy: positive, but because the state is carried in
  primitive form the g-pressure resets E (dE/E ~ 5 % at 32³).
* resolve H, or a coupling that decelerates the transported mass (open).

## Costs

Accuracy (smooth advected waves, double precision, `smooth1d.py` /
`smooth2d.py`), all variants 5th order:

| wave, Mach | baseline L1 (N=64) | PP-WENO |
|---|---|---|
| shear (vortical), 0.08 | 4.25e-7 | 4.38e-7 (unchanged: own speed) |
| entropy (contact), 0.77 | 8.5e-7 (N=64) | 2.3x |
| entropy (contact), 0.08 | 8.5e-7 | 14x = (|v|+c)/|v| |

So the only accuracy price is the dissipation coefficient of low-Mach entropy
waves (density contrasts advected at low Mach), roughly 1.7x in linear
resolution at Mach 0.08 and nothing at high Mach.

Runtime, native kernel, A100, one x-flux at 64³: isothermal MHD 1.20x;
ideal MHD 3.2x with the first (bisection) version, see `bench.py` for the
closed-form version.

## Side findings

* 1D finite-difference runs with periodic boundaries converge at **first
  order**. The 1D config keeps ghost-cell handling with too few ghost cells,
  whereas 2D/3D switch to `PERIODIC_ROLL`. `smooth1d.py` forces the roll.
* Isothermal hydro in 1D does not run (`momentum_index.x` on an int).
* `_lsrk4_with_ct` never calls the flux blending (survey).

## Open

* Pallas port. Both options are native-only (an OPTIMAL_BACKEND request
  resolves to NATIVE_JAX, explicit PALLAS is refused). The tangent of the
  Pallas path is already native, so only the forward kernels need the change.
* Entropy-wave splitting speed at low Mach: the density of `w` stays positive
  with the entropy field on its own speed; only its pressure can drop
  (second order in the cell-to-face velocity difference). An adaptive choice is
  possible but not done.
* The gravity coupling (above).

## Reproduce

All scripts take `WENO_VARIANT=baseline|face|pp`; GPU jobs go through
`stability_lab/run.sh` and `pq`.

    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python stability_lab/riemann1d.py
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python stability_lab/smooth2d.py shear 0.1
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python stability_lab/evrard.py --n 32
    WENO_VARIANT=pp pq sub -t a100 -n 1 --name t64 -- stability_lab/run.sh stability_lab/turb.py --N 64 --tag pp64
