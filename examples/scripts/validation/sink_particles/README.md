# Sink particles: hard baseline on the collapse setups

`hard_baseline.py` runs the four isothermal collapse setups of
`astronomix/test_setups/self_gravity` with the formation and accretion of
Federrath et al. (2010) at the hard limit of the smoothing parameters, and
records

* the **reference**: the sinks at 100 snapshots of a run with adaptive steps
  (number, masses, positions, formation times);
* the **hard gradients**: `jax.grad` against central finite differences at the
  relative steps 1e-1 to 1e-4, for the total sink mass and a fixed random
  combination of the mass-weighted position moments Σ m cos(2π x_d / L),
  Σ m sin(2π x_d / L), shortly after the first formation and at the end of
  the run.

```
python hard_baseline.py                                   # every setup, both parts
python hard_baseline.py --setups rotating_core --part gradients
```

Each setup writes `results/<setup>.npz`; `results/report.md` gathers the
numbers.

| Setup | Input of the gradients | End time |
| --- | --- | --- |
| `bonnor_ebert`: Bonnor–Ebert sphere, contrast 20, density enhanced ×2 | density contrast | 0.3 |
| `singular_isothermal`: ρ = A c_s² / (2πG r²), A = 2 | overdensity factor A | 0.2 |
| `rotating_core`: Boss & Bodenheimer (1979), α ≈ 0.26, β ≈ 0.16 | rotation rate | 0.2 |
| `turbulent_cluster`: 64 Jeans masses, Mach 3 | Mach number | 0.1 |

All runs use 32³ cells, G = c_s = 1, float64, the native JAX backend and the
finite-volume scheme (RK2, HLL, minmod), in periodic boxes.

## Choices behind the runs

* **Finite volume.** The default finite-difference scheme breaks down in the
  turbulent cluster: next to the heaviest sink a nearly empty cell reaches
  Mach 40–70, alternating from step to step, and the state turns non-finite at
  t ≈ 0.05 (t ≈ 0.07 with the positivity-preserving WENO flux). The time step
  is not the cause: it is about 1e-4, against a free-fall time of the gas next
  to that sink of about 3e-3.
* **Periodic boxes.** With open boundaries the clouds draw gas in through the
  edges of the box: gas plus sink mass grows by 83 % in the Bonnor–Ebert run
  and by 6.5 % in the rotating core. Periodic boxes keep it constant to
  rounding, and the first sinks form at nearly the same times
  (Bonnor–Ebert 0.096 against 0.102, rotating core 0.115 against 0.113).
* **Triggered collapses.** At 32³ the equilibria do not collapse: without the
  enhancement of its density the Bonnor–Ebert sphere (contrast 20 to 300)
  forms no sink by t = 0.6, and the singular isothermal sphere with A = 1 keeps a
  pressure-supported core, since the grid softens the gravity of its centre;
  its centre then fails the Jeans check of the formation
  (|E_grav| / 2E_th ≈ 0.6). The comparison with Shu's accretion rate, which
  needs A = 1, is not possible at this resolution.
* **Checkpointed segments.** The fixed-step loop of `time_integration` is a
  plain `fori_loop` whatever `differentiation_mode` says, so reverse mode keeps
  every step (about 160 MB per step at 32³ in float64). A gradient run is
  therefore a `lax.scan` over segments of 25 steps, each under
  `jax.checkpoint`; the setups do not depend on time, so every segment starts
  from t = 0.
