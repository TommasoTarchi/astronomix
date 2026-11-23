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
  the run. Next to the value of each fixed-step run, the report gives the same
  output of the adaptive reference at that time, interpolated between its
  snapshots; the two must agree, except where sinks form between the two
  snapshots around that time (marked with *), since the output jumps there.

```
python hard_baseline.py                                   # every setup, both parts
python hard_baseline.py --setups rotating_core --part gradients
python hard_baseline.py --parameters smooth --diagnostics # the smooth scheme
```

Each setup writes `results/<setup>.npz`; `results/report.md` gathers the
numbers.

With `--parameters smooth` the same runs use the default (smooth) parameters
of the sink formation and accretion and write to `results/smooth`. The
gradients are compared at the times of the hard baseline, and the report gives
the hard reference next to the smooth one. A sink is a slot with activation
above ½, and its formation time is the first snapshot at which its activation
exceeds ½. `--diagnostics` prints, at every step of the reference runs, the
candidates created and recycled with their activation and mass, the number of
occupied slots, and the gas above the density threshold far from every slot.

| Setup | Input of the gradients | End time |
| --- | --- | --- |
| `bonnor_ebert`: Bonnor–Ebert sphere, contrast 20, density enhanced ×2 | density contrast | 0.3 |
| `singular_isothermal`: ρ = A c_s² / (2πG (r² + r_c²)), A = 4, r_c = 2.5 cells | overdensity factor A | 0.2 |
| `rotating_core`: Boss & Bodenheimer (1979), α ≈ 0.26, β ≈ 0.16 | rotation rate | 0.2 |
| `turbulent_cluster`: 64 Jeans masses, Mach 3 | Mach number | 0.1 |
| `bonnor_ebert_offset`: the Bonnor–Ebert sphere with its centre 0.25 cells off the cell centre along x | the offset, in cells | 0.3 |

All runs use 32³ cells, G = c_s = 1, float64, the native JAX backend and the
finite-volume scheme (RK2, HLL, minmod), in periodic boxes.

## Choices behind the runs

* **Finite volume.** The default finite-difference scheme breaks down in the
  turbulent cluster with or without sinks, once the collapse is no longer
  resolved: the state turns non-finite by t ≈ 0.046 without sinks and by
  t ≈ 0.051 with them (t ≈ 0.067 with sinks and the positivity-preserving WENO
  flux). The finite-volume scheme runs the cluster with sinks to t = 0.1.
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
* **Core of the singular isothermal sphere.** Its core of r_c = 2.5 cells
  keeps the centre below the density threshold at the start, also at
  A = 4.4, the largest value of the finite differences (the centre is below
  the threshold for r_c > r_acc √(2A) / π). With a core of half a cell and
  A = 2 the centre starts at 10 times the threshold and the sink forms in the
  first step, so no formation decision depends on A. A core of 2 cells with
  A = 2 first expands and forms its sink only at t ≈ 0.36; A = 4 forms it at
  t ≈ 0.08.
* **Positions.** In the symmetric setups (Bonnor–Ebert sphere, singular
  isothermal sphere, rotating core) the single sink stays on the central
  cell, so the position moments are the total mass times a constant; only the
  turbulent cluster measures positions with them. `bonnor_ebert_offset`
  measures the position of a single sink instead: its outputs are the total
  sink mass and the mass-weighted mean x of the sinks, in cells from the
  centre of the central cell, and its input is the offset of the sphere's
  centre from there.
* **Checkpointed segments.** The fixed-step loop of `time_integration` is a
  plain `fori_loop` whatever `differentiation_mode` says, so reverse mode keeps
  every step (about 160 MB per step at 32³ in float64). A gradient run is
  therefore a `lax.scan` over segments of 25 steps, each under
  `jax.checkpoint`; the setups do not depend on time, so every segment starts
  from t = 0.
