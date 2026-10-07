# Roadmap

Where the next studies start in this repository, what is already in place for
them, and what is known to be open. The detailed notes live next to the code
they describe; this file only points to them.

## 1. AthenaPK's second-order scheme in astronomix against AthenaPK

**In place.** `time_integrator=VL2` (with `HLLD`, GLM cleaning, first-order flux
correction and floors) re-implements AthenaPK's default scheme, native and Pallas,
single- and multi-GPU (`_pallas_call_sharded`).
[`examples/scripts/validation/athenapk_vl2`](examples/scripts/validation/athenapk_vl2)
holds the AthenaPK runner, the 19-case comparison (round-off agreement, the scale
being AthenaPK's own GPU-vs-CPU difference), the CP Alfvén convergence study and
the A100 benchmark (Pallas 9.25 ms per step at 2 x 128³ in double precision
against AthenaPK's 25.4 ms). `pytests/mhd/vl2_athenapk_regression.py` guards it.

**Next.** The same benchmark on H100/H200 and at larger N (the scheme is
FP64-compute bound on the A100), strong and weak scaling on several GPUs against
AthenaPK's own scaling, single precision, and the turbulence driving of the
dynamo study (below) under VL2.

## 2. Cas A at high resolution

**In place.** The calibrated pipeline, the JAX observation model, the X-ray fit and
the 4D-Var live in
[`examples/gallery/supernova_showcase`](examples/gallery/supernova_showcase); start
with its `OVERVIEW.md`, `ROADMAP.md` and `HANDOFF_LARGER_CLUSTER.md`. The scripts
run on the positivity-preserving WENO (`weno_positivity_preserving=True`, set by
`_common.make_fd_config`), whose gradients are finite in single precision. The
`run.sh` wrapper imports the checkout it lives in.

**Next** (details in `HANDOFF_LARGER_CLUSTER.md`): the gates G0-G2 on a GPU (the
PP re-baseline at 128³ and the long-window tangent test), the PP-native cold-face
tangent only if G2 needs it (P2), a separate frozen-theta switch (P3), then the
448³ and 512³ runs. Callers outside the repository still pass the removed
`--positivity` flag.

## 3. Second-order FV against fifth-order FD: spectra, dynamo, Prandtl number

**In place.** [`examples/scripts/forward/mhd/turbulence`](examples/scripts/forward/mhd/turbulence)
(start with `README.md` and `DYNAMO_MECHANISM.md`): the small-scale dynamo with
astronomix WENO5/WENO-Z + CT against AthenaPK PLM/PPM/WENO-Z + GLM, in-flight
spectra and transfer functions, the numerical viscosity and resistivity measured
at matched E_B/E_K (Pm_num is a scheme constant: CT ~1.21, every GLM scheme
0.48-0.63), explicit-diffusivity calibration ladders, AthenaPK-style few-modes
forcing (`TurbulentForcingConfig.forcing_modes`). The hydro counterpart against
AthenaK is in `examples/scripts/forward/hydro/turbulence`.

**Next.** Run the same study with astronomix's own VL2 scheme, so FV and FD share
the code, the forcing realisation and the diagnostics: `dynamo_convergence.py`
needs a `--scheme vl2` path (ideal gas with gamma close to 1 as in
`athenapk_turb.py`, the GLM psi instead of face fields in the reduction, the
VL2 time step in the step-count reconstruction). Then compare spectra, dynamo
growth rates and Pm_num of the two discretisations, also against the literature
(e.g. arXiv:astro-ph/0109497).

## Known open issues

- Reflective boundaries applied to the stacked face magnetic field or velocity
  (`MAGNETIC_FIELD_ONLY` / `VELOCITY_ONLY`) negate the component with index equal
  to the array axis rather than the normal component, and mirror face-centred
  values like cell-centred ones; this matters for finite-difference MHD with
  reflective walls.
- `BackendConfig(pallas_ct=True)` is not differentiable (the CT Pallas kernels
  have no native tangent wrapper); use `pallas_ct=False` for gradients.
- Self-gravity with the isothermal equation of state is not rejected by
  `finalize_config`, although the energy source then has no energy variable.
- The multi-GPU fast path of the LSRK4 + CT step (kept x halo, fused CT parts) is
  used only without positivity-preserving WENO, dual energy and the cold-crush
  blend; porting it to the SSPRK4 integrator and to the PP data flow is open.
- The pairwise `all_to_all` halo exchange of the multi-GPU scaling work
  (esovetkin/optim_scaling, 256483e) helped from 32 GPUs on but is not merged
  (it fails with integer mesh-axis names); a pairwise-`ppermute` variant is the
  candidate to benchmark.
