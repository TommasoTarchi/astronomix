# Draft: request for remnant-stage 3D Cas A states (Orlando et al.)

**Status: DRAFT, NOT SENT.** Rewritten 2026-09-02 after H.-T. Janka's reply
(`janka_mails.md`). Sending it is the author's call.

## What changed, and why the request is now addressed to Salvatore Orlando

The first draft asked the MPA group for the mapped explosion model
`W15-2-cw-IIb`. Janka's answer settles three things:

1. The explosion data *can* be shared (some 3D outputs are in the Garching
   archive), but the model is more than ten years old and "has shortcomings".
2. Salvatore Orlando has evolved that model with PLUTO to the present-day
   remnant, with the long-term physics included, and **Janka suggests those
   later states are the more suitable starting point for us.**
3. His substantive objection is not about data: it is that "optimising initial
   conditions" leaves open what the initial state *means* and whether it is
   unique, and that the observables are projections set by local radiation
   conditions. `ROADMAP.md` §0 is the answer to that objection (the initial
   state is the remnant-phase state at the first Chandra epoch, inferred with a
   physics prior and a forward model that includes the radiation conditions),
   and this email should say so in two sentences rather than argue it.

So the request is for **remnant-stage 3D states** (any epoch between ~100 yr
and the present, from Orlando et al. 2016 / 2021 / 2022 / 2025), which serve two
purposes: a direct replacement for our statistical ejecta seed, and — more
importantly for the vision — independent samples of what a physically produced
remnant-phase state looks like, against which our own Route-B ensemble can be
tested as a prior.

## The email

**To:** Salvatore Orlando (INAF – Osservatorio Astronomico di Palermo); cc
H.-Thomas Janka. *Verify the current address on arXiv:2503.00130 before
sending.*

**Subject:** Cas A remnant-stage 3D model states as priors for a differentiable
reconstruction

Dear Salvatore, dear Thomas,

thank you for the reply, and for the pointer toward the evolved remnant models
rather than the early explosion data. Let me answer the question of what we are
actually proposing, since it was not clear from my first message.

We are not trying to infer the explosion, and I agree that "the initial state"
of a core-collapse event is not one thing. What we mean by the initial state is
the **remnant-phase hydrodynamic state at the epoch of the first Chandra
observation (2000)**: density, velocity, pressure, composition and shock
history on a grid. From there the physics is adiabatic hydrodynamics plus known
microphysics, and Chandra has since observed the remnant for 23 years, so the
question "which states at 2000 evolve into what was seen in 2004 … 2023" is
well posed and the ambiguities are quantifiable. The physics before 2000 enters
as a *prior*: our own calibrated 1D-then-3D model (Route B of Orlando et al.
2016, calibrated to the shock radii, post-shock density and unshocked mass),
and — this is the request — your evolved 3D states, which are the only
physically produced samples of such a remnant-phase state that exist.

On the forward model: we agree that the observables are projections governed by
local conditions. Our pipeline carries per-cell composition (μ, μ_e), Coulomb
electron–ion equilibration, non-equilibrium ionization from a carried ionization
age, interstellar dust scattering, and the real ACIS response, and produces a
synthetic event list binned identically to the real `evt2` data; the comparison
is done in count rates through the same response, not on images. Per-element
electron temperatures and ionization ages from XRISM/Resolve (Vink et al. 2026)
are what we score the plasma against. We have found — and documented — several
places where our own model was wrong, so we do not expect a mapped state to
solve anything by itself; we expect it to tell us how far our statistical prior
is from a physical one.

Concretely, I would be grateful for **one or more 3D states of your Cas A model
at remnant ages between roughly 100 yr and the present**, on whatever grid and in
whatever format is convenient (density, velocity, pressure, species mass
fractions; anything else optional). If your outputs at ~150 yr exist, that epoch
is where our own pipeline maps from and would allow the most direct comparison.
We can handle spherical or Cartesian layouts and interpolate ourselves.

On terms: we will cite the model papers, acknowledge provenance explicitly, and
are glad to agree to an embargo, non-redistribution, or a collaborative
arrangement including co-authorship — whichever you prefer. What a
differentiable solver adds downstream is exact gradients of observables with
respect to the state and the parameters through the hydrodynamic evolution,
which turns the calibration into a gradient-based fit and, over the 23-year
Chandra baseline, into an inference of the state itself with uncertainties. If
that is of interest to you, I would very much like to discuss it.

With best regards,

[NAME]
[POSITION, INSTITUTION, EMAIL]

## Notes on the draft

* Deliberately short on our results and long on what we mean by "initial
  state" — that was the actual objection.
* Do not quote spectral agreement numbers: two soft-band systematics the size
  of the residual were found on 2026-09-02 (`CALIBRATION.md` Result 26).
* The sub-grid clumping and synchrotron layers are not mentioned: both are
  interpretation layers with fitted parameters.
