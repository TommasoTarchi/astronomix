"""
Passive scalars (advected per-parcel labels) for the finite-difference solver.

A passive scalar ``C`` is a label carried by the fluid without acting back on it:

    d(rho C)/dt + div(rho C v) = 0            <=>      dC/dt + v.grad C = 0

Passive scalars are what lets a single-fluid simulation be compared with a
spectrum rather than only with an image: chemical stratification per element
(Orlando et al.'s ``C_el``), an ejecta-versus-circumstellar discriminator, and,
through the shock bookkeeping below, the non-equilibrium ionization age
``n_e t`` and the electron/ion temperature relaxation.

**Why this is operator-split rather than an extra characteristic field.**
The WENO flux is characteristic-wise: adding scalars to it means extending the
eigenstructure (they ride the entropy wave at speed ``u``) in the native kernel,
in the hand-written adjoints and in the Pallas kernels, all of which must stay
bit-compatible with each other. The dual-energy density ``g`` set the precedent
for carrying an extra advected field outside that machinery, and the same choice
is made here, at the cost of an O(dt) splitting error in the scalars only.

Unlike for ``g``, first-order upwind is not good enough here. Every use of these
scalars is about a *contact discontinuity* (the ejecta / circumstellar
interface, the boundary of a metal-rich knot), and first-order upwind smears a
contact over ~sqrt(N_steps) cells, which over the ~10^4 steps of a long run
erases exactly the structure the scalars exist to track. So the advection here
is fifth-order WENO in space with SSP-RK3 in time, the third-order strong
stability preserving companion of WENO5 (Shu & Osher 1988), which is enough for
an operator-split tracer.

**Boundedness comes from a consistent mass flux.** A companion density ``rho~``
is advected alongside, and the scalar flux is built as

    F_s = F_rho * C_face

i.e. the *same* numerical mass flux that updates ``rho~``, multiplied by the
upwind reconstructed value of the RATIO. The update of ``s = rho~ C`` is then a
combination weighted consistently with the update of ``rho~``, so ``C = s /
rho~`` inherits a maximum principle. Reconstructing ``rho C`` independently does
not bound ``C``: the two reconstructions use different data and therefore
different nonlinear weights, and at a persistent contact discontinuity (one that
sits in the same cells for the whole run) the resulting error is one-signed and
accumulates step after step until it overflows. Sharing the WENO weights between
the two reconstructions does not cure this, because the WENO5 reconstruction has
negative coefficients and is not monotone whatever weights it is given; the
consistency has to come from the flux.

Reconstructing the ratio rather than ``rho C`` is better physics too: ``C`` is
smooth across a shock, where ``rho`` is not.

**Shock bookkeeping (``config.track_shock_history``).** Four further scalars are
managed by the library, implementing the ionization-age proxy of Dwarkadas, Dewey
& Bauer (2010) as used by Orlando et al. (2015). Rather than integrating an
ionization network, each parcel carries how long it has been shocked and how much
electron column it has swept (their positions in the block are the ``*_SLOT``
constants of the variable registry):

* ``entropy_initial`` -- the parcel's specific entropy ``log(p / rho^gamma)`` at
  ``t = 0``, advected and never rewritten. Entropy is constant along particle
  paths in smooth adiabatic flow and rises only across shocks, so comparing the
  current entropy against the parcel's own initial value is a clean, dt- and
  resolution-independent test of "has this parcel been through a shock", with no
  need to store a per-step history or tune a per-step threshold. Since the
  Rankine-Hugoniot jump fixes the entropy rise as a function of Mach number
  alone, the threshold ``config.shock_entropy_jump`` is really a minimum shock
  strength: the default ``log(2)`` corresponds to Mach 3.3 at ``gamma = 5/3``,
  against Mach ~100 (7.1 nats) for a young remnant's shocks and 0.06 nats for a
  Sod tube. Weak compressions and sound waves, which carry no ionization, are
  correctly ignored. The label is unbounded, so a non-finite or runaway value
  (more than ``ENTROPY_LABEL_WINDOW`` nats from the current entropy) is reset to
  the current entropy every step: see :func:`sanitize_entropy_label`.
* ``shocked_fraction`` -- how much of the parcel has been through a shock,
  advected and saturating at 1. It is what latches a parcel as shocked after it
  stops converging, and it is a FRACTION rather than a boolean flag on purpose:
  see :func:`update_shock_history`.
* ``time_since_shock`` -- accumulates ``shocked_fraction * dt``; this is
  Orlando's ``Delta t_j = t - t_sh,j``, which drives the Coulomb electron/ion
  relaxation.
* ``density_time`` -- accumulates ``shocked_fraction * rho dt``. In code units
  this is the ionization age up to a constant: ``n_e t = density_time *
  unit_density * unit_time / (mu_e m_p)``. Accumulating the integral directly
  avoids having to carry a shock *time* stamp, which mixes badly under advection
  (averaging a never-shocked parcel with a long-shocked one would produce a
  meaningless intermediate stamp).

All four are *history* variables that ride with the parcel, not instantaneous
flags, and that distinction matters when interpreting them: a parcel shocked at
an earlier step keeps its accumulated record, so later mixing with unshocked
material can leave a cell holding a positive ionization age while its *current*
entropy contrast has fallen back below the threshold. That is the intended
behaviour (it is what "this material has been through a shock" means once the
material moves and mixes), but it means a test of the form "every flagged cell
is currently above the threshold" will fail, and should.

**Differentiability.** Three things make this module differentiable in reverse
mode as well as forward mode, all without changing the primal:

* the sub-cycling loop has a traced trip count, which reverse mode cannot
  differentiate; under ``differentiation_mode == BACKWARDS`` it becomes a
  static, masked loop (``config.passive_scalar_substep_loop``, see
  :func:`advect_passive_scalars`);
* every bound clamp keeps derivative 1 ON its bound (``jnp.clip`` and
  ``jnp.maximum`` give 0.5 at a tie, and these fields sit exactly on their
  bounds almost everywhere, so a plain clamp would halve the tangent every
  step);
* ``config.ad_smooth_shock_latch`` gives the boolean shock latch a
  straight-through surrogate derivative, so that moving a shock changes the
  shock history in the derivative too (see :func:`update_shock_history`).
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    AD_REMAT_AXIS,
    AD_REMAT_NONE,
    BACKWARDS,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    STATE_TYPE,
    SUBSTEPS_AUTO,
    SUBSTEPS_MASKED,
)
from astronomix.variable_registry.registered_variables import (
    DENSITY_TIME_SLOT,
    ENTROPY_INITIAL_SLOT,
    NUM_SHOCK_HISTORY_SCALARS,
    SHOCKED_FRACTION_SLOT,
    TIME_SINCE_SHOCK_SLOT,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._finite_difference._interface_fluxes._weno_weights import (
    _weno_omega_weights_ad,
    _weno_omega_weights_z,
)

#: Floor on the sum of the WENO alphas (the ``tiny`` argument of
#: ``_weno_omega_weights*``). It only keeps the weight quotient finite where
#: every smoothness indicator of a stencil is huge.
_WENO_ALPHA_SUM_FLOOR = 1e-40

#: Floor keeping densities, pressures, speeds and divergences strictly positive
#: before they are divided by or logged.
_POSITIVE_FLOOR = 1e-30

#: Largest believable gap, in nats, between a parcel's advected
#: ``entropy_initial`` label and its current specific entropy. A strong shock
#: raises ``ln(p / rho^gamma)`` by ~2 ln M, so 50 nats would take a Mach ~1e10
#: shock, and radiative cooling at fixed pressure from 1e9 K to 1e4 K lowers it by
#: ~19. Cold ejecta held at a pressure floor and later shocked reach physical gaps
#: of ~40 nats. A label further away than this is the ratio-recovery instability
#: (see :func:`sanitize_entropy_label`).
ENTROPY_LABEL_WINDOW = 50.0


# -----------------------------------------------------------------------------
# ======== ↓ Bound clamps whose derivative survives at the bound ↓ ===========
# -----------------------------------------------------------------------------

# ``jnp.clip`` and ``jnp.maximum`` split the derivative evenly at a tie:
# ``grad(clip)(x = lo) = 0.5`` and ``grad(maximum(x, 0))(0) = 0.5``. For these
# scalars a tie is the common case, not an edge case: in a remnant most cells
# hold a mass fraction of EXACTLY 0 or exactly 1, and a never-shocked parcel's
# shock history is exactly 0. Advection keeps them exactly on the bound (the
# ratio of identical operators is exact), so the clamp ties again every step and
# the tangent / cotangent of those cells is halved every step: 0.5^30 ~ 1e-9
# over a 30-step window. The clamps below have the same primal (the very same
# ``jnp`` call, so bit for bit) and pass the derivative of ``x`` through whenever
# ``x`` is inside the bounds OR on them; only a strictly out-of-bounds value
# takes the bound's.

@jax.custom_jvp
def _clip_keep_derivative(x, lo, hi):
    """``jnp.clip(x, lo, hi)`` with derivative 1 inside and ON the bounds."""
    return jnp.clip(x, lo, hi)


@_clip_keep_derivative.defjvp
def _clip_keep_derivative_jvp(primals, tangents):
    x, lo, hi = primals
    dx, dlo, dhi = tangents
    out = jnp.clip(x, lo, hi)
    dout = jnp.where(x < lo, dlo, jnp.where(x > hi, dhi, dx))
    return out, jnp.broadcast_to(dout, out.shape).astype(out.dtype)


@jax.custom_jvp
def _max_keep_derivative(x, floor):
    """``jnp.maximum(x, floor)`` with the derivative of ``x`` at a tie."""
    return jnp.maximum(x, floor)


@_max_keep_derivative.defjvp
def _max_keep_derivative_jvp(primals, tangents):
    x, floor = primals
    dx, dfloor = tangents
    out = jnp.maximum(x, floor)
    dout = jnp.where(x < floor, dfloor, dx)
    return out, jnp.broadcast_to(dout, out.shape).astype(out.dtype)


@jax.custom_jvp
def _latch_straight_through(carried, latch, soft):
    """
    The shocked-fraction latch ``clip(max(carried, latch), 0, 1)``, with a
    straight-through tangent (``config.ad_smooth_shock_latch``).

    ``latch`` is the boolean "freshly shocked" test as 0/1, ``soft`` its smooth
    surrogate. The primal is exactly the hard latch. The tangent is that of a
    probabilistic OR of the carried fraction with the surrogate,
    ``(1 - latch) d carried + (1 - carried) d soft``: where the latch does not
    fire it is the exact derivative (transport) plus the surrogate's
    sensitivity to firing; where it fires the carried fraction is overwritten,
    so only the surrogate's term remains.
    """
    return jnp.clip(jnp.maximum(carried, latch), 0.0, 1.0)


@_latch_straight_through.defjvp
def _latch_straight_through_jvp(primals, tangents):
    carried, latch, soft = primals
    d_carried, _, d_soft = tangents
    out = jnp.clip(jnp.maximum(carried, latch), 0.0, 1.0)
    dout = (1.0 - latch) * d_carried + (1.0 - carried) * d_soft
    return out, jnp.broadcast_to(dout, out.shape).astype(out.dtype)

# -----------------------------------------------------------------------------
# ======== ↑ Bound clamps whose derivative survives at the bound ↑ ===========
# -----------------------------------------------------------------------------


def _masked_substeps(config) -> bool:
    """Whether the passive-scalar sub-cycling uses the static, masked loop
    (reverse-differentiable) rather than the traced trip count."""
    mode = config.passive_scalar_substep_loop
    if mode == SUBSTEPS_AUTO:
        return config.differentiation_mode == BACKWARDS
    return mode == SUBSTEPS_MASKED


def _scalar_lean(config) -> bool:
    """Whether the lean reverse-mode memory layout of the passive-scalar block
    is active: ``config.ad_scalar_lean`` under reverse-mode rematerialisation."""
    return bool(config.ad_scalar_lean) and config.ad_remat != AD_REMAT_NONE


def _velocity_components(primitive_state, config, registered_variables):
    """The velocity components of the primitive state, as a list per dimension."""
    if config.dimensionality == 1:
        return [primitive_state[registered_variables.velocity_index]]
    if config.dimensionality == 2:
        return [
            primitive_state[registered_variables.velocity_index.x],
            primitive_state[registered_variables.velocity_index.y],
        ]
    return [
        primitive_state[registered_variables.velocity_index.x],
        primitive_state[registered_variables.velocity_index.y],
        primitive_state[registered_variables.velocity_index.z],
    ]


def _split_companion_stack(conserved_stack):
    """
    Split the advected conserved stack into its two parts.

    The stack is ``(rho~, rho~ C_0, ..., rho~ C_{n-1})`` along the leading
    axis: the companion density first, then the scalar densities.

    Returns:
        ``(companion_density, scalar_densities)``.
    """
    return conserved_stack[0], conserved_stack[1:]


def specific_entropy(primitive_state, gamma, registered_variables):
    """``log(p / rho^gamma)``: constant along particle paths except across shocks."""
    density = jnp.maximum(
        primitive_state[registered_variables.density_index],
        _POSITIVE_FLOOR,
    )
    pressure = jnp.maximum(
        primitive_state[registered_variables.pressure_index],
        _POSITIVE_FLOOR,
    )
    return jnp.log(pressure) - gamma * jnp.log(density)


def sanitize_entropy_label(entropy_initial, entropy_now, window=ENTROPY_LABEL_WINDOW):
    """
    Reset a non-finite or runaway ``entropy_initial`` label to the current entropy.

    ``entropy_initial`` is the only library scalar with no bound: the mass
    fractions are clipped to their declared range and the two accumulators to
    their own global maximum (:func:`_recover_ratios`), but the label can take
    any value. Where the companion density ``rho~`` nearly collapses without
    crossing the ``1e-6 rho`` guard, ``rho~ s0 / rho~`` amplifies WENO overshoot
    every step, as an odd-even pair of opposite sign. Left alone, such a label
    eventually overflows; the NaN then spreads over the whole label field
    through the WENO stencils, makes every reverse-mode gradient NaN, and
    freezes the shock latch (``entropy_now - s0 > jump`` is False for NaN), so
    no parcel is flagged afterwards and the shocked fraction and ionization age
    go stale.

    The reset value is the current entropy, i.e. "not shocked relative to now"
    (entropy rise 0, so it cannot fire the latch), with no derivative: the cell
    is pathological, and its label's tangent (the WENO overshoot it came from)
    is meaningless. A label within ``window`` nats of the current entropy is
    returned bit for bit, so a run that never trips this is unchanged; the test
    ``|s_now - s0| <= window`` is False for NaN and +-inf as well.

    Args:
        entropy_initial: The advected ``entropy_initial`` label.
        entropy_now: The current specific entropy (:func:`specific_entropy`).
        window: The largest accepted gap between the two, in nats.

    Returns:
        The label, with pathological values replaced by the current entropy.
    """
    label_is_believable = jnp.abs(entropy_now - entropy_initial) <= window
    return jnp.where(
        label_is_believable,
        entropy_initial,
        jax.lax.stop_gradient(entropy_now),
    )


def _weno5_left_biased(q0, q1, q2, q3, q4, epsilon, omega_weights):
    """
    WENO5 reconstruction at the right face of the middle cell.

    ``q0..q4`` are the values at ``i-2 .. i+2``; the result is the state at
    ``i+1/2`` reconstructed from the left, i.e. the upwind value for a positive
    face velocity. The optimal linear weights are ``(1, 6, 3)/10``, matching the
    convention of :func:`_weno_omega_weights`, which returns ``(omega_0,
    omega_2)``.

    Args:
        q0, q1, q2, q3, q4: The cell values at ``i-2 .. i+2``.
        epsilon: The WENO epsilon of the smoothness denominators.
        omega_weights: The nonlinear-weight function (one of the
            ``_weno_omega_weights*`` variants).

    Returns:
        The reconstructed value at ``i+1/2``.
    """
    # The three third-order candidate reconstructions at i+1/2.
    candidate_0 = (2.0 * q0 - 7.0 * q1 + 11.0 * q2) / 6.0
    candidate_1 = (-q1 + 5.0 * q2 + 2.0 * q3) / 6.0
    candidate_2 = (2.0 * q2 + 5.0 * q3 - q4) / 6.0

    # Smoothness indicators of the three candidate stencils (Jiang & Shu 1996).
    thirteen_twelfths = 13.0 / 12.0
    smoothness_0 = (
        thirteen_twelfths * (q0 - 2.0 * q1 + q2) ** 2
        + 0.25 * (q0 - 4.0 * q1 + 3.0 * q2) ** 2
    )
    smoothness_1 = (
        thirteen_twelfths * (q1 - 2.0 * q2 + q3) ** 2
        + 0.25 * (q1 - q3) ** 2
    )
    smoothness_2 = (
        thirteen_twelfths * (q2 - 2.0 * q3 + q4) ** 2
        + 0.25 * (3.0 * q2 - 4.0 * q3 + q4) ** 2
    )

    weight_0, weight_2 = omega_weights(
        smoothness_0,
        smoothness_1,
        smoothness_2,
        epsilon,
        _WENO_ALPHA_SUM_FLOOR,
    )
    weight_1 = 1.0 - weight_0 - weight_2
    return weight_0 * candidate_0 + weight_1 * candidate_1 + weight_2 * candidate_2


def _advection_rhs(
    conserved_stack,
    velocities,
    grid_spacing,
    epsilon,
    omega_weights,
    per_axis_remat=False,
):
    """
    ``-div(conserved_stack * v)`` for the advected conserved stack.

    The stack carries a leading field axis, so spatial axis ``a`` is array axis
    ``a + 1``. The face velocity is the mean of the two neighbours and the
    interface value is upwinded on its sign, both sides reconstructed to fifth
    order. The scalar fluxes are the companion density's mass flux times the
    upwind value of the ratio (see the module docstring).

    Args:
        conserved_stack: The conserved stack ``(rho~, rho~ C_0, ...)``.
        velocities: The velocity components, one field per dimension.
        grid_spacing: The cell width.
        epsilon: The WENO epsilon.
        omega_weights: The nonlinear-weight function.
        per_axis_remat: Wrap each axis' reconstruction and flux difference in
            ``jax.checkpoint`` and reconstruct the scalars one at a time inside
            it, which lowers the reverse-mode memory (``config.ad_remat ==
            "axis"``); the arithmetic is the same.

    Returns:
        The right-hand side, same shape as ``conserved_stack``.
    """
    # The flux needs the ratio, so it is recovered from the conserved stack at
    # every Runge-Kutta stage.
    companion_density, scalar_densities = _split_companion_stack(conserved_stack)
    safe_density = jnp.where(companion_density > 0.0, companion_density, 1.0)
    ratios = scalar_densities / safe_density[None, ...]
    density_rhs = jnp.zeros_like(companion_density)
    scalar_rhs = jnp.zeros_like(ratios)

    def upwind_face_values(stack, array_axis, face_velocity):
        """Upwind WENO5 value at i+1/2 of a (possibly stacked) field."""
        q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3 = [
            _shift(stack, shift, axis=array_axis) for shift in (2, 1, 0, -1, -2, -3)
        ]
        from_left = _weno5_left_biased(
            q_im2,
            q_im1,
            q_i,
            q_ip1,
            q_ip2,
            epsilon,
            omega_weights,
        )
        from_right = _weno5_left_biased(
            q_ip3,
            q_ip2,
            q_ip1,
            q_i,
            q_im1,
            epsilon,
            omega_weights,
        )
        return jnp.where(face_velocity >= 0.0, from_left, from_right)

    def flux_differences(companion_density, ratios, velocity, grid_spacing, axis):
        """This axis' ``(F_{i+1/2} - F_{i-1/2}) / dx`` for rho~ and for s."""
        face_velocity = 0.5 * (velocity + _shift(velocity, -1, axis=axis))

        # ONE mass flux, built from the density reconstruction ...
        density_face = upwind_face_values(companion_density, axis, face_velocity)
        mass_flux = face_velocity * density_face

        # ... and the scalar flux is that SAME mass flux times the upwind value
        # of the RATIO. This consistent mass flux is what bounds the recovered
        # C; see the module docstring.
        density_flux_difference = (
            mass_flux - _shift(mass_flux, 1, axis=axis)
        ) / grid_spacing
        if per_axis_remat:
            # One scalar at a time (a rematerialised lax.map): the backward
            # pass then holds a single field's WENO internals rather than the
            # whole stack's, which matters because the scalar stack can
            # dominate the state and its reconstruction dominates the
            # advection's backward memory.
            def single_scalar_flux_difference(ratio):
                scalar_flux = mass_flux * upwind_face_values(ratio, axis, face_velocity)
                return (scalar_flux - _shift(scalar_flux, 1, axis=axis)) / grid_spacing

            return density_flux_difference, jax.lax.map(
                jax.checkpoint(single_scalar_flux_difference),
                ratios,
            )
        ratio_face = upwind_face_values(ratios, axis + 1, face_velocity[None, ...])
        scalar_flux = mass_flux[None, ...] * ratio_face

        return density_flux_difference, (
            scalar_flux - _shift(scalar_flux, 1, axis=axis + 1)
        ) / grid_spacing

    for axis, velocity in enumerate(velocities):
        axis_flux_differences = partial(flux_differences, axis=axis)
        if per_axis_remat:
            axis_flux_differences = jax.checkpoint(axis_flux_differences)
        density_flux_difference, scalar_flux_difference = axis_flux_differences(
            companion_density,
            ratios,
            velocity,
            grid_spacing,
        )
        density_rhs = density_rhs - density_flux_difference
        scalar_rhs = scalar_rhs - scalar_flux_difference

    return jnp.concatenate([density_rhs[None, ...], scalar_rhs], axis=0)


def _substep_count(velocities, grid_spacing, dt, config):
    """
    How many sub-steps this advection needs, from the flow itself.

    Two conditions have to hold over a sub-step ``h``:

    * the advection CFL, ``max|u| h / dx <= cfl``;
    * positivity of the companion density, ``h max|div v| <= 0.5``: the
      explicit update ``rho~ - h div(rho v)`` is what goes negative in a strong
      compression, and a negative denominator is what makes the recovered ratio
      diverge.

    Returned as a traced integer so ``lax.fori_loop`` runs exactly once on a
    benign step; the cap ``config.max_passive_scalar_substeps`` keeps a
    pathological cell from stalling the run.

    Args:
        velocities: The velocity components, one field per dimension.
        grid_spacing: The cell width.
        dt: The hydro time step.
        config: The simulation configuration.

    Returns:
        The number of sub-steps, an int32 scalar in
        ``[1, config.max_passive_scalar_substeps]``.
    """
    inverse_dx = 1.0 / grid_spacing
    max_speed = jnp.max(jnp.stack([jnp.max(jnp.abs(velocity)) for velocity in velocities]))
    velocity_divergence = sum(
        0.5 * (_shift(velocity, -1, axis=axis) - _shift(velocity, 1, axis=axis)) * inverse_dx
        for axis, velocity in enumerate(velocities)
    )
    max_abs_divergence = jnp.max(jnp.abs(velocity_divergence))

    advective_substep = (
        config.passive_scalar_cfl * grid_spacing / jnp.maximum(max_speed, _POSITIVE_FLOOR)
    )
    compressive_substep = 0.5 / jnp.maximum(max_abs_divergence, _POSITIVE_FLOOR)
    safe_substep = jnp.minimum(advective_substep, compressive_substep)
    num_substeps = jnp.ceil(dt / jnp.maximum(safe_substep, _POSITIVE_FLOOR))
    return jnp.clip(num_substeps, 1, config.max_passive_scalar_substeps).astype(jnp.int32)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def advect_passive_scalars(
    scalars: Float[Array, "..."],
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    grid_spacing: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Advance the passive scalars by one operator-split step.

    Args:
        scalars: The per-parcel labels, shape ``(n_scalars,) + grid``.
        primitive_state: The primitive state at the start of the step, whose
            density and velocity define the advecting flow (frozen over ``dt``).
        dt: The hydro time step. It is NOT automatically safe for this
            advection (see :func:`_substep_count`), so the step is sub-cycled
            as the flow requires.
        grid_spacing: The cell width.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The updated scalars, same shape as ``scalars``.
    """

    # -------------------------------------------------------------
    # ======= ↓ WENO weights and the advected conserved stack ↓ ====
    # -------------------------------------------------------------

    epsilon = config.weno_epsilon
    # The overflow-free ``_weno_omega_weights_ad`` gives the same weights bit
    # for bit, but a derivative that cannot overflow. These weights are always
    # differentiated, and an unbounded label such as ``entropy_initial`` with a
    # large jump across the stencil makes the float32 Jiang-Shu weight VJP NaN
    # even for a zero cotangent.
    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights_ad

    density = primitive_state[registered_variables.density_index]
    velocities = _velocity_components(primitive_state, config, registered_variables)

    # Advect (rho~, rho~ C_0, ..., rho~ C_{n-1}) as one stack through the
    # identical operator; see the module docstring for why the companion density
    # rather than the hydro solver's updated density is what keeps C bounded.
    conserved_stack = jnp.concatenate(
        [density[None, ...], scalars * density[None, ...]],
        axis=0,
    )

    # -------------------------------------------------------------
    # ======= ↑ WENO weights and the advected conserved stack ↑ ====
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ RHS rematerialisation ↓ ================
    # -------------------------------------------------------------

    # Reverse-mode rematerialisation of the right-hand side (``config.ad_remat``):
    # the backward pass keeps only each Runge-Kutta stage's input and recomputes
    # the WENO internals (with ``"axis"`` additionally one axis at a time).
    @jax.checkpoint
    def checkpointed_advection_rhs(stack, velocities, grid_spacing):
        return _advection_rhs(
            stack,
            velocities,
            grid_spacing,
            epsilon,
            omega_weights,
            per_axis_remat=config.ad_remat == AD_REMAT_AXIS,
        )

    def rematerialised_advection_rhs(stack):
        return checkpointed_advection_rhs(stack, velocities, grid_spacing)

    if config.ad_remat == AD_REMAT_NONE:
        def advection_rhs(stack):
            return _advection_rhs(stack, velocities, grid_spacing, epsilon, omega_weights)
    else:
        advection_rhs = rematerialised_advection_rhs

    # -------------------------------------------------------------
    # =================== ↑ RHS rematerialisation ↑ ================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Sub-cycled SSP-RK3 update ↓ ==============
    # -------------------------------------------------------------

    def ssprk3_step(stack, substep_dt, rhs=None):
        """One SSP-RK3 step (Shu & Osher 1988), the companion of WENO5."""
        rhs = advection_rhs if rhs is None else rhs
        stage_1 = stack + substep_dt * rhs(stack)
        stage_2 = 0.75 * stack + 0.25 * (stage_1 + substep_dt * rhs(stage_1))
        return (1.0 / 3.0) * stack + (2.0 / 3.0) * (stage_2 + substep_dt * rhs(stage_2))

    # SUB-CYCLING. The hydro time step is limited by |u| + c, which bounds this
    # advection's |u| but NOT the companion density's positivity: the explicit
    # update stays positive only while dt |div v| < 1, and at a typical hydro
    # C_cfl = 0.3 the margin is only ~0.6. Where strong compression (e.g. a
    # radiatively cooling shell hit by a fast piston) spends that margin, rho~
    # crosses zero and the recovered ratio runs away.
    #
    # The sub-step count is computed from the flow rather than configured, so a
    # benign step costs exactly one SSP-RK3 step (the loop runs once); only the
    # violent steps pay.
    #
    # The count is a traced integer, so ``fori_loop(0, num_substeps)`` is a
    # while loop, which reverse-mode AD cannot differentiate. Under BACKWARDS
    # (see ``config.passive_scalar_substep_loop``) the loop instead runs to the
    # static cap ``max_passive_scalar_substeps`` and skips the surplus sub-steps
    # with a ``lax.cond`` (only the taken branch executes): the same arithmetic
    # on the same data. With a cap of 1 this is just one straight-line sub-step.
    # Otherwise every sub-step, the first included, sits in ONE scan whose body
    # is ``jax.checkpoint``-ed, with its right-hand side rematerialised too,
    # whatever ``config.ad_remat`` says: XLA allocates the buffers of a
    # conditional and of a loop body whether or not they run, and adds them to
    # the rest of the step's, so a separately differentiated first sub-step plus
    # a masked loop for the rest would need nearly twice the backward memory of
    # the single rematerialised scan. The price is one extra evaluation of the
    # advection in the backward pass. The count itself carries no derivative (it
    # is an integer; the flow it is derived from is held constant for it
    # explicitly).
    if _masked_substeps(config):
        num_substeps = _substep_count(
            [jax.lax.stop_gradient(velocity) for velocity in velocities],
            jax.lax.stop_gradient(grid_spacing),
            jax.lax.stop_gradient(dt),
            config,
        )
        substep_dt = dt / num_substeps.astype(conserved_stack.dtype)
        max_substeps = int(config.max_passive_scalar_substeps)
        if max_substeps == 1:
            advected_stack = ssprk3_step(conserved_stack, substep_dt)
        elif _scalar_lean(config):
            # With ``config.ad_scalar_lean`` the flow-derived count runs in
            # equinox's checkpointed while loop: the same sub-steps, but the
            # backward pass stores two scalar-stack checkpoints instead of one
            # per cap slot (the masked scan's stack of checkpoints can be the
            # largest buffer of a reverse-mode step); a benign step
            # (num_substeps = 1) recomputes nothing.
            from equinox.internal._loop.checkpointed import checkpointed_while_loop

            def substeps_remaining(carry):
                substep_index, _ = carry
                return substep_index < num_substeps

            def lean_substep(carry):
                substep_index, stack = carry
                return (
                    substep_index + 1,
                    ssprk3_step(stack, substep_dt, rematerialised_advection_rhs),
                )

            _, advected_stack = checkpointed_while_loop(
                substeps_remaining,
                lean_substep,
                (jnp.zeros((), jnp.int32), conserved_stack),
                checkpoints=min(2, max_substeps),
            )
        else:
            @jax.checkpoint
            def masked_substep(substep_index, stack):
                def take_substep(stack):
                    return ssprk3_step(stack, substep_dt, rematerialised_advection_rhs)

                def skip_substep(stack):
                    return stack

                return jax.lax.cond(
                    substep_index < num_substeps,
                    take_substep,
                    skip_substep,
                    stack,
                )

            # Python-int bounds make this fori_loop a scan, not a while loop,
            # so reverse mode can differentiate it.
            advected_stack = jax.lax.fori_loop(
                0,
                max_substeps,
                masked_substep,
                conserved_stack,
            )
    else:
        num_substeps = _substep_count(velocities, grid_spacing, dt, config)
        substep_dt = dt / num_substeps.astype(conserved_stack.dtype)
        advected_stack = jax.lax.fori_loop(
            0,
            num_substeps,
            lambda _, stack: ssprk3_step(stack, substep_dt),
            conserved_stack,
        )

    # -------------------------------------------------------------
    # ================= ↑ Sub-cycled SSP-RK3 update ↑ ==============
    # -------------------------------------------------------------

    recover_ratios = partial(_recover_ratios, config=config)
    if _scalar_lean(config):
        # With ``config.ad_scalar_lean`` the recovery's clip masks are
        # recomputed in the backward pass instead of being stored.
        recover_ratios = jax.checkpoint(recover_ratios)
    return recover_ratios(advected_stack, density, scalars)


def _recover_ratios(advected_stack, density, scalars, *, config):
    """
    The scalar ratios ``s / rho~`` of an advected conserved stack, guarded where
    the companion density collapsed and clipped to the known bounds.

    Args:
        advected_stack: The advected conserved stack ``(rho~, rho~ C_0, ...)``.
        density: The density the stack was built from (start of the step).
        scalars: The scalars before the advection, whose global ranges bound
            the guarded cells and the shock-history accumulators.
        config: The simulation configuration.

    Returns:
        The advected scalars, same shape as ``scalars``.
    """
    advected_density, advected_scalar_densities = _split_companion_stack(advected_stack)

    # -------------------------------------------------------------
    # ==================== ↓ Collapse guard ↓ ======================
    # -------------------------------------------------------------

    # The recovered ratio ``s / rho~`` is meaningless wherever the companion
    # density has collapsed, as it does in near-vacuum regions (e.g. blast-wave
    # interiors, many orders of magnitude below the shell density). There
    # numerator and denominator are both tiny, WENO overshoot dominates their
    # ratio, and the result feeds back through ``s = rho C`` on the next step,
    # so it amplifies until every scalar is NaN and the accumulators blow up in
    # a few cells, while their median stays perfectly correct (a failure that
    # survives a glance at a figure).
    #
    # The guard is deliberately TARGETED. ``rho~`` is advected from ``rho`` over
    # a single step by a consistent operator, so it cannot legitimately fall by
    # six orders of magnitude; a cell where it has is pathological, and only
    # there is the high-order result replaced by the global range bound (which
    # advection cannot exceed anyway). Applying that bound everywhere instead
    # looks harmless and is not: it clips smooth extrema every step, a one-sided
    # error that accumulates and destroys the convergence order.
    density_floor = 1e-6 * jnp.abs(density)
    safe_density = jnp.maximum(advected_density, density_floor)
    safe_density = jnp.where(safe_density > 0.0, safe_density, 1.0)
    recovered = advected_scalar_densities / safe_density[None, ...]

    spatial_axes = tuple(range(1, scalars.ndim))
    range_min = jnp.min(scalars, axis=spatial_axes, keepdims=True)
    range_max = jnp.max(scalars, axis=spatial_axes, keepdims=True)
    untrustworthy = (advected_density < density_floor)[None, ...]
    recovered = jnp.where(
        untrustworthy,
        _clip_keep_derivative(jnp.nan_to_num(recovered, nan=0.0), range_min, range_max),
        recovered,
    )

    # -------------------------------------------------------------
    # ==================== ↑ Collapse guard ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ==================== ↓ Declared bounds ↓ =====================
    # -------------------------------------------------------------

    # Physical bounds, where the caller has declared them. This is the backstop
    # that actually holds in practice: the guard above only fires when the
    # companion density has collapsed to 1e-6 of the density it was advected
    # from, and a ratio is already meaningless well before that. A mass fraction
    # is bounded in [0, 1] by definition, so enforcing it costs nothing, unlike
    # clipping to a scalar's own current range, which clips smooth extrema and
    # destroys the order. ``_clip_keep_derivative`` has the primal of
    # ``jnp.clip``, but a scalar sitting exactly on its bound (most of the grid)
    # keeps derivative 1.
    bounds = _scalar_bounds(config, scalars.shape[0])
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        recovered = _clip_keep_derivative(recovered, lower_bounds, upper_bounds)

    # -------------------------------------------------------------
    # ==================== ↑ Declared bounds ↑ =====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================== ↓ Accumulator bounds ↓ ====================
    # -------------------------------------------------------------

    # The two shock-history ACCUMULATORS have no declarable upper bound (they
    # grow with the run), but they do have an exact one available here:
    # advection cannot raise a field's global maximum, only the source term can,
    # and the source is applied separately in ``update_shock_history``. Clipping
    # them to their own pre-advection global maximum is therefore free of
    # physical content, and it stops the same ``s / rho~`` pathology that the
    # declared bounds stop for the mass fractions; without it an ionization age
    # can grow by orders of magnitude in a few cells while every bounded field
    # looks perfect.
    #
    # This is safe here for the reason the same clip was NOT safe applied to
    # every scalar: these are monotone diagnostics, so they have no smooth
    # extremum whose clipping would bias the solution.
    if config.track_shock_history:
        # The accumulators (time_since_shock, density_time) are the last two
        # slots of the shock-history block, which ends the scalar stack.
        accumulator_start = (
            scalars.shape[0] - NUM_SHOCK_HISTORY_SCALARS + TIME_SINCE_SHOCK_SLOT
        )
        accumulators = recovered[accumulator_start:]
        accumulator_max = jnp.max(
            scalars[accumulator_start:],
            axis=spatial_axes,
            keepdims=True,
        )
        recovered = jnp.concatenate(
            [
                recovered[:accumulator_start],
                _clip_keep_derivative(accumulators, 0.0, accumulator_max),
            ],
            axis=0,
        )

    # -------------------------------------------------------------
    # ================== ↑ Accumulator bounds ↑ ====================
    # -------------------------------------------------------------

    return recovered


def _scalar_bounds(config, num_scalars):
    """
    ``(lower, upper)`` bound arrays broadcastable over the scalar stack, or
    ``None`` when no scalar has a bound.

    User scalars take their bounds from ``config.passive_scalar_bounds``; the
    library-managed shock-history block knows its own: the shocked fraction is
    in [0, 1] and both accumulators are non-negative, while the initial-entropy
    label is unbounded.

    ``num_scalars`` comes from the scalar array itself, NOT from the registry:
    by the time this runs the registry has had the scalar block stripped off
    it, so its counts read zero.

    Args:
        config: The simulation configuration.
        num_scalars: The number of scalars in the stack (user plus history).

    Returns:
        The bounds, each of shape ``(num_scalars,) + (1,) * dimensionality``,
        or ``None``.
    """
    num_history = NUM_SHOCK_HISTORY_SCALARS if config.track_shock_history else 0
    num_user = num_scalars - num_history

    declared = tuple(config.passive_scalar_bounds or ())
    if not declared and num_history == 0:
        return None

    lower_bounds, upper_bounds = [], []
    for scalar_number in range(num_user):
        declared_bound = declared[scalar_number] if scalar_number < len(declared) else None
        lower_bounds.append(-jnp.inf if declared_bound is None else declared_bound[0])
        upper_bounds.append(jnp.inf if declared_bound is None else declared_bound[1])

    if num_history:
        history_lower_bounds = [None] * NUM_SHOCK_HISTORY_SCALARS
        history_upper_bounds = [None] * NUM_SHOCK_HISTORY_SCALARS
        history_lower_bounds[ENTROPY_INITIAL_SLOT] = -jnp.inf
        history_upper_bounds[ENTROPY_INITIAL_SLOT] = jnp.inf
        history_lower_bounds[SHOCKED_FRACTION_SLOT] = 0.0
        history_upper_bounds[SHOCKED_FRACTION_SLOT] = 1.0
        history_lower_bounds[TIME_SINCE_SHOCK_SLOT] = 0.0
        history_upper_bounds[TIME_SINCE_SHOCK_SLOT] = jnp.inf
        history_lower_bounds[DENSITY_TIME_SLOT] = 0.0
        history_upper_bounds[DENSITY_TIME_SLOT] = jnp.inf
        lower_bounds += history_lower_bounds
        upper_bounds += history_upper_bounds

    shape = (num_scalars,) + (1,) * int(config.dimensionality)
    return jnp.asarray(lower_bounds).reshape(shape), jnp.asarray(upper_bounds).reshape(shape)


@partial(jax.jit, static_argnames=["config"])
def _fill_scalar_ghost_cells(scalars, config: SimulationConfig):
    """
    Fill the ghost zones of the passive-scalar block.

    The generic boundary handler cannot be reused: it negates the variable whose
    index equals the axis (the normal velocity), which for a scalar block would
    silently flip a composition field. Passive scalars are true scalars, so all
    three boundary types reduce to a copy (mirrored for reflective, wrapped for
    periodic, edge-extended for open) with no sign change anywhere. Other
    boundary types leave the ghost cells untouched.

    Args:
        scalars: The scalar block, shape ``(n_scalars,) + padded grid``.
        config: The simulation configuration.

    Returns:
        The scalar block with filled ghost cells.
    """
    num_ghost = config.num_ghost_cells
    if num_ghost == 0:
        return scalars

    boundary_settings = config.boundary_settings
    if config.dimensionality == 1:
        per_axis_settings = (boundary_settings,)
    else:
        per_axis_settings = (
            boundary_settings.x,
            boundary_settings.y,
            boundary_settings.z,
        )[:config.dimensionality]

    for axis, axis_settings in enumerate(per_axis_settings):
        # The spatial axis within the scalar block (axis 0 holds the scalars).
        array_axis = axis + 1

        def axis_slice(start, stop):
            index = [slice(None)] * scalars.ndim
            index[array_axis] = slice(start, stop)
            return tuple(index)

        left = axis_settings.left_boundary
        right = axis_settings.right_boundary
        if left == PERIODIC_BOUNDARY and right == PERIODIC_BOUNDARY:
            scalars = scalars.at[axis_slice(0, num_ghost)].set(
                scalars[axis_slice(-2 * num_ghost, -num_ghost)]
            )
            scalars = scalars.at[axis_slice(-num_ghost, None)].set(
                scalars[axis_slice(num_ghost, 2 * num_ghost)]
            )
            continue

        if left == OPEN_BOUNDARY:
            scalars = scalars.at[axis_slice(0, num_ghost)].set(
                scalars[axis_slice(num_ghost, num_ghost + 1)]
            )
        elif left == REFLECTIVE_BOUNDARY:
            scalars = scalars.at[axis_slice(0, num_ghost)].set(
                jnp.flip(scalars[axis_slice(num_ghost, 2 * num_ghost)], axis=array_axis)
            )

        if right == OPEN_BOUNDARY:
            scalars = scalars.at[axis_slice(-num_ghost, None)].set(
                scalars[axis_slice(-num_ghost - 1, -num_ghost)]
            )
        elif right == REFLECTIVE_BOUNDARY:
            scalars = scalars.at[axis_slice(-num_ghost, None)].set(
                jnp.flip(scalars[axis_slice(-2 * num_ghost, -num_ghost)], axis=array_axis)
            )

    return scalars


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def update_shock_history(
    history: Float[Array, "..."],
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    entropy_jump: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Advance the ``NUM_SHOCK_HISTORY_SCALARS`` (four) shock-history scalars
    after the hydro update.

    ``history`` holds ``entropy_initial``, ``shocked_fraction``,
    ``time_since_shock`` and ``density_time`` at the positions given by the
    ``*_SLOT`` constants of the variable registry (see the module docstring). A
    parcel counts as shocked once its specific entropy has risen by more than
    ``entropy_jump`` nats above the value it carried at ``t = 0`` while the
    flow converges; ``entropy_jump = log(2)`` corresponds to a factor-two
    entropy rise, comfortably above numerical noise and well below the ~10 nats
    a strong remnant shock produces.

    Args:
        history: The shock-history scalars, shape
            ``(NUM_SHOCK_HISTORY_SCALARS,) + grid``.
        primitive_state: The primitive state AFTER the hydro update.
        dt: The time step just taken.
        gamma: The adiabatic index.
        entropy_jump: The entropy-rise threshold, in nats.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The updated shock-history scalars, same shape as ``history``.
    """

    # -------------------------------------------------------------
    # ===================== ↓ Entropy label ↓ ======================
    # -------------------------------------------------------------

    entropy_now = specific_entropy(primitive_state, gamma, registered_variables)
    # The label is unbounded; a non-finite or runaway one is reset BEFORE the
    # latch reads it (bit for bit unchanged otherwise; see
    # ``sanitize_entropy_label``).
    entropy_initial = sanitize_entropy_label(history[ENTROPY_INITIAL_SLOT], entropy_now)

    # -------------------------------------------------------------
    # ===================== ↑ Entropy label ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Fresh-shock criterion ↓ ==================
    # -------------------------------------------------------------

    # Two conditions for a FRESH shock crossing, then a latch.
    #
    # The entropy test alone over-triggers. ``entropy_initial`` is advected, so
    # wherever the initial state itself has an entropy discontinuity the
    # reconstruction smears it, and material sitting next to high-entropy gas
    # inherits a too-low label and looks shocked. In a strong shock tube that
    # flags almost the whole box, at entropy rises far below the threshold.
    #
    # Requiring the flow to be CONVERGING as well removes it: rarefaction fans
    # diverge and contact discontinuities have div(v) ~ 0, so only genuine
    # compressions can arm the flag. This is the standard shock criterion
    # (Pfrommer et al. 2017 use converging flow plus aligned temperature and
    # density gradients).
    #
    # The latch is what makes it a *history*: a parcel is no longer converging
    # once it is well downstream, but it is still shocked material, and its
    # ionization age must keep accumulating. The latch is the advected
    # ``shocked_fraction`` below: it rides with the parcel, so the record
    # follows the material rather than the grid.
    #
    # Only the sign of the divergence enters the hard criterion, so it is left
    # undivided by dx: ``velocity_divergence_times_dx`` is dx div(v), the
    # velocity difference across a cell, which is also the scale the smooth
    # surrogate below compares against the sound speed.
    velocities = _velocity_components(primitive_state, config, registered_variables)
    velocity_divergence_times_dx = sum(
        0.5 * (_shift(velocity, -1, axis=axis) - _shift(velocity, 1, axis=axis))
        for axis, velocity in enumerate(velocities)
    )
    entropy_rise = entropy_now - entropy_initial
    newly_shocked = (entropy_rise > entropy_jump) & (velocity_divergence_times_dx < 0.0)

    # -------------------------------------------------------------
    # ================= ↑ Fresh-shock criterion ↑ ==================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ Shocked-fraction latch ↓ ==================
    # -------------------------------------------------------------

    # The latch is a FRACTION, not a boolean, and this matters more than it
    # looks. A boolean latch on "has this cell any accumulated time?" is
    # triggered by the infinitesimal advective leakage the scheme spreads into
    # neighbouring cells: a cell holding 1e-18 from numerical diffusion would
    # then accumulate at the FULL rate, and within a hundred steps the whole box
    # is flagged. Carrying the shocked fraction as its own advected scalar
    # instead makes the accumulation proportional, so leakage contributes in
    # proportion to how much shocked material actually arrived.
    #
    # The clamps use the tie-preserving variants (same primal as
    # jnp.maximum / jnp.clip): a never-shocked parcel sits exactly on the bound
    # 0, and a tie would halve its derivative every step.
    latch = jnp.where(newly_shocked, 1.0, 0.0)
    if config.ad_smooth_shock_latch:
        # Straight-through: the primal is the boolean latch above, the
        # derivative that of a sigmoid surrogate of the same two criteria (see
        # ``SimulationConfig.ad_smooth_shock_latch``).
        floored_density = jnp.maximum(
            primitive_state[registered_variables.density_index],
            _POSITIVE_FLOOR,
        )
        floored_pressure = jnp.maximum(
            primitive_state[registered_variables.pressure_index],
            _POSITIVE_FLOOR,
        )
        sound_speed = jax.lax.stop_gradient(jnp.sqrt(gamma * floored_pressure / floored_density))
        soft_latch = (
            jax.nn.sigmoid(
                (entropy_rise - entropy_jump) / config.ad_shock_latch_entropy_width
            )
            * jax.nn.sigmoid(
                -velocity_divergence_times_dx
                / (config.ad_shock_latch_compression_width * sound_speed)
            )
        )
        shocked_fraction = _latch_straight_through(
            history[SHOCKED_FRACTION_SLOT],
            latch,
            soft_latch,
        )
    else:
        shocked_fraction = _clip_keep_derivative(
            _max_keep_derivative(history[SHOCKED_FRACTION_SLOT], latch),
            0.0,
            1.0,
        )

    # -------------------------------------------------------------
    # ================ ↑ Shocked-fraction latch ↑ ==================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ===================== ↓ Accumulators ↓ =======================
    # -------------------------------------------------------------

    density = primitive_state[registered_variables.density_index]
    # Both accumulators are monotone non-decreasing along a particle path by
    # construction, so any negative value is WENO ringing where the advected
    # field meets its own sharp edge at the shock front. Clamp it: a negative
    # ionization age or a negative time since shocking is meaningless
    # downstream, and the clamp cannot mask a real effect because the true value
    # can never be below zero.
    time_since_shock = _max_keep_derivative(
        history[TIME_SINCE_SHOCK_SLOT] + shocked_fraction * dt,
        0.0,
    )
    density_time = _max_keep_derivative(
        history[DENSITY_TIME_SLOT] + shocked_fraction * density * dt,
        0.0,
    )

    # -------------------------------------------------------------
    # ===================== ↑ Accumulators ↑ =======================
    # -------------------------------------------------------------

    # ``entropy_initial`` is a t = 0 label: advected, never rewritten (except
    # for the reset of a pathological label above).
    updated_history = [None] * NUM_SHOCK_HISTORY_SCALARS
    updated_history[ENTROPY_INITIAL_SLOT] = entropy_initial
    updated_history[SHOCKED_FRACTION_SLOT] = shocked_fraction
    updated_history[TIME_SINCE_SHOCK_SLOT] = time_since_shock
    updated_history[DENSITY_TIME_SLOT] = density_time
    return jnp.stack(updated_history, axis=0)
