"""Positivity-preserving recombination of the WENO split fluxes.

The finite-difference WENO flux at an interface is the sum of two
reconstructed split fluxes, ``F_hat = f_hat^+ + f_hat^-``, upwinded from the
left and from the right. Take ``alpha`` = the largest wave speed on the
stencil. Splitting characteristic field ``s`` with speed ``alpha_s <= alpha``
makes each split flux a scaled vector

    f^+- = +-(alpha / 2) w^+-,      w^+- = w~^+- + z^+-,
    w~^+- = q +- F(q) / alpha,      z^+- = sum_s (alpha_s / alpha - 1) R_s (L_s q).

``w~`` is a physically admissible state (positive density, and pressure for
an ideal gas) whenever ``alpha >= |v_n| + c`` for the Euler equations. For
ideal MHD it is not (Wu 2018), and the scalings act on the cell's own
mirror pairs and on its inflow states summed over the axes instead (see
``_joint_inflow_scalings``). Write ``theta`` for the
scaling of the WENO face value toward its upwind cell. Following Zhang & Shu
(2012, J. Comput. Phys. 231, 2245), the forward-Euler update of a cell is a
convex combination of admissible states if all of these hold:

    w + theta d   admissible   (the face value, which flows into the neighbour),
    w~ - theta d  admissible   (the mirror about the unshifted state),
    lambda (alpha_left + alpha_right) <= 1,

with ``d = w_hat - w``. The cell's own flux cancels between the two faces, so
face-local speeds are allowed. Two scalings in [0, 1] enforce this, both
inside the reconstruction:

* every field that carries mass is split with the common speed ``alpha``,
  the stencil's spectral radius. That makes the frozen-basis splitting
  monotone for every cell of the stencil, however differently the basis
  represents it (a low-density cell with a large fast speed next to dense
  gas). Per-field speeds chosen only to keep the split states admissible
  are positive but not robust: Mach-10 MHD turbulence still blows up
  (commit 96a191f). Fields that carry no mass (hydrodynamic shear waves,
  isothermal-MHD Alfven waves) keep their own speed. Their ``z`` leaves the
  density unchanged and only raises the pressure, so vortical modes keep the
  default dissipation;
* the face value is pulled toward its upwind state, ``w_hat -> w + theta d``,
  by the largest ``theta`` meeting the two conditions above. ``theta < 1``
  mixes the first-order candidate into the WENO combination with a weight set
  by admissibility instead of smoothness; ``theta = 1`` in smooth flow.

Both scalings use the same primitive: the largest ``t in [0, 1]`` with
``base + t * step`` admissible. Density is linear in ``t``. The pressure is
concave along any line in conserved space, so the chord between the two ends
lies below it and the chord's root is admissible: one closed-form step.
"""

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift


def mass_free_modes(config: SimulationConfig) -> tuple:
    """Indices of the characteristic fields that carry no mass and no
    energy-coupled density, in the mode order of the eigensystem modules: the
    shear waves of the Euler equations (both equations of state) and the
    Alfven waves of isothermal MHD. They keep their own splitting speed.

    Args:
        config: The simulation configuration.

    Returns:
        A tuple of mode indices.
    """
    if config.mhd:
        return (1, 4) if config.equation_of_state == ISOTHERMAL else ()
    if config.equation_of_state == ISOTHERMAL:
        return tuple(range(1, config.dimensionality))
    return tuple(range(2, config.dimensionality + 1))


def stencil_maximum(cell_field):
    """Maximum of a cell field over the six-point WENO stencil of each
    interface (cells i - 2 ... i + 3 for the interface at i + 1/2)."""
    return jnp.max(
        jnp.stack([_shift(cell_field, offset, axis=0) for offset in (2, 1, 0, -1, -2, -3)]),
        axis=0,
    )


# -----------------------------------------------------------------------------
# ↓ Array form (native kernel; variable axis leading) ↓
# -----------------------------------------------------------------------------


@jax.custom_jvp
def _floored_ratio(numerator, denominator):
    """
    Return ``numerator / max(denominator, 1e-30)``.

    The forward value is exactly the guarded division it replaces; only the
    derivative differs (see ``_floored_ratio_jvp``).
    """
    return numerator / jnp.maximum(denominator, 1e-30)


@_floored_ratio.defjvp
def _floored_ratio_jvp(primals, tangents):
    """
    Derivative of the guarded division that stays finite in single precision.

    JAX differentiates ``x / y`` through ``-x y**-2``. For the 1e-30 guard (in
    fact for any ``y`` below ~5e-20) ``y**-2`` overflows float32 to infinity,
    and the tangent multiplying it is exactly zero wherever ``jnp.maximum``
    picks the guard or a later ``where`` / ``clip`` discards the lane, so the
    product is ``0 * inf = NaN``, which reverse mode then spreads. The
    derivative is therefore only formed where ``y**-2`` is finite; elsewhere
    it is the guard's own derivative, zero.
    """
    numerator, denominator = primals
    numerator_tangent, denominator_tangent = tangents
    ratio = numerator / jnp.maximum(denominator, 1e-30)
    smallest_safe_denominator = 2.0 * float(
        np.sqrt(1.0 / np.finfo(jnp.result_type(denominator)).max)
    )
    differentiable = denominator > smallest_safe_denominator
    inverse_denominator = 1.0 / jnp.where(differentiable, denominator, 1.0)
    ratio_tangent = jnp.where(
        differentiable,
        (numerator_tangent - jnp.where(differentiable, ratio, 0.0) * denominator_tangent)
        * inverse_denominator,
        0.0,
    )
    return ratio, ratio_tangent


def _gas_pressure(state, gamma, config: SimulationConfig, registered_variables: RegisteredVariables):
    """Gas pressure of a conserved-state-like vector (no floors)."""
    density = state[registered_variables.density_index]
    energy = state[registered_variables.energy_index]

    if config.dimensionality == 1 and not config.mhd:
        momentum_squared = state[registered_variables.momentum_index] ** 2
    else:
        momentum_squared = state[registered_variables.momentum_index.x] ** 2
        if config.dimensionality >= 2 or config.mhd:
            momentum_squared = momentum_squared + state[registered_variables.momentum_index.y] ** 2
        if config.dimensionality == 3 or config.mhd:
            momentum_squared = momentum_squared + state[registered_variables.momentum_index.z] ** 2

    internal_energy = energy - _floored_ratio(0.5 * momentum_squared, density)
    if config.mhd:
        internal_energy = internal_energy - 0.5 * (
            state[registered_variables.magnetic_index.x] ** 2
            + state[registered_variables.magnetic_index.y] ** 2
            + state[registered_variables.magnetic_index.z] ** 2
        )
    return (gamma - 1.0) * internal_energy


def _admissible_fraction(
    base,
    step,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Largest t in [0, 1] with ``base + t * step`` admissible.

    The floors are the configured minima, capped at half of ``base``'s own
    value so a state sitting below a floor is not forced to t = 0. A
    non-admissible ``base`` gives t = 0.

    Args:
        base: The starting state (an admissible split state).
        step: The direction of the segment.
        params: The simulation parameters (floors, gamma).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The fraction t, one per interface.
    """

    density_index = registered_variables.density_index
    base_density = base[density_index]
    density_floor = jnp.minimum(params.minimum_density, 0.5 * base_density)
    density_step = step[density_index]
    fraction = jnp.where(
        density_step < 0.0,
        _floored_ratio(base_density - density_floor, -density_step),
        1.0,
    )
    fraction = jnp.clip(jnp.where(base_density > 0.0, fraction, 0.0), 0.0, 1.0)

    if config.equation_of_state != IDEAL_GAS:
        return fraction

    gamma = params.gamma
    base_pressure = _gas_pressure(base, gamma, config, registered_variables)
    pressure_floor = jnp.minimum(params.minimum_pressure, 0.5 * jnp.maximum(base_pressure, 0.0))
    base_margin = base_pressure - pressure_floor
    end_margin = _gas_pressure(base + fraction[None] * step, gamma, config, registered_variables) - pressure_floor
    chord_root = _floored_ratio(fraction * base_margin, base_margin - end_margin)
    fraction = jnp.where(end_margin >= 0.0, fraction, chord_root)
    return jnp.where(base_margin > 0.0, fraction, 0.0)


def _upwind_split_states(conserved_state, cell_flux, common_speed, axis=0):
    """The unshifted split states ``w~`` of the two upwind cells of each
    interface: ``q_i + F_i / alpha`` (for f^+) and ``q_{i+1} - F_{i+1} / alpha``
    (for f^-); ``axis`` is the spatial sweep axis."""
    alpha = jnp.maximum(common_speed, 1e-30)[None]
    plus_state = conserved_state + cell_flux / alpha
    minus_state = (
        _shift(conserved_state, -1, axis=axis + 1) - _shift(cell_flux, -1, axis=axis + 1) / alpha
    )
    return plus_state, minus_state


# Ideal MHD: the split states q +- F / alpha of one cell are often not
# admissible at the fast speed (Wu 2018), but the per-cell decomposition of
# the forward-Euler update only needs two weighted PAIRS to be:
#
#   q_i^{n+1} = (1 - lambda S) q_i + (lambda S / 2) (own_i + in_i),   S = alpha_L + alpha_R,
#   own_i = q_i - (alpha_R theta+_{i+1/2} d+_{i+1/2} + alpha_L theta-_{i-1/2} d-_{i-1/2}) / S,
#   in_i  = (alpha_L w+_{i-1} + alpha_R w-_{i+1}) / S
#           + (alpha_L theta+_{i-1/2} d+_{i-1/2} + alpha_R theta-_{i+1/2} d-_{i+1/2}) / S.
#
# The cell's own flux cancels in own_i, whose base is q_i itself; in_i's base is
# the first-order Lax-Friedrichs inflow, in which the magnetic-tension terms of
# the two neighbours cancel up to their B_n difference (Wu's generalized
# splitting property; random low-beta pairs fail at the fast speed in ~1e-4
# of cases, single split states in ~70 %). Each theta enters one pair of each
# of its two cells; a pair is admissible for every theta in [0, t]^2 once the
# corners (t, 0), (0, t), (t, t) are (convexity), and a face takes the
# smaller t of its two pairs.


def _pair_fraction(base, first, second, params, config, registered_variables):
    """Largest t with base + a first + b second admissible for all a, b in [0, t]."""
    return jnp.minimum(
        jnp.minimum(
            _admissible_fraction(base, first, params, config, registered_variables),
            _admissible_fraction(base, second, params, config, registered_variables),
        ),
        _admissible_fraction(base, first + second, params, config, registered_variables),
    )


def _paired_scalings(
    conserved_state, common_speed, plus_unshifted, minus_unshifted, plus_step, minus_step,
    params, config, registered_variables, axis=0,
):
    """theta+ and theta- per interface (aligned with cell i) from the own and
    inflow pairs of the cells on both sides."""
    alpha_right = jnp.maximum(common_speed, 1e-30)
    alpha_left = _shift(alpha_right, 1, axis=axis)
    weight_sum = alpha_left + alpha_right
    left_weight = (alpha_left / weight_sum)[None]
    right_weight = (alpha_right / weight_sum)[None]

    own_plus = -right_weight * plus_step
    own_minus = -left_weight * _shift(minus_step, 1, axis=axis + 1)
    own_fraction = _pair_fraction(conserved_state, own_plus, own_minus, params, config, registered_variables)

    inflow_base = left_weight * _shift(plus_unshifted, 1, axis=axis + 1) + right_weight * minus_unshifted
    inflow_plus = left_weight * _shift(plus_step, 1, axis=axis + 1)
    inflow_minus = right_weight * minus_step
    inflow_fraction = _pair_fraction(inflow_base, inflow_plus, inflow_minus, params, config, registered_variables)

    plus_theta = jnp.minimum(own_fraction, _shift(inflow_fraction, -1, axis=axis))
    minus_theta = jnp.minimum(inflow_fraction, _shift(own_fraction, -1, axis=axis))
    return plus_theta, minus_theta


# The per-axis inflow pair above is not the right object in multi-D. Its
# base fails wherever B_n varies along the axis (~7 % of the cells of
# Mach-20 turbulence), although that variation is mostly divergence-free and
# cancels between the axes. What the update needs is the cell's inflow summed
# over all axes,
#
#   in_i = B_i + sum_k theta_k s_k,  B_i = sum_d S_d base_d / sum_d S_d,
#   s_k = alpha_face d_face / sum_d S_d   (the 2 dim inflow faces),
#
# whose first-order part B_i was admissible in every cell of the failing
# states. Splitting it convexly, in_i = sum_k (B_i + n theta_k s_k) / n with
# n = 2 dim, gives each inflow face its own bound theta_k <= frac(B_i, n s_k).
# The own pairs stay per axis (base q_i).


def mhd_inflow_reference(conserved_state, params: SimulationParams, config: SimulationConfig,
                         registered_variables: RegisteredVariables):
    """The axis-summed first-order inflow state of every cell and the summed
    splitting speeds (ideal MHD, untransposed layout):

        B_i = sum_d [alpha_L q_{i-1} + alpha_R q_{i+1} + F_d(q_{i-1}) - F_d(q_{i+1})] / sum_d S_d,

    with alpha_L, alpha_R the splitting speeds of the faces i -+ 1/2 along d
    (stencil maxima of |v_d| + c_f, as in the kernels) and S_d = alpha_L + alpha_R.

    Returns:
        ``(B, sum_d S_d)``.
    """
    rv = registered_variables
    gamma = params.gamma
    density = conserved_state[rv.density_index]
    floored_density = jnp.maximum(density, params.minimum_density)
    pressure = jnp.maximum(_gas_pressure(conserved_state, gamma, config, rv), params.minimum_pressure)
    field = [conserved_state[index] for index in rv.magnetic_index]
    field_squared = sum(component * component for component in field) / floored_density
    sound_squared = gamma * pressure / floored_density
    numerator = jnp.zeros_like(conserved_state)
    speed_sum = jnp.zeros_like(density)
    for axis in range(config.dimensionality):
        discriminant = (sound_squared + field_squared) ** 2 - 4.0 * sound_squared * field[axis] ** 2 / floored_density
        fast_speed = jnp.sqrt(0.5 * (sound_squared + field_squared + jnp.sqrt(jnp.maximum(discriminant, 0.0))))
        radius = jnp.abs(conserved_state[rv.momentum_index[axis]] / density) + fast_speed
        alpha_right = jnp.max(
            jnp.stack([_shift(radius, offset, axis=axis) for offset in (2, 1, 0, -1, -2, -3)]), axis=0
        )
        alpha_left = _shift(alpha_right, 1, axis=axis)
        flux = mhd_physical_flux(conserved_state, gamma, rv, axis)
        numerator = numerator + (
            alpha_left[None] * _shift(conserved_state, 1, axis=axis + 1)
            + alpha_right[None] * _shift(conserved_state, -1, axis=axis + 1)
            + _shift(flux, 1, axis=axis + 1) - _shift(flux, -1, axis=axis + 1)
        )
        speed_sum = speed_sum + alpha_left + alpha_right
    return numerator / speed_sum[None], speed_sum


def _joint_inflow_scalings(
    conserved_state, common_speed, plus_step, minus_step, inflow_reference,
    params, config, registered_variables, axis=0,
):
    """theta+ and theta- per interface: own pairs per axis, inflow faces
    against the cell's axis-summed inflow ``inflow_reference = (B, sum S)``
    (in the sweep's layout)."""
    reference_state, speed_sum = inflow_reference
    alpha_right = jnp.maximum(common_speed, 1e-30)
    alpha_left = _shift(alpha_right, 1, axis=axis)
    own_total = alpha_left + alpha_right
    own_plus = -(alpha_right / own_total)[None] * plus_step
    own_minus = -(alpha_left / own_total)[None] * _shift(minus_step, 1, axis=axis + 1)
    own_fraction = _pair_fraction(conserved_state, own_plus, own_minus, params, config, registered_variables)

    share = (2.0 * config.dimensionality / speed_sum)[None]
    left_inflow = share * alpha_left[None] * _shift(plus_step, 1, axis=axis + 1)
    right_inflow = share * alpha_right[None] * minus_step
    left_fraction = _admissible_fraction(reference_state, left_inflow, params, config, registered_variables)
    right_fraction = _admissible_fraction(reference_state, right_inflow, params, config, registered_variables)

    plus_theta = jnp.minimum(own_fraction, _shift(left_fraction, -1, axis=axis))
    minus_theta = jnp.minimum(right_fraction, _shift(own_fraction, -1, axis=axis))
    return plus_theta, minus_theta


def positivity_preserving_interface_flux(
    conserved_state,
    cell_flux,
    common_speed,
    plus_face_flux,
    minus_face_flux,
    plus_shift,
    minus_shift,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int = 0,
    inflow_reference=None,
):
    """Assemble the interface flux from admissibility-scaled split fluxes.

    Args:
        conserved_state: Conserved state (variables first).
        cell_flux: Physical flux at the cell centres.
        common_speed: The largest wave speed on each interface's stencil.
        plus_face_flux: The reconstructed ``f_hat^+`` (central part included).
        minus_face_flux: The reconstructed ``f_hat^-``.
        plus_shift: ``z^+`` at the speeds actually used.
        minus_shift: ``z^-`` likewise.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The spatial sweep axis (0 for the native kernel, which moves the
            active axis to the front).
        inflow_reference: Ideal MHD: ``mhd_inflow_reference`` in the sweep's
            layout; the inflow faces are then limited jointly per cell. Without
            it, the per-axis pairs are used.

    Returns:
        The interface flux at i + 1/2, aligned with cell i.
    """

    alpha = jnp.maximum(common_speed, 1e-30)[None]
    plus_unshifted, minus_unshifted = _upwind_split_states(conserved_state, cell_flux, common_speed, axis)
    plus_owner = plus_unshifted + plus_shift
    minus_owner = minus_unshifted + minus_shift
    plus_step = 2.0 * plus_face_flux / alpha - plus_owner
    minus_step = -2.0 * minus_face_flux / alpha - minus_owner

    def face_scaling(owner, unshifted, step):
        return jnp.minimum(
            _admissible_fraction(owner, step, params, config, registered_variables),
            _admissible_fraction(unshifted, -step, params, config, registered_variables),
        )

    if config.mhd and config.equation_of_state == IDEAL_GAS and inflow_reference is not None:
        plus_theta, minus_theta = _joint_inflow_scalings(
            conserved_state, common_speed, plus_step, minus_step, inflow_reference,
            params, config, registered_variables, axis,
        )
    elif config.mhd and config.equation_of_state == IDEAL_GAS:
        plus_theta, minus_theta = _paired_scalings(
            conserved_state, common_speed, plus_unshifted, minus_unshifted, plus_step, minus_step,
            params, config, registered_variables, axis,
        )
    else:
        plus_theta = face_scaling(plus_owner, plus_unshifted, plus_step)
        minus_theta = face_scaling(minus_owner, minus_unshifted, minus_step)
    if config.weno_ad_frozen_weights:
        # theta is a nonlinear weight like the WENO omegas: freeze it with them
        plus_theta = jax.lax.stop_gradient(plus_theta)
        minus_theta = jax.lax.stop_gradient(minus_theta)

    plus_flux = 0.5 * alpha * (plus_owner + plus_theta[None] * plus_step)
    minus_flux = -0.5 * alpha * (minus_owner + minus_theta[None] * minus_step)
    # MHD: the normal-field component is the Lax-Friedrichs diffusion of B_n.
    # It must stay: the convex decomposition behind positivity includes it,
    # and with B_n held the energy would carry the magnetic energy of a B_n
    # never applied (at low beta enough to make p negative). CT ignores this
    # component and the cell-centred B is rebuilt from the faces, pressure held.
    return plus_flux + minus_flux


# -----------------------------------------------------------------------------
# ↓ Pallas-kernel form (per-cell local component tuples) ↓
# -----------------------------------------------------------------------------
# The Pallas kernels work on tuples of the LOCAL components (rho, m_normal,
# m_tangential..., [B_normal, B_tangential...], [E]) of one cell column; these
# mirror the array form operation for operation.


def _local_gas_pressure(state, gm1, magnetic_slots=()):
    """Ideal-gas pressure of a local (rho, m..., [B...], E) tuple (no floors);
    the magnetic components sit in ``magnetic_slots``."""
    density = jnp.maximum(state[0], 1e-30)
    momentum_slots = [slot for slot in range(1, len(state) - 1) if slot not in magnetic_slots]
    momentum_squared = state[momentum_slots[0]] * state[momentum_slots[0]]
    for slot in momentum_slots[1:]:
        momentum_squared = momentum_squared + state[slot] * state[slot]
    internal_energy = state[-1] - 0.5 * momentum_squared / density
    for slot in magnetic_slots:
        internal_energy = internal_energy - 0.5 * state[slot] * state[slot]
    return gm1 * internal_energy


def _local_admissible_fraction(base, step, gm1, rhomin, pgmin, ideal_gas, magnetic_slots):
    """``_admissible_fraction`` for local tuples (density only when not an
    ideal gas)."""
    base_density = base[0]
    density_floor = jnp.minimum(rhomin, 0.5 * base_density)
    fraction = jnp.where(
        step[0] < 0.0,
        (base_density - density_floor) / jnp.maximum(-step[0], 1e-30),
        1.0,
    )
    fraction = jnp.clip(jnp.where(base_density > 0.0, fraction, 0.0), 0.0, 1.0)
    if not ideal_gas:
        return fraction

    base_pressure = _local_gas_pressure(base, gm1, magnetic_slots)
    pressure_floor = jnp.minimum(pgmin, 0.5 * jnp.maximum(base_pressure, 0.0))
    base_margin = base_pressure - pressure_floor
    end_state = tuple(base[slot] + fraction * step[slot] for slot in range(len(step)))
    end_margin = _local_gas_pressure(end_state, gm1, magnetic_slots) - pressure_floor
    chord_root = fraction * base_margin / jnp.maximum(base_margin - end_margin, 1e-30)
    fraction = jnp.where(end_margin >= 0.0, fraction, chord_root)
    return jnp.where(base_margin > 0.0, fraction, 0.0)


def _local_upwind_split_states(left_state, right_state, left_flux, right_flux, common_speed):
    """``_upwind_split_states`` for local tuples."""
    alpha = jnp.maximum(common_speed, 1e-30)
    plus_state = tuple(left_state[slot] + left_flux[slot] / alpha for slot in range(len(left_state)))
    minus_state = tuple(right_state[slot] - right_flux[slot] / alpha for slot in range(len(right_state)))
    return plus_state, minus_state


def positivity_preserving_flux_local(
    left_state,
    right_state,
    left_flux,
    right_flux,
    plus_face_flux,
    minus_face_flux,
    plus_shift,
    minus_shift,
    common_speed,
    gm1,
    rhomin,
    pgmin,
    ideal_gas=True,
    magnetic_slots=(),
):
    """``positivity_preserving_interface_flux`` for one interface of a
    Pallas WENO kernel.

    Args:
        left_state, right_state: Local conserved tuples of cells i and i + 1.
        left_flux, right_flux: Their physical fluxes.
        plus_face_flux, minus_face_flux: The reconstructed split fluxes
            (central part included).
        plus_shift, minus_shift: ``z^+-`` at the speeds actually used.
        common_speed: The largest wave speed on the stencil.
        gm1: gamma - 1 (unused when not an ideal gas).
        rhomin, pgmin: The density and pressure floors.
        ideal_gas: Whether the pressure constraint applies.
        magnetic_slots: Local slots of the magnetic field (MHD); the first is
            the normal field, whose flux is set to zero.

    Returns:
        The interface flux as a list of local components.
    """
    ncomp = len(left_state)
    alpha = jnp.maximum(common_speed, 1e-30)
    plus_unshifted, minus_unshifted = _local_upwind_split_states(
        left_state, right_state, left_flux, right_flux, common_speed
    )
    plus_owner = tuple(plus_unshifted[slot] + plus_shift[slot] for slot in range(ncomp))
    minus_owner = tuple(minus_unshifted[slot] + minus_shift[slot] for slot in range(ncomp))
    plus_step = tuple(2.0 * plus_face_flux[slot] / alpha - plus_owner[slot] for slot in range(ncomp))
    minus_step = tuple(-2.0 * minus_face_flux[slot] / alpha - minus_owner[slot] for slot in range(ncomp))

    def face_scaling(owner, unshifted, step):
        mirror_step = tuple(-component for component in step)
        return jnp.minimum(
            _local_admissible_fraction(owner, step, gm1, rhomin, pgmin, ideal_gas, magnetic_slots),
            _local_admissible_fraction(unshifted, mirror_step, gm1, rhomin, pgmin, ideal_gas, magnetic_slots),
        )

    plus_theta = face_scaling(plus_owner, plus_unshifted, plus_step)
    minus_theta = face_scaling(minus_owner, minus_unshifted, minus_step)
    interface_flux = [
        0.5 * alpha * (plus_owner[slot] + plus_theta * plus_step[slot])
        - 0.5 * alpha * (minus_owner[slot] + minus_theta * minus_step[slot])
        for slot in range(ncomp)
    ]
    # (the normal-field component stays; see positivity_preserving_interface_flux)
    return interface_flux


def mhd_physical_flux(conserved_state, gamma, registered_variables: RegisteredVariables, axis: int):
    """Ideal-MHD flux along spatial ``axis`` in the registered variable layout
    (no floors, like the Pallas kernel's cell fluxes)."""
    density = conserved_state[registered_variables.density_index]
    momentum = [conserved_state[index] for index in registered_variables.momentum_index]
    field = [conserved_state[index] for index in registered_variables.magnetic_index]
    energy = conserved_state[registered_variables.energy_index]
    velocity = [component / density for component in momentum]
    magnetic_pressure = 0.5 * sum(component * component for component in field)
    total_pressure = (gamma - 1.0) * (
        energy - 0.5 * sum(m * v for m, v in zip(momentum, velocity)) - magnetic_pressure
    ) + magnetic_pressure
    v_dot_b = sum(v * b for v, b in zip(velocity, field))
    flux = jnp.zeros_like(conserved_state)
    flux = flux.at[registered_variables.density_index].set(momentum[axis])
    for k, (m_index, b_index) in enumerate(zip(registered_variables.momentum_index, registered_variables.magnetic_index)):
        momentum_flux = momentum[k] * velocity[axis] - field[axis] * field[k]
        flux = flux.at[m_index].set(momentum_flux + total_pressure if k == axis else momentum_flux)
        flux = flux.at[b_index].set(velocity[axis] * field[k] - field[axis] * velocity[k])
    return flux.at[registered_variables.energy_index].set(
        (energy + total_pressure) * velocity[axis] - field[axis] * v_dot_b
    )
