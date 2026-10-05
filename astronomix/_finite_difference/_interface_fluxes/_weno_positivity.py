"""Positivity-preserving recombination of the WENO split fluxes.

The finite-difference WENO flux at an interface is the sum of two
reconstructed split fluxes, ``F_hat = f_hat^+ + f_hat^-``, upwinded from the
left and from the right. Take ``alpha`` = the largest wave speed on the
stencil. Splitting characteristic field ``s`` with speed ``alpha_s <= alpha``
makes each split flux a scaled vector

    f^+- = +-(alpha / 2) w^+-,      w^+- = w~^+- + z^+-,
    w~^+- = q +- F(q) / alpha,      z^+- = sum_s (alpha_s / alpha - 1) R_s (L_s q).

``w~`` is a physically admissible state (positive density, and pressure for
an ideal gas) whenever ``alpha >= |v_n| + c`` for the Euler equations; ideal
MHD needs a larger, state-dependent speed (see the admissible-splitting-speed
section at the end of this module). Write ``theta`` for the
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


def _gas_pressure(state, gamma, config: SimulationConfig, registered_variables: RegisteredVariables):
    """Gas pressure of a conserved-state-like vector (no floors)."""
    density = jnp.maximum(state[registered_variables.density_index], 1e-30)
    energy = state[registered_variables.energy_index]

    if config.dimensionality == 1 and not config.mhd:
        momentum_squared = state[registered_variables.momentum_index] ** 2
    else:
        momentum_squared = state[registered_variables.momentum_index.x] ** 2
        if config.dimensionality >= 2 or config.mhd:
            momentum_squared = momentum_squared + state[registered_variables.momentum_index.y] ** 2
        if config.dimensionality == 3 or config.mhd:
            momentum_squared = momentum_squared + state[registered_variables.momentum_index.z] ** 2

    internal_energy = energy - 0.5 * momentum_squared / density
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
        (base_density - density_floor) / jnp.maximum(-density_step, 1e-30),
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
    chord_root = fraction * base_margin / jnp.maximum(base_margin - end_margin, 1e-30)
    fraction = jnp.where(end_margin >= 0.0, fraction, chord_root)
    return jnp.where(base_margin > 0.0, fraction, 0.0)


def _upwind_split_states(conserved_state, cell_flux, common_speed):
    """The unshifted split states ``w~`` of the two upwind cells of each
    interface: ``q_i + F_i / alpha`` (for f^+) and ``q_{i+1} - F_{i+1} / alpha``
    (for f^-)."""
    alpha = jnp.maximum(common_speed, 1e-30)[None]
    plus_state = conserved_state + cell_flux / alpha
    minus_state = _shift(conserved_state, -1, axis=1) - _shift(cell_flux, -1, axis=1) / alpha
    return plus_state, minus_state


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
):
    """Assemble the interface flux from admissibility-scaled split fluxes.

    Args:
        conserved_state: Conserved state, the active axis leading the spatial axes.
        cell_flux: Physical flux at the cell centres.
        common_speed: The largest wave speed on each interface's stencil.
        plus_face_flux: The reconstructed ``f_hat^+`` (central part included).
        minus_face_flux: The reconstructed ``f_hat^-``.
        plus_shift: ``z^+`` at the speeds actually used.
        minus_shift: ``z^-`` likewise.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The interface flux at i + 1/2, aligned with cell i.
    """

    alpha = jnp.maximum(common_speed, 1e-30)[None]
    plus_unshifted, minus_unshifted = _upwind_split_states(conserved_state, cell_flux, common_speed)
    plus_owner = plus_unshifted + plus_shift
    minus_owner = minus_unshifted + minus_shift
    plus_step = 2.0 * plus_face_flux / alpha - plus_owner
    minus_step = -2.0 * minus_face_flux / alpha - minus_owner

    def face_scaling(owner, unshifted, step):
        return jnp.minimum(
            _admissible_fraction(owner, step, params, config, registered_variables),
            _admissible_fraction(unshifted, -step, params, config, registered_variables),
        )

    plus_theta = face_scaling(plus_owner, plus_unshifted, plus_step)
    minus_theta = face_scaling(minus_owner, minus_unshifted, minus_step)
    if config.weno_ad_frozen_weights:
        # theta is a nonlinear weight like the WENO omegas: freeze it with them
        plus_theta = jax.lax.stop_gradient(plus_theta)
        minus_theta = jax.lax.stop_gradient(minus_theta)

    plus_flux = 0.5 * alpha * (plus_owner + plus_theta[None] * plus_step)
    minus_flux = -0.5 * alpha * (minus_owner + minus_theta[None] * minus_step)
    interface_flux = plus_flux + minus_flux
    if config.mhd:
        # The normal field has no flux in the unlimited scheme (neither F nor
        # any right eigenvector has a B_normal component); unequal scalings of
        # the two split states would otherwise diffuse it. CT owns B_normal.
        interface_flux = interface_flux.at[registered_variables.magnetic_index.x].set(0.0)
    return interface_flux


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
    if magnetic_slots:
        # no normal-field flux (see positivity_preserving_interface_flux)
        interface_flux[magnetic_slots[0]] = interface_flux[0] * 0.0
    return interface_flux


# -----------------------------------------------------------------------------
# ↓ Admissible splitting speed (ideal MHD) ↓
# -----------------------------------------------------------------------------
# For the Euler equations the split states q +- F / alpha are admissible as
# soon as alpha >= |v_n| + c. For ideal MHD they are not, at any multiple of
# the fast speed (Wu 2018, SIAM J. Numer. Anal. 56, 2124): with B along the
# normal and v = 0,
#
#     p(q + F / alpha) / (gamma - 1) = p / (gamma - 1) - (p - B^2 / 2)^2 / (2 rho alpha^2),
#
# negative at low beta for alpha = c_f. The theta-scaling then has no
# admissible base, returns theta = 0, and the face falls back to first-order
# Rusanov in smooth flow (CP Alfven wave: order 0.2-0.6 instead of 5).
#
# The remedy inside the same argument is a larger splitting speed. The
# pressure of q + s F is concave in s, so {s : p(q + s F) >= kappa p(q)} is an
# interval [0, s*]: every alpha >= 1 / s* keeps both split states admissible,
# with a fraction kappa of the cell's pressure left as room for the WENO step.
# s* is bracketed between the chord root from s = 0 (admissible, as the chord
# of a concave function lies below it) and 1 / spectral radius, and the
# bracket is closed by geometric bisection (it can span many decades at low
# beta) plus a final chord; the lower end is admissible throughout. The proof's CFL
# condition lambda (alpha_left + alpha_right) <= 1 then holds with these
# speeds, so the time step uses them too. At low beta and slow flow
# alpha ~ 0.58 v_A / sqrt(beta) (kappa = 1/2): the price of provable positivity.

ADMISSIBLE_PRESSURE_FRACTION = 0.5
ADMISSIBLE_SPEED_ITERATIONS = 12
MHD_MAGNETIC_SLOTS = (4, 5, 6)


def local_mhd_normal_flux(state, gm1):
    """Ideal-MHD flux along the normal of a local (rho, m_n, m_t1, m_t2, B_n,
    B_t1, B_t2, E) tuple (no floors)."""
    density, mn, mt1, mt2, bn, bt1, bt2, energy = state
    inverse_density = 1.0 / density
    vn, vt1, vt2 = mn * inverse_density, mt1 * inverse_density, mt2 * inverse_density
    total_pressure = (
        _local_gas_pressure(state, gm1, MHD_MAGNETIC_SLOTS) + 0.5 * (bn * bn + bt1 * bt1 + bt2 * bt2)
    )
    v_dot_b = vn * bn + vt1 * bt1 + vt2 * bt2
    return (
        mn,
        mn * vn + total_pressure - bn * bn,
        mt1 * vn - bn * bt1,
        mt2 * vn - bn * bt2,
        0.0 * bn,
        vn * bt1 - bn * vt1,
        vn * bt2 - bn * vt2,
        (energy + total_pressure) * vn - bn * v_dot_b,
    )


def local_admissible_speed(state, flux, spectral_radius, gm1, magnetic_slots=MHD_MAGNETIC_SLOTS):
    """Splitting speed of a cell, at least its spectral radius, at which both
    split states ``q +- F / alpha`` keep ``ADMISSIBLE_PRESSURE_FRACTION`` of the
    cell's gas pressure (``state`` and ``flux`` are local tuples, energy last).
    A cell without positive pressure keeps its spectral radius."""
    pressure = _local_gas_pressure(state, gm1, magnetic_slots)
    target = ADMISSIBLE_PRESSURE_FRACTION * jnp.maximum(pressure, 0.0)
    s_radius = 1.0 / jnp.maximum(spectral_radius, 1e-30)

    def margin(s, sign):
        moved = tuple(state[slot] + (sign * s) * flux[slot] for slot in range(len(state)))
        return _local_gas_pressure(moved, gm1, magnetic_slots) - target

    speed = spectral_radius
    for sign in (1.0, -1.0):
        g_radius = margin(s_radius, sign)
        g_zero = pressure - target
        # bracket [s_lower, s_upper] of the root: the chord from s = 0 is
        # admissible by concavity, s_radius is not
        s_lower = s_radius * jnp.clip(g_zero / jnp.maximum(g_zero - g_radius, 1e-30), 0.0, 1.0)
        s_upper = s_radius
        for _ in range(ADMISSIBLE_SPEED_ITERATIONS):
            # geometric bisection: the bracket ratio can be many decades
            s_middle = jnp.sqrt(s_lower * s_upper)
            admissible = margin(s_middle, sign) >= 0.0
            s_lower = jnp.where(admissible, s_middle, s_lower)
            s_upper = jnp.where(admissible, s_upper, s_middle)
        # final chord across the bracket, admissible by concavity
        g_lower = margin(s_lower, sign)
        g_upper = margin(s_upper, sign)
        s_final = s_lower + (s_upper - s_lower) * jnp.clip(
            g_lower / jnp.maximum(g_lower - g_upper, 1e-30), 0.0, 1.0
        )
        needs_raise = (g_radius < 0.0) & (pressure > 0.0)
        s_admissible = jnp.where(needs_raise, s_final, s_radius)
        speed = jnp.maximum(speed, 1.0 / jnp.maximum(s_admissible, 1e-30))
    return speed


def _mhd_local_tuple(state, registered_variables: RegisteredVariables):
    """(rho, m_x, m_y, m_z, B_x, B_y, B_z, E) of an x-normal conserved array."""
    return (
        state[registered_variables.density_index],
        state[registered_variables.momentum_index.x],
        state[registered_variables.momentum_index.y],
        state[registered_variables.momentum_index.z],
        state[registered_variables.magnetic_index.x],
        state[registered_variables.magnetic_index.y],
        state[registered_variables.magnetic_index.z],
        state[registered_variables.energy_index],
    )


def admissible_splitting_speed(
    conserved_state,
    cell_flux,
    spectral_radius,
    common_speed,
    gamma,
    registered_variables: RegisteredVariables,
):
    """Raise the common splitting speed of each interface (ideal MHD, x normal)
    to the admissible speeds of its two upwind cells, i and i + 1."""
    cell_speed = local_admissible_speed(
        _mhd_local_tuple(conserved_state, registered_variables),
        _mhd_local_tuple(cell_flux, registered_variables),
        spectral_radius,
        gamma - 1.0,
    )
    return jnp.maximum(common_speed, jnp.maximum(cell_speed, _shift(cell_speed, -1, axis=0)))


def mhd_admissible_signal_speed(density, velocity, magnetic, pressure, spectral_radius, gamma, axis):
    """Per-cell admissible splitting speed along ``axis`` from primitives, for
    the time step (``velocity`` and ``magnetic`` are (x, y, z) triples)."""
    order = (axis,) + tuple(k for k in range(3) if k != axis)
    momentum = tuple(density * velocity[k] for k in order)
    field = tuple(magnetic[k] for k in order)
    energy = (
        pressure / (gamma - 1.0)
        + 0.5 * density * sum(velocity[k] * velocity[k] for k in range(3))
        + 0.5 * sum(magnetic[k] * magnetic[k] for k in range(3))
    )
    state = (density,) + momentum + field + (energy,)
    return local_admissible_speed(state, local_mhd_normal_flux(state, gamma - 1.0), spectral_radius, gamma - 1.0)
