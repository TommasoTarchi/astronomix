"""Positivity-preserving recombination of the WENO split fluxes.

The finite-difference WENO flux at an interface is the sum of two
reconstructed split fluxes, ``F_hat = f_hat^+ + f_hat^-``, upwinded from the
left and from the right. Take ``alpha`` = the largest wave speed on the
stencil. Splitting every characteristic field ``s`` with its own speed
``alpha_s <= alpha`` makes each split flux a scaled vector

    f^+- = +-(alpha / 2) w^+-,      w^+- = w~^+- + z^+-,
    w~^+- = q +- F(q) / alpha,      z^+- = sum_s (alpha_s / alpha - 1) R_s (L_s q).

``w~`` is a physically admissible state (positive density, and pressure for
an ideal gas) whenever ``alpha >= |v_n| + c``. Write ``theta`` for the
scaling of the WENO face value toward its upwind cell. Following Zhang & Shu
(2012, J. Comput. Phys. 231, 2245), the forward-Euler update of a cell is a
convex combination of admissible states if all of these hold:

    w + theta d   admissible   (the face value, which flows into the neighbour),
    w~ - theta d  admissible   (the mirror about the unshifted state),
    lambda (alpha_left + alpha_right) <= 1,

with ``d = w_hat - w``. The cell's own flux cancels between the two faces, so
face-local speeds are allowed. Two scalings in [0, 1] enforce this, both
inside the reconstruction:

* the fraction ``eta`` of the way from the common speed ``alpha`` to the
  per-field speeds ``alpha_s`` (``alpha_s -> alpha - eta (alpha - alpha_s)``)
  is the largest for which both upwind split states ``w~ + eta z`` are
  admissible. It is fixed per interface before the reconstruction, because
  the speeds enter the WENO weights. In smooth subsonic flow ``eta = 1``, i.e.
  the ordinary per-field splitting;
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
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift


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


def admissible_speed_fraction(
    conserved_state,
    cell_flux,
    common_speed,
    full_plus_shift,
    full_minus_shift,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """The fraction eta of the way from the common splitting speed to the
    per-field speeds that keeps both upwind split states admissible.

    Args:
        conserved_state: Conserved state, the active axis leading the spatial axes.
        cell_flux: Physical flux at the cell centres.
        common_speed: The largest wave speed on each interface's stencil.
        full_plus_shift: ``z^+`` with every field on its own speed (eta = 1).
        full_minus_shift: ``z^-`` likewise.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        eta in [0, 1], one per interface.
    """
    plus_state, minus_state = _upwind_split_states(conserved_state, cell_flux, common_speed)
    return jnp.minimum(
        _admissible_fraction(plus_state, full_plus_shift, params, config, registered_variables),
        _admissible_fraction(minus_state, full_minus_shift, params, config, registered_variables),
    )


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


def admissible_speed_fraction_local(
    left_state, right_state, left_flux, right_flux, common_speed,
    full_plus_shift, full_minus_shift, gm1, rhomin, pgmin, ideal_gas=True, magnetic_slots=(),
):
    """``admissible_speed_fraction`` for one interface of a Pallas kernel."""
    plus_state, minus_state = _local_upwind_split_states(
        left_state, right_state, left_flux, right_flux, common_speed
    )
    return jnp.minimum(
        _local_admissible_fraction(plus_state, full_plus_shift, gm1, rhomin, pgmin, ideal_gas, magnetic_slots),
        _local_admissible_fraction(minus_state, full_minus_shift, gm1, rhomin, pgmin, ideal_gas, magnetic_slots),
    )


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
