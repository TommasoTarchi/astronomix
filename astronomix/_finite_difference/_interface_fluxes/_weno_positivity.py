"""Positivity-preserving recombination of the WENO split fluxes.

The finite-difference WENO flux at an interface is the sum of two
reconstructed split fluxes, ``F_hat = f_hat^+ + f_hat^-``, the first upwinded
from the left cell, the second from the right. With ONE splitting speed
``alpha`` for every field that carries mass, each split flux is a scaled
STATE,

    f^+- = +-(alpha / 2) w^+-,        w^+- = q +- F(q) / alpha,

and ``w^+-`` is a physically admissible state (positive density, and positive
pressure for an ideal gas) whenever ``alpha >= |v_n| + c``. Zhang & Shu (2012,
J. Comput. Phys. 231, 2245) showed that the high-order update is then
positivity preserving under a CFL condition provided every reconstructed face
value ``w_hat`` and its mirror ``2 w - w_hat`` about the upwind cell's state
``w`` are admissible as well. Both are enforced here by the scaling

    w_hat  <-  w + theta (w_hat - w),        theta in [0, 1] maximal,

applied separately to the two split fluxes of every interface. This is a
change to the reconstruction itself: ``theta < 1`` mixes the first-order
(single-cell) candidate into the WENO combination with a weight set by
admissibility rather than smoothness, and ``theta = 1`` in smooth flow, where
the scheme is unchanged.

Fields that carry no mass (hydrodynamic shear waves, Alfven waves of the
isothermal system) keep their own, smaller splitting speed. Their contribution
shifts ``w`` along the field's right eigenvector, which leaves the density
untouched and only raises the pressure, so ``w`` stays admissible.
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
    """Indices of the characteristic fields whose right eigenvector has no
    density component, in the mode order of the eigensystem modules.

    Only the fields for which the admissibility argument is proven are
    listed: the shear waves of the Euler equations (both equations of state)
    and the Alfven waves of isothermal MHD. Ideal-MHD Alfven waves also
    carry energy and stay on the common splitting speed.

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


def _admissible_scaling(
    owner_state,
    step,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Largest theta in [0, 1] keeping ``owner_state +- theta * step``
    admissible.

    Density is linear along the segment, so its bound is closed form; the
    pressure bound uses the concavity of the pressure along the segment.
    The floors are the configured minima, capped at half the owner's own
    value so a cell sitting below a floor is not forced to first order.

    Args:
        owner_state: The split state ``w`` of the upwind cell.
        step: The reconstruction increment ``w_hat - w``.
        params: The simulation parameters (floors, gamma).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The scaling factor theta, one per interface.
    """

    density_index = registered_variables.density_index
    owner_density = owner_state[density_index]
    density_floor = jnp.minimum(params.minimum_density, 0.5 * owner_density)
    theta = (owner_density - density_floor) / jnp.maximum(jnp.abs(step[density_index]), 1e-30)
    theta = jnp.clip(jnp.where(owner_density > 0.0, theta, 0.0), 0.0, 1.0)

    if config.equation_of_state != IDEAL_GAS:
        return theta

    # The pressure is concave along any line in conserved space, so on
    # [0, theta] it lies above the chord between its end values. Where the
    # density-limited theta leaves the pressure (of the face value or of its
    # mirror) below the floor, the chord's root is therefore admissible: one
    # closed-form step, no iteration, slightly cautious only where the
    # limiter acts anyway.
    gamma = params.gamma
    owner_pressure = _gas_pressure(owner_state, gamma, config, registered_variables)
    pressure_floor = jnp.minimum(params.minimum_pressure, 0.5 * jnp.maximum(owner_pressure, 0.0))
    owner_margin = owner_pressure - pressure_floor

    for direction in (1.0, -1.0):
        end_margin = _gas_pressure(
            owner_state + direction * theta[None] * step, gamma, config, registered_variables
        ) - pressure_floor
        chord_root = theta * owner_margin / jnp.maximum(owner_margin - end_margin, 1e-30)
        theta = jnp.where(end_margin >= 0.0, theta, chord_root)

    return jnp.where(owner_margin > 0.0, theta, 0.0)


def positivity_preserving_interface_flux(
    conserved_state,
    cell_flux,
    common_speed,
    plus_correction,
    minus_correction,
    plus_owner_shift,
    minus_owner_shift,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Assemble the interface flux from admissibility-scaled split fluxes.

    Args:
        conserved_state: Conserved state, the active axis leading the spatial axes.
        cell_flux: Physical flux at the cell centres.
        common_speed: The common splitting speed alpha of each interface.
        plus_correction: Sum over fields of the high-order part of ``f_hat^+``
            (the WENO term ``-phi^+`` back-projected, plus the central-part
            correction of the fields that keep their own speed).
        minus_correction: The same for ``f_hat^-``.
        plus_owner_shift: Shift of the left cell's ``w^+`` caused by the
            fields that keep their own speed.
        minus_owner_shift: The same for the right cell's ``w^-``.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The interface flux at i + 1/2, aligned with cell i.
    """

    alpha = jnp.maximum(common_speed, 1e-30)[None]

    # Fourth-order central part of the two split fluxes; together they give
    # the familiar 1/12 (-F + 7F + 7F - F) term of the unsplit scheme.
    def central(field):
        return (1.0 / 12.0) * (
            -_shift(field, 1, axis=1)
            + 7.0 * field
            + 7.0 * _shift(field, -1, axis=1)
            - _shift(field, -2, axis=1)
        )

    central_flux = central(cell_flux)
    central_state = central(conserved_state)
    plus_face_flux = 0.5 * (central_flux + alpha * central_state) + plus_correction
    minus_face_flux = 0.5 * (central_flux - alpha * central_state) + minus_correction

    # The split states of the upwind cells (left for f^+, right for f^-) and
    # the reconstruction increments toward the WENO face values.
    plus_owner = conserved_state + cell_flux / alpha + plus_owner_shift
    minus_owner = (
        _shift(conserved_state, -1, axis=1) - _shift(cell_flux, -1, axis=1) / alpha + minus_owner_shift
    )
    plus_step = 2.0 * plus_face_flux / alpha - plus_owner
    minus_step = -2.0 * minus_face_flux / alpha - minus_owner

    plus_theta = _admissible_scaling(plus_owner, plus_step, params, config, registered_variables)
    minus_theta = _admissible_scaling(minus_owner, minus_step, params, config, registered_variables)
    if config.weno_ad_frozen_weights:
        # theta is a nonlinear weight like the WENO omegas: freeze it with them
        plus_theta = jax.lax.stop_gradient(plus_theta)
        minus_theta = jax.lax.stop_gradient(minus_theta)

    plus_flux = 0.5 * alpha * (plus_owner + plus_theta[None] * plus_step)
    minus_flux = -0.5 * alpha * (minus_owner + minus_theta[None] * minus_step)
    return plus_flux + minus_flux


# -----------------------------------------------------------------------------
# ↓ Pallas-kernel form (per-cell local component tuples) ↓
# -----------------------------------------------------------------------------
# The Pallas hydro kernel works on tuples of the LOCAL components
# (rho, m_normal, m_tangential..., E) of one cell column; these mirror
# _gas_pressure / _admissible_scaling / positivity_preserving_interface_flux
# operation for operation.


def _local_gas_pressure(state, gm1):
    """Ideal-gas pressure of a local (rho, m_n, m_t..., E) tuple (no floors)."""
    density = jnp.maximum(state[0], 1e-30)
    momentum_squared = state[1] * state[1]
    for slot in range(2, len(state) - 1):
        momentum_squared = momentum_squared + state[slot] * state[slot]
    return gm1 * (state[-1] - 0.5 * momentum_squared / density)


def _local_admissible_scaling(owner_state, step, gm1, rhomin, pgmin):
    """``_admissible_scaling`` for local tuples of an ideal gas."""
    owner_density = owner_state[0]
    density_floor = jnp.minimum(rhomin, 0.5 * owner_density)
    theta = (owner_density - density_floor) / jnp.maximum(jnp.abs(step[0]), 1e-30)
    theta = jnp.clip(jnp.where(owner_density > 0.0, theta, 0.0), 0.0, 1.0)

    owner_pressure = _local_gas_pressure(owner_state, gm1)
    pressure_floor = jnp.minimum(pgmin, 0.5 * jnp.maximum(owner_pressure, 0.0))
    owner_margin = owner_pressure - pressure_floor
    for direction in (1.0, -1.0):
        end_state = tuple(owner_state[slot] + direction * theta * step[slot] for slot in range(len(step)))
        end_margin = _local_gas_pressure(end_state, gm1) - pressure_floor
        chord_root = theta * owner_margin / jnp.maximum(owner_margin - end_margin, 1e-30)
        theta = jnp.where(end_margin >= 0.0, theta, chord_root)
    return jnp.where(owner_margin > 0.0, theta, 0.0)


def positivity_preserving_flux_local(
    left_state,
    right_state,
    left_flux,
    right_flux,
    plus_face_flux,
    minus_face_flux,
    plus_owner_shift,
    minus_owner_shift,
    common_speed,
    gm1,
    rhomin,
    pgmin,
):
    """``positivity_preserving_interface_flux`` for one interface of the
    Pallas hydro kernel.

    Args:
        left_state, right_state: Local conserved tuples of cells i and i + 1.
        left_flux, right_flux: Their physical fluxes.
        plus_face_flux, minus_face_flux: The reconstructed split fluxes
            (central part included).
        plus_owner_shift, minus_owner_shift: Split-state shifts from the
            fields on their own splitting speed.
        common_speed: The common splitting speed.
        gm1: gamma - 1.
        rhomin, pgmin: The density and pressure floors.

    Returns:
        The interface flux as a list of local components.
    """
    ncomp = len(left_state)
    alpha = jnp.maximum(common_speed, 1e-30)
    plus_owner = tuple(
        left_state[slot] + left_flux[slot] / alpha + plus_owner_shift[slot] for slot in range(ncomp)
    )
    minus_owner = tuple(
        right_state[slot] - right_flux[slot] / alpha + minus_owner_shift[slot] for slot in range(ncomp)
    )
    plus_step = tuple(2.0 * plus_face_flux[slot] / alpha - plus_owner[slot] for slot in range(ncomp))
    minus_step = tuple(-2.0 * minus_face_flux[slot] / alpha - minus_owner[slot] for slot in range(ncomp))

    plus_theta = _local_admissible_scaling(plus_owner, plus_step, gm1, rhomin, pgmin)
    minus_theta = _local_admissible_scaling(minus_owner, minus_step, gm1, rhomin, pgmin)
    return [
        0.5 * alpha * (plus_owner[slot] + plus_theta * plus_step[slot])
        - 0.5 * alpha * (minus_owner[slot] + minus_theta * minus_step[slot])
        for slot in range(ncomp)
    ]
