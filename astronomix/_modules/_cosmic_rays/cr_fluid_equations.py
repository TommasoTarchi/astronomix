"""
Two-fluid (gas + cosmic-ray) equation-of-state helpers.

These routines convert between primitive and conserved quantities for a fluid
that carries a cosmic-ray component alongside the thermal gas. The cosmic rays
are tracked through ``n_cr = P_cr ** (1 / gamma_cr)`` (an advected scalar), so
the cosmic-ray pressure is recovered as ``P_cr = n_cr ** gamma_cr``. The total
pressure stored in ``pressure_index`` is the sum of the gas and cosmic-ray
pressures.

NOTE: The conversions that involve the velocity read a single velocity
component, so they only support 1D setups; generalising them to 2D / 3D needs
the full kinetic energy.
"""

# general
from functools import partial

# typing
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables

# WARNING: the adiabatic indices are fixed here rather than read from
# ``SimulationParams``. They should eventually be sourced from the simulation
# parameters so a run can override them consistently.
gamma_gas = 5 / 3
gamma_cr = 4 / 3


# -------------------------------------------------------------
# ============ ↓ AD- and round-off-safe conversions ↓ ==========
# -------------------------------------------------------------


def cosmic_ray_pressure_from_n(n_cr, adiabatic_index_cr=gamma_cr):
    """
    ``P_cr = n_cr ** gamma_cr`` with ``n_cr`` clipped at zero.

    A reconstruction or Riemann-solver undershoot can leave ``n_cr`` a hair
    below zero next to a CR front, and a non-integer power of a negative
    number is NaN. The clip only affects cells that would otherwise be NaN.
    The derivative ``gamma_cr * n ** (gamma_cr - 1)`` is finite (zero) at
    ``n = 0``, so this is AD-safe as it stands.

    Args:
        n_cr: The advected cosmic-ray scalar ``P_cr ** (1 / gamma_cr)``.
        adiabatic_index_cr: The adiabatic index of the cosmic-ray fluid.

    Returns:
        The cosmic-ray pressure.
    """
    return jnp.maximum(n_cr, 0.0) ** adiabatic_index_cr


def cosmic_ray_n_from_pressure(p_cr, adiabatic_index_cr=gamma_cr):
    """
    ``n_cr = P_cr ** (1 / gamma_cr)``, AD-safe at ``P_cr = 0``.

    The naive power has an infinite derivative at zero, which turns every
    CR-free cell into a NaN tangent (inf * 0) under ``jax.jvp`` / ``jax.grad``,
    and CR-free regions are common (e.g. any initial condition without CRs).
    The double-``where`` evaluates the power only on strictly positive
    pressures and returns ``n = 0`` with a zero derivative elsewhere; the
    forward value is unchanged for every ``P_cr >= 0``.

    Args:
        p_cr: The cosmic-ray pressure.
        adiabatic_index_cr: The adiabatic index of the cosmic-ray fluid.

    Returns:
        The advected cosmic-ray scalar ``n_cr``.
    """
    positive = p_cr > 0.0
    p_safe = jnp.where(positive, p_cr, 1.0)
    return jnp.where(positive, p_safe ** (1.0 / adiabatic_index_cr), 0.0)


# -------------------------------------------------------------
# ============ ↑ AD- and round-off-safe conversions ↑ ==========
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["registered_variables"])
def total_energy_from_primitives_with_crs(
    primitive_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculate the total energy density from the primitive variables of a
    fluid with cosmic rays.

    Args:
        primitive_state: The primitive state array.
        registered_variables: The registered variables.

    Returns:
        The total (gas kinetic + gas thermal + cosmic-ray) energy density.
    """

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index]
    )

    # Cosmic-ray energy density from its (relativistic) equation of state.
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # The stored pressure is the total; the gas pressure is what remains after
    # removing the cosmic-ray contribution.
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )

    # Gas energy density: internal (thermal) plus kinetic.
    density = primitive_state[registered_variables.density_index]
    velocity = primitive_state[registered_variables.velocity_index]
    gas_energy = gas_pressure / (gamma_gas - 1) + 0.5 * density * velocity**2

    # Total energy density is the sum of the two components.
    total_energy = gas_energy + cosmic_ray_energy

    return total_energy


@partial(jax.jit, static_argnames=["registered_variables"])
def gas_pressure_from_primitives_with_crs(
    primitive_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculate the gas pressure from the primitive state of a fluid with
    cosmic rays.

    Args:
        primitive_state: The primitive state array.
        registered_variables: The registered variables.

    Returns:
        The gas pressure.
    """

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index]
    )

    # The stored pressure is the total, so subtract the cosmic-ray part.
    return primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure


@partial(jax.jit, static_argnames=["registered_variables"])
def total_pressure_from_conserved_with_crs(
    conserved_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculate the total pressure from the conserved state of a fluid with
    cosmic rays.

    Args:
        conserved_state: The conserved state array (the energy slot holds the
            total energy density).
        registered_variables: The registered variables.

    Returns:
        The total (gas + cosmic-ray) pressure.
    """

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        conserved_state[registered_variables.cosmic_ray_n_index]
    )

    # Cosmic-ray energy density from its equation of state.
    cosmic_ray_energy = cosmic_ray_pressure / (gamma_cr - 1)

    # In the conserved state the energy slot holds the total energy density;
    # the gas energy is what remains after removing the cosmic-ray part.
    gas_energy = (
        conserved_state[registered_variables.pressure_index] - cosmic_ray_energy
    )

    # Back out the gas pressure from the gas energy by removing the kinetic part.
    density = conserved_state[registered_variables.density_index]
    velocity = conserved_state[registered_variables.velocity_index] / density
    gas_pressure = (gas_energy - 0.5 * density * velocity**2) * (gamma_gas - 1)

    # The total pressure is the sum of both pressure components.
    total_pressure = cosmic_ray_pressure + gas_pressure

    return total_pressure


@partial(jax.jit, static_argnames=["registered_variables"])
def speed_of_sound_crs(
    primitive_state: Float[Array, "num_vars num_cells"],
    registered_variables: RegisteredVariables,
) -> Float[Array, "num_cells"]:
    """
    Calculate the sound speed of a fluid with cosmic rays,
    ``c_s = sqrt((gamma_gas * P_gas + gamma_cr * P_cr) / rho)``.

    Args:
        primitive_state: The primitive state array.
        registered_variables: The registered variables.

    Returns:
        The effective sound speed of the composite fluid.
    """

    # Recover the cosmic-ray pressure from the advected scalar n_cr.
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index]
    )

    # The stored pressure is the total, so subtract the cosmic-ray part.
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )

    # Both components stiffen the fluid, so the effective sound speed mixes the
    # gas and cosmic-ray pressures weighted by their adiabatic indices.
    return jnp.sqrt(
        (gamma_gas * gas_pressure + gamma_cr * cosmic_ray_pressure)
        / primitive_state[registered_variables.density_index]
    )
