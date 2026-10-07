"""
Operator-split advection of the dual-energy internal-energy density.

In high-Mach or low-beta flows the internal energy recovered from the total
energy, ``E - kinetic - magnetic``, is a small difference of large numbers and
is destroyed by floating-point cancellation. The dual-energy formalism (Bryan
et al. 1995) therefore evolves a separate internal-energy density ``g = rho e``
with its own gas-energy equation

    d g / d t + div(g v) = - p div(v)

and, where the total-energy value is unreliable, recovers the pressure from
``g`` instead. This module holds the advection of ``g`` used by the
finite-difference solver (hydro and MHD). The switch itself lives in the
pressure recoveries of the finite-difference scheme (the primitive recovery and
the coupled WENO recovery), and ``_evolve_state_fd`` re-syncs ``g`` from the
switched pressure at the end of every step.
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
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift


def _momentum_indices(config, registered_variables):
    """The momentum-density indices of the conserved state, one per dimension."""
    if config.dimensionality == 1:
        return [registered_variables.momentum_index]
    if config.dimensionality == 2:
        return [registered_variables.momentum_index.x, registered_variables.momentum_index.y]
    return [
        registered_variables.momentum_index.x,
        registered_variables.momentum_index.y,
        registered_variables.momentum_index.z,
    ]


def _velocities_from_conserved(conserved_state, config, registered_variables):
    """The velocity components ``momentum / rho`` of the conserved state, with
    the density floored to a tiny positive value."""
    density = jnp.maximum(conserved_state[registered_variables.density_index], 1e-30)
    return [
        conserved_state[momentum_index] / density
        for momentum_index in _momentum_indices(config, registered_variables)
    ]


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def advect_internal_energy(
    internal_energy_density: Float[Array, "..."],
    conserved_state: STATE_TYPE,
    pressure: Float[Array, "..."],
    dt: Union[float, Float[Array, ""]],
    grid_spacing: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    One operator-split update of ``g`` over ``dt``, solving
    ``d g / d t + div(g v) = - p div(v)``.

    The conservative advection ``div(g v)`` is first-order upwind with the face
    velocity ``(v_i + v_{i+1}) / 2``; the work term ``p div(v)`` uses a central
    velocity divergence. The stencils are periodic shifts; with ghost-cell
    boundaries the ghost cells supply the boundary values. First order is
    dissipative but stable and conservative, and deliberately simple: the dual
    energy only matters where the total-energy internal energy is unusable,
    and there a robust low-order ``g`` beats a cancellation-destroyed
    high-order one.

    Args:
        internal_energy_density: The dual-energy density ``g`` (one field).
        conserved_state: The conserved state at the start of the step, whose
            momentum and density define the advecting velocity.
        pressure: The pressure at the start of the step (for the work term).
        dt: The time step.
        grid_spacing: The cell width.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The updated internal-energy density.
    """
    velocities = _velocities_from_conserved(conserved_state, config, registered_variables)
    dt_over_dx = dt / grid_spacing

    internal_energy_flux_divergence = jnp.zeros_like(internal_energy_density)
    velocity_divergence = jnp.zeros_like(internal_energy_density)
    for axis, velocity in enumerate(velocities):
        # The field carries no leading variable axis, so the spatial axis is
        # the array axis.
        internal_energy_right = _shift(internal_energy_density, -1, axis=axis)
        velocity_right = _shift(velocity, -1, axis=axis)

        # Face velocity at i+1/2 and the first-order upwind value of g there.
        face_velocity = 0.5 * (velocity + velocity_right)
        internal_energy_face = jnp.where(
            face_velocity >= 0.0,
            internal_energy_density,
            internal_energy_right,
        )
        flux_right_face = face_velocity * internal_energy_face
        flux_left_face = _shift(flux_right_face, 1, axis=axis)
        internal_energy_flux_divergence = internal_energy_flux_divergence + (
            flux_right_face - flux_left_face
        )

        # Central divergence of the velocity for the pdV work.
        velocity_divergence = velocity_divergence + 0.5 * (
            velocity_right - _shift(velocity, 1, axis=axis)
        )

    return (
        internal_energy_density
        - dt_over_dx * internal_energy_flux_divergence
        - dt_over_dx * pressure * velocity_divergence
    )
