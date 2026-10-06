"""
GLM divergence cleaning for the VL2 finite-volume MHD scheme.

AthenaPK controls the divergence of the cell-centred magnetic field with the
hyperbolic-parabolic generalised Lagrange multiplier (GLM) method of Dedner et
al. (2002). A scalar psi carries divergence errors away at the cleaning speed
``c_h`` — this happens inside the Riemann solver, which solves the decoupled
``(B_normal, psi)`` subsystem exactly — and psi is damped every stage by the
source term implemented here. As in AthenaPK, ``c_h`` is the largest
hyperbolic signal speed on the grid, so the cleaning waves never restrict the
time step.
"""

# typing
from jaxtyping import Array, Float

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
    _maximum_signal_speed_vl2,
)


def _glm_cleaning_speed(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """
    The hyperbolic divergence-cleaning speed ``c_h`` of the coming step: the
    largest signal speed ``|v_d| + c_fast,d`` of the state at its start.

    Args:
        primitive_state: The primitive state at the start of the step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The cleaning speed ``c_h``.
    """
    return _maximum_signal_speed_vl2(primitive_state, config, params, registered_variables)


def _psi_damping_factor(
    cleaning_speed,
    stage_time_step,
    config: SimulationConfig,
    params: SimulationParams,
):
    """
    The parabolic damping factor of psi for one stage,
    ``exp(-alpha * c_h * beta dt / dx)`` (Mignone & Tzeferacos 2010, eq. 27),
    where ``alpha`` (``params.glm_alpha``) is the ratio of the diffusive to the
    advective time scale of the cleaning.

    Args:
        cleaning_speed: The cleaning speed ``c_h``.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        The multiplicative damping factor.
    """
    return jnp.exp(-params.glm_alpha * cleaning_speed * stage_time_step / config.grid_spacing)


def _centered_difference(field, axis: int):
    """``field[i+1] - field[i-1]`` along a spatial axis of a single field."""
    return jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)


def _dedner_source(
    conserved_state: STATE_TYPE,
    primitive_state: STATE_TYPE,
    damping_factor,
    stage_time_step,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Apply AthenaPK's Dedner source to the freshly updated conserved state.

    The plain variant (AthenaPK's default ``dedner_plain``) only damps psi. The
    extended variant (``dedner_extended``) additionally adds the
    non-conservative ``-(div B) B`` momentum and ``-B . grad(psi)`` energy
    sources (Dedner et al. 2002), evaluated with central differences from the
    primitive state the stage started from, as in AthenaPK.

    Args:
        conserved_state: The conserved state after the stage's flux update.
        primitive_state: The primitive state at the start of the stage.
        damping_factor: The stage's psi damping factor (``_psi_damping_factor``).
        stage_time_step: The stage's time-step weight ``beta * dt``.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state with the source applied.
    """
    psi_index = registered_variables.magnetic_psi_index

    if config.glm_extended_source:
        magnetic_indices = tuple(registered_variables.magnetic_index)
        momentum_indices = tuple(registered_variables.momentum_index)
        psi = primitive_state[psi_index]

        # div(B) and B . grad(psi) from central differences over the active axes
        divergence_sum = 0.0
        field_dot_psi_gradient = 0.0
        for axis in range(config.dimensionality):
            field_component = primitive_state[magnetic_indices[axis]]
            divergence_sum = divergence_sum + _centered_difference(field_component, axis) / config.grid_spacing
            field_dot_psi_gradient = field_dot_psi_gradient + (
                field_component * _centered_difference(psi, axis) / config.grid_spacing
            )
        magnetic_divergence = 0.5 * divergence_sum
        field_dot_psi_gradient = 0.5 * field_dot_psi_gradient

        for component in range(3):
            conserved_state = conserved_state.at[momentum_indices[component]].add(
                -stage_time_step * magnetic_divergence * primitive_state[magnetic_indices[component]]
            )
        conserved_state = conserved_state.at[registered_variables.energy_index].add(
            -stage_time_step * field_dot_psi_gradient
        )

    return conserved_state.at[psi_index].multiply(damping_factor)
