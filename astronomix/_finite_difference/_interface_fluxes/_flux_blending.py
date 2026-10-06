"""Cold-crush blending of the WENO interface flux toward first-order Lax-Friedrichs.

    F_hat_{i+1/2} = (1 - w_{i+1/2}) F_WENO_{i+1/2} + w_{i+1/2} F_LLF_{i+1/2},

with an LLF weight ``w in [0, 1]`` from a temperature ramp on the colder
adjacent cell under compression (``PositivityConfig.coldcrush_blend``; see
``_coldcrush_blend_weight``). It damps the runaway compression of radiatively
cooled, ram-pressure-crushed gas once the grid resolves the cooling layer --
a dissipation of unresolved physics, not a positivity fix (positivity is
``weno_positivity_preserving``). Native-JAX post-process on the assembled
interface flux, applied before the divergence. CT-safe for MHD (CT rebuilds
single-valued edge EMFs from whatever face fluxes it is given).
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix functions
from astronomix._fluid_equations._dual_energy_switch import dual_energy_internal_energy
from astronomix._stencil_operations._stencil_operations import _shift


def _momentum_indices(config, registered_variables):
    """
    The registry indices of the momentum components, one per spatial dimension.

    Args:
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The list of momentum indices (x, y, z order, ``config.dimensionality``
        entries).
    """
    if config.dimensionality == 1:
        return [registered_variables.velocity_index]
    return [
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
    ][:config.dimensionality]


# -------------------------------------------------------------
# ===== ↓ Shared first-order Lax-Friedrichs interface flux ↓ ===
# -------------------------------------------------------------


def _local_lax_friedrichs_flux(
    conserved_state,
    axis,
    params,
    config,
    registered_variables,
    internal_energy_density=None,
):
    """
    First-order local Lax-Friedrichs (Rusanov) interface flux along ``axis``.

    Covers hydro and MHD, ideal-gas and isothermal. ``F_LLF[..., i]`` is the
    flux at interface ``i+1/2`` (cells ``i`` and ``i+1``), matching the WENO
    convention so the blended array feeds the existing
    ``-dt/dx (F_{i+1/2} - F_{i-1/2})`` divergence unchanged.

    The dual-energy switch matters here: without it the raw ``E - KE``
    recovery is destroyed by float32 cancellation in cold, kinetic-energy
    dominated cells, and a blend that activates there would inject fluxes
    built from a corrupted pressure.

    Args:
        conserved_state: The conserved state.
        axis: The spatial axis of the interfaces.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None. When given, the pressure recovery is switched exactly like
            in the WENO path.

    Returns:
        The full interface-flux array.
    """
    density_index = registered_variables.density_index
    minimum_density = params.minimum_density
    is_ideal_gas = (config.equation_of_state == IDEAL_GAS)
    is_mhd = bool(config.mhd)

    momentum_indices = _momentum_indices(config, registered_variables)
    normal_momentum_index = momentum_indices[axis]
    transverse_momentum_indices = [
        index for component, index in enumerate(momentum_indices) if component != axis
    ]

    if is_mhd:
        magnetic_indices = [
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        ]
        normal_magnetic_index = magnetic_indices[axis]
        transverse_magnetic_indices = [
            magnetic_indices[component] for component in range(3) if component != axis
        ]

    def right_neighbour(field):
        return _shift(field, -1, axis=axis)

    def right_neighbour_state(state):
        return _shift(state, -1, axis=axis + 1)

    # -------------------------------------------------------------
    # ============== ↓ Left and right face states ↓ ===============
    # -------------------------------------------------------------

    density_left = jnp.maximum(conserved_state[density_index], minimum_density)
    density_right = jnp.maximum(right_neighbour(conserved_state[density_index]), minimum_density)
    normal_momentum_left = conserved_state[normal_momentum_index]
    normal_momentum_right = right_neighbour(conserved_state[normal_momentum_index])
    normal_velocity_left = normal_momentum_left / density_left
    normal_velocity_right = normal_momentum_right / density_right

    transverse_velocities_left = [
        conserved_state[index] / density_left for index in transverse_momentum_indices
    ]
    transverse_velocities_right = [
        right_neighbour(conserved_state[index]) / density_right
        for index in transverse_momentum_indices
    ]

    if is_mhd:
        normal_field_left = conserved_state[normal_magnetic_index]
        normal_field_right = right_neighbour(conserved_state[normal_magnetic_index])
        transverse_fields_left = [
            conserved_state[index] for index in transverse_magnetic_indices
        ]
        transverse_fields_right = [
            right_neighbour(conserved_state[index]) for index in transverse_magnetic_indices
        ]
        field_squared_left = normal_field_left * normal_field_left
        field_squared_right = normal_field_right * normal_field_right
        for field_left, field_right in zip(transverse_fields_left, transverse_fields_right):
            field_squared_left = field_squared_left + field_left * field_left
            field_squared_right = field_squared_right + field_right * field_right

    if is_ideal_gas:
        gamma = params.gamma
        energy_left = conserved_state[registered_variables.energy_index]
        energy_right = right_neighbour(energy_left)
        kinetic_energy_left = 0.5 * (normal_momentum_left * normal_momentum_left) / density_left
        kinetic_energy_right = (
            0.5 * (normal_momentum_right * normal_momentum_right) / density_right
        )
        for velocity in transverse_velocities_left:
            kinetic_energy_left = kinetic_energy_left + 0.5 * density_left * velocity * velocity
        for velocity in transverse_velocities_right:
            kinetic_energy_right = (
                kinetic_energy_right + 0.5 * density_right * velocity * velocity
            )
        internal_energy_left = energy_left - kinetic_energy_left
        internal_energy_right = energy_right - kinetic_energy_right
        if is_mhd:
            internal_energy_left = internal_energy_left - 0.5 * field_squared_left
            internal_energy_right = internal_energy_right - 0.5 * field_squared_right
        if internal_energy_density is not None:
            # The Bryan et al. (1995) dual-energy switch, as in the WENO-side
            # pressure recovery.
            internal_energy_left = dual_energy_internal_energy(
                internal_energy_left,
                energy_left,
                internal_energy_density,
                config.dual_energy_eta,
            )
            internal_energy_right = dual_energy_internal_energy(
                internal_energy_right,
                energy_right,
                _shift(internal_energy_density, -1, axis=axis),
                config.dual_energy_eta,
            )
        pressure_left = jnp.maximum((gamma - 1.0) * internal_energy_left, params.minimum_pressure)
        pressure_right = jnp.maximum(
            (gamma - 1.0) * internal_energy_right,
            params.minimum_pressure,
        )
        sound_speed_squared_left = gamma * pressure_left / density_left
        sound_speed_squared_right = gamma * pressure_right / density_right
    else:
        sound_speed = params.isothermal_sound_speed
        sound_speed_squared_left = sound_speed * sound_speed
        sound_speed_squared_right = sound_speed * sound_speed
        pressure_left = sound_speed_squared_left * density_left
        pressure_right = sound_speed_squared_right * density_right

    # -------------------------------------------------------------
    # ============== ↑ Left and right face states ↑ ===============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================== ↓ Splitting speed ↓ ======================
    # -------------------------------------------------------------

    if is_mhd:
        def fast_magnetosonic_speed(field_squared, density, normal_field, sound_speed_squared):
            field_squared_over_rho = field_squared / density
            normal_field_squared_over_rho = (normal_field * normal_field) / density
            discriminant = jnp.maximum(
                (field_squared_over_rho + sound_speed_squared) ** 2
                - 4.0 * normal_field_squared_over_rho * sound_speed_squared,
                0.0,
            )
            return jnp.sqrt(
                jnp.maximum(
                    0.5 * (
                        field_squared_over_rho
                        + sound_speed_squared
                        + jnp.sqrt(discriminant)
                    ),
                    0.0,
                )
            )
        wave_speed_left = fast_magnetosonic_speed(
            field_squared_left,
            density_left,
            normal_field_left,
            sound_speed_squared_left,
        )
        wave_speed_right = fast_magnetosonic_speed(
            field_squared_right,
            density_right,
            normal_field_right,
            sound_speed_squared_right,
        )
    else:
        wave_speed_left = jnp.sqrt(sound_speed_squared_left)
        wave_speed_right = jnp.sqrt(sound_speed_squared_right)

    splitting_speed = jnp.maximum(
        jnp.abs(normal_velocity_left) + wave_speed_left,
        jnp.abs(normal_velocity_right) + wave_speed_right,
    )
    if config.weno_ad_frozen_weights:
        # Frozen like the WENO splitting speed: d c / d p ~ 1 / c blows up in
        # cold gas.
        splitting_speed = jax.lax.stop_gradient(splitting_speed)

    # -------------------------------------------------------------
    # ================== ↑ Splitting speed ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Physical fluxes of both face states ↓ ========
    # -------------------------------------------------------------

    conserved_state_right = right_neighbour_state(conserved_state)
    flux_left = jnp.zeros_like(conserved_state)
    flux_right = jnp.zeros_like(conserved_state)

    flux_left = flux_left.at[density_index].set(normal_momentum_left)
    flux_right = flux_right.at[density_index].set(normal_momentum_right)

    normal_momentum_flux_left = normal_momentum_left * normal_velocity_left + pressure_left
    normal_momentum_flux_right = normal_momentum_right * normal_velocity_right + pressure_right
    if is_mhd:
        normal_momentum_flux_left = (
            normal_momentum_flux_left + 0.5 * field_squared_left
            - normal_field_left * normal_field_left
        )
        normal_momentum_flux_right = (
            normal_momentum_flux_right + 0.5 * field_squared_right
            - normal_field_right * normal_field_right
        )
    flux_left = flux_left.at[normal_momentum_index].set(normal_momentum_flux_left)
    flux_right = flux_right.at[normal_momentum_index].set(normal_momentum_flux_right)

    for component, index in enumerate(transverse_momentum_indices):
        transverse_momentum_flux_left = (
            normal_momentum_left * transverse_velocities_left[component]
        )
        transverse_momentum_flux_right = (
            normal_momentum_right * transverse_velocities_right[component]
        )
        if is_mhd:
            transverse_momentum_flux_left = (
                transverse_momentum_flux_left
                - normal_field_left * transverse_fields_left[component]
            )
            transverse_momentum_flux_right = (
                transverse_momentum_flux_right
                - normal_field_right * transverse_fields_right[component]
            )
        flux_left = flux_left.at[index].set(transverse_momentum_flux_left)
        flux_right = flux_right.at[index].set(transverse_momentum_flux_right)

    if is_mhd:
        flux_left = flux_left.at[normal_magnetic_index].set(jnp.zeros_like(normal_field_left))
        flux_right = flux_right.at[normal_magnetic_index].set(jnp.zeros_like(normal_field_right))
        for component, index in enumerate(transverse_magnetic_indices):
            flux_left = flux_left.at[index].set(
                transverse_fields_left[component] * normal_velocity_left
                - normal_field_left * transverse_velocities_left[component]
            )
            flux_right = flux_right.at[index].set(
                transverse_fields_right[component] * normal_velocity_right
                - normal_field_right * transverse_velocities_right[component]
            )

    if is_ideal_gas:
        energy_index = registered_variables.energy_index
        if is_mhd:
            velocity_dot_field_left = normal_velocity_left * normal_field_left
            velocity_dot_field_right = normal_velocity_right * normal_field_right
            for component in range(len(transverse_momentum_indices)):
                velocity_dot_field_left = (
                    velocity_dot_field_left
                    + transverse_velocities_left[component] * transverse_fields_left[component]
                )
                velocity_dot_field_right = (
                    velocity_dot_field_right
                    + transverse_velocities_right[component] * transverse_fields_right[component]
                )
            flux_left = flux_left.at[energy_index].set(
                (energy_left + pressure_left + 0.5 * field_squared_left) * normal_velocity_left
                - normal_field_left * velocity_dot_field_left
            )
            flux_right = flux_right.at[energy_index].set(
                (energy_right + pressure_right + 0.5 * field_squared_right) * normal_velocity_right
                - normal_field_right * velocity_dot_field_right
            )
        else:
            flux_left = flux_left.at[energy_index].set(
                (energy_left + pressure_left) * normal_velocity_left
            )
            flux_right = flux_right.at[energy_index].set(
                (energy_right + pressure_right) * normal_velocity_right
            )

    # -------------------------------------------------------------
    # ============ ↑ Physical fluxes of both face states ↑ ========
    # -------------------------------------------------------------

    return (
        0.5 * (flux_left + flux_right)
        - 0.5 * splitting_speed * (conserved_state_right - conserved_state)
    )


# -------------------------------------------------------------
# ===== ↑ Shared first-order Lax-Friedrichs interface flux ↑ ===
# -------------------------------------------------------------


# -------------------------------------------------------------
# ========== ↓ Activation: cold-crush temperature ramp ↓ =======
# -------------------------------------------------------------


def _face_min_specific_pressure(
    conserved_state,
    axis,
    params,
    config,
    registered_variables,
    internal_energy_density=None,
):
    """
    The smaller specific pressure ``min(p_L/rho_L, p_R/rho_R)`` per interface.

    The pressures are recovered with the dual-energy switch when ``g`` is
    given: the raw recovery is destroyed by cancellation in exactly the cold
    cells the cold-crush gate has to classify.

    Args:
        conserved_state: The conserved state.
        axis: The spatial axis of the interfaces.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None.

    Returns:
        The interface minimum of ``p/rho``, the floored left and right
        densities, and the momentum indices (for the convergence gate).
    """
    density_index = registered_variables.density_index
    gamma = params.gamma
    minimum_density = params.minimum_density

    def right_neighbour(field):
        return _shift(field, -1, axis=axis)

    density_left = jnp.maximum(conserved_state[density_index], minimum_density)
    density_right = jnp.maximum(right_neighbour(conserved_state[density_index]), minimum_density)

    momentum_indices = _momentum_indices(config, registered_variables)
    kinetic_energy_left = (
        sum(conserved_state[index] ** 2 for index in momentum_indices) * 0.5 / density_left
    )
    kinetic_energy_right = (
        sum(right_neighbour(conserved_state[index]) ** 2 for index in momentum_indices)
        * 0.5 / density_right
    )

    energy_index = registered_variables.energy_index
    energy_left = conserved_state[energy_index]
    energy_right = right_neighbour(energy_left)
    internal_energy_left = energy_left - kinetic_energy_left
    internal_energy_right = energy_right - kinetic_energy_right
    if config.mhd:
        field_squared_left = sum(
            conserved_state[index] ** 2 for index in (
                registered_variables.magnetic_index.x,
                registered_variables.magnetic_index.y,
                registered_variables.magnetic_index.z,
            )
        )
        internal_energy_left = internal_energy_left - 0.5 * field_squared_left
        internal_energy_right = (
            internal_energy_right - 0.5 * _shift(field_squared_left, -1, axis=axis)
        )

    if internal_energy_density is not None:
        internal_energy_left = dual_energy_internal_energy(
            internal_energy_left,
            energy_left,
            internal_energy_density,
            config.dual_energy_eta,
        )
        internal_energy_right = dual_energy_internal_energy(
            internal_energy_right,
            energy_right,
            _shift(internal_energy_density, -1, axis=axis),
            config.dual_energy_eta,
        )

    pressure_left = jnp.maximum((gamma - 1.0) * internal_energy_left, params.minimum_pressure)
    pressure_right = jnp.maximum((gamma - 1.0) * internal_energy_right, params.minimum_pressure)
    face_min_specific_pressure = jnp.minimum(
        pressure_left / density_left,
        pressure_right / density_right,
    )
    return face_min_specific_pressure, density_left, density_right, momentum_indices


def _coldcrush_blend_weight(
    conserved_state,
    axis,
    params,
    config,
    registered_variables,
    internal_energy_density=None,
):
    """
    LLF weight for radiatively crushed cells: interfaces that are both
    SUB-floor cold and CONVERGING.

    Two gates, both per interface:

    * temperature ramp — on the COLDER of the two adjacent cells' recovered
      ``p/rho``, ramping from 1 at (or below) the effective temperature
      floor ``minimum_specific_pressure`` down to 0 at
      ``coldcrush_blend_factor`` times it. Any interface with a cold side
      under compression gets the diffusive flux: cold-cold isothermal
      collapse AND the boundary faces of a cold dense clump being crushed
      by hot surroundings (a gate on the hotter side would leave exactly
      those faces unprotected). The price is that shock fronts advancing
      into cold ambient gas are handled at first order locally — the
      classic first-order flux-correction trade.
    * convergence gate — the normal velocity must be compressive across the
      interface (``v_L > v_R``), ramped over the floor sound speed. Freely
      expanding cold gas is divergent and never activates, so seeded density
      structure in it is not diffused away; static cold gas has no
      convergence and is untouched.

    Args:
        conserved_state: The conserved state.
        axis: The spatial axis of the interfaces.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None.

    Returns:
        The blend weight in [0, 1] per interface.
    """
    gamma = params.gamma
    # The specific pressure p/rho at the floor temperature.
    floor_specific_pressure = params.minimum_specific_pressure

    def right_neighbour(field):
        return _shift(field, -1, axis=axis)

    face_min_specific_pressure, density_left, density_right, momentum_indices = (
        _face_min_specific_pressure(
            conserved_state,
            axis,
            params,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )
    )

    # Temperature ramp on the colder side: 1 at (or below) the floor
    # temperature, 0 at factor * floor, so any compressed cold side qualifies.
    ramp_top_specific_pressure = (
        config.positivity_config.coldcrush_blend_factor * floor_specific_pressure
    )
    temperature_weight = jnp.clip(
        (ramp_top_specific_pressure - face_min_specific_pressure)
        / jnp.maximum(ramp_top_specific_pressure - floor_specific_pressure, 1e-30),
        0.0,
        1.0,
    )

    # Convergence gate: a compressive normal velocity, ramped over the floor
    # sound speed so that it switches on smoothly.
    normal_momentum_index = momentum_indices[axis]
    normal_velocity_left = conserved_state[normal_momentum_index] / density_left
    normal_velocity_right = right_neighbour(conserved_state[normal_momentum_index]) / density_right
    floor_sound_speed = jnp.sqrt(gamma * jnp.maximum(floor_specific_pressure, 1e-30))
    convergence_weight = jnp.clip(
        (normal_velocity_left - normal_velocity_right) / floor_sound_speed,
        0.0,
        1.0,
    )

    return temperature_weight * convergence_weight


# -------------------------------------------------------------
# ========== ↑ Activation: cold-crush temperature ramp ↑ =======
# -------------------------------------------------------------


# -------------------------------------------------------------
# ======================= ↓ Entry point ↓ =====================
# -------------------------------------------------------------


def _blend_interface_flux(dF_weno, conserved_state, axis, dtdx, params, config,
                          registered_variables, internal_energy_density=None):
    """
    Blend the WENO interface flux toward LLF along ``axis`` at cold interfaces
    under compression (``coldcrush_blend``; ideal gas only).

    Args:
        dF_weno: The WENO interface flux along ``axis``.
        conserved_state: The conserved state.
        axis: The spatial axis of the interfaces.
        dtdx: Unused; the blend needs no time-step information.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None.

    Returns:
        The blended interface flux, or ``dF_weno`` unchanged when the blend is
        off.
    """
    if not (config.positivity_config.coldcrush_blend and config.equation_of_state == IDEAL_GAS):
        return dF_weno
    lax_friedrichs_flux = _local_lax_friedrichs_flux(
        conserved_state,
        axis,
        params,
        config,
        registered_variables,
        internal_energy_density=internal_energy_density,
    )
    blend_weight = _coldcrush_blend_weight(
        conserved_state,
        axis,
        params,
        config,
        registered_variables,
        internal_energy_density=internal_energy_density,
    )
    # The blend weight is a switching function (limiter activation): its
    # derivative carries no physical sensitivity, so the limiter is frozen at
    # its current activation for differentiation. The primal is untouched.
    blend_weight = jax.lax.stop_gradient(blend_weight)[None, ...]
    return dF_weno * (1.0 - blend_weight) + lax_friedrichs_flux * blend_weight


# -------------------------------------------------------------
# ======================= ↑ Entry point ↑ =====================
# -------------------------------------------------------------
