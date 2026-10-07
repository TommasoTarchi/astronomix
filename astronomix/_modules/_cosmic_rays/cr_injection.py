"""
Cosmic-ray injection at shock fronts (diffusive shock acceleration).

Locates the selected shock (the strongest or the outermost flagged one, see
``CosmicRayConfig.shock_selection``), estimates its upstream Mach number from
the jump conditions, and converts a configurable fraction of the dissipated
energy into cosmic-ray pressure. The injected energy is distributed across the
broadened numerical shock layer. The scheme follows Pfrommer et al. (2017) and
Dubois et al. (2019); see ``inject_crs_at_strongest_shock`` for the references.

Safeguards that keep every step well defined:

* a step is applied only if it is physically meaningful: the squared Mach
  number must exceed 1, the compression ratio must exceed 1, and the dissipated
  energy and the sum of the distribution weights must be positive and finite.
  Otherwise (a shock that has not formed yet, a one-cell pressure pulse, a
  mis-identified zone) the step injects nothing.
* the distribution weights are the cells' energy-density excess over the
  upstream reference cell times their own volume, ``(e_i - e_ref) V_i``,
  clipped at zero, so no cell receives negative CR energy and a uniform field
  injects nothing.
* a single step removes at most ``max_thermal_fraction_per_step`` of a cell's
  thermal energy, so the gas pressure can never be driven negative.
* ``P_cr ** (1 / gamma_cr)`` is evaluated AD-safely (finite, zero tangent at
  ``P_cr = 0``), so tangents stay finite in CR-free cells.
* ``escape_fraction`` removes that fraction of the freshly accelerated CR
  energy from the system (upstream escape).

The Mach number comes from ``shock_finder.mach_number_squared`` (exact
general-EOS Rankine-Hugoniot inversion with the upstream effective index in
the prefactor), and the shock zone and pre-shock cell from
``shock_finder.find_shock_zone``.

NOTE: This routine only supports 1D setups; generalising it to 2D / 3D needs
the multi-axis shock finder and a shock-normal estimate.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    SPHERICAL,
    STATE_TYPE,
)

# astronomix containers
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_n_from_pressure,
    cosmic_ray_pressure_from_n,
)
from astronomix.shock_finder.shock_finder import (
    find_shock_zone,
    mach_number_squared,
)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def inject_crs_at_strongest_shock(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    helper_data: HelperData,
    cosmic_ray_params: CosmicRayParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """
    Cosmic-ray injection at a shock front.

    The injection happens at a single shock: the strongest flagged shock
    (default, hence the name) or the outermost one, as selected by
    ``CosmicRayConfig.shock_selection``.

    The implementation generally follows

    Pfrommer, Christoph, et al. "Simulating cosmic ray physics on a moving mesh."
    Monthly Notices of the Royal Astronomical Society 465.4 (2017): 4500-4529.
    https://arxiv.org/abs/1604.07399

    and

    Dubois, Yohan, et al. "Shock-accelerated cosmic rays and streaming instability
    in the adaptive mesh refinement code Ramses."
    Astronomy & Astrophysics 631 (2019): A121.
    https://arxiv.org/abs/1907.04300

    Args:
        primitive_state: The primitive state array.
        gamma: The adiabatic index.
        helper_data: The helper data.
        cosmic_ray_params: The cosmic ray parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        dt: The time step.

    Returns:
        The primitive state array with injected cosmic rays.

    """

    num_cells = primitive_state.shape[1]

    # The injection efficiency (fraction of dissipated energy that goes into
    # cosmic rays) is supplied by the user. In future this could be replaced by
    # a Mach-dependent model, e.g. the ones in
    # https://github.com/LudwigBoess/DiffusiveShockAccelerationModels.jl/tree/main/src/mach_models
    injection_efficiency = cosmic_ray_params.diffusive_shock_acceleration_efficiency

    # The adiabatic indices for the two fluids are currently hard-coded; the gas
    # index follows the user-supplied ``gamma``.
    gamma_cr = 4 / 3
    gamma_gas = gamma

    # -------------------------------------------------------------
    # ============== ↓ Locate the injection shock ↓ ===============
    # -------------------------------------------------------------

    max_shock_index, left_index, right_index = find_shock_zone(
        primitive_state,
        config,
        registered_variables,
        helper_data,
        shock_selection=config.cosmic_ray_config.shock_selection,
        outermost_shock_window_cells=config.cosmic_ray_config.outermost_shock_window_cells,
        gamma_gas=gamma_gas,
    )

    # NOTE: shifting ``left_index`` outward by +2 smooths the transition of the
    # different pressure components across the shock, but causes problems at
    # lower resolutions. There is also the subtlety that cosmic-ray pressure
    # injected into the broadened shock layer experiences P_CR * div(u) forces
    # (Dubois et al. 2019), which can lead to effective over-injection.

    # We only consider a shock moving from left to right, so the pre-shock state
    # is upstream and the post-shock state is downstream in the shock frame.
    # The indices are clipped to the grid, because JAX would otherwise clamp an
    # out-of-range index silently.
    pre_shock_index = jnp.minimum(right_index + 1, num_cells - 1)
    post_shock_index = jnp.maximum(left_index - 1, 0)
    reference_index = jnp.minimum(right_index, num_cells - 1)

    # -------------------------------------------------------------
    # ============== ↑ Locate the injection shock ↑ ===============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ======== ↓ Pre- and post-shock fluid quantities ↓ =========
    # -------------------------------------------------------------

    # Pre-shock (upstream) state: density, total / CR / gas pressures and the
    # gas and CR energy densities.
    upstream_density = primitive_state[registered_variables.density_index, pre_shock_index]
    upstream_pressure = primitive_state[registered_variables.pressure_index, pre_shock_index]
    upstream_cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index, pre_shock_index]
    )
    upstream_gas_pressure = upstream_pressure - upstream_cosmic_ray_pressure
    upstream_gas_energy_density = upstream_gas_pressure / (gamma_gas - 1)
    upstream_cosmic_ray_energy_density = upstream_cosmic_ray_pressure / (gamma_cr - 1)

    # Post-shock (downstream) state.
    downstream_density = primitive_state[registered_variables.density_index, post_shock_index]
    downstream_pressure = primitive_state[registered_variables.pressure_index, post_shock_index]
    downstream_cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index, post_shock_index]
    )
    downstream_gas_pressure = downstream_pressure - downstream_cosmic_ray_pressure
    downstream_gas_energy_density = downstream_gas_pressure / (gamma_gas - 1)
    downstream_cosmic_ray_energy_density = downstream_cosmic_ray_pressure / (gamma_cr - 1)

    # -------------------------------------------------------------
    # ======== ↑ Pre- and post-shock fluid quantities ↑ =========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Mach number and dissipated flux ↓ ===========
    # -------------------------------------------------------------

    # Effective adiabatic index of the upstream mixture, and the corresponding
    # upstream sound speed.
    upstream_effective_gamma = (
        gamma_cr * upstream_cosmic_ray_pressure + gamma_gas * upstream_gas_pressure
    ) / upstream_pressure
    upstream_sound_speed = jnp.sqrt(
        upstream_effective_gamma * upstream_pressure / upstream_density
    )

    # Compression ratio x_s across the shock.
    compression_ratio = downstream_density / upstream_density

    # Squared upstream Mach number from the general-EOS Rankine-Hugoniot
    # inversion (Pfrommer et al. 2017, Sec. 3.1; Dubois et al. 2019, Eq. 16,
    # with the upstream effective index in the prefactor, see
    # ``mach_number_squared``). Pfrommer et al. (2017) use the simpler
    # expression
    #     M_1^2 = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))
    # for the injection itself (it is only a lower bound there). That simpler
    # form led to more crashes in spherical-geometry setups, so the full
    # inversion is used here.
    upstream_mach_squared = mach_number_squared(
        upstream_pressure,
        upstream_cosmic_ray_pressure,
        downstream_pressure,
        downstream_cosmic_ray_pressure,
        gamma_gas=gamma_gas,
        gamma_cr=gamma_cr,
        denominator_floor=1e-12,
    )

    # Dissipated energy density and the corresponding dissipated energy flux
    # through the shock surface.
    dissipated_energy_density = (
        downstream_gas_energy_density
        - upstream_gas_energy_density * compression_ratio**gamma_gas
        + downstream_cosmic_ray_energy_density
        - upstream_cosmic_ray_energy_density * compression_ratio**gamma_cr
    )

    # A step is only well posed for a compressive (M > 1), dissipative jump;
    # anything else (a shock that has not formed yet, a mis-identified zone)
    # injects nothing instead of propagating a NaN from sqrt(M^2 < 0).
    step_valid = (
        (upstream_mach_squared > 1.0)
        & (dissipated_energy_density > 0.0)
        & (compression_ratio > 1.0)
        & jnp.isfinite(upstream_mach_squared)
        & jnp.isfinite(dissipated_energy_density)
        & jnp.isfinite(upstream_sound_speed)
    )
    upstream_mach = jnp.sqrt(jnp.where(step_valid, upstream_mach_squared, 1.0))
    dissipated_energy_flux = jnp.where(
        step_valid,
        dissipated_energy_density
        * upstream_mach
        * upstream_sound_speed
        / compression_ratio,
        0.0,
    )

    # -------------------------------------------------------------
    # ============ ↑ Mach number and dissipated flux ↑ ===========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Distribute the injected energy ↓ ============
    # -------------------------------------------------------------

    # Shock surface area: a sphere in spherical geometry, otherwise the
    # transverse cell area implied by the grid spacing and dimensionality.
    if config.geometry == SPHERICAL:
        shock_radius = helper_data.geometric_centers[max_shock_index]
        shock_surface = 4 * jnp.pi * shock_radius**2
    else:
        shock_surface = config.grid_spacing ** (config.dimensionality - 1)

    # Total energy to be injected as cosmic-ray pressure over this time step.
    injected_energy = dissipated_energy_flux * shock_surface * dt * injection_efficiency

    # Build a mask for the broadened shock zone over which the energy is spread.
    # NOTE: Pfrommer et al. (2017) use ``post_shock_index`` here instead of
    # ``left_index`` as the lower bound.
    indices = jnp.arange(num_cells)
    shock_zone_mask = (indices >= left_index) & (indices <= max_shock_index)

    # Distribute the injected energy across the shock zone in proportion to each
    # cell's total energy excess relative to the upstream reference cell.
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index]
    )
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )
    thermal_energy_density = gas_pressure / (gamma_gas - 1)
    cosmic_ray_energy_density = cosmic_ray_pressure / (gamma_cr - 1)
    total_energy_density = thermal_energy_density + cosmic_ray_energy_density

    # Weights: each cell's excess over the upstream reference energy density,
    # times the cell's own volume, clipped at zero (a cell below the reference
    # must not receive negative CR energy) and normalised by a guarded sum.
    # Energy densities are compared, not volume-integrated energies: the cells
    # differ in volume in spherical geometry, and comparing e_i V_i with
    # e_ref V_ref (V_ref > V_i) would make all weights of a real shock near the
    # origin negative, so the clipped step would inject nothing.
    weights = jnp.where(
        shock_zone_mask,
        jnp.maximum(total_energy_density - total_energy_density[reference_index], 0.0)
        * helper_data.cell_volumes,
        0.0,
    )
    total_weight = jnp.sum(weights)
    step_valid = step_valid & (total_weight > 0.0) & jnp.isfinite(total_weight)
    total_weight_safe = jnp.where(step_valid, total_weight, 1.0)
    injected_energy_per_cell = jnp.where(
        step_valid,
        injected_energy * weights / total_weight_safe,
        0.0,
    )

    # Safety cap: never remove more than a fixed fraction of a cell's thermal
    # energy in one step (keeps P_gas > 0 whatever the zone identification).
    max_thermal_energy_removal = (
        cosmic_ray_params.max_thermal_fraction_per_step
        * jnp.maximum(thermal_energy_density * helper_data.cell_volumes, 0.0)
    )
    injected_energy_per_cell = jnp.minimum(
        injected_energy_per_cell,
        max_thermal_energy_removal,
    )

    # -------------------------------------------------------------
    # ============ ↑ Distribute the injected energy ↑ ============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Apply the cosmic-ray injection ↓ ============
    # -------------------------------------------------------------

    # The gas loses ``injected_energy_per_cell``; the CR fluid keeps
    # (1 - f_esc) of it and the rest escapes upstream (leaves the system).
    retained_fraction = 1.0 - cosmic_ray_params.escape_fraction
    retained_cosmic_ray_energy = retained_fraction * injected_energy_per_cell

    # Updated cosmic-ray pressure after injection.
    new_cosmic_ray_pressure = (
        cosmic_ray_pressure
        + retained_cosmic_ray_energy / helper_data.cell_volumes * (gamma_cr - 1)
    )

    # The cosmic rays are tracked through n_cr = P_CR ** (1 / gamma_cr), so the
    # updated pressure is converted back to the advected scalar before storing
    # (AD-safe: finite derivative at P_cr = 0).
    new_cosmic_ray_n = cosmic_ray_n_from_pressure(new_cosmic_ray_pressure, gamma_cr)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index].set(
        new_cosmic_ray_n
    )

    # We want energy (not pressure) conservation, so removing thermal energy and
    # converting it into cosmic-ray energy requires adapting the stored total
    # pressure accordingly.
    gas_pressure_change = (
        injected_energy_per_cell / helper_data.cell_volumes * (gamma_gas - 1)
    )
    new_gas_pressure = (
        primitive_state[registered_variables.pressure_index]
        - cosmic_ray_pressure
        - gas_pressure_change
    )
    new_total_pressure = new_gas_pressure + new_cosmic_ray_pressure

    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        new_total_pressure
    )

    # -------------------------------------------------------------
    # ============ ↑ Apply the cosmic-ray injection ↑ ============
    # -------------------------------------------------------------

    return primitive_state
