"""
Shock detection on a 1D fluid state.

Implements the smoothness sensor and the multi-criterion shock test of
Pfrommer et al. (2017) used to flag shocks (e.g. for diffusive shock
acceleration of cosmic rays), plus a routine that broadens the selected shock
(the strongest or the outermost flagged one) into a numerical shock zone.

* ``shock_sensor`` is the centred WENO-JS smoothness indicator (Jiang & Shu
  1996) of the pressure. Both of its terms are squares, so it is non-negative
  and its maximum sits on the jump rather than on a concave pressure maximum.
* ``mach_number_squared`` is the exact general-EOS Rankine-Hugoniot inversion
  for a composite gas + cosmic-ray fluid, with the upstream effective index in
  the prefactor. It is shared by the shock criterion and the injection, and
  it is exact for any Rankine-Hugoniot jump of the composite.
* ``shock_criteria`` flags converging cells with aligned temperature and
  density gradients whose upstream Mach number exceeds a threshold.
* ``find_shock_zone`` picks the sensor maximum among the flagged cells (or
  among those near the outermost flagged cell) and extends the zone to both
  sides until the pressure stops falling or the jump becomes small.

Cosmic-ray pressures are read through the clipped, AD-safe helper
``cosmic_ray_pressure_from_n``.

NOTE: Only 1D is supported; generalising requires multi-axis stencils and
dropping the assumption of a shock moving from left to right.
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from jaxtyping import Array, Int

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
)
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    OUTERMOST_SHOCK,
    STRONGEST_SHOCK,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)


@partial(jax.jit, static_argnames=["config"])
def _calculate_1d_divergence(
    field: FIELD_TYPE,
    config: SimulationConfig,
    radii: FIELD_TYPE,
) -> FIELD_TYPE:
    """
    Central-difference 1D divergence of ``field`` for the active geometry.

    Args:
        field: The 1D field whose divergence is taken.
        config: The simulation configuration (selects the geometry).
        radii: The radial cell-centre coordinates (used in spherical geometry).

    Returns:
        The divergence of the field; ghost cells at the ends are left at zero.
    """
    # Approximate the 1D divergence with a simple central difference.
    div_field = jnp.zeros_like(field)
    if config.geometry == CARTESIAN:
        div_field = div_field.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * config.grid_spacing)
        )
    elif config.geometry == SPHERICAL:
        # This is not exactly correct, since the field values live at the
        # volumetric rather than the geometric cell centres, but it is accurate
        # enough for the purposes of the shock finder.
        div_field = div_field.at[1:-1].set(
            (radii[2:] ** 2 * field[2:] - radii[:-2] ** 2 * field[:-2])
            / (2 * config.grid_spacing * radii[1:-1] ** 2)
        )
    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for the shock finder."
        )
    return div_field


def mach_number_squared(
    upstream_pressure: Union[float, jnp.ndarray],
    upstream_cosmic_ray_pressure: Union[float, jnp.ndarray],
    downstream_pressure: Union[float, jnp.ndarray],
    downstream_cosmic_ray_pressure: Union[float, jnp.ndarray],
    gamma_gas: Union[float, jnp.ndarray] = 5 / 3,
    gamma_cr: Union[float, jnp.ndarray] = 4 / 3,
    denominator_floor: float = 1e-6,
) -> jnp.ndarray:
    """
    Upstream Mach number squared of a composite gas + CR shock.

    The general-EOS Rankine-Hugoniot inversion of Pfrommer et al. (2017,
    Sec. 3.1; quoted as Eq. 16 of Dubois et al. 2019), from the total
    pressures P1 (upstream) / P2 (downstream) and the CR pressures:

        M1^2 = (y - 1) C / (gamma_eff1 [C - ((g1 + 1) + (g1 - 1) y)(g2 - 1)]),
        C = ((g2 + 1) y + g2 - 1)(g1 - 1),  y = P2/P1,  g_i = P_i/e_i + 1,

    with ``gamma_eff1 = (gamma_cr P_cr1 + gamma_gas P_gas1)/P1`` the upstream
    sound-speed index (c1^2 = gamma_eff1 P1/rho1, Pfrommer Eq. 30). It follows
    from mass, momentum and enthalpy conservation with the actual energy
    indices on both sides, so it is exact for any Rankine-Hugoniot jump of the
    composite (with or without injection), and reduces to
    ``((gamma+1) y + gamma - 1) / (2 gamma)`` without CRs. Returns values
    < 1 for jumps with P2 < P1 (an upstream on the other side).

    NOTE: Dubois et al. (2019) print the prefactor with the downstream index
    (their gamma_e). That underestimates M by sqrt(gamma_eff1 / gamma_eff2),
    e.g. 9.51 instead of 10.00 for the Pfrommer et al. (2017) shock tube (CRs
    upstream), and overestimates it when the upstream is CR-free and the
    downstream is not (every diffusive-shock-acceleration shock). The upstream
    index is the one the Rankine-Hugoniot derivation requires.

    Args:
        upstream_pressure: The upstream total pressure P1.
        upstream_cosmic_ray_pressure: The upstream cosmic-ray pressure.
        downstream_pressure: The downstream total pressure P2.
        downstream_cosmic_ray_pressure: The downstream cosmic-ray pressure.
        gamma_gas: The adiabatic index of the thermal gas.
        gamma_cr: The adiabatic index of the cosmic-ray fluid.
        denominator_floor: Denominators with a magnitude at or below this are
            replaced by +denominator_floor (the denominator vanishes for
            P2 = P1).

    Returns:
        M1^2 (array like the inputs).
    """
    upstream_gas_pressure = upstream_pressure - upstream_cosmic_ray_pressure
    downstream_gas_pressure = downstream_pressure - downstream_cosmic_ray_pressure
    upstream_energy_density = (
        upstream_gas_pressure / (gamma_gas - 1)
        + upstream_cosmic_ray_pressure / (gamma_cr - 1)
    )
    downstream_energy_density = (
        downstream_gas_pressure / (gamma_gas - 1)
        + downstream_cosmic_ray_pressure / (gamma_cr - 1)
    )
    upstream_effective_gamma = (
        gamma_cr * upstream_cosmic_ray_pressure + gamma_gas * upstream_gas_pressure
    ) / upstream_pressure

    # The energy indices g1, g2 and the pressure ratio y of the formula above;
    # ``C`` keeps the paper's symbol.
    energy_index_upstream = upstream_pressure / upstream_energy_density + 1
    energy_index_downstream = downstream_pressure / downstream_energy_density + 1
    pressure_ratio = downstream_pressure / upstream_pressure
    C = (
        (energy_index_downstream + 1) * pressure_ratio + energy_index_downstream - 1
    ) * (energy_index_upstream - 1)
    denominator = C - (
        (energy_index_upstream + 1) + (energy_index_upstream - 1) * pressure_ratio
    ) * (energy_index_downstream - 1)
    denominator = jnp.where(
        jnp.abs(denominator) > denominator_floor,
        denominator,
        denominator_floor,
    )
    return 1 / upstream_effective_gamma * (pressure_ratio - 1) * C / denominator


@jax.jit
def shock_sensor(pressure: FIELD_TYPE) -> FIELD_TYPE:
    """
    WENO-JS 1D smoothness indicator for shock detection.

    Args:
        pressure: The 1D pressure.

    Returns:
        The shock sensor, large where the pressure jumps.
    """

    # beta = 1/4 (first difference)^2 + 13/12 (second difference)^2, i.e. the
    # centred WENO-JS indicator (Jiang & Shu 1996). Both terms are squares, so
    # the sensor is non-negative.
    shock_sensors = jnp.zeros_like(pressure)
    shock_sensors = shock_sensors.at[1:-1].set(
        1 / 4 * (pressure[2:] - pressure[:-2]) ** 2
        + 13 / 12 * (pressure[2:] - 2 * pressure[1:-1] + pressure[:-2]) ** 2
    )

    return shock_sensors


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def shock_criteria(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma_gas: Union[float, jnp.ndarray] = 5 / 3,
    gamma_cr: Union[float, jnp.ndarray] = 4 / 3,
    mach_min: Union[float, jnp.ndarray] = 1.3,
) -> jnp.ndarray:
    """
    Implement the shock criteria from Pfrommer et al, 2017.
    https://arxiv.org/abs/1604.07399

    A cell is flagged if (i) the flow converges, (ii) the temperature and
    density gradients are aligned (rejects contacts / tangential
    discontinuities) and (iii) the Mach number estimated from the jump between
    its two neighbours exceeds ``mach_min``. The Mach estimate is
    ``mach_number_squared`` (Pfrommer et al. 2017, Sec. 3.1, with the upstream
    effective index in the prefactor) for a composite gas + CR fluid, with the
    upstream state on the right (a shock moving towards +x / +r). Shocks whose
    upstream is on the left (e.g. the reverse shock of a remnant) have
    P2/P1 < 1 there, hence M < 1, and are never flagged.

    Args:
        primitive_state: The 1D primitive state (pressure slot = P_gas + P_cr).
        config: The simulation configuration.
        registered_variables: The registered variables.
        helper_data: The helper data (geometric centres for the divergence).
        gamma_gas: The adiabatic index of the thermal gas.
        gamma_cr: The adiabatic index of the CR fluid.
        mach_min: The minimum upstream Mach number of a flagged shock.

    Returns:
        Boolean mask of shocked cells.
    """

    velocity = primitive_state[registered_variables.velocity_index]

    # The cosmic-ray pressure is clipped at zero, which keeps it AD-safe and
    # robust against small undershoots of the advected scalar.
    cosmic_ray_pressure = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index], gamma_cr
    )

    # --------------- ↓ (i) Converging flow ↓ ----------------
    # The flow converges into a shock, div(v) < 0.
    velocity_divergence = _calculate_1d_divergence(
        velocity,
        config,
        helper_data.geometric_centers,
    )
    converging_flow_criterion = velocity_divergence < 0
    # --------------- ↑ (i) Converging flow ↑ ----------------

    # --------------- ↓ (ii) Aligned gradients ↓ ----------------
    # Temperature and density rise in the same direction across a shock,
    # grad(T) . grad(rho) > 0, but not across a contact discontinuity.
    pseudo_temperature = (
        primitive_state[registered_variables.pressure_index]
        / primitive_state[registered_variables.density_index]
    )
    temperature_gradient = jnp.zeros_like(pseudo_temperature)
    temperature_gradient = temperature_gradient.at[1:-1].set(
        (pseudo_temperature[2:] - pseudo_temperature[:-2]) / 2
    )
    density_gradient = jnp.zeros_like(primitive_state[registered_variables.density_index])
    density_gradient = density_gradient.at[1:-1].set(
        (
            primitive_state[registered_variables.density_index][2:]
            - primitive_state[registered_variables.density_index][:-2]
        )
        / 2
    )
    aligned_gradients_criterion = temperature_gradient * density_gradient > 0
    # --------------- ↑ (ii) Aligned gradients ↑ ----------------

    # --------------- ↓ (iii) Mach number ↓ ----------------
    # The upstream Mach number exceeds the threshold, M1 > mach_min.
    # Only shocks moving from left to right are considered, so the downstream
    # state is the left neighbour and the upstream state the right neighbour.
    downstream_pressure = primitive_state[registered_variables.pressure_index, :-2]
    downstream_cosmic_ray_pressure = cosmic_ray_pressure[:-2]
    upstream_pressure = primitive_state[registered_variables.pressure_index, 2:]
    upstream_cosmic_ray_pressure = cosmic_ray_pressure[2:]

    # NOTE: The simple estimate M1^2 = (P2 / P1 - 1) x_s / (gamma_eff1 (x_s - 1))
    # with x_s = rho2 / rho1 is singular where x_s = 1, so the exact inversion
    # is used instead.
    upstream_mach_squared = mach_number_squared(
        upstream_pressure,
        upstream_cosmic_ray_pressure,
        downstream_pressure,
        downstream_cosmic_ray_pressure,
        gamma_gas=gamma_gas,
        gamma_cr=gamma_cr,
        denominator_floor=1e-6,
    )

    mach_number_criterion = jnp.zeros_like(converging_flow_criterion, dtype=jnp.bool_)
    mach_number_criterion = mach_number_criterion.at[1:-1].set(
        upstream_mach_squared > mach_min**2
    )
    # --------------- ↑ (iii) Mach number ↑ ----------------

    return converging_flow_criterion & aligned_gradients_criterion & mach_number_criterion


@partial(
    jax.jit,
    static_argnames=[
        "registered_variables",
        "config",
        "shock_selection",
        "outermost_shock_window_cells",
    ],
)
def find_shock_zone(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_selection: int = STRONGEST_SHOCK,
    outermost_shock_window_cells: int = 8,
    gamma_gas: Union[float, jnp.ndarray] = 5 / 3,
) -> Tuple[
    Union[int, Int[Array, ""]], Union[int, Int[Array, ""]], Union[int, Int[Array, ""]]
]:
    """
    Find the numerically broadened region of the selected shock (strongest or
    outermost flagged), based on the shock sensor and the pressure difference
    between adjacent cells. Assumes a shock front moving from left to right.

    Args:
        primitive_state: The 1D primitive state.
        config: The simulation configuration.
        registered_variables: The registered variables.
        helper_data: The helper data.
        shock_selection: ``STRONGEST_SHOCK`` (largest sensor among all flagged
            cells; default) or ``OUTERMOST_SHOCK`` (largest sensor among the
            flagged cells within ``outermost_shock_window_cells`` of the
            outermost flagged cell, i.e. the forward shock).
        outermost_shock_window_cells: See ``shock_selection``; the default
            matches ``CosmicRayConfig.outermost_shock_window_cells``.
        gamma_gas: The adiabatic index of the gas (for the Mach criterion).

    Returns:
        The index of the shock-sensor maximum, the left boundary of the
        broadened shock and the right boundary of the broadened shock.
    """

    pressure = primitive_state[registered_variables.pressure_index]
    num_cells = pressure.shape[0]

    # NOTE: The shock is placed at the maximum of the shock sensor. Pfrommer et
    # al. (2017) use the flagged cell of maximum compression (minimum velocity
    # divergence) instead.
    sensors = shock_sensor(pressure)

    shock_flags = shock_criteria(
        primitive_state,
        config,
        registered_variables,
        helper_data,
        gamma_gas=gamma_gas,
    )

    indices = jnp.arange(num_cells)
    if shock_selection == OUTERMOST_SHOCK:
        # Restrict the search to the outermost flagged shock: the flagged
        # cells at most ``outermost_shock_window_cells`` inside the outermost
        # flagged cell.
        outermost_flagged_index = jnp.max(jnp.where(shock_flags, indices, -1))
        candidates = shock_flags & (
            indices >= outermost_flagged_index - outermost_shock_window_cells
        )
    elif shock_selection == STRONGEST_SHOCK:
        candidates = shock_flags
    else:
        raise ValueError(f"unknown shock_selection {shock_selection}")

    max_shock_index = jnp.argmax(jnp.where(candidates, sensors, -1))

    # Backward pressure differences, pressure_differences[i] = p[i] - p[i - 1];
    # the first entry stays zero.
    pressure_differences = jnp.zeros_like(pressure)
    pressure_differences = pressure_differences.at[1:].set(pressure[1:] - pressure[:-1])

    # A pressure change between adjacent cells counts as small if it is below a
    # tenth of the pressure jump at the shock-sensor maximum.
    small_jump_threshold = 0.1 * jnp.abs(pressure_differences[max_shock_index])

    # For a right-moving shock the pressure falls (dp < 0) inside the jump, so
    # both edges of the broadened shock are the closest cells where dp turns
    # positive or falls below the small-jump threshold.
    left_edge_candidates = jnp.where(
        (indices < max_shock_index)
        & (
            (jnp.abs(pressure_differences) < small_jump_threshold)
            | (pressure_differences > 0)
        ),
        indices,
        -1,
    )
    right_edge_candidates = jnp.where(
        (indices > max_shock_index)
        & (
            (jnp.abs(pressure_differences) < small_jump_threshold)
            | (pressure_differences > 0)
        ),
        indices,
        num_cells,
    )
    left_index = jnp.max(left_edge_candidates)
    right_index = jnp.min(right_edge_candidates)

    return max_shock_index, left_index, right_index
