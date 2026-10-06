"""
Shock detection on a 1D fluid state.

Implements the smoothness sensor and the multi-criterion shock test of
Pfrommer et al. (2017) used to flag shocks (e.g. for diffusive shock
acceleration of cosmic rays), plus a routine that broadens the strongest shock
into a numerical shock zone.

NOTE: the routines here currently only support 1D setups; TODO: generalise.

Fixes of 2026-09-25 (audit ``cr_code.md`` section 1):

* ``shock_sensor``: the 13/12 curvature term of the WENO-JS smoothness
  indicator is now squared, as in Jiang & Shu (1996). Unsquared it is
  sign-indefinite, so the argmax that picks the shock could land on a
  concave-down pressure maximum instead of the jump.
* ``shock_criteria``: the downstream gas ENERGY density was computed as
  ``P2 - P2_CR`` (a pressure; the ``/(gamma - 1)`` was missing), which made the
  downstream "energy" index 2 even without CRs, and ``gamma_eff2`` used the
  total instead of the gas pressure. The resulting Mach estimate was wrong and
  non-monotone (true M = 1.2 -> 4.3, 1.3 -> 2.17, 10 -> 10.6, and a pole at
  P2/P1 = 1.5), so the M > 1.3 gate passed weak compressions.
* ``mach_number_squared`` (shared by the criterion and the injection): the
  prefactor uses the UPSTREAM effective index, as the Rankine-Hugoniot
  derivation requires (Dubois+19 Eq. 16 prints the downstream one). The
  estimate is now exact for Rankine-Hugoniot jumps (tested for
  M = 1.05 ... 100 without CRs and on both Pfrommer+17 tube jumps with CRs).
* ``find_shock_zone``: the right (upstream) edge of the broadened shock was
  detected by ``dp < 0`` -- the shock's own sign -- so it always stopped one
  cell outside the sensor maximum and the "pre-shock" state was read inside
  the shock. It now stops where the pressure stops falling (``dp > 0``) or the
  jump becomes small, mirroring the left edge.
* ``find_shock_zone`` can select the OUTERMOST flagged shock (the forward
  shock of an ejecta-driven remnant) instead of the strongest one.
* CR pressures are read through the clipped, AD-safe helper.
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from jaxtyping import Array, Int, jaxtyped
from beartype import beartype as typechecker

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

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._modules._cosmic_rays.cosmic_ray_options import (
    OUTERMOST_SHOCK,
    STRONGEST_SHOCK,
)
from astronomix._modules._cosmic_rays.cr_fluid_equations import (
    cosmic_ray_pressure_from_n,
)


@partial(jax.jit, static_argnames=["config"])
def _calculate_1d_divergence(
    field: FIELD_TYPE, config: SimulationConfig, r: FIELD_TYPE
) -> FIELD_TYPE:
    """Central-difference 1D divergence of ``field`` for the active geometry.

    Args:
        field: The 1D field whose divergence is taken.
        config: The simulation configuration (selects the geometry).
        r: The radial cell-centre coordinates (used in spherical geometry).

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
        div_field = jnp.zeros_like(field)
        # This is not exactly correct, since the field values live at the
        # volumetric rather than the geometric cell centres, but it is accurate
        # enough for the purposes of the shock finder.
        div_field = div_field.at[1:-1].set(
            (r[2:] ** 2 * field[2:] - r[:-2] ** 2 * field[:-2])
            / (2 * config.grid_spacing * r[1:-1] ** 2)
        )
    else:
        raise NotImplementedError(
            "Only Cartesian and Spherical geometry supported for the shock finder."
        )
    return div_field


def mach_number_squared(
    P1, P1_CRs, P2, P2_CRs,
    gamma_gas=5 / 3, gamma_cr=4 / 3, denominator_floor=1e-6,
):
    """Upstream Mach number squared of a composite gas + CR shock.

    The general-EOS Rankine-Hugoniot inversion of Pfrommer et al. (2017,
    Sec. 3.1; quoted as Eq. 16 of Dubois et al. 2019), from the TOTAL
    pressures ``P1`` (upstream) / ``P2`` (downstream) and the CR pressures:

        M1^2 = (y - 1) C / (gamma_eff1 [C - ((g1 + 1) + (g1 - 1) y)(g2 - 1)]),
        C = ((g2 + 1) y + g2 - 1)(g1 - 1),  y = P2/P1,  g_i = P_i/e_i + 1,

    with ``gamma_eff1 = (gamma_cr P_cr1 + gamma_gas P_gas1)/P1`` the UPSTREAM
    sound-speed index (c1^2 = gamma_eff1 P1/rho1, Pfrommer Eq. 30). It follows
    from mass, momentum and enthalpy conservation with the actual energy
    indices on both sides, so it is exact for any Rankine-Hugoniot jump of the
    composite (with or without injection), and reduces to
    ``((gamma+1) y + gamma - 1) / (2 gamma)`` without CRs. Returns values
    < 1 for jumps with P2 < P1 (an upstream on the other side).

    FIX (2026-09-25): Dubois et al. (2019) write the prefactor with the
    DOWNSTREAM index (their gamma_e) and this code followed them; that
    underestimates M by sqrt(gamma_eff1/gamma_eff2), e.g. 9.51 instead of
    10.00 for the Pfrommer+17 shock tube (CRs upstream), and overestimates it
    when the upstream is CR-free and the downstream is not (every DSA shock).

    Args:
        P1, P1_CRs: upstream total and CR pressure.
        P2, P2_CRs: downstream total and CR pressure.
        gamma_gas, gamma_cr: adiabatic indices of the two components.
        denominator_floor: the (signed-agnostic) floor of the denominator,
            which vanishes for P2 = P1.

    Returns:
        M1^2 (array like the inputs).
    """
    P1_gas = P1 - P1_CRs
    P2_gas = P2 - P2_CRs
    e1 = P1_gas / (gamma_gas - 1) + P1_CRs / (gamma_cr - 1)
    e2 = P2_gas / (gamma_gas - 1) + P2_CRs / (gamma_cr - 1)
    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    gamma1 = P1 / e1 + 1
    gamma2 = P2 / e2 + 1
    gammat = P2 / P1
    C = ((gamma2 + 1) * gammat + gamma2 - 1) * (gamma1 - 1)
    denominator = C - ((gamma1 + 1) + (gamma1 - 1) * gammat) * (gamma2 - 1)
    denominator = jnp.where(
        jnp.abs(denominator) > denominator_floor, denominator, denominator_floor
    )
    return 1 / gamma_eff1 * (gammat - 1) * C / denominator


@jax.jit
def shock_sensor(pressure: FIELD_TYPE) -> FIELD_TYPE:
    """
    WENO-JS 1D smoothness indicator for shock detection.

    Args:
        pressure: the 1d pressure

    Returns:
        shock sensors, high where large pressure jumps

    """

    # beta = 1/4 (first difference)^2 + 13/12 (second difference)^2, i.e. the
    # centred WENO-JS indicator (Jiang & Shu 1996). Both terms are squares, so
    # the sensor is non-negative; the second one used to be unsquared.
    shock_sensors = jnp.zeros_like(pressure)
    shock_sensors = shock_sensors.at[1:-1].set(
        1 / 4 * (pressure[2:] - pressure[:-2]) ** 2
        + 13 / 12 * (pressure[2:] - 2 * pressure[1:-1] + pressure[:-2]) ** 2
    )

    return shock_sensors


# @jaxtyped(typechecker=typechecker)
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
    its two neighbours exceeds ``mach_min``. The Mach estimate is Eq. 16 of
    Dubois et al. (2019) for a composite gas + CR fluid, with the upstream
    state on the RIGHT (a shock moving towards +x / +r). Shocks whose upstream
    is on the left (e.g. the reverse shock of a remnant) have P2/P1 < 1 there,
    hence M < 1, and are never flagged.

    # NOTE: for now only 1D

    Args:
        primitive_state: the 1D primitive state (pressure slot = P_gas + P_cr).
        config: the simulation configuration.
        registered_variables: the registered variables.
        helper_data: helper data (geometric centres for the divergence).
        gamma_gas: adiabatic index of the thermal gas.
        gamma_cr: adiabatic index of the CR fluid.
        mach_min: minimum upstream Mach number of a flagged shock.

    Returns:
        boolean mask of shocked cells.
    """

    # get the velocity
    velocity = primitive_state[registered_variables.velocity_index]

    # get the cosmic ray pressure (clipped at zero: AD- and undershoot-safe)
    P_CRs = cosmic_ray_pressure_from_n(
        primitive_state[registered_variables.cosmic_ray_n_index], gamma_cr
    )

    # i) \nabla \cdot \vec{v} < 0
    div_v = _calculate_1d_divergence(velocity, config, helper_data.geometric_centers)
    converging_flow_criterion = div_v < 0

    # ii) \nabla T \cdot \nabla \rho > 0
    pseudo_temperature = (
        primitive_state[registered_variables.pressure_index]
        / primitive_state[registered_variables.density_index]
    )
    div_T = jnp.zeros_like(pseudo_temperature)
    div_T = div_T.at[1:-1].set((pseudo_temperature[2:] - pseudo_temperature[:-2]) / 2)
    div_rho = jnp.zeros_like(primitive_state[registered_variables.density_index])
    div_rho = div_rho.at[1:-1].set(
        (
            primitive_state[registered_variables.density_index][2:]
            - primitive_state[registered_variables.density_index][:-2]
        )
        / 2
    )
    no_spurious_shocks = div_T * div_rho > 0

    # iii) M1 > Mmin
    # NOTE: currently we only consider shocks moving left to right
    # (downstream = left neighbour, upstream = right neighbour)
    P2 = primitive_state[registered_variables.pressure_index, :-2]
    P2_CRs = P_CRs[:-2]
    P1 = primitive_state[registered_variables.pressure_index, 2:]
    P1_CRs = P_CRs[2:]

    # advanced Mach number calculation, formula 16 from Dubois et al, 2019.
    # FIX (2026-09-25): the downstream gas energy used to be ``P2 - P2_CRs``
    # (a pressure; the /(gamma - 1) was missing, so gamma2 = 2 without CRs)
    # and gamma_eff2 used the total P2 instead of P2_gas; see
    # ``mach_number_squared`` for the corrected, exact form.
    M1sq = mach_number_squared(
        P1, P1_CRs, P2, P2_CRs, gamma_gas=gamma_gas, gamma_cr=gamma_cr,
        denominator_floor=1e-6,
    )

    # simple Mach number calculation, crashes
    # the simulation where x_s = 1, better just evaluate
    # this where the other criterions hold / add a numerical
    # safeguard
    # x_s = rho2 / rho1
    # M1sq = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))

    mach_number_criterion = jnp.zeros_like(converging_flow_criterion, dtype=jnp.bool_)

    mach_number_criterion = mach_number_criterion.at[1:-1].set(M1sq > mach_min**2)

    return converging_flow_criterion & no_spurious_shocks & mach_number_criterion


@partial(
    jax.jit,
    static_argnames=[
        "registered_variables", "config", "shock_selection", "outermost_window_cells",
    ],
)
def find_shock_zone(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_selection: int = STRONGEST_SHOCK,
    outermost_window_cells: int = 8,
    gamma_gas: Union[float, jnp.ndarray] = 5 / 3,
) -> Tuple[
    Union[int, Int[Array, ""]], Union[int, Int[Array, ""]], Union[int, Int[Array, ""]]
]:
    """
    Find a numerically broadened shock region based of the strongest shock based
    on the result of the shock_sensor function and the pressure difference
    between adjacent cells. Assumes a shock front moving left to right.

    Args:
        primitive_state: the 1D primitive state.
        config: the simulation configuration.
        registered_variables: the registered variables.
        helper_data: helper data.
        shock_selection: ``STRONGEST_SHOCK`` (largest sensor among all flagged
            cells; the original behaviour) or ``OUTERMOST_SHOCK`` (largest
            sensor among the flagged cells within ``outermost_window_cells``
            of the outermost flagged cell, i.e. the forward shock).
        outermost_window_cells: see ``shock_selection``.
        gamma_gas: adiabatic index of the gas (for the Mach criterion).

    Returns:
        index of max shock sensor,
        left boundary of broadened shock,
        right boundary of broadened shock

    """

    pressure = primitive_state[registered_variables.pressure_index]
    num_cells = pressure.shape[0]

    # one can either use the maximum of the shock sensor
    sensors = shock_sensor(pressure)
    # or the cell with maximum compression, as in Pfrommer et al 2017
    # div_v = _calculate_1d_divergence(primitive_state[registered_variables.velocity_index], config, helper_data.geometric_centers)

    shock_crit = shock_criteria(
        primitive_state, config, registered_variables, helper_data,
        gamma_gas=gamma_gas,
    )

    indices = jnp.arange(num_cells)
    if shock_selection == OUTERMOST_SHOCK:
        # restrict the search to the outermost flagged shock: the flagged
        # cells within ``outermost_window_cells`` inside the outermost one
        outermost_flag = jnp.max(jnp.where(shock_crit, indices, -1))
        candidates = shock_crit & (indices >= outermost_flag - outermost_window_cells)
    elif shock_selection == STRONGEST_SHOCK:
        candidates = shock_crit
    else:
        raise ValueError(f"unknown shock_selection {shock_selection}")

    max_shock_idx = jnp.argmax(jnp.where(candidates, sensors, -1))
    # max_shock_idx = jnp.argmin(jnp.where(shock_crit, div_v, 1))

    # calculate differences in pressure
    pressure_differences = jnp.zeros_like(pressure)
    # 0 <- 1 - 0
    pressure_differences = pressure_differences.at[1:].set(pressure[1:] - pressure[:-1])

    # bound on the change in pressure between adjacent cells compared
    # to the pressure jump at the max_shock_index
    bound_diff = 0.1 * jnp.abs(pressure_differences[max_shock_idx])

    # left index: closest left index where |pressure_difference| < bound_diff or switched sign
    # right index: closest right index where |pressure_difference| < bound_diff or switched sign
    # (for a right-moving shock dp < 0 inside the jump, so "switched sign" is
    # dp > 0 on BOTH sides; the right edge used to test dp < 0, the shock's own
    # sign, and therefore always stopped at max_shock_idx + 1)
    left_indices = jnp.where(
        (indices < max_shock_idx)
        & ((jnp.abs(pressure_differences) < bound_diff) | (pressure_differences > 0)),
        indices,
        -1,
    )
    right_indices = jnp.where(
        (indices > max_shock_idx)
        & ((jnp.abs(pressure_differences) < bound_diff) | (pressure_differences > 0)),
        indices,
        num_cells,
    )
    left_idx = jnp.max(left_indices)
    right_idx = jnp.min(right_indices)

    return max_shock_idx, left_idx, right_idx
