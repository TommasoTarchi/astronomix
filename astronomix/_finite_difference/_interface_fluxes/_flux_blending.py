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
from astronomix._stencil_operations._stencil_operations import _shift


# ---------------------------------------------------------------------------
# Shared first-order Lax-Friedrichs interface flux (hydro & MHD, iso & ideal)
# ---------------------------------------------------------------------------

def _local_lax_friedrichs_flux(conserved_state, axis, params, config,
                               registered_variables,
                               internal_energy_density=None):
    """First-order local Lax-Friedrichs (Rusanov) interface flux along ``axis``.

    ``F_LLF[..., i]`` is the flux at interface ``i+1/2`` (cells ``i`` and ``i+1``),
    matching the WENO convention so the blended array feeds the existing
    ``-dt/dx (F_{i+1/2} - F_{i-1/2})`` divergence unchanged. Returns the full
    interface-flux array.

    ``internal_energy_density`` (dual-energy ``g``), when given, switches the
    pressure recovery exactly like the WENO path does. Without it, the raw
    ``E - KE`` recovery is float32 cancellation garbage in cold KE-dominated
    cells, and a blend that activates there would inject fluxes built from a
    corrupted pressure — observed to blow up runs that blend broadly.
    """
    ndim = config.dimensionality
    di = registered_variables.density_index
    rhomin = params.minimum_density
    is_ideal = (config.equation_of_state == IDEAL_GAS)
    is_mhd = bool(config.mhd)

    if ndim == 1:
        mom_all = [registered_variables.velocity_index]
    else:
        mom_all = [
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        ][:ndim]
    md = mom_all[axis]
    mom_others = [m for i, m in enumerate(mom_all) if i != axis]

    if is_mhd:
        B_all = [
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        ]
        Bd = B_all[axis]
        B_others = [B_all[i] for i in range(3) if i != axis]

    def R(a):
        return _shift(a, -1, axis=axis)

    def R_state(a):
        return _shift(a, -1, axis=axis + 1)

    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(R(conserved_state[di]), rhomin)
    mdL = conserved_state[md]
    mdR = R(conserved_state[md])
    vdL = mdL / rhoL
    vdR = mdR / rhoR

    veL = [conserved_state[m] / rhoL for m in mom_others]
    veR = [R(conserved_state[m]) / rhoR for m in mom_others]

    if is_mhd:
        BdL = conserved_state[Bd]
        BdR = R(conserved_state[Bd])
        BeL = [conserved_state[b] for b in B_others]
        BeR = [R(conserved_state[b]) for b in B_others]
        b2L = BdL * BdL
        b2R = BdR * BdR
        for bl, br in zip(BeL, BeR):
            b2L = b2L + bl * bl
            b2R = b2R + br * br

    if is_ideal:
        gamma = params.gamma
        EL = conserved_state[registered_variables.energy_index]
        ER = R(EL)
        keL = 0.5 * (mdL * mdL) / rhoL
        keR = 0.5 * (mdR * mdR) / rhoR
        for ve in veL:
            keL = keL + 0.5 * rhoL * ve * ve
        for ve in veR:
            keR = keR + 0.5 * rhoR * ve * ve
        eL = EL - keL
        eR = ER - keR
        if is_mhd:
            eL = eL - 0.5 * b2L
            eR = eR - 0.5 * b2R
        if internal_energy_density is not None:
            # Bryan+95 dual-energy switch, mirroring the WENO-side recovery
            eta = config.dual_energy_eta
            gL = internal_energy_density
            gR = _shift(internal_energy_density, -1, axis=axis)
            relL = (eL > eta * jnp.maximum(EL, 1e-30)) & (eL == eL)
            relR = (eR > eta * jnp.maximum(ER, 1e-30)) & (eR == eR)
            eL = jnp.where(relL, eL, gL)
            eR = jnp.where(relR, eR, gR)
        pL = jnp.maximum((gamma - 1.0) * eL, params.minimum_pressure)
        pR = jnp.maximum((gamma - 1.0) * eR, params.minimum_pressure)
        cs2L = gamma * pL / rhoL
        cs2R = gamma * pR / rhoR
    else:
        cs = params.isothermal_sound_speed
        cs2L = cs * cs
        cs2R = cs * cs
        pL = cs2L * rhoL
        pR = cs2R * rhoR

    if is_mhd:
        def cfast(b2, rho, Bn, cs2):
            b2_over_rho = b2 / rho
            bn2_over_rho = (Bn * Bn) / rho
            disc = jnp.maximum((b2_over_rho + cs2) ** 2 - 4.0 * bn2_over_rho * cs2, 0.0)
            return jnp.sqrt(jnp.maximum(0.5 * (b2_over_rho + cs2 + jnp.sqrt(disc)), 0.0))
        cL = cfast(b2L, rhoL, BdL, cs2L)
        cR = cfast(b2R, rhoR, BdR, cs2R)
    else:
        cL = jnp.sqrt(cs2L)
        cR = jnp.sqrt(cs2R)

    alpha = jnp.maximum(jnp.abs(vdL) + cL, jnp.abs(vdR) + cR)
    if config.weno_ad_frozen_weights:
        # frozen with the WENO splitting speed: d c / d p ~ 1 / c in cold gas
        alpha = jax.lax.stop_gradient(alpha)

    qR = R_state(conserved_state)
    FL = jnp.zeros_like(conserved_state)
    FR = jnp.zeros_like(conserved_state)

    FL = FL.at[di].set(mdL)
    FR = FR.at[di].set(mdR)

    fmdL = mdL * vdL + pL
    fmdR = mdR * vdR + pR
    if is_mhd:
        fmdL = fmdL + 0.5 * b2L - BdL * BdL
        fmdR = fmdR + 0.5 * b2R - BdR * BdR
    FL = FL.at[md].set(fmdL)
    FR = FR.at[md].set(fmdR)

    for k, m in enumerate(mom_others):
        feL = mdL * veL[k]
        feR = mdR * veR[k]
        if is_mhd:
            feL = feL - BdL * BeL[k]
            feR = feR - BdR * BeR[k]
        FL = FL.at[m].set(feL)
        FR = FR.at[m].set(feR)

    if is_mhd:
        FL = FL.at[Bd].set(jnp.zeros_like(BdL))
        FR = FR.at[Bd].set(jnp.zeros_like(BdR))
        for k, b in enumerate(B_others):
            FL = FL.at[b].set(BeL[k] * vdL - BdL * veL[k])
            FR = FR.at[b].set(BeR[k] * vdR - BdR * veR[k])

    if is_ideal:
        ei = registered_variables.energy_index
        if is_mhd:
            vdotBL = vdL * BdL
            vdotBR = vdR * BdR
            for k in range(len(mom_others)):
                vdotBL = vdotBL + veL[k] * BeL[k]
                vdotBR = vdotBR + veR[k] * BeR[k]
            FL = FL.at[ei].set((EL + pL + 0.5 * b2L) * vdL - BdL * vdotBL)
            FR = FR.at[ei].set((ER + pR + 0.5 * b2R) * vdR - BdR * vdotBR)
        else:
            FL = FL.at[ei].set((EL + pL) * vdL)
            FR = FR.at[ei].set((ER + pR) * vdR)

    return 0.5 * (FL + FR) - 0.5 * alpha * (qR - conserved_state)


# ---------------------------------------------------------------------------
# Activation: cold-crush temperature ramp
# ---------------------------------------------------------------------------

def _face_min_specific_pressure(conserved_state, axis, params, config,
                                registered_variables, internal_energy_density=None):
    """``min(p_L/rho_L, p_R/rho_R)`` per interface, with the dual-energy
    pressure recovery (for the cold-crush gate)."""
    di = registered_variables.density_index
    gamma = params.gamma
    rhomin = params.minimum_density

    def R(a):
        return _shift(a, -1, axis=axis)

    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(R(conserved_state[di]), rhomin)

    if config.dimensionality == 1:
        mom_all = [registered_variables.velocity_index]
    else:
        mom_all = [
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        ][:config.dimensionality]
    keL = sum(conserved_state[m] ** 2 for m in mom_all) * 0.5 / rhoL
    keR = sum(R(conserved_state[m]) ** 2 for m in mom_all) * 0.5 / rhoR

    ei = registered_variables.energy_index
    EL = conserved_state[ei]
    ER = R(EL)
    eL = EL - keL
    eR = ER - keR
    if config.mhd:
        b2L = sum(conserved_state[b] ** 2 for b in (
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z))
        eL = eL - 0.5 * b2L
        eR = eR - 0.5 * _shift(b2L, -1, axis=axis)

    if internal_energy_density is not None:
        # dual-energy switch: the raw e recovery is cancellation garbage in
        # exactly the cold cells this gate needs to classify
        eta = config.dual_energy_eta
        gL = internal_energy_density
        gR = _shift(internal_energy_density, -1, axis=axis)
        relL = (eL > eta * jnp.maximum(EL, 1e-30)) & (eL == eL)
        relR = (eR > eta * jnp.maximum(ER, 1e-30)) & (eR == eR)
        eL = jnp.where(relL, eL, gL)
        eR = jnp.where(relR, eR, gR)

    pL = jnp.maximum((gamma - 1.0) * eL, params.minimum_pressure)
    pR = jnp.maximum((gamma - 1.0) * eR, params.minimum_pressure)
    return jnp.minimum(pL / rhoL, pR / rhoR), rhoL, rhoR, mom_all


def _coldcrush_blend_weight(conserved_state, axis, params, config,
                            registered_variables,
                            internal_energy_density=None):
    """LLF weight for radiatively crushed cells: interfaces that are both
    SUB-floor cold and CONVERGING.

    Two gates, both per interface:

    * temperature ramp — on the COLDER of the two adjacent cells' recovered
      ``p/rho``, ramping from 1 at (or below) the effective temperature
      floor ``minimum_specific_pressure`` down to 0 at
      ``coldcrush_blend_factor`` times it. Any interface with a cold side
      under compression gets the diffusive flux: cold-cold isothermal
      collapse AND the boundary faces of a cold dense clump being crushed
      by hot surroundings (the hero-4 failure mode — a hotter-side gate
      left exactly those faces unprotected). The price is that shock fronts
      advancing into cold ambient gas are handled at first order locally —
      the classic FOFC trade, and what the reference Athena SNR setups do.
    * convergence gate — the normal velocity must be compressive across the
      interface (``v_L > v_R``), ramped over the floor sound speed. The
      cold, freely-expanding ejecta core is divergent and never activates,
      so its seeded clump structure is not diffused away; the static cold
      ambient has no convergence and is untouched.
    """
    gamma = params.gamma
    tfloor = params.minimum_specific_pressure  # p/rho at the floor temperature

    def R(a):
        return _shift(a, -1, axis=axis)

    T_face, rhoL, rhoR, mom_all = _face_min_specific_pressure(
        conserved_state, axis, params, config, registered_variables,
        internal_energy_density=internal_energy_density)

    # temperature ramp on the colder side: 1 at (or below) the floor
    # temperature, 0 at factor * floor — any compressed cold side qualifies
    blend_thr = config.positivity_config.coldcrush_blend_factor * tfloor
    w_T = jnp.clip(
        (blend_thr - T_face) / jnp.maximum(blend_thr - tfloor, 1e-30), 0.0, 1.0
    )

    # convergence gate: compressive normal velocity, ramped over the floor
    # sound speed so it switches on smoothly
    ma = mom_all[axis] if config.dimensionality > 1 else mom_all[0]
    vdL = conserved_state[ma] / rhoL
    vdR = R(conserved_state[ma]) / rhoR
    c_floor = jnp.sqrt(gamma * jnp.maximum(tfloor, 1e-30))
    w_conv = jnp.clip((vdL - vdR) / c_floor, 0.0, 1.0)

    return w_T * w_conv


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def _blend_interface_flux(dF_weno, conserved_state, axis, dtdx, params, config,
                          registered_variables, internal_energy_density=None):
    """Blend the WENO interface flux toward LLF along ``axis`` at cold
    interfaces under compression (``coldcrush_blend``; ideal gas). Returns
    ``dF_weno`` unchanged otherwise. ``dtdx`` is unused (kept for the call
    sites of the integrators)."""
    if not (config.positivity_config.coldcrush_blend and config.equation_of_state == IDEAL_GAS):
        return dF_weno
    F_llf = _local_lax_friedrichs_flux(
        conserved_state, axis, params, config, registered_variables,
        internal_energy_density=internal_energy_density)
    w = _coldcrush_blend_weight(
        conserved_state, axis, params, config, registered_variables,
        internal_energy_density=internal_energy_density)
    # The blend weight is a switching function (limiter activation): its
    # derivative carries no physical sensitivity, so the limiter is frozen at
    # its current activation for differentiation. The primal is untouched.
    w = jax.lax.stop_gradient(w)[None, ...]
    return dF_weno * (1.0 - w) + F_llf * w
