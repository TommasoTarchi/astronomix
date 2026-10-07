"""
Analytic solution of the two-fluid (thermal gas + cosmic-ray) Riemann problem.

Pfrommer, Pakmor, Schaal, Simpson & Springel (2017), "Simulating cosmic ray
physics on a moving mesh", MNRAS 465, 4500 (arXiv:1604.07399), Appendix C:
a left state (5) and a right state (1) of a composite fluid whose two
components have adiabatic indices ``gamma_th`` (gas) and ``gamma_cr`` (CRs).
The solution consists of a rarefaction fan (both components adiabatic), a
contact discontinuity (region 3 | region 2) and a shock (region 2 | 1). At the
shock the CRs are compressed adiabatically and, with ``injection_efficiency``
zeta > 0, additionally receive the fraction zeta of the energy dissipated at
the shock (``P_inj``), which the gas loses -- exactly the model implemented by
``astronomix._modules._cosmic_rays.cr_injection``.

Table 1 of the paper (runs "CR" and "CR+inj"): rho 1 | 0.125,
P_th 17.172 | 0.05, P_cr 34.344 | 0.05 (i.e. P_cr / P_th = 2 | 1), zeta 0.5,
t = 0.35, box [0, 10], membrane at 5. The paper quotes a shock compression of
x_s = 3.90 (CR) and 4.78 (CR+inj) and Mach numbers 10.00 / 9.56.

The code is a transcription of the paper's Mathematica notebook (Appendix C).
Its symbols follow the paper's notation and region numbering (1 = upstream of
the shock, 2 = post-shock, 3 = post-rarefaction, 5 = left state): ``xr`` =
rho3 / rho5 and ``xs`` = rho2 / rho1 are the rarefaction and shock compression
ratios, ``eps`` the total energy density, ``ACR5`` / ``Ath5`` the adiabats
P / rho^gamma of the left state, ``Xinj`` = zeta / (1 - zeta), ``gCRi`` the
adiabatic index of the freshly injected CRs, ``c5`` the sound speed of the
left state and ``vt`` the speed of the rarefaction tail.
"""

# typing
from typing import Dict, Tuple

# numerics
import numpy as np
from scipy.integrate import quad
from scipy.optimize import (
    root,
    root_scalar,
)


def _riemann_invariant_integrand(
    density: float,
    ACR5: float,
    Ath5: float,
    gamma_th: float,
    gamma_cr: float,
) -> float:
    """Integrand c(rho) / rho of the Riemann invariant of the composite polytrope."""
    value = (
        gamma_cr * ACR5 * density ** (gamma_cr - 3.0)
        + gamma_th * Ath5 * density ** (gamma_th - 3.0)
    )
    return np.sqrt(max(value, 0.0))


def _riemann_invariant_integral(
    density: float,
    ACR5: float,
    Ath5: float,
    gamma_th: float,
    gamma_cr: float,
) -> float:
    """Riemann invariant integral of c(rho) / rho from 0 to ``density``."""
    return quad(
        _riemann_invariant_integrand,
        0,
        density,
        args=(ACR5, Ath5, gamma_th, gamma_cr),
    )[0]


def _composite_sound_speed(
    density: float,
    ACR5: float,
    Ath5: float,
    gamma_th: float,
    gamma_cr: float,
) -> float:
    """Sound speed of the composite polytrope on the left-state adiabats."""
    return np.sqrt(
        gamma_cr * ACR5 * density ** (gamma_cr - 1.0)
        + gamma_th * Ath5 * density ** (gamma_th - 1.0)
    )


def cosmic_ray_shock_tube_regions(
    thermal_pressure_left: float = 17.172,
    thermal_pressure_right: float = 0.05,
    density_left: float = 1.0,
    density_right: float = 1.0 / 8.0,
    cosmic_ray_to_thermal_pressure_ratio_left: float = 2.0,
    cosmic_ray_to_thermal_pressure_ratio_right: float = 1.0,
    injection_efficiency: float = 0.5,
    gamma_th: float = 5.0 / 3.0,
    gamma_cr: float = 4.0 / 3.0,
) -> Dict[str, float]:
    """
    Solve the jump conditions and return the constant-state values and speeds.

    Args:
        thermal_pressure_left: The thermal pressure of the left state (5).
        thermal_pressure_right: The thermal pressure of the right state (1).
        density_left: The density of the left state.
        density_right: The density of the right state.
        cosmic_ray_to_thermal_pressure_ratio_left: P_cr / P_th of the left state.
        cosmic_ray_to_thermal_pressure_ratio_right: P_cr / P_th of the right state.
        injection_efficiency: The fraction zeta of the energy dissipated at the
            shock that is injected into CRs.
        gamma_th: The adiabatic index of the thermal gas.
        gamma_cr: The adiabatic index of the cosmic rays.

    Returns:
        A dict with the rarefaction ratio ``xr = rho3/rho5``, the shock
        compression ``xs = rho2/rho1``, the region-3 and region-2 states
        (``rho3, P_th3, P_cr3, rho2, P_th2, P_cr2``), the post-shock velocity
        ``v3`` (= v2), the shock speed ``vs``, the rarefaction head / tail
        speeds ``c5`` and ``vt`` and the injected CR pressure ``P_inj``. The
        left-state adiabats ``ACR5``, ``Ath5`` and the left-state Riemann
        invariant integral ``V_left`` are included for
        ``get_cosmic_ray_analytic_solution``.
    """
    # Initial guess for the root find (the paper's values).
    xr0, xs0 = 0.3, 3.8

    # Adiabatic index of the freshly injected CRs (the same population here).
    gCRi = gamma_cr

    eth1 = thermal_pressure_right / (gamma_th - 1.0)
    PCR5 = cosmic_ray_to_thermal_pressure_ratio_left * thermal_pressure_left
    PCR1 = cosmic_ray_to_thermal_pressure_ratio_right * thermal_pressure_right
    eCR1 = PCR1 / (gamma_cr - 1.0)
    Xinj = injection_efficiency / (1.0 - injection_efficiency)

    P5 = PCR5 + thermal_pressure_left
    P1 = PCR1 + thermal_pressure_right
    eps1 = eCR1 + eth1

    if P5 != 0:
        g5 = (gamma_cr * PCR5 + gamma_th * thermal_pressure_left) / P5
    else:
        g5 = (gamma_cr + gamma_th) / 2.0
    c5 = np.sqrt(g5 * P5 / density_left)

    def P3(xr):
        return PCR5 * xr**gamma_cr + thermal_pressure_left * xr**gamma_th

    def PCR2(xs):
        return PCR1 * xs**gamma_cr

    def eCR2(xs):
        return PCR2(xs) / (gamma_cr - 1.0)

    def adiabatic_thermal_energy(xs):
        return eth1 * xs**gamma_th

    injection_energy_factor = 1.0 / (
        (1.0 - injection_efficiency) * (gamma_th - 1.0) / (gCRi - 1.0)
        + injection_efficiency
    )

    def eps2(xr, xs):
        return (
            injection_energy_factor
            * (
                P3(xr) / (gCRi - 1.0)
                + Xinj * adiabatic_thermal_energy(xs)
                - (gamma_cr - 1.0) / (gCRi - 1.0) * eCR2(xs)
            )
            + eCR2(xs)
            - Xinj * adiabatic_thermal_energy(xs)
        )

    ACR5 = PCR5 * density_left ** (-gamma_cr)
    Ath5 = thermal_pressure_left * density_left ** (-gamma_th)

    def riemann_invariant_integral(density):
        return _riemann_invariant_integral(density, ACR5, Ath5, gamma_th, gamma_cr)

    V_left = riemann_invariant_integral(density_left)

    def jump_condition_residuals(compression_ratios):
        xr, xs = compression_ratios
        # Penalise unphysical trial states so the Levenberg-Marquardt solver
        # steps back.
        if xr <= 0 or xs <= 1:
            return [1e10, 1e10]
        p3_val = P3(xr)
        diff_V = V_left - riemann_invariant_integral(xr * density_left)
        velocity_residual = (
            (p3_val - P1) * (xs - 1.0) - density_right * xs * diff_V**2
        )
        energy_residual = eps2(xr, xs) - xs * eps1 - 0.5 * (p3_val + P1) * (xs - 1.0)
        return [velocity_residual, energy_residual]

    solution = root(jump_condition_residuals, [xr0, xs0], method="lm", tol=1e-12)
    if not solution.success:
        raise RuntimeError(
            f"CR shock tube: root finding did not converge ({solution.message})"
        )
    xr_sol, xs_sol = solution.x

    rho3 = xr_sol * density_left
    rho2 = xs_sol * density_right
    p3_sol = P3(xr_sol)

    pressure_diff_term = p3_sol - P1
    if pressure_diff_term >= 0:
        v3 = np.sqrt(pressure_diff_term * (xs_sol - 1.0) / (xs_sol * density_right))
    else:
        v3 = np.nan
    vs = v3 if rho2 == density_right else rho2 * v3 / (rho2 - density_right)

    c_rho3 = _composite_sound_speed(rho3, ACR5, Ath5, gamma_th, gamma_cr)
    vt = -(V_left - riemann_invariant_integral(rho3)) + c_rho3

    PCR3 = PCR5 * (rho3 / density_left) ** gamma_cr
    Pth3 = p3_sol - PCR3
    Pth2 = 1.0 / (1.0 + injection_efficiency) * (
        p3_sol
        + Xinj * (gCRi - 1.0) / (gamma_th - 1.0) * thermal_pressure_right * xs_sol**gamma_th
        - PCR1 * xs_sol**gamma_cr
    )
    Pinj = (
        Xinj
        * (gCRi - 1.0)
        / (gamma_th - 1.0)
        * (Pth2 - thermal_pressure_right * xs_sol**gamma_th)
    )

    return dict(
        xr=xr_sol,
        xs=xs_sol,
        rho3=rho3,
        P_th3=Pth3,
        P_cr3=PCR3,
        rho2=rho2,
        P_th2=Pth2,
        P_cr2=PCR2(xs_sol) + Pinj,
        P_inj=Pinj,
        v3=v3,
        vs=vs,
        c5=c5,
        vt=vt,
        ACR5=ACR5,
        Ath5=Ath5,
        V_left=V_left,
    )


def get_cosmic_ray_analytic_solution(
    thermal_pressure_left: float = 17.172,
    thermal_pressure_right: float = 0.05,
    density_left: float = 1.0,
    density_right: float = 1.0 / 8.0,
    cosmic_ray_to_thermal_pressure_ratio_left: float = 2.0,
    cosmic_ray_to_thermal_pressure_ratio_right: float = 1.0,
    injection_efficiency: float = 0.5,
    t_end: float = 0.35,
    left_boundary: float = 0.0,
    right_boundary: float = 10.0,
    initial_shock_pos: float = 5.0,
    num_rarefaction_evals: int = 51,
    gamma_th: float = 5.0 / 3.0,
    gamma_cr: float = 4.0 / 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Piecewise analytic profiles at ``t_end``.

    Args:
        thermal_pressure_left: The thermal pressure of the left state (5).
        thermal_pressure_right: The thermal pressure of the right state (1).
        density_left: The density of the left state.
        density_right: The density of the right state.
        cosmic_ray_to_thermal_pressure_ratio_left: P_cr / P_th of the left state.
        cosmic_ray_to_thermal_pressure_ratio_right: P_cr / P_th of the right state.
        injection_efficiency: The fraction zeta of the energy dissipated at the
            shock that is injected into CRs.
        t_end: The time at which the profiles are evaluated.
        left_boundary: The left edge of the domain.
        right_boundary: The right edge of the domain.
        initial_shock_pos: The position of the initial membrane.
        num_rarefaction_evals: The number of samples across the rarefaction fan.
        gamma_th: The adiabatic index of the thermal gas.
        gamma_cr: The adiabatic index of the cosmic rays.

    Returns:
        ``(x, rho, v, P_th, P_cr, P_tot)``. The arrays are piecewise with
        duplicated x at the jumps (plot them directly, or sample constant
        states via :func:`cosmic_ray_shock_tube_regions`).
    """
    regions = cosmic_ray_shock_tube_regions(
        thermal_pressure_left=thermal_pressure_left,
        thermal_pressure_right=thermal_pressure_right,
        density_left=density_left,
        density_right=density_right,
        cosmic_ray_to_thermal_pressure_ratio_left=cosmic_ray_to_thermal_pressure_ratio_left,
        cosmic_ray_to_thermal_pressure_ratio_right=cosmic_ray_to_thermal_pressure_ratio_right,
        injection_efficiency=injection_efficiency,
        gamma_th=gamma_th,
        gamma_cr=gamma_cr,
    )
    ACR5 = regions["ACR5"]
    Ath5 = regions["Ath5"]
    c5 = regions["c5"]
    vt = regions["vt"]
    v3 = regions["v3"]
    vs = regions["vs"]
    rho3 = regions["rho3"]
    rho2 = regions["rho2"]
    PCR5 = cosmic_ray_to_thermal_pressure_ratio_left * thermal_pressure_left
    PCR1 = cosmic_ray_to_thermal_pressure_ratio_right * thermal_pressure_right

    # --------------- ↓ Self-similar rarefaction fan ↓ ----------------
    # The fan spans x / t from -c5 (head) to -vt (tail); x is measured from the
    # initial membrane.
    fan_samples = np.linspace(0, num_rarefaction_evals, num_rarefaction_evals + 1)
    x = -c5 * t_end + (-vt + c5) * t_end * fan_samples / num_rarefaction_evals
    rarefaction_density = np.full(num_rarefaction_evals, np.nan)
    for sample_index in range(1, num_rarefaction_evals + 1):
        x_sample = x[sample_index]

        def rarefaction_residual(density):
            if density <= 0 or density > density_left:
                return 1e10
            sound_speed = _composite_sound_speed(density, ACR5, Ath5, gamma_th, gamma_cr)
            return (
                _riemann_invariant_integral(density, ACR5, Ath5, gamma_th, gamma_cr)
                - regions["V_left"]
                + x_sample / t_end
                + sound_speed
            )

        density_lower = max(rho3 * (1 - 1e-6), 1e-9)
        density_upper = density_left * (1 + 1e-6)
        root_result = root_scalar(
            rarefaction_residual,
            bracket=(density_lower, density_upper),
            method="brentq",
            xtol=1e-12,
        )
        if not root_result.converged:
            raise RuntimeError(
                f"CR shock tube: rarefaction root failed at x/t = {x_sample / t_end}"
            )
        rarefaction_density[sample_index - 1] = min(max(root_result.root, 0.0), density_left)

    rarefaction_thermal_pressure = Ath5 * rarefaction_density**gamma_th
    rarefaction_cosmic_ray_pressure = ACR5 * rarefaction_density**gamma_cr
    rarefaction_velocity = x[1:] / t_end + np.sqrt(
        gamma_cr * ACR5 * rarefaction_density ** (gamma_cr - 1)
        + gamma_th * Ath5 * rarefaction_density ** (gamma_th - 1)
    )
    # --------------- ↑ Self-similar rarefaction fan ↑ ----------------

    x_rarefaction_left = initial_shock_pos - c5 * t_end
    x_rarefaction_right = initial_shock_pos - vt * t_end
    x_contact = initial_shock_pos + v3 * t_end
    x_shock = initial_shock_pos + vs * t_end

    rho_full = np.concatenate(
        (
            [density_left, density_left],
            rarefaction_density,
            [rho3, rho3],
            [rho2, rho2],
            [density_right, density_right],
        )
    )
    velocity_full = np.concatenate(
        (
            [0.0, 0.0],
            rarefaction_velocity,
            [v3, v3],
            [v3, v3],
            [0.0, 0.0],
        )
    )
    thermal_pressure_full = np.concatenate(
        (
            [thermal_pressure_left, thermal_pressure_left],
            rarefaction_thermal_pressure,
            [regions["P_th3"], regions["P_th3"]],
            [regions["P_th2"], regions["P_th2"]],
            [thermal_pressure_right, thermal_pressure_right],
        )
    )
    cosmic_ray_pressure_full = np.concatenate(
        (
            [PCR5, PCR5],
            rarefaction_cosmic_ray_pressure,
            [regions["P_cr3"], regions["P_cr3"]],
            [regions["P_cr2"], regions["P_cr2"]],
            [PCR1, PCR1],
        )
    )
    x_full = np.concatenate(
        (
            [left_boundary, x_rarefaction_left],
            x[1:] + initial_shock_pos,
            [x_rarefaction_right, x_contact],
            [x_contact, x_shock],
            [x_shock, right_boundary],
        )
    )
    return (
        x_full,
        rho_full,
        velocity_full,
        thermal_pressure_full,
        cosmic_ray_pressure_full,
        thermal_pressure_full + cosmic_ray_pressure_full,
    )
