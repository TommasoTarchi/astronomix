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

Ported from the ``cosmic_rays`` branch (``tests/cosmic_ray_tests/
analytic_solution.py``, a transcription of the paper's Mathematica notebook)
on 2026-09-25. Changes: the misleadingly named
``thermal_to_cosmic_ray_pressure_ratio_*`` arguments (they were always
P_cr / P_th) are now ``cosmic_ray_to_thermal_pressure_ratio_*``; failed root
finds raise instead of printing; the region-2/3 plateau values are also
returned as a dict (``cosmic_ray_shock_tube_regions``) for tests.
"""

# general
from typing import Dict, Tuple

# numerics
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, root_scalar


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
    """Solve the jump conditions; return the constant-state values and speeds.

    Returns a dict with the rarefaction ratio ``xr = rho3/rho5``, the shock
    compression ``xs = rho2/rho1``, the region-3 and region-2 states
    (``rho3, P_th3, P_cr3, rho2, P_th2, P_cr2``), the post-shock velocity
    ``v3`` (= v2), the shock speed ``vs``, the rarefaction head/tail speeds
    ``c5`` and ``vt`` and the injected CR pressure ``P_inj``.
    """
    # initial guess for the root find (the paper's values)
    xr0, xs0 = 0.3, 3.8

    gth, gCR = gamma_th, gamma_cr
    # adiabatic index of the freshly injected CRs (the same population here)
    gCRi = gamma_cr

    eth1 = thermal_pressure_right / (gth - 1.0)
    PCR5 = cosmic_ray_to_thermal_pressure_ratio_left * thermal_pressure_left
    PCR1 = cosmic_ray_to_thermal_pressure_ratio_right * thermal_pressure_right
    eCR1 = PCR1 / (gCR - 1.0)
    Xinj = injection_efficiency / (1.0 - injection_efficiency)

    P5 = PCR5 + thermal_pressure_left
    P1 = PCR1 + thermal_pressure_right
    eps1 = eCR1 + eth1

    g5 = (gCR * PCR5 + gth * thermal_pressure_left) / P5 if P5 != 0 else (gCR + gth) / 2.0
    c5 = np.sqrt(g5 * P5 / density_left)

    def P3(xr):
        return PCR5 * xr**gCR + thermal_pressure_left * xr**gth

    def PCR2(xs):
        return PCR1 * xs**gCR

    def eCR2(xs):
        return PCR2(xs) / (gCR - 1.0)

    def ethad(xs):
        return eth1 * xs**gth

    fac = 1.0 / ((1.0 - injection_efficiency) * (gth - 1.0) / (gCRi - 1.0) + injection_efficiency)

    def eps2(xr, xs):
        return (
            fac * (P3(xr) / (gCRi - 1.0) + Xinj * ethad(xs) - (gCR - 1.0) / (gCRi - 1.0) * eCR2(xs))
            + eCR2(xs) - Xinj * ethad(xs)
        )

    ACR5 = PCR5 * density_left ** (-gCR)
    Ath5 = thermal_pressure_left * density_left ** (-gth)

    # Riemann invariant integrand of the composite polytrope
    def integrand(r):
        value = gCR * ACR5 * r ** (gCR - 3.0) + gth * Ath5 * r ** (gth - 3.0)
        return np.sqrt(max(value, 0.0))

    def Int_rho(rho):
        return quad(integrand, 0, rho)[0]

    V_left = Int_rho(density_left)

    def equations(vars_):
        xr, xs = vars_
        if xr <= 0 or xs <= 1:
            return [1e10, 1e10]
        p3_val = P3(xr)
        diff_V = V_left - Int_rho(xr * density_left)
        f1 = (p3_val - P1) * (xs - 1.0) - density_right * xs * diff_V**2
        f2 = eps2(xr, xs) - xs * eps1 - 0.5 * (p3_val + P1) * (xs - 1.0)
        return [f1, f2]

    sol = root(equations, [xr0, xs0], method="lm", tol=1e-12)
    if not sol.success:
        raise RuntimeError(f"CR shock tube: root finding did not converge ({sol.message})")
    xr_sol, xs_sol = sol.x

    rho3 = xr_sol * density_left
    rho2 = xs_sol * density_right
    p3_sol = P3(xr_sol)

    pressure_diff_term = p3_sol - P1
    v3 = np.sqrt(pressure_diff_term * (xs_sol - 1.0) / (xs_sol * density_right)) \
        if pressure_diff_term >= 0 else np.nan
    vs = v3 if rho2 == density_right else rho2 * v3 / (rho2 - density_right)

    c_rho3 = np.sqrt(gCR * ACR5 * rho3 ** (gCR - 1.0) + gth * Ath5 * rho3 ** (gth - 1.0))
    vt = -(V_left - Int_rho(rho3)) + c_rho3

    PCR3 = PCR5 * (rho3 / density_left) ** gCR
    Pth3 = p3_sol - PCR3
    Pth2 = 1.0 / (1.0 + injection_efficiency) * (
        p3_sol + Xinj * (gCRi - 1.0) / (gth - 1.0) * thermal_pressure_right * xs_sol**gth
        - PCR1 * xs_sol**gCR
    )
    Pinj = Xinj * (gCRi - 1.0) / (gth - 1.0) * (Pth2 - thermal_pressure_right * xs_sol**gth)

    return dict(
        xr=xr_sol, xs=xs_sol,
        rho3=rho3, P_th3=Pth3, P_cr3=PCR3,
        rho2=rho2, P_th2=Pth2, P_cr2=PCR2(xs_sol) + Pinj, P_inj=Pinj,
        v3=v3, vs=vs, c5=c5, vt=vt,
        ACR5=ACR5, Ath5=Ath5, V_left=V_left,
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
    """Piecewise analytic profiles at ``t_end``.

    Returns ``(x, rho, v, P_th, P_cr, P_tot)``. The arrays are piecewise with
    duplicated x at the jumps (plot them directly, or sample constant states
    via :func:`cosmic_ray_shock_tube_regions`).
    """
    reg = cosmic_ray_shock_tube_regions(
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
    gth, gCR = gamma_th, gamma_cr
    ACR5, Ath5 = reg["ACR5"], reg["Ath5"]
    c5, vt, v3, vs = reg["c5"], reg["vt"], reg["v3"], reg["vs"]
    rho3, rho2 = reg["rho3"], reg["rho2"]
    PCR5 = cosmic_ray_to_thermal_pressure_ratio_left * thermal_pressure_left
    PCR1 = cosmic_ray_to_thermal_pressure_ratio_right * thermal_pressure_right

    def integrand(r):
        value = gCR * ACR5 * r ** (gCR - 3.0) + gth * Ath5 * r ** (gth - 3.0)
        return np.sqrt(max(value, 0.0))

    def Int_rho(rho):
        return quad(integrand, 0, rho)[0]

    # self-similar rarefaction fan, x/t from -c5 to -vt
    t = t_end
    x = -c5 * t + (-vt + c5) * t * np.linspace(0, num_rarefaction_evals, num_rarefaction_evals + 1) / num_rarefaction_evals
    rhorf = np.full(num_rarefaction_evals, np.nan)
    for j in range(1, num_rarefaction_evals + 1):
        x_j = x[j]

        def rf(rho):
            if rho <= 0 or rho > density_left:
                return 1e10
            c_rho = np.sqrt(gCR * ACR5 * rho ** (gCR - 1.0) + gth * Ath5 * rho ** (gth - 1.0))
            return Int_rho(rho) - reg["V_left"] + x_j / t + c_rho

        lo = max(rho3 * (1 - 1e-6), 1e-9)
        hi = density_left * (1 + 1e-6)
        sol_rf = root_scalar(rf, bracket=(lo, hi), method="brentq", xtol=1e-12)
        if not sol_rf.converged:
            raise RuntimeError(f"CR shock tube: rarefaction root failed at x/t = {x_j / t}")
        rhorf[j - 1] = min(max(sol_rf.root, 0.0), density_left)

    Pthrf = Ath5 * rhorf**gth
    PCRrf = ACR5 * rhorf**gCR
    vrf = x[1:] / t + np.sqrt(gCR * ACR5 * rhorf ** (gCR - 1) + gth * Ath5 * rhorf ** (gth - 1))

    x_rarefaction_left = initial_shock_pos - c5 * t_end
    x_rarefaction_right = initial_shock_pos - vt * t_end
    x_contact = initial_shock_pos + v3 * t_end
    x_shock = initial_shock_pos + vs * t_end

    rho_full = np.concatenate((
        [density_left, density_left], rhorf, [rho3, rho3], [rho2, rho2],
        [density_right, density_right],
    ))
    velocity_full = np.concatenate(([0.0, 0.0], vrf, [v3, v3], [v3, v3], [0.0, 0.0]))
    thermal_pressure_full = np.concatenate((
        [thermal_pressure_left, thermal_pressure_left], Pthrf,
        [reg["P_th3"], reg["P_th3"]], [reg["P_th2"], reg["P_th2"]],
        [thermal_pressure_right, thermal_pressure_right],
    ))
    cosmic_ray_pressure_full = np.concatenate((
        [PCR5, PCR5], PCRrf, [reg["P_cr3"], reg["P_cr3"]],
        [reg["P_cr2"], reg["P_cr2"]], [PCR1, PCR1],
    ))
    x_full = np.concatenate((
        [left_boundary, x_rarefaction_left], x[1:] + initial_shock_pos,
        [x_rarefaction_right, x_contact], [x_contact, x_shock],
        [x_shock, right_boundary],
    ))
    return (
        x_full,
        rho_full,
        velocity_full,
        thermal_pressure_full,
        cosmic_ray_pressure_full,
        thermal_pressure_full + cosmic_ray_pressure_full,
    )
