"""
Exact self-similar Sedov-Taylor blast wave of a gas + cosmic-ray fluid.

With a constant fraction ``zeta`` of the energy dissipated at the (strong)
shock put into CRs, the problem has no scale beyond E and rho_0, so it stays
self-similar (Chevalier 1983): R(t) = (E / (alpha rho_0))^(1/5) t^(2/5).
Pfrommer et al. (2017, Sec. 4.2) state that no such solution exists and
compare with a single-fluid gamma = 7/5 Sedov instead; it is, however, a
four-variable ODE in xi = r / R, integrated here from the shock inwards:

    D, U, P_th, P_cr   with   rho = rho_0 D,  v = Rdot U,  p_i = rho_0 Rdot^2 P_i

    (U - xi) D' + D U' = -2 U D / xi                          (mass)
    (U - xi) U' + (P_th' + P_cr') / D = 3/2 U                 (momentum)
    (U - xi) P_i' - gamma_i P_i (U - xi) D'/D = 3 P_i         (entropy of each fluid)

with the strong-shock jump of the composite fluid at xi = 1: the downstream
CR-to-gas pressure ratio is X0 = (gamma_cr - 1) zeta / ((gamma_th - 1)(1 - zeta))
(= zeta / [2 (1 - zeta)] for 5/3, 4/3), the mixture's energy index gamma_s
follows from e = P_th/(gamma_th-1) + P_cr/(gamma_cr-1), the compression is
(gamma_s + 1)/(gamma_s - 1), and U(1) = P(1) = 1 - 1/compression. The energy
integral gives alpha = (16 pi / 25) * int xi^2 [D U^2 / 2 + P_th/(gamma_th-1)
+ P_cr/(gamma_cr-1)] dxi.

The solution reproduces the classical single-fluid constants
xi_0 = alpha^(-1/5) = 1.15167 (gamma = 5/3) and 1.03278 (7/5) of Sedov (1959).
"""

# typing
from typing import Any, Dict

# numerics
import numpy as np
from scipy.integrate import (
    quad,
    solve_ivp,
)


def sedov_two_fluid(
    zeta: float = 0.0,
    gamma_th: float = 5.0 / 3.0,
    gamma_cr: float = 4.0 / 3.0,
    xi_min: float = 1e-6,
) -> Dict[str, Any]:
    """
    Self-similar two-fluid Sedov-Taylor solution for a constant injection
    efficiency.

    Args:
        zeta: The fraction of the energy dissipated at the shock that is put
            into cosmic rays (0 gives the single-fluid gamma_th solution).
        gamma_th: The adiabatic index of the thermal gas.
        gamma_cr: The adiabatic index of the cosmic rays.
        xi_min: The inner end of the ODE integration in xi = r / R (the ODE is
            singular at xi = 0).

    Returns:
        ``dict(alpha, X0, compression, gamma_shock, E_cr_fraction, solution)``:
        the energy constant alpha of R(t), the post-shock CR-to-gas pressure
        ratio X0, the shock compression, the energy index of the post-shock
        mixture, the (time-independent) fraction of the blast energy held by
        the CRs, and ``solution(xi)``, which returns ``(D, U, P_th, P_cr)``.
    """
    if zeta > 0:
        X0 = (gamma_cr - 1.0) * zeta / ((gamma_th - 1.0) * (1.0 - zeta))
    else:
        X0 = 0.0
    mixture_inverse_gamma_minus_one = (
        (1.0 / (1.0 + X0)) / (gamma_th - 1.0) + (X0 / (1.0 + X0)) / (gamma_cr - 1.0)
    )
    gamma_s = 1.0 + 1.0 / mixture_inverse_gamma_minus_one
    compression = (gamma_s + 1.0) / (gamma_s - 1.0)
    shock_velocity = 1.0 - 1.0 / compression
    shock_pressure = shock_velocity
    shock_state = [
        compression,
        shock_velocity,
        shock_pressure / (1.0 + X0),
        shock_pressure * X0 / (1.0 + X0),
    ]

    def rhs(xi, state):
        """Derivatives d/dxi of (D, U, P_th, P_cr) from the ODE system above."""
        D, U, P_th, P_cr = state
        relative_velocity = U - xi
        P = P_th + P_cr
        sound_speed_squared = (gamma_th * P_th + gamma_cr * P_cr) / D
        dD_dxi = D * (
            1.5 * U
            + 2.0 * U * relative_velocity / xi
            - 3.0 * P / (D * relative_velocity)
        ) / (sound_speed_squared - relative_velocity * relative_velocity)
        dU_dxi = (-2.0 * U * D / xi - relative_velocity * dD_dxi) / D
        dP_th_dxi = 3.0 * P_th / relative_velocity + gamma_th * P_th * dD_dxi / D
        dP_cr_dxi = 3.0 * P_cr / relative_velocity + gamma_cr * P_cr * dD_dxi / D
        return [dD_dxi, dU_dxi, dP_th_dxi, dP_cr_dxi]

    ode_solution = solve_ivp(
        rhs,
        [1.0, xi_min],
        shock_state,
        method="LSODA",
        rtol=1e-11,
        atol=1e-14,
        dense_output=True,
    )
    if not ode_solution.success:
        raise RuntimeError(f"two-fluid Sedov ODE failed: {ode_solution.message}")

    def energy_integrand(xi):
        D, U, P_th, P_cr = ode_solution.sol(xi)
        return xi**2 * (0.5 * D * U**2 + P_th / (gamma_th - 1.0) + P_cr / (gamma_cr - 1.0))

    def cosmic_ray_energy_integrand(xi):
        return xi**2 * ode_solution.sol(xi)[3] / (gamma_cr - 1.0)

    total_energy_integral = quad(
        energy_integrand,
        xi_min,
        1.0,
        limit=400,
        epsabs=1e-13,
        epsrel=1e-11,
    )[0]
    cr_energy_integral = quad(
        cosmic_ray_energy_integrand,
        xi_min,
        1.0,
        limit=400,
    )[0]
    return dict(
        alpha=16.0 * np.pi / 25.0 * total_energy_integral,
        X0=X0,
        compression=compression,
        gamma_shock=gamma_s,
        E_cr_fraction=cr_energy_integral / total_energy_integral,
        solution=ode_solution.sol,
    )


def sedov_radius(
    alpha: float,
    energy: float = 1.0,
    density: float = 1.0,
    time: float = 0.1,
) -> float:
    """
    Shock radius R = (E / (alpha rho))^(1/5) t^(2/5) of a self-similar blast.

    Args:
        alpha: The energy constant of the self-similar solution.
        energy: The blast energy E.
        density: The ambient density rho.
        time: The time t.

    Returns:
        The shock radius.
    """
    return (energy / (alpha * density)) ** 0.2 * time**0.4
