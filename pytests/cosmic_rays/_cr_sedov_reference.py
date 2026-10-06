"""
Exact self-similar Sedov-Taylor blast wave of a gas + cosmic-ray fluid.

With a CONSTANT fraction ``zeta`` of the energy dissipated at the (strong)
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

Checked against the classical single-fluid constants: xi_0 = alpha^(-1/5) =
1.15167 (gamma = 5/3) and 1.03278 (7/5), Sedov (1959).
"""

import numpy as np
from scipy.integrate import quad, solve_ivp


def sedov_two_fluid(zeta=0.0, gamma_th=5.0 / 3.0, gamma_cr=4.0 / 3.0, xi_min=1e-6):
    """Return ``dict(alpha, X0, compression, gamma_shock, E_cr_fraction, solution)``.

    ``solution(xi)`` gives ``(D, U, P_th, P_cr)``; ``E_cr_fraction`` is the
    fraction of the blast energy held by the CRs (time-independent).
    """
    X0 = (gamma_cr - 1.0) * zeta / ((gamma_th - 1.0) * (1.0 - zeta)) if zeta > 0 else 0.0
    inv = (1.0 / (1.0 + X0)) / (gamma_th - 1.0) + (X0 / (1.0 + X0)) / (gamma_cr - 1.0)
    gamma_s = 1.0 + 1.0 / inv
    compression = (gamma_s + 1.0) / (gamma_s - 1.0)
    U1 = P1 = 1.0 - 1.0 / compression
    y0 = [compression, U1, P1 / (1.0 + X0), P1 * X0 / (1.0 + X0)]

    def rhs(xi, y):
        D, U, Pt, Pc = y
        w = U - xi
        P = Pt + Pc
        c2 = (gamma_th * Pt + gamma_cr * Pc) / D
        dD = D * (1.5 * U + 2.0 * U * w / xi - 3.0 * P / (D * w)) / (c2 - w * w)
        dU = (-2.0 * U * D / xi - w * dD) / D
        dPt = 3.0 * Pt / w + gamma_th * Pt * dD / D
        dPc = 3.0 * Pc / w + gamma_cr * Pc * dD / D
        return [dD, dU, dPt, dPc]

    sol = solve_ivp(rhs, [1.0, xi_min], y0, method="LSODA", rtol=1e-11,
                    atol=1e-14, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"two-fluid Sedov ODE failed: {sol.message}")

    def energy_density(s):
        D, U, Pt, Pc = sol.sol(s)
        return s**2 * (0.5 * D * U**2 + Pt / (gamma_th - 1.0) + Pc / (gamma_cr - 1.0))

    total = quad(energy_density, xi_min, 1.0, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    cr = quad(lambda s: s**2 * sol.sol(s)[3] / (gamma_cr - 1.0), xi_min, 1.0, limit=400)[0]
    return dict(
        alpha=16.0 * np.pi / 25.0 * total,
        X0=X0,
        compression=compression,
        gamma_shock=gamma_s,
        E_cr_fraction=cr / total,
        solution=sol.sol,
    )


def sedov_radius(alpha, energy=1.0, density=1.0, time=0.1):
    """Shock radius R = (E / (alpha rho))^(1/5) t^(2/5)."""
    return (energy / (alpha * density)) ** 0.2 * time**0.4
