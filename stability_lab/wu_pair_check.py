"""Admissibility of single vs paired Lax-Friedrichs split states of ideal MHD.

Random admissible state pairs with equal B_n at plasma beta 1e-1, 1e-3, 1e-5:
how often q + F/alpha is inadmissible at the fast speed, and how often the
weighted inflow pair (q1 + F1/alpha + q2 - F2/alpha)/2 is (Wu 2018).

    python stability_lab/wu_pair_check.py
"""
import numpy as np
rng = np.random.default_rng(0)
g = 5/3
def flux(rho, v, B, p):
    pt = p + 0.5 * (B**2).sum(0); E = p/(g-1) + 0.5*rho*(v**2).sum(0) + 0.5*(B**2).sum(0)
    F = np.zeros((8,) + rho.shape)
    F[0] = rho*v[0]
    for k in range(3):
        F[1+k] = rho*v[k]*v[0] - B[0]*B[k] + (pt if k == 0 else 0)
        F[4+k] = v[0]*B[k] - B[0]*v[k]
    F[7] = (E + pt)*v[0] - B[0]*(v*B).sum(0)
    q = np.concatenate([rho[None], rho*v, B, E[None]])
    return q, F
def pressure(q):
    return (g-1)*(q[7] - 0.5*(q[1:4]**2).sum(0)/q[0] - 0.5*(q[4:7]**2).sum(0))
def cf(rho, B, p):
    a2 = g*p/rho; b2 = (B**2).sum(0)/rho; bn2 = B[0]**2/rho
    return np.sqrt(0.5*(a2 + b2 + np.sqrt(np.maximum((a2+b2)**2 - 4*a2*bn2, 0))))
n = 200000
for logbeta in (-1, -3, -5):
    bad_single = bad_pair = 0
    worst = 1.0
    rho1, rho2 = np.exp(rng.normal(0, 1, n)), np.exp(rng.normal(0, 1, n))
    v1, v2 = rng.normal(0, 1, (3, n)), rng.normal(0, 1, (3, n))
    Bn = rng.normal(0, 1, n)
    Bt1, Bt2 = rng.normal(0, 1, (2, n)), rng.normal(0, 1, (2, n))
    B1 = np.stack([Bn, Bt1[0], Bt1[1]]); B2 = np.stack([Bn, Bt2[0], Bt2[1]])
    p1 = 0.5 * 10**logbeta * (B1**2).sum(0) * np.exp(rng.normal(0, 1, n))
    p2 = 0.5 * 10**logbeta * (B2**2).sum(0) * np.exp(rng.normal(0, 1, n))
    q1, F1 = flux(rho1, v1, B1, p1); q2, F2 = flux(rho2, v2, B2, p2)
    alpha = np.maximum(np.abs(v1[0]) + cf(rho1, B1, p1), np.abs(v2[0]) + cf(rho2, B2, p2))
    single = pressure(q1 + F1/alpha) / p1
    pair = pressure(0.5*(q1 + F1/alpha + q2 - F2/alpha)) / (0.5*(p1 + p2))
    print(f"beta ~ 1e{logbeta}: single split state inadmissible {np.mean(single < 0)*100:5.1f} %  "
          f"pair inadmissible {np.mean(pair < 0)*100:5.2f} %  min pair p/p {pair.min():+.2e}")
print("--- pair admissibility vs alpha factor")
for logbeta in (-1, -3, -5):
    rho1, rho2 = np.exp(rng.normal(0, 1, n)), np.exp(rng.normal(0, 1, n))
    v1, v2 = rng.normal(0, 1, (3, n)), rng.normal(0, 1, (3, n))
    Bn = rng.normal(0, 1, n)
    Bt1, Bt2 = rng.normal(0, 1, (2, n)), rng.normal(0, 1, (2, n))
    B1 = np.stack([Bn, Bt1[0], Bt1[1]]); B2 = np.stack([Bn, Bt2[0], Bt2[1]])
    p1 = 0.5 * 10**logbeta * (B1**2).sum(0) * np.exp(rng.normal(0, 1, n))
    p2 = 0.5 * 10**logbeta * (B2**2).sum(0) * np.exp(rng.normal(0, 1, n))
    q1, F1 = flux(rho1, v1, B1, p1); q2, F2 = flux(rho2, v2, B2, p2)
    alpha0 = np.maximum(np.abs(v1[0]) + cf(rho1, B1, p1), np.abs(v2[0]) + cf(rho2, B2, p2))
    out = []
    for factor in (1.0, 1.05, 1.1, 1.2, 1.5):
        alpha = factor * alpha0
        pair = pressure(0.5*(q1 + F1/alpha + q2 - F2/alpha))
        out.append(f"x{factor}: {int((pair < 0).sum())}")
    print(f"beta ~ 1e{logbeta}: inadmissible pairs (of {n}) " + "  ".join(out))
