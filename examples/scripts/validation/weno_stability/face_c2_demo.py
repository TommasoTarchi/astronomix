"""Toy check: face sound speed from the current averaging vs consistent averaging."""
import numpy as np

gamma = 5.0 / 3.0


def face_c2_current(rho_l, v_l, p_l, rho_r, v_r, p_r):
    h_l = (p_l / (gamma - 1) + 0.5 * rho_l * v_l**2 + p_l) / rho_l
    h_r = (p_r / (gamma - 1) + 0.5 * rho_r * v_r**2 + p_r) / rho_r
    rho_f = 0.5 * (rho_l + rho_r)
    v_f = 0.5 * (rho_l * v_l + rho_r * v_r) / rho_f
    return (gamma - 1) * (0.5 * (h_l + h_r) - 0.5 * v_f**2)


for M in [1, 3, 10, 30, 100]:
    for ratio in [1, 10, 100, 1e4]:
        # cold dense fast side next to slow tenuous side, both with c = 1
        rho_l, rho_r = 1.0, 1.0 / ratio
        p_l, p_r = rho_l / gamma, rho_r / gamma
        c2 = face_c2_current(rho_l, M, p_l, rho_r, 0.0, p_r)
        # same pair in a frame moving with the dense side
        c2_boost = face_c2_current(rho_l, 0.0, p_l, rho_r, -M, p_r)
        print(f"M={M:5} ratio={ratio:8.0e}  c2_face={c2:10.3e}  "
              f"boosted frame={c2_boost:10.3e}  (true c2 = 1)")
