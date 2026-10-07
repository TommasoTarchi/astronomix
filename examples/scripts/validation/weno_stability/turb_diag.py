"""Forensics on the last finite turbulence state: where is min rho, what is around it."""
import sys
import numpy as np
d = np.load(sys.argv[1])
s = d["state"]
print("t =", float(d["t"]), "shape", s.shape)
rho = s[0]
v = s[1:4]
B = s[4:7]
speed = np.sqrt((v**2).sum(0))
i, j, k = np.unravel_index(np.argmin(rho), rho.shape)
print("min rho", rho.min(), "at", (i, j, k), " max|v| there", speed[i, j, k])
print("max |v|", speed.max(), "at", np.unravel_index(np.argmax(speed), speed.shape))
for frac in [1e-1, 3e-2, 1e-2]:
    print(f"cells with rho < {frac}: {(rho < frac).sum()}")
n = rho.shape[0]
for name, arr in [("rho", rho), ("vx", v[0]), ("vy", v[1]), ("vz", v[2]), ("Bx", B[0]), ("By", B[1]), ("Bz", B[2])]:
    for ax, sl in [("x", lambda a, o: a[(i + o) % n, j, k]), ("y", lambda a, o: a[i, (j + o) % n, k]), ("z", lambda a, o: a[i, j, (k + o) % n])]:
        print(f"{name:3s} along {ax}:", np.array2string(np.array([sl(arr, o) for o in range(-4, 5)]), precision=3, max_line_width=200))
