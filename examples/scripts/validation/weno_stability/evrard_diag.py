"""Locate the first non-admissible cells in an Evrard snapshot series."""
import sys
import numpy as np
d = np.load(sys.argv[1])
states, times = d["states"], d["times"]
n = states.shape[-1]
dx = 4.0 / n
centers = (np.arange(n) + 0.5) * dx - 2.0
X, Y, Z = np.meshgrid(centers, centers, centers, indexing="ij")
R = np.sqrt(X**2 + Y**2 + Z**2)
for k in range(len(times)):
    rho, p = states[k, 0], states[k, 4]
    vr = (states[k, 1] * X + states[k, 2] * Y + states[k, 3] * Z) / np.maximum(R, 1e-9)
    bad = (rho <= 0) | (p <= 0) | ~np.isfinite(rho) | ~np.isfinite(p)
    print(f"t={times[k]:.4f} min rho={np.nanmin(rho):.3e} min p={np.nanmin(p):.3e} "
          f"nbad={bad.sum()} max|v|={np.nanmax(np.abs(states[k,1:4])):.3f} "
          f"min vr={np.nanmin(vr):.3f}" + (f" bad r: {np.round(np.unique(R[bad])[:6],3)}" if bad.any() else ""))
    if bad.any() and "--profile" in sys.argv:
        i, j, l = np.argwhere(bad)[0]
        print(" first bad cell", i, j, l, "r=", R[i, j, l])
        for name, arr in [("rho", rho), ("p", p), ("vx", states[k, 1]), ("vy", states[k,2]), ("vz", states[k,3])]:
            print(f"  {name} along x:", np.array2string(arr[max(i-4,0):i+5, j, l], precision=3))
            print(f"  {name} along y:", np.array2string(arr[i, max(j-4,0):j+5, l], precision=3))
        break
