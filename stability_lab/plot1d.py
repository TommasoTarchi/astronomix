"""Plot saved 1D runs against the exact Riemann solution."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "stability_lab")
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
from riemann1d import PROBLEMS
from astronomix.test_setups.reference_solutions.riemann_solver import _exact_riemann_ideal_gas

name = sys.argv[1]
files = sys.argv[2:]
problem = [p for p in PROBLEMS if p.name == name][0]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for f in files:
    d = np.load(f)
    x, s = d["x"], d["state"]
    for k, ax in enumerate(axes):
        ax.plot(x, s[k], ".-", ms=2, lw=0.6, label=os.path.basename(f))
xf = np.linspace(0, problem.box_size, 4000)
ex = _exact_riemann_ideal_gas(problem.rho_L, problem.u_L, problem.p_L, problem.rho_R, problem.u_R, problem.p_R, problem.gamma, xf, problem.t_end, problem.x0)
for k, ax in enumerate(axes):
    ax.plot(xf, np.asarray(ex[k]), "k-", lw=0.8, label="exact")
    ax.set_yscale("log" if k != 1 else "linear")
axes[0].legend(fontsize=7)
for ax, t in zip(axes, ["rho", "v", "p"]):
    ax.set_title(f"{name}: {t}")
plt.tight_layout()
plt.savefig(f"stability_lab/out/{name}.png", dpi=110)
print("saved", f"stability_lab/out/{name}.png")
