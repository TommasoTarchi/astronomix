"""Census of the first-order inflow states over a whole 3D MHD state.

For every cell: the per-axis inflow-pair bases (alpha_L w+_{i-1} + alpha_R w-_{i+1}) / S_d
and their axis-weighted sum sum_d S_d base_d / sum_d S_d. Per-axis bases fail where B_n
varies along the axis; the sum is what a cell-level (all axes jointly) limiter would
need, and it can only fail through the central divergence of the cell-centred field
(Wu 2018), which is where a divergence-consistent source term would be required.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/inflow_census.py STATE.npy|FAILSTEP.npz
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
import jax.numpy as jnp

from astronomix import SimulationConfig, get_registered_variables
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd
from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_physical_flux
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 5.0 / 3.0
path = sys.argv[1]
loaded = np.load(path)
state = loaded["state"] if path.endswith(".npz") else loaded
primitive = jnp.asarray(state.astype(np.float64))
rv = get_registered_variables(SimulationConfig(equation_of_state=IDEAL_GAS, dimensionality=3, mhd=True))
q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
n = q.shape[-1]
dx = 1.0 / n


def pressure(s):
    kinetic = 0.5 * sum(s[k] ** 2 for k in rv.momentum_index) / s[rv.density_index]
    magnetic = 0.5 * sum(s[k] ** 2 for k in rv.magnetic_index)
    return (GAMMA - 1.0) * (s[rv.energy_index] - kinetic - magnetic)


p = pressure(q)
density = q[rv.density_index]
field = [q[k] for k in rv.magnetic_index]
velocity = [q[k] / density for k in rv.momentum_index]
b2 = sum(f * f for f in field)

weighted = jnp.zeros_like(q)
weights = jnp.zeros_like(p)
axis_bad = []
for axis in range(3):
    a2 = GAMMA * p / density
    fast = jnp.sqrt(0.5 * (a2 + b2 / density + jnp.sqrt(jnp.maximum((a2 + b2 / density) ** 2
                                                                     - 4.0 * a2 * field[axis] ** 2 / density, 0.0))))
    radius = jnp.abs(velocity[axis]) + fast
    alpha_right = jnp.max(jnp.stack([jnp.roll(radius, -o, axis) for o in (-2, -1, 0, 1, 2, 3)]), axis=0)
    alpha_left = jnp.roll(alpha_right, 1, axis)
    flux = mhd_physical_flux(q, GAMMA, rv, axis)
    plus_left = jnp.roll(q, 1, axis + 1) + jnp.roll(flux, 1, axis + 1) / alpha_left[None]
    minus_right = jnp.roll(q, -1, axis + 1) - jnp.roll(flux, -1, axis + 1) / alpha_right[None]
    total = alpha_left + alpha_right
    base = (alpha_left[None] * plus_left + alpha_right[None] * minus_right) / total[None]
    axis_bad.append(np.asarray(pressure(base) <= 0))
    weighted = weighted + total[None] * base
    weights = weights + total
summed = weighted / weights[None]
summed_bad = np.asarray(pressure(summed) <= 0)
any_axis_bad = axis_bad[0] | axis_bad[1] | axis_bad[2]
div_central = sum((jnp.roll(field[a], -1, a) - jnp.roll(field[a], 1, a)) / (2 * dx) for a in range(3))
relative_div = np.asarray(jnp.abs(div_central) * dx / jnp.sqrt(b2 + 1e-300))

cells = q[0].size
print(f"{os.path.basename(path)}: {cells} cells")
for axis in range(3):
    print(f"  axis {axis}: inadmissible first-order inflow base in {axis_bad[axis].sum():8d} cells")
print(f"  some axis inadmissible:                 {any_axis_bad.sum():8d} cells")
print(f"  axis-weighted sum inadmissible:          {summed_bad.sum():8d} cells "
      f"(median |div B| dx/|B| there: {np.median(relative_div[summed_bad]) if summed_bad.any() else float('nan'):.3f}; "
      f"everywhere: {np.median(relative_div):.4f})")
