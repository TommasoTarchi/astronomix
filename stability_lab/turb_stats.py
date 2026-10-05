"""Compare turbulence statistics of two runs from their state dumps.

    PYTHONPATH=. JAX_PLATFORMS=cpu python stability_lab/turb_stats.py pstats_pp128 pstats_recipe128
"""

# general
import glob
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ruff: noqa: E402
import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astronomix.analysis_helpers.energy_spectrum import (
    get_kinetic_energy_spectrum,
    vector_field_energy_spectrum,
)

STATE_DIR = "/export/data/lstorcks/weno_stability"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
T_CROSS = 0.5
SNAPSHOT_SPACING = 2.5 / 99  # turb.py: nsnap = 100 over 5 t_cross

tags = sys.argv[1:]
labels = {tag: tag for tag in tags}
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for tag in tags:
    files = sorted(glob.glob(os.path.join(STATE_DIR, f"{tag}_snap*.npy")))
    late = [f for f in files if int(f[-7:-4]) * SNAPSHOT_SPACING / T_CROSS >= 2.0]
    if not late:
        print(tag, "no late snapshots")
        continue
    log_density, density_spectra, kinetic_spectra = [], [], []
    for path in late:
        state = np.load(path)
        rho = state[0]
        log_density.append(np.log(np.maximum(rho, 1e-30)).ravel())
        k, power_rho = vector_field_energy_spectrum(
            jnp.asarray(rho - rho.mean()), jnp.zeros_like(rho), jnp.zeros_like(rho)
        )
        _, power_kinetic = get_kinetic_energy_spectrum(
            jnp.asarray(state[1]), jnp.asarray(state[2]), jnp.asarray(state[3]), jnp.asarray(rho)
        )
        density_spectra.append(np.asarray(power_rho))
        kinetic_spectra.append(np.asarray(power_kinetic))
    log_density = np.concatenate(log_density)
    k = np.asarray(k)
    density_spectrum = np.mean(density_spectra, axis=0)
    kinetic_spectrum = np.mean(kinetic_spectra, axis=0)
    sigma_s = float(np.std(log_density))
    print(f"{tag}: {len(late)} snapshots, sigma_ln_rho = {sigma_s:.3f}, "
          f"min rho = {np.exp(log_density.min()):.2e}, "
          f"mass fraction below 0.02 = "
          f"{np.mean(np.exp(log_density) * (np.exp(log_density) < 0.02)) / np.mean(np.exp(log_density)):.2e}")

    histogram, edges = np.histogram(log_density, bins=120, range=(-22, 6), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    axes[0].semilogy(centers, np.maximum(histogram, 1e-12), label=f"{tag} (sigma={sigma_s:.2f})")
    axes[1].loglog(k[1:], density_spectrum[1:], label=tag)
    axes[2].loglog(k[1:], kinetic_spectrum[1:] * k[1:] ** 2, label=tag)

    diag = np.loadtxt(os.path.join(OUT_DIR, "turb", f"diag_{tag}.txt"))
    axes[3].plot(diag[:, 0] / T_CROSS, diag[:, 4], label=tag)

axes[0].set_xlabel("ln rho"); axes[0].set_ylabel("PDF (t > 2 t_c)"); axes[0].set_ylim(1e-7, 1)
axes[0].axvline(np.log(0.02), color="k", lw=0.6, ls=":")
axes[1].set_xlabel("k"); axes[1].set_ylabel("density power")
axes[2].set_xlabel("k"); axes[2].set_ylabel("k^2 E_kin(k)")
axes[3].set_xlabel("t / t_c"); axes[3].set_ylabel("v_rms")
for axis in axes:
    axis.legend(fontsize=7)
plt.tight_layout()
out = os.path.join(OUT_DIR, f"stats_{'_vs_'.join(tags)}.png")
plt.savefig(out, dpi=110)
print("saved", out)
