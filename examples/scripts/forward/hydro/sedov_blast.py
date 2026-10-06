"""Sedov-Taylor blast wave — paper figure (4x3 layout at 128^3).

Generates ``figures/sedov_blast_<N>.png`` (and a ``.pdf`` copy): radial
profiles (density, |v|, pressure) of the spherical Sedov-Taylor blast at
t = 0.1 compared with the exact self-similar solution, for four solvers, one
per row, all at the same resolution:

    row 0:  FV (HLL, minmod)
    row 1:  FV (HLLC, minmod)
    row 2:  FV (AM-HLLC, minmod)
    row 3:  FD (WENO)

Each run is cached separately under ``data/sedov_<key>_<N>.npz`` (binned radial
profiles + a scatter subsample), so the figure can be re-styled without
re-running any simulation. Pass ``--rerun`` to recompute.

Resolution can be overridden for a quick smoke test:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/sedov_blast.py --res 32 --rerun

Default run (the paper figure):

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/sedov_blast.py --rerun
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys

# jax
import jax
import jax.numpy as jnp
import jax.random as jr

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# exact solution
from exactpack.solvers.sedov.sedov import Sedov

# astronomix constants
from astronomix import (
    CARTESIAN,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLL,
    HLLC,
    MINMOD,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import (
    AM_HLLC,
    HYBRID_HLLC,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)

# astronomix functions
from astronomix import (
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
)


# shared hydro-figure helpers
from _common import (
    DATA_DIR,
    FIG_DIR,
    FD_COLOR,
    FV_AM_HLLC_COLOR,
    FV_HLLC_COLOR,
    FV_HLL_COLOR,
    rerun_requested,
)

# ---- physical setup (matches tests/hydro_tests/sedov3D.py) ----------------
T_END = 0.1
GAMMA = 5.0 / 3.0
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
# Injection region: a well-resolved sphere (~8 cells radius at 256^3) with a
# smooth tanh taper of ~2 cells, instead of a single-cell sharp top-hat. The
# taper removes the grid-imprint noise the sharp injection leaves near the
# origin, and the explosion pressure is renormalised (below) so the total
# deposited thermal energy is exactly E_EXPLOSION regardless of the smoothing.
R_EXPLOSION = 0.03
SMOOTH_CELLS = 2.0
NUM_BINS = 200
NUM_SCATTER = 80_000

# One entry per solver row in the figure, as
# (key, label, color, solver_mode, riemann_solver).
RUNS = [
    ("fv_hll", "FV (HLL, minmod)", FV_HLL_COLOR, FINITE_VOLUME, HLL),
    ("fv_hllc", "FV (HLLC, minmod)", FV_HLLC_COLOR, FINITE_VOLUME, HLLC),
    ("fv_amhllc", "FV (AM-HLLC, minmod)", FV_AM_HLLC_COLOR, FINITE_VOLUME, AM_HLLC),
    ("fd", "FD", FD_COLOR, FINITE_DIFFERENCE, HYBRID_HLLC),
]


def _resolution_from_argv():
    """Return the requested cubic resolution.

    Returns:
        The integer following a ``--res`` command-line flag, or the default
        256 (the paper resolution) when the flag is absent.
    """
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def make_config(solver_mode, riemann_solver, num_cells):
    """Build the simulation configuration for one Sedov solver row.

    Args:
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver to use (e.g. HLL, HLLC).
        num_cells: The number of cells per dimension (cubic domain).

    Returns:
        The assembled :class:`SimulationConfig`.
    """
    is_finite_difference = solver_mode == FINITE_DIFFERENCE
    kwargs = dict(
        geometry=CARTESIAN,
        solver_mode=solver_mode,
        riemann_solver=riemann_solver,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=num_cells,
        exact_end_time=True,
        progress_bar=True,
        mhd=is_finite_difference,  # the FD/WENO backend runs in MHD mode (B = 0 here)
    )
    if is_finite_difference:
        # Periodic box + positivity-preserving WENO for the strong point
        # explosion. The FD/WENO backend runs ~10x faster through the Pallas
        # (Triton) backend; results are bit-compatible with native JAX.
        kwargs.update(
            weno_positivity_preserving=True,
            boundary_settings=BoundarySettings(
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            ),
        )
    return SimulationConfig(**kwargs)


def simulate(solver_mode, riemann_solver, num_cells):
    """Run one 3D Sedov simulation and reduce it to radial diagnostics.

    Args:
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver to use (e.g. HLL, HLLC).
        num_cells: The number of cells per dimension (cubic domain).

    Returns:
        A dict of numpy arrays holding the binned radial means (density,
        |v|, pressure), a reproducible scatter subsample of the raw cells, and
        the maximum domain radius — everything the figure needs, cached to disk.
    """

    # -------------------------------------------------------------
    # ============ ↓ Initial condition (energy injection) ↓ =======
    # -------------------------------------------------------------

    config = make_config(solver_mode, riemann_solver, num_cells)
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (num_cells, num_cells, num_cells)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    # Smoothly-tapered spherical injection weight in [0, 1] (≈1 inside
    # R_EXPLOSION, tanh taper of SMOOTH_CELLS cells), then renormalise the
    # over-pressure so the deposited thermal energy above ambient,
    #   E = Σ (p - p_amb)/(γ-1) · ΔV ,
    # is exactly E_EXPLOSION (independent of the taper / resolution).
    dx = 1.0 / num_cells  # box_size = 1.0
    smooth_width = SMOOTH_CELLS * dx
    radius = helper_data.r
    weight = 0.5 * (1.0 - jnp.tanh((radius - R_EXPLOSION) / smooth_width))
    cell_volume = dx**3
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * cell_volume)
    gas_pressure = P_AMBIENT + delta_p * weight

    fields = dict(
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    if config.mhd:
        fields.update(
            magnetic_field_x=zeros, magnetic_field_y=zeros, magnetic_field_z=zeros
        )

    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables, **fields
    )
    config = finalize_config(config, initial_state.shape)

    params = SimulationParams(t_end=T_END, gamma=GAMMA)
    if config.mhd:
        params = params._replace(minimum_density=1e-6, minimum_pressure=1e-6)

    # -------------------------------------------------------------
    # ============ ↑ Initial condition (energy injection) ↑ =======
    # -------------------------------------------------------------

    result = time_integration(initial_state, config, params, registered_variables)

    # -------------------------------------------------------------
    # ================= ↓ Radial reduction / binning ↓ ============
    # -------------------------------------------------------------

    radius_flat = helper_data.r.flatten()
    density_flat = result[registered_variables.density_index].flatten()
    pressure_flat = result[registered_variables.pressure_index].flatten()
    velocity_index = registered_variables.velocity_index
    velocity_magnitude = jnp.sqrt(
        result[velocity_index.x] ** 2
        + result[velocity_index.y] ** 2
        + result[velocity_index.z] ** 2
    ).flatten()

    # Bin every cell by radius and average each field within a bin. Empty bins
    # would divide by zero, so their count is forced to one (their mean stays 0).
    domain_max_r = float(jnp.max(radius_flat))
    bins = jnp.linspace(0, domain_max_r, NUM_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_index = jnp.clip(
        jnp.searchsorted(bins, radius_flat, side="right") - 1, 0, NUM_BINS - 1
    )

    counts = jnp.zeros(NUM_BINS).at[bin_index].add(1.0)
    counts = jnp.where(counts == 0, 1.0, counts)
    mean_rho = jnp.zeros(NUM_BINS).at[bin_index].add(density_flat) / counts
    mean_v = jnp.zeros(NUM_BINS).at[bin_index].add(velocity_magnitude) / counts
    mean_p = jnp.zeros(NUM_BINS).at[bin_index].add(pressure_flat) / counts

    # Reproducible scatter subsample of the raw cells for the background cloud.
    perm = jr.permutation(jr.PRNGKey(42), radius_flat.shape[0])[:NUM_SCATTER]

    # -------------------------------------------------------------
    # ================= ↑ Radial reduction / binning ↑ ============
    # -------------------------------------------------------------

    return dict(
        bin_centers=np.asarray(bin_centers),
        mean_rho=np.asarray(mean_rho),
        mean_v=np.asarray(mean_v),
        mean_p=np.asarray(mean_p),
        r_scatter=np.asarray(radius_flat[perm]),
        rho_scatter=np.asarray(density_flat[perm]),
        v_scatter=np.asarray(velocity_magnitude[perm]),
        p_scatter=np.asarray(pressure_flat[perm]),
        domain_max_r=np.asarray(domain_max_r),
    )


def cache_path(key, num_cells):
    """Return the ``.npz`` cache path for run ``key`` at ``num_cells``^3."""
    return DATA_DIR / f"sedov_{key}_{num_cells}.npz"


def get_run(key, solver_mode, riemann_solver, num_cells, rerun):
    """Load the cached diagnostics for one run, computing them if needed.

    Args:
        key: The short run identifier used in the cache filename.
        solver_mode: The solver family, FINITE_VOLUME or FINITE_DIFFERENCE.
        riemann_solver: The Riemann solver to use (e.g. HLL, HLLC).
        num_cells: The number of cells per dimension (cubic domain).
        rerun: When True, recompute and overwrite the cache instead of reusing it.

    Returns:
        The dict of radial diagnostics produced by :func:`simulate`.
    """
    path = cache_path(key, num_cells)
    if path.exists() and not rerun:
        return dict(np.load(path))
    print(f"running Sedov {key} at {num_cells}^3 ...")
    data = simulate(solver_mode, riemann_solver, num_cells)
    np.savez(path, **data)
    print(f"  cached -> {path}")
    return data


def exact_solution(domain_max_r):
    """Evaluate the exact self-similar Sedov solution out to ``domain_max_r``.

    Args:
        domain_max_r: The maximum radius to sample the analytic solution over.

    Returns:
        A tuple ``(r_exact, sol)`` of the sample radii and the ExactPack
        solution object (fields ``density``, ``velocity``, ``pressure``).
    """
    solver = Sedov(geometry=3, eblast=E_EXPLOSION / RHO_AMBIENT, gamma=GAMMA, omega=0.0)
    r_exact = np.linspace(0.0, domain_max_r, 500)
    sol = solver(r=r_exact, t=T_END)
    return r_exact, sol


def plot(num_cells, rerun):
    """Draw the 4x3 Sedov figure (one solver per row, three fields per column).

    Args:
        num_cells: The number of cells per dimension (cubic domain).
        rerun: When True, recompute each run's cache before plotting.
    """
    fig, axes = plt.subplots(
        len(RUNS), 3, figsize=(12, 3.2 * len(RUNS)), sharex=True
    )

    col_labels = ["density", r"$|v|$", "pressure"]

    for row_index, (key, label, color, solver_mode, riemann_solver) in enumerate(RUNS):
        run_data = get_run(key, solver_mode, riemann_solver, num_cells, rerun)
        domain_max_r = float(run_data["domain_max_r"])
        r_exact, sol = exact_solution(domain_max_r)

        # Per column: the scatter cloud, the binned mean, the exact profile, and
        # whether the y-axis is logarithmic (pressure spans several decades).
        profiles = [
            ("rho", run_data["rho_scatter"], run_data["mean_rho"], sol["density"], False),
            ("v", run_data["v_scatter"], run_data["mean_v"], sol["velocity"], False),
            ("p", run_data["p_scatter"], run_data["mean_p"], sol["pressure"], True),
        ]
        for col_index, (_, scatter, mean, exact, log_y) in enumerate(profiles):
            ax = axes[row_index, col_index]
            ax.scatter(
                run_data["r_scatter"],
                scatter,
                color="lightgray",
                alpha=0.5,
                s=1,
                rasterized=True,
                label="simulation (sampled)",
            )
            ax.plot(
                run_data["bin_centers"], mean, "-", color=color, lw=1.8,
                label="binned mean",
            )
            ax.plot(r_exact, exact, "--", color="black", lw=1.5, label="exact solution")
            ax.grid(True, ls=":", alpha=0.6)
            ax.set_xlim(0, domain_max_r)
            if log_y:
                ax.set_yscale("log")
                ax.set_ylim(bottom=P_AMBIENT / 2)
            else:
                ax.set_ylim(0, None)
            ax.set_ylabel(col_labels[col_index])
        # The solver label goes on the left, outside the axes (a row label, not
        # an axes title), rotated to run alongside the row.
        axes[row_index, 0].annotate(
            label,
            xy=(0, 0.5),
            xytext=(-axes[row_index, 0].yaxis.labelpad - 28, 0),
            xycoords=axes[row_index, 0].yaxis.label,
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
        )

    for col_index in range(3):
        axes[-1, col_index].set_xlabel("radius")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005)
    )

    plt.tight_layout(rect=[0.04, 0.03, 1, 1])
    out = FIG_DIR / f"sedov_blast_{num_cells}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    num_cells = _resolution_from_argv()
    plot(num_cells, rerun_requested())
