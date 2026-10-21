"""1D Sod shock-tube — paper figure (``shock_tube1D_test.svg``).

Runs the Sod shock tube at 200 cells with three solver configurations,

    FV (VL2, HLLC)       the finite-volume VL2 scheme (AthenaPK's van Leer
                         predictor-corrector with PLM and van Leer slopes),
                         HLLC Riemann solver, C_cfl = 0.5 as AthenaPK's Sod
    FV (VL2, HLLE)       the same with the HLLE Riemann solver
    FD (WENO-Z)          the finite-difference scheme with WENO-Z weights at
                         C_cfl = 0.75 (Sod needs no positivity-preserving WENO)

overlays each on the analytic Riemann solution and writes
``figures/shock_tube1D_test.svg`` (density / velocity / pressure with a zoomed
inset around the contact discontinuity). The exact density decreases
monotonically from left to right, so every rise of the numerical density
between neighbouring cells is a spurious oscillation; for every configuration
the script prints the largest such rise and their sum.

The same problem runs as a fast correctness check in
``pytests/hydrodynamics/shock_tube1D.py``; this script is the figure generator.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/shock_tube1D.py
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    HLL,
    HLLC,
    VL2,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
)
from astronomix.plotting_helpers.inset_box import add_inset_box
from astronomix.test_setups.hydrodynamics.shock_tube1D import (
    setup_sod_shock_tube,
    sod_shock_tube_solution,
)


# figures are written to the local figures/ directory
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

NUM_CELLS = 200

# The configurations shown, as (legend label, configuration, CFL number): the
# two Riemann solvers of the finite-volume VL2 scheme and the
# finite-difference scheme. VL2 always reconstructs with the van Leer slope,
# so the solvers are the finite-volume variants worth comparing.
RUNS = [
    (
        "FV (VL2, HLLC)",
        SimulationConfig(
            solver_mode=FINITE_VOLUME,
            time_integrator=VL2,
            riemann_solver=HLLC,
            num_cells=NUM_CELLS,
        ),
        0.5,
    ),
    (
        "FV (VL2, HLLE)",
        SimulationConfig(
            solver_mode=FINITE_VOLUME,
            time_integrator=VL2,
            riemann_solver=HLL,
            num_cells=NUM_CELLS,
        ),
        0.5,
    ),
    (
        "FD (WENO-Z)",
        SimulationConfig(
            weno_z=True,
            num_cells=NUM_CELLS,
        ),
        0.75,
    ),
]


def density_rises(density):
    """
    Rises of the density between neighbouring cells, which the monotonically
    decreasing exact Sod density does not have.

    Args:
        density: The numerical density on the cell centres.

    Returns:
        A tuple ``(largest_rise, total_rise)``: the largest increase of the
        density from one cell to its right neighbour and the sum of all
        increases (both >= 0).
    """
    rises = np.maximum(np.diff(density), 0.0)
    return rises.max(), rises.sum()


def run_shock_tube1D():
    """Run every configuration and write the Sod shock-tube overview figure."""

    # -------------------------------------------------------------
    # =========== ↓ High-resolution reference solution ↓ ==========
    # -------------------------------------------------------------

    fig, (ax_density, ax_velocity, ax_pressure) = plt.subplots(1, 3, figsize=(15, 5))

    # A finely resolved analytic solution serves as the visual reference the
    # coarser runs are plotted against.
    config_high_res = SimulationConfig(
        dimensionality=1,
        num_cells=1000,
    )

    registered_variables = get_registered_variables(config_high_res)
    helper_data_high_res = get_helper_data(config_high_res)
    _, config_high_res, params_high_res = setup_sod_shock_tube(
        config_high_res,
        registered_variables,
        SimulationParams(),
        helper_data_high_res,
    )
    reference_solution_high_res = sod_shock_tube_solution(
        config_high_res,
        registered_variables,
        params_high_res,
        helper_data_high_res,
    )

    ax_density.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.density_index],
        label="Reference",
        color="black",
    )
    ax_velocity.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.velocity_index],
        label="Reference",
        color="black",
    )
    ax_pressure.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.pressure_index],
        label="Reference",
        color="black",
    )
    ax_density.set_xlabel("x")
    ax_density.set_ylabel("Density")
    ax_velocity.set_xlabel("x")
    ax_velocity.set_ylabel("Velocity")
    ax_pressure.set_xlabel("x")
    ax_pressure.set_ylabel("Pressure")
    ax_density.set_title("Density")
    ax_velocity.set_title("Velocity")
    ax_pressure.set_title("Pressure")

    # -------------------------------------------------------------
    # =========== ↑ High-resolution reference solution ↑ ==========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Per-configuration run and plot ↓ =============
    # -------------------------------------------------------------

    for label, config, cfl in RUNS:
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        initial_state, config, params = setup_sod_shock_tube(
            config,
            registered_variables,
            SimulationParams(C_cfl=cfl),
            helper_data,
        )

        final_state = time_integration(
            initial_state,
            config,
            params,
            registered_variables,
        )

        largest_rise, total_rise = density_rises(
            np.asarray(final_state[registered_variables.density_index])
        )
        print(
            f"{label}: largest density rise between neighbouring cells "
            f"{largest_rise:.3e}, sum of the rises {total_rise:.3e}"
        )

        ax_density.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.density_index],
            label=label,
        )
        ax_velocity.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.velocity_index],
            label=label,
        )
        ax_pressure.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.pressure_index],
            label=label,
        )

    # -------------------------------------------------------------
    # ============ ↑ Per-configuration run and plot ↑ =============
    # -------------------------------------------------------------

    # A zoomed inset around the contact discontinuity highlights how the
    # schemes differ where the solutions are hardest to resolve.
    add_inset_box(ax_density, x1=0.62, x2=0.72, y1=0.20, y2=0.30, connect_loc1=1)

    handles, labels = ax_density.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIG_DIR / "shock_tube1D_test.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    run_shock_tube1D()
