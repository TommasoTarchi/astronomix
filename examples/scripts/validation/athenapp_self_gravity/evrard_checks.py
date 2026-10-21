"""
Checks of the VL2 self-gravity coupling in Evrard's collapse.

Uses the collapse setup of ``examples/scripts/forward/self_gravity/_collapse.py``
with the finite-volume scheme (VL2, HLLC, C_cfl = 0.3) and prints

* how far the cold collapse at 128^3 gets without AthenaPK's floors,
* the relative total-energy error at 128^3 for two sets of floors, with
  Athena++'s and with the time-centred energy source,
* the relative total-energy error of the cold (e0 = 0.05) and the mild
  (e0 = 0.2) collapse at 32^3 for fixed time steps, with both energy sources.

    python evrard_checks.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys
from pathlib import Path

# jax
import jax

jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# The collapse setup is shared with the figure scripts of the paper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "forward" / "self_gravity"))

# astronomix constants
from astronomix.option_classes.simulation_config import (
    ATHENA_SECOND_ORDER,
    POSITIVITY_NONE,
)

# astronomix containers
from astronomix import PositivityConfig

# astronomix functions
from astronomix import (
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# figure helpers
import _collapse as collapse

END_TIME = 1.2
NUM_SNAPSHOTS = 25


def run(
    num_cells,
    time_centred_energy_source,
    floors=True,
    fixed_num_timesteps=None,
    initial_energy=collapse.COLD_INITIAL_ENERGY,
    minimum_density=collapse.MINIMUM_DENSITY,
    minimum_pressure=collapse.MINIMUM_PRESSURE,
):
    """
    Run the collapse with the finite-volume scheme.

    Args:
        num_cells: Cells per dimension.
        time_centred_energy_source: Whether to use the time-centred energy
            source of Mullen, Hanawa & Gammie (2021).
        floors: Whether AthenaPK's floors are on.
        fixed_num_timesteps: The number of equal steps, or ``None`` for the
            CFL step.
        initial_energy: The initial thermal energy per unit mass.
        minimum_density: The density floor.
        minimum_pressure: The pressure floor.

    Returns:
        ``(snapshots, relative total-energy error per snapshot)``.
    """
    scheme = collapse.CollapseScheme(
        self_gravity_version=ATHENA_SECOND_ORDER,
        time_centred_energy_source=time_centred_energy_source,
    )
    config = collapse.collapse_config(
        num_cells,
        scheme,
        num_snapshots=NUM_SNAPSHOTS,
        fixed_num_timesteps=fixed_num_timesteps,
    )
    if not floors:
        config = config._replace(positivity_config=PositivityConfig(per_step_mode=POSITIVITY_NONE))
    params = collapse.collapse_params(
        END_TIME,
        collapse.FINITE_VOLUME_COURANT_NUMBER,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    initial_state = collapse.collapse_initial_state(
        config,
        params,
        initial_energy,
        helper_data,
        registered_variables,
    )
    config = finalize_config(config, initial_state.shape)
    snapshots = jax.block_until_ready(time_integration(initial_state, config, params, registered_variables))
    total_energy = np.asarray(snapshots.total_energy)
    return snapshots, np.abs(total_energy - total_energy[0]) / abs(total_energy[0])


def main():
    """Run the three checks and print the results."""
    snapshots, energy_error = run(128, time_centred_energy_source=False, floors=False)
    finite = np.isfinite(energy_error)
    print(
        f"128^3 without floors: last finite snapshot at t = {np.asarray(snapshots.time_points)[finite][-1]:.2f}",
        flush=True,
    )

    for minimum_density, minimum_pressure in ((1e-5, 3e-6), (1e-6, 1e-8)):
        for time_centred in (False, True):
            snapshots, energy_error = run(
                128,
                time_centred,
                minimum_density=minimum_density,
                minimum_pressure=minimum_pressure,
            )
            print(
                f"128^3 floors ({minimum_density:.0e}, {minimum_pressure:.0e}) time-centred {time_centred}: "
                f"{int(snapshots.num_iterations)} steps, max |dE|/|E0| = {energy_error.max():.3e}",
                flush=True,
            )

    for initial_energy, label in ((collapse.COLD_INITIAL_ENERGY, "cold"), (collapse.MILD_INITIAL_ENERGY, "mild")):
        for time_centred in (False, True):
            for num_steps in (250, 500, 1000, 2000, 4000):
                _, energy_error = run(
                    32,
                    time_centred,
                    fixed_num_timesteps=num_steps,
                    initial_energy=initial_energy,
                )
                print(
                    f"{label} 32^3 time-centred {time_centred} dt = {END_TIME / num_steps:.2e}: "
                    f"final |dE|/|E0| = {energy_error[-1]:.3e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
