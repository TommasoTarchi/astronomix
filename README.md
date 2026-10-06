# astronomix - differentiable mhd for astrophysics in JAX

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/848159116.svg)](https://doi.org/10.5281/zenodo.15052815)

`astronomix` (formerly `jf1uids`) is a differentiable hydrodynamics and magnetohydrodynamics code
written in `JAX` with a focus on astrophysical applications. `astronomix` is easy to use, well-suited for
fast method development, scales to multiple GPUs and its differentiability 
opens the door for gradient-based inverse modeling and sampling as well 
as surrogate / solver-in-the-loop training.

## Features

- [x] 1D, 2D and 3D hydrodynamics and magnetohydrodynamics simulations scaling to multiple GPUs and nodes
- [x] a 5th order finite difference constrained transport WENO MHD scheme following [HOW-MHD by Seo & Ryu 2023](https://arxiv.org/abs/2304.04360) as well as the provably divergence free and provably positivity preserving
finite volume approach of [Pang and Wu (2024)](https://arxiv.org/abs/2410.05173) (the WENO scheme is also available standalone for hydrodynamics)
- [x] isothermal hydrodynamics and magnetohydrodynamics are also supported (currently only in the finite difference scheme)
- [x] for finite volume simulations the basic Lax-Friedrichs, HLL and HLLC Riemann solvers as well as the HLLC-LM ([Fleischmann et al., 2020](https://www.sciencedirect.com/science/article/pii/S0021999120305362)) and HYBRID-HLLC & AM-HLLC ([Hu et al., 2025](https://www.sciencedirect.com/science/article/pii/S1007570425005891)) (sequels to HLLC-LM) variants
- [x] novel semi-discretely energy conserving self-gravity scheme
- [x] spherically symmetric simulations such that mass and energy are conserved based on the scheme of [Crittenden and Balachandar (2018)](https://doi.org/10.1007/s00193-017-0784-y)
- [x] backwards and forwards differentiable with adaptive timestepping
- [x] turbulent driving, simple stellar wind, simple radiative cooling modules
- [x] easily extensible, all code is open source

## Contents

- [Features](#features)
- [Installation](#installation)
- [Hello World! Your first astronomix simulation](#hello-world-your-first-astronomix-simulation)
- [Examples for Getting Started](#examples-for-getting-started)
- [Showcase](#showcase)
- [Performance](#performance)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
  - [How to store intermediate information during the simulation?](#how-to-store-intermediate-information-during-the-simulation)
  - [How to fit larger simulations into GPU memory?](#how-to-fit-larger-simulations-into-gpu-memory)
  - [What if my simulation crashes?](#what-if-my-simulation-crashes)
- [Documentation](#documentation)
- [Citing astronomix](#citing-astronomix)

## Installation

`astronomix` can be installed via `pip`

```bash
pip install astronomix
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Hello World! Your first astronomix simulation

Below is a minimal example of a 1D hydrodynamics shock tube simulation using `astronomix`.

```python
import jax.numpy as jnp
from astronomix import (
    SimulationConfig, SimulationParams,
    get_helper_data, finalize_config,
    get_registered_variables, construct_primitive_state,
    time_integration
)

# the SimulationConfig holds static 
# configuration parameters
config = SimulationConfig(
    box_size = 1.0,
    num_cells = 101,
    progress_bar = True
)

# the SimulationParams can be changed
# without causing re-compilation
params = SimulationParams(
    t_end = 0.2,
)

# the variable registry allows for the principled
# access of simulation variables from the state array
registered_variables = get_registered_variables(config)

# next we set up the initial state using the helper data
helper_data = get_helper_data(config)
shock_pos = 0.5
r = helper_data.geometric_centers
rho = jnp.where(r < shock_pos, 1.0, 0.125)
u = jnp.zeros_like(r)
p = jnp.where(r < shock_pos, 1.0, 0.1)

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u,
    gas_pressure = p,
)

# finalize and check the config
config = finalize_config(config, initial_state.shape)

# now we run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# the final_state holds the final primitive state, the 
# variables can be accessed via the registered_variables
rho_final = final_state[registered_variables.density_index]
u_final = final_state[registered_variables.velocity_index]
p_final = final_state[registered_variables.pressure_index]
```

You've just run your first `astronomix` simulation! You can continue with
the notebooks below and we have also prepared a more advanced use-case
(stellar wind in driven MHD tubulence) which you can
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Pg98IPGnoejaGvzmNNZiwmf1JnXwYAJH?usp=sharing).

## Examples for Getting Started

Every example is available both as a runnable script (`examples/scripts/…`) and
as an equivalent notebook (`examples/notebooks/…`).

- forward simulations
  - [1D Sod shock tube](examples/notebooks/forward/hydro/shock_tube.ipynb)
  - [2D Kelvin-Helmholtz instability](examples/notebooks/forward/hydro/khi.ipynb)
  - [3D MHD jet](examples/notebooks/forward/mhd/jet.ipynb)
  - [3D driven MHD turbulence](examples/notebooks/forward/mhd/turbulence.ipynb)
  - [3D self-gravitating collapse](examples/notebooks/forward/self_gravity/collapse.ipynb)
- differentiability
  - [Field-level inference](examples/notebooks/differentiability/field_level_inference.ipynb)
  - [KHI eigenmode initialization](examples/notebooks/differentiability/eigen_initialization.ipynb)
  - [Solver-in-the-loop correction network](examples/notebooks/differentiability/solver_in_the_loop.ipynb)
- output options
  - [On-the-fly movie via callback](examples/notebooks/output_options/callback.ipynb)
  - [Orbax checkpointing and restart](examples/notebooks/output_options/orbax_checkpointing.ipynb)
  - [In-memory snapshot diagnostics](examples/notebooks/output_options/return_snapshots.ipynb)
- multi-GPU
  - [Sharded multi-GPU run](examples/notebooks/multi_gpu/multi_gpu.ipynb)
  - [Multi-node run](examples/notebooks/multi_gpu/multi_node.ipynb)

More involved, physics-module showcase scripts — turbulent radiative mixing
layer, stellar wind, cosmic rays, cooling and more — live in
[`examples/gallery/`](examples/gallery). The faithful methods-paper figure
generators live under [`examples/scripts/`](examples/scripts); regenerate every
paper figure with `pytest pytests/test_reproduce_paper.py --reproduce-paper`.
The fast correctness tests live under [`pytests/`](pytests).

## Showcase

| ![wind in driven turbulence](readme_resources/readme_figures/interm_driven_turb_wind4.png) |
|:---------------------------------------------------------------------------------:|
| Magnetohydrodynamics simulation with driven turbulence at a resolution of 512³ cells in a fifth order CT MHD scheme run on 4 H200 GPUs. |

| ![wind in driven turbulence](readme_resources/readme_figures/driven_turb_wind4.png) |
|:---------------------------------------------------------------------------------:|
| Magnetohydrodynamics simulation with driven turbulence and stellar wind at a resolution of 512³ cells in a fifth order CT MHD scheme run on 4 H200 GPUs. |

| ![3D MHD jet](readme_resources/readme_figures/mhd_jet.png) |
|:----------------------------------------------------------:|
| 3D MHD jet propagating into a magnetized medium. |

| ![Semi-Discretely Energy Conserving Self Gravity Scheme](readme_resources/readme_figures/collapse_energy_evolution_comparison.svg) |
|:-----------------------------------------------------------------------------------------------------------:|
| Novel semi-discretely energy conserving self-gravity scheme. |

## Performance

Methods paper incoming :)

## Frequently Asked Questions (FAQ)

### How to store intermediate information during the simulation?

There are two main options for storing intermediate states:

- activate `return_snapshots` in the `SimulationConfig`, set either `num_snapshots` in the `SimulationConfig` for equidistant snapshots or `snapshot_timepoints` in the `SimulationParams` for custom snapshot timepoints, then activate specific variables to be stored via `snapshot_settings` in the `SimulationConfig` (e.g. the full states, the total energy, ..., see [here](https://astronomix-mhd.web.app/source/astronomix.option_classes.simulation_config.html#astronomix.option_classes.simulation_config.SnapshotSettings)) - these snapshots are stored on the GPU directly, which avoids host-device transfer and is e.g. very useful for losses over intermediate states
- activate `snapshot_callable` in the `SimulationConfig` and provide a callable to the `time_integration` function which is called at the same timepoints as the snapshots in the first option, this function can then offload specific data (possibly after processing) to the host via `jax.debug.callback` and save it appropriately (e.g. directly make plots, save to disk in the preferred format, ...)

### How to fit larger simulations into GPU memory?

Make sure that you only have the initial state in GPU memory when you start the simulation,
e.g. have a function which constructs the initial state (otherwise e.g. the intermediate helper data
you used to construct the state might still be in GPU memory). To further save storage,
you can donate the initial state to the time integration (activate `donate_state` in the `SimulationConfig`),
which allows `JAX` to reuse the same memory for the state throughout the simulation (but you can also no
longer access the initial state after the simulation has started).

### What if my simulation crashes?

First of all check if the initial conditions are valid. For the finite
difference scheme, turn on the positivity-preserving WENO reconstruction
(`weno_positivity_preserving=True` in the `SimulationConfig`), which keeps
density and pressure positive for `C_cfl <= 0.75`. For the finite volume
scheme, use a positivity-preserving limiter such as `VAN_ALBADA_PP`; the
`PositivityConfig` in the `SimulationConfig` additionally offers a per-step
density / pressure floor.

## Documentation

See [here](https://astronomix-mhd.web.app/).

## Citing astronomix

If you use `astronomix` in your research, please cite via

```bibtex
@misc{storcks_astronomix_2025,
  doi = {10.5281/ZENODO.17782162},
  url = {https://zenodo.org/doi/10.5281/zenodo.17782162},
  author = {Storcks, Leonard},
  title = {astronomix - differentiable MHD in JAX},
  publisher = {Zenodo},
  year = {2025},
  copyright = {MIT License}
}
```

There is also a workshop paper on an earlier stage of the project:

[Storcks, L., & Buck, T. (2024). Differentiable Conservative Radially Symmetric Fluid Simulations and Stellar Winds--jf1uids. arXiv preprint arXiv:2410.23093.](https://arxiv.org/abs/2410.23093)