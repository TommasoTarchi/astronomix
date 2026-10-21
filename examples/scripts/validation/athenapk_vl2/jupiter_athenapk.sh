#!/bin/bash
# Run a command with the JUPITER GH200 AthenaPK builds available to the scripts
# of this directory, e.g. inside a single-GPU job:
#
#     sbatch --exclusive ... pytests/runners/gpu1.sh \
#         bash $PWD/examples/scripts/validation/athenapk_vl2/jupiter_athenapk.sh \
#         python $PWD/examples/scripts/validation/athenapk_vl2/benchmark.py ...
#
# Sets ATHENAPK_GPU_BIN (double precision), ATHENAPK_GPU_SINGLE_BIN
# (PARTHENON_SINGLE_PRECISION=ON) and ATHENAPK_LIBRARY_PATH. The builds (Kokkos
# CUDA for Hopper, no MPI, serial HDF5) link CUDA 13 and HDF5 from the module
# system; their library path is handed over in ATHENAPK_LIBRARY_PATH and only the
# athenaPK processes run with it (athenapk_runner.athenapk_environment), because
# the Python process keeps the CUDA libraries its JAX installation brings along.

# no `set -e`: module shell functions misbehave under it
set -u

athenapk_build_directory="${ATHENAPK_BUILD_DIRECTORY:-/e/project1/astronomix/storcks1/athenapk_build}"
export ATHENAPK_GPU_BIN="$athenapk_build_directory/build-gh200-double/bin/athenaPK"
export ATHENAPK_GPU_SINGLE_BIN="$athenapk_build_directory/build-gh200-single/bin/athenaPK"

if ! type module >/dev/null 2>&1; then
    source "${MODULESHOME:-/e/software/default/lmod/8.7.64}/init/bash"
fi
# The modules are loaded in a subshell only, to read off their library path.
ATHENAPK_LIBRARY_PATH="$(module load CUDA/13 HDF5/1.14.6-serial >/dev/null 2>&1; echo "${LD_LIBRARY_PATH:-}")"
export ATHENAPK_LIBRARY_PATH

echo "=== AthenaPK builds: $ATHENAPK_GPU_BIN, $ATHENAPK_GPU_SINGLE_BIN ==="
exec "$@"
