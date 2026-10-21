#!/bin/bash
#SBATCH --job-name=gpu1
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Run one command on one GH200 of a booster node, from the repository root.
# The command and its arguments are the script's own arguments::
#
#     sbatch --job-name=sound_wave --time=00:30:00 pytests/runners/gpu1.sh \
#         python examples/scripts/forward/hydro/sound_wave_convergence.py
#
# A script that should run in its own directory (relative data/ and figures/
# paths) is given with RUN_DIR::
#
#     sbatch --export=ALL,RUN_DIR=examples/scripts/forward/hydro \
#         pytests/runners/gpu1.sh python sound_wave_convergence.py
#
# _env.sh puts the repository root (ASTRONOMIX_REPO) on PYTHONPATH, so the
# scripts import that checkout's astronomix. autocvd picks the allocated GPU.

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

# The weak-scaling environment preallocates 92 % of the GPU; single-GPU
# figure generators allocate on demand instead, so the memory they report is
# their own.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION

eval "$(autocvd -e -q -l)"

if [ -n "${RUN_DIR:-}" ]; then
    cd "$RUN_DIR"
fi

echo "=== host $(hostname), GPU $CUDA_VISIBLE_DEVICES, cwd $(pwd) ==="
echo "=== command: $* ==="
"$@"
rc=$?
echo "=== exit code: $rc ==="
exit $rc
