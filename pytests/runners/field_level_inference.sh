#!/bin/bash
#SBATCH --job-name=fli_adam128
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# One segment of the paper's field-level inference (plain Adam on the 3 x 128^3
# initial velocity) on one GH200. optimize_velocity.py resumes from its run
# directory and stops cleanly before the wall budget (and once the target loss
# reduction or the step count is reached), so a run longer than the 12 h limit
# is a chain of these jobs:
#
#     export ASTRONOMIX_REPO=$PWD
#     first=$(sbatch --parsable pytests/runners/field_level_inference.sh \
#         --resolution 128 --num-steps 300 --target-reduction 600 --wall-budget 42000)
#     sbatch --dependency=afterok:$first pytests/runners/field_level_inference.sh \
#         --resolution 128 --num-steps 300 --target-reduction 600 --wall-budget 42000
#
# The arguments are optimize_velocity.py's. With the Pallas adjoint of the
# WENO flux (the default) the value-and-grad of the 128^3 problem plans ~9 GiB,
# with the native tangent (--flux-adjoint native) ~41 GiB of one contiguous
# pool; the pool is preallocated (as in _env.sh) because on-demand allocation
# fragments it.

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

eval "$(autocvd -e -q -l)"

cd "$ASTRONOMIX_REPO/examples/scripts/differentiability/field_level_inference"

echo "=== host $(hostname), GPU $CUDA_VISIBLE_DEVICES, cwd $(pwd) ==="
echo "=== command: python optimize_velocity.py $* ==="
python optimize_velocity.py "$@"
rc=$?
echo "=== exit code: $rc ==="
exit $rc
