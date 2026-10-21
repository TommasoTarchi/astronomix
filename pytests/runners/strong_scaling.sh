#!/bin/bash
#SBATCH --job-name=strong_scaling
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Strong-scaling points of one (scheme, physics) on G = nodes *
# ntasks-per-node GPUs, one process per GPU, for a list of resolutions N
# (global grid 2N x N x N)::
#
#     sbatch --nodes=2 --ntasks-per-node=4 --exclusive \
#         --export=ALL,SCHEME=fv,PHYSICS=mhd,RESOLUTIONS=128x192x256x384x512 \
#         pytests/runners/strong_scaling.sh
#
# RESOLUTIONS uses 'x' separators (commas inside --export values are eaten by
# Slurm); the 1-GPU baseline job lists only the resolutions that fit one GPU.

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

SCHEME="${SCHEME:-fd}"
PHYSICS="${PHYSICS:-hydro}"
RESOLUTIONS="${RESOLUTIONS:-128x192x256x384x512}"
MIN_STEPS="${MIN_STEPS:-20}"
TAG="${TAG:-}"

# One process per node by default (see sharded_layout in _env.sh).
sharded_layout
DRIVER_GPU_ARGS="$DRIVER_GPU_ARGS --halo-exchange ${HALO_EXCHANGE:-pair_all_to_all}"
DRIVER_ARGS="$DRIVER_GPU_ARGS"
# DEFAULT_CT=1: FD MHD with the default (roll-based) constrained transport.
if [ "${DEFAULT_CT:-0}" = "1" ]; then
    DRIVER_ARGS="$DRIVER_ARGS --default-ct"
fi

srun $SRUN_LAYOUT python examples/scripts/scaling/strong_scaling_speedup.py \
    --scheme "$SCHEME" --physics "$PHYSICS" --resolutions ${RESOLUTIONS//x/ } \
    --min-steps "$MIN_STEPS" --tag "$TAG" $DRIVER_ARGS
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
