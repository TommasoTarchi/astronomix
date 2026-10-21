#!/bin/bash
#SBATCH --job-name=validate_sharded
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Sharded-versus-single-GPU correctness check of the four scaling
# configurations (FD WENO-Z and FV VL2, hydro and MHD); one process per GPU,
# G = nodes * ntasks-per-node::
#
#     sbatch --nodes=2 --ntasks-per-node=4 pytests/runners/validate_sharded.sh
#
# RESOLUTION sets N of the (2N, N, N) global grid.

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

RESOLUTION="${RESOLUTION:-32}"
STEPS="${STEPS:-20}"

# One process per node by default (see sharded_layout in _env.sh).
sharded_layout
DRIVER_GPU_ARGS="$DRIVER_GPU_ARGS --halo-exchange ${HALO_EXCHANGE:-pair_all_to_all}"
srun $SRUN_LAYOUT python examples/scripts/scaling/validate_sharded_schemes.py \
    --resolution "$RESOLUTION" --steps "$STEPS" $DRIVER_GPU_ARGS
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
