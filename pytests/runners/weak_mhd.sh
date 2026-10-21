#!/bin/bash
#SBATCH --job-name=weak_mhd
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Weak-scaling rung for the Pallas MHD benchmark on JUPITER, finite difference
# (SCHEME=fd: WENO-Z, LSRK4, constrained transport; PALLAS_CT=1 for the staged
# Pallas CT kernels) or finite volume (SCHEME=fv: VL2, HLLD + GLM).  See
# weak_hydro.sh for the launch pattern; BLOCK uses 'x' separators::
#
#     sbatch --nodes=4 --ntasks-per-node=4 --exclusive \
#         --export=ALL,SCHEME=fd,PALLAS_CT=1,BX=128,BY=2048,BZ=1024,TAG=gh200zct \
#         pytests/runners/weak_mhd.sh

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

SCHEME="${SCHEME:-fd}"
BX="${BX:-128}"
BY="${BY:-2048}"
BZ="${BZ:-1024}"
STEPS="${STEPS:-10}"
BLOCK="${BLOCK:-}"
TAG="${TAG:-gh200}"

EXTRA_ARGS=""
if [ "${PALLAS_CT:-0}" = "1" ]; then
    EXTRA_ARGS="--pallas-ct"
fi

# One process per node by default (see sharded_layout in _env.sh).
sharded_layout
DRIVER_GPU_ARGS="$DRIVER_GPU_ARGS --halo-exchange ${HALO_EXCHANGE:-pair_all_to_all}"
DRIVER_ARGS="$DRIVER_GPU_ARGS"

srun $SRUN_LAYOUT python examples/scripts/scaling/weak_scaling_mhd.py \
    --scheme "$SCHEME" --bx "$BX" --by "$BY" --bz "$BZ" \
    --steps "$STEPS" --block-shape "$BLOCK" --tag "$TAG" $EXTRA_ARGS $DRIVER_ARGS
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
