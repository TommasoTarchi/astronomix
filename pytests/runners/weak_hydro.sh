#!/bin/bash
#SBATCH --job-name=weak_hydro
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Weak-scaling rung for the Pallas hydro benchmark on JUPITER, finite
# difference (SCHEME=fd: WENO-Z, LSRK4) or finite volume (SCHEME=fv: VL2,
# HLLC).  The geometry is controlled by environment variables so the same
# runner serves every rung of the ladder (override the Slurm shape on the
# command line)::
#
#     sbatch --nodes=4 --ntasks-per-node=4 --exclusive \
#         --export=ALL,SCHEME=fd,BX=128,BY=2048,BZ=2048,STEPS=10,TAG=gh200z \
#         pytests/runners/weak_hydro.sh
#
# BLOCK (default: the scheme's tuned block) uses 'x' separators because
# commas inside --export values are eaten by Slurm's own list parsing; the
# driver accepts either.
#
# G = nodes * ntasks-per-node GPUs (one process per node by default, see
# sharded_layout in _env.sh); global grid (BX * G, BY, BZ).

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

SCHEME="${SCHEME:-fd}"
BX="${BX:-128}"
BY="${BY:-2048}"
BZ="${BZ:-2048}"
STEPS="${STEPS:-10}"
BLOCK="${BLOCK:-}"
TAG="${TAG:-gh200}"

# WENO_JS=1 selects the Jiang-Shu weights (the original FD configuration).
EXTRA_ARGS=""
if [ "${WENO_JS:-0}" = "1" ]; then
    EXTRA_ARGS="--weno-js"
fi
# One process per node by default (see sharded_layout in _env.sh).
sharded_layout
DRIVER_GPU_ARGS="$DRIVER_GPU_ARGS --halo-exchange ${HALO_EXCHANGE:-pair_all_to_all}"

srun $SRUN_LAYOUT python examples/scripts/scaling/weak_scaling_hydro.py \
    --scheme "$SCHEME" --bx "$BX" --by "$BY" --bz "$BZ" \
    --steps "$STEPS" --block-shape "$BLOCK" --tag "$TAG" $EXTRA_ARGS $DRIVER_GPU_ARGS
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
