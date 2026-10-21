#!/bin/bash
#SBATCH --job-name=mhd_jet
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Run a script with one process per GPU (srun, --gpu-bind=none so that every
# task sees its node's GPUs and picks its own by SLURM_LOCALID), by default
# the MHD jet figure generator; its arguments are this script's arguments:
#
#     export ASTRONOMIX_REPO=$PWD
#     sbatch pytests/runners/mhd_jet.sh --res 1024 --rerun
#     sbatch --export=ALL,SCRIPT=pytests/runners/mhd_jet_decomposition_check.py \
#         pytests/runners/mhd_jet.sh --res 128 --t-end 0.5 --out eight.npz

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

SCRIPT="${SCRIPT:-examples/scripts/forward/mhd/mhd_jet.py}"

echo "=== nodes $SLURM_JOB_NODELIST, $SLURM_NTASKS tasks, script $SCRIPT $* ==="
srun --gpu-bind=none python "$SCRIPT" "$@"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
