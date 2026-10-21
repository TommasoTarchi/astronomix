#!/bin/bash
#SBATCH --job-name=turb_slices_series
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Sweep every checkpoint of a turbulence_dist.py run and extract mid-plane
# magnetic-energy slices for animation (one small NPZ per step).  2 nodes
# suffice for 2048^3: the sharded restore is the expensive part (~1 min per
# step), the gathered slices are tiny.  Already-extracted steps are skipped,
# so resubmitting after a walltime hit just continues.

source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh

N="${N:?set N}"
CKPT="${CKPT:?set CKPT}"
OUTDIR="${OUTDIR:?set OUTDIR (per-step npz directory)}"
EVERY="${EVERY:-1}"

srun --gpu-bind=none python examples/scripts/forward/mhd/turbulence/turb_slices_series.py \
    --ckpt "$CKPT" --N "$N" --every "$EVERY" --outdir "$OUTDIR"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
