#!/bin/bash
#SBATCH --job-name=turb_paper
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# One case of the paper's driven isothermal-MHD turbulence
# (examples/scripts/forward/mhd/turbulence/paper_turbulence.py) on one GH200.
# The arguments are paper_turbulence.py's own::
#
#     cd <repository> && export ASTRONOMIX_REPO=$PWD
#     sbatch --job-name=turb_ISM512 pytests/runners/turb_paper.sh \
#         --mturb 10 --beta 0.1 --tcross 5 --N 512 --tag ISM_N512
#
# The driver checkpoints after every snapshot interval (to
# /e/scratch/astronomix/revision/turbulence/<tag>) and resumes from there, so
# a run longer than one walltime is a chain of identical submissions::
#
#     first=$(sbatch --parsable --job-name=turb_ICM512 pytests/runners/turb_paper.sh <args>)
#     sbatch --dependency=afterany:$first --job-name=turb_ICM512 pytests/runners/turb_paper.sh <args>
#
# Each job stops cleanly before its walltime (it does not start an interval
# it could not finish); a job that finds the run complete only rewrites the
# reduced npz in the script's data/ directory.

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

# _env.sh preallocates 92 % of the GPU as one pool. That is kept here: a run
# of many intervals allocates and frees state-sized buffers (integration,
# checkpoint read, reduction) over and over, and the on-demand pool fragments
# under that pattern.

eval "$(autocvd -e -q -l)"

# The wall-time budget handed to the driver: the job's remaining time at
# start-up minus 15 minutes for the last interval's checkpoint and the exit.
time_left="$(squeue -h -j "${SLURM_JOB_ID}" -o %L)"
stop_after_hours="$(python - "$time_left" <<'PY'
import sys
days, _, clock = sys.argv[1].rpartition("-")
fields = [int(part) for part in clock.split(":")]
hours, minutes, seconds = [0] * (3 - len(fields)) + fields
total_hours = 24 * int(days or 0) + hours + minutes / 60 + seconds / 3600
print(f"{max(total_hours - 0.25, 0.1):.3f}")
PY
)"

cd examples/scripts/forward/mhd/turbulence
echo "=== host $(hostname), GPU $CUDA_VISIBLE_DEVICES, time left $time_left, budget ${stop_after_hours} h ==="
echo "=== command: python paper_turbulence.py $* ==="
python paper_turbulence.py "$@" --stop-after-hours "$stop_after_hours"
rc=$?
echo "=== exit code: $rc ==="
exit $rc
