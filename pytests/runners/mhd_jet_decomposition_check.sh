#!/bin/bash
#SBATCH --job-name=jet_check
#SBATCH --account=astronomix
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/astronomix/storcks1/runs/%x_%j.out
#SBATCH --error=/e/project1/astronomix/storcks1/runs/%x_%j.err

# Validate the domain-decomposed positivity-preserving MHD jet: run it on one
# and on four GPUs of one node (one process each time) and compare the final
# states, early (t = 0.5) and at the end (t = 5); compare periodic with open
# boundaries on one GPU. Optionally probe the memory and speed of a
# production size on four GPUs (PROBE_RES, PROBE_FLAGS, e.g. "--lsrk4").
#
#     export ASTRONOMIX_REPO=$PWD
#     sbatch --export=ALL,CHECK_RES=128 pytests/runners/mhd_jet_decomposition_check.sh
#     sbatch --export=ALL,CHECK_RES=0,PROBE_RES=1024 pytests/runners/mhd_jet_decomposition_check.sh

source "${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}/pytests/runners/_env.sh"

CHECK_RES="${CHECK_RES:-128}"
PROBE_RES="${PROBE_RES:-0}"
PROBE_FLAGS="${PROBE_FLAGS:-}"
PROBE_T_END="${PROBE_T_END:-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-/e/scratch/astronomix/revision/forward_tests/jet_check}"
mkdir -p "$OUTPUT_DIR"

CHECK=pytests/runners/mhd_jet_decomposition_check.py
echo "=== host $(hostname), repo $ASTRONOMIX_REPO, output $OUTPUT_DIR ==="

if [ "$CHECK_RES" -gt 0 ]; then
    for t_end in 0.5 5.0; do
        python $CHECK --res "$CHECK_RES" --gpus 1 --t-end "$t_end" \
            --out "$OUTPUT_DIR/one_gpu_${CHECK_RES}_t${t_end}.npz"
        python $CHECK --res "$CHECK_RES" --gpus 4 --t-end "$t_end" \
            --out "$OUTPUT_DIR/four_gpus_${CHECK_RES}_t${t_end}.npz"
        echo "=== 1 vs 4 GPUs, ${CHECK_RES}^3, t = $t_end ==="
        python $CHECK --compare "$OUTPUT_DIR/one_gpu_${CHECK_RES}_t${t_end}.npz" \
            "$OUTPUT_DIR/four_gpus_${CHECK_RES}_t${t_end}.npz"
    done
    python $CHECK --res "$CHECK_RES" --gpus 1 --t-end 5.0 --boundary open \
        --out "$OUTPUT_DIR/one_gpu_open_${CHECK_RES}_t5.0.npz"
    echo "=== periodic vs open boundaries, ${CHECK_RES}^3, t = 5 ==="
    python $CHECK --compare "$OUTPUT_DIR/one_gpu_${CHECK_RES}_t5.0.npz" \
        "$OUTPUT_DIR/one_gpu_open_${CHECK_RES}_t5.0.npz"
fi

if [ "$PROBE_RES" -gt 0 ]; then
    echo "=== probe: ${PROBE_RES}^3 on 4 GPUs to t = $PROBE_T_END, flags: $PROBE_FLAGS ==="
    python $CHECK --res "$PROBE_RES" --gpus 4 --t-end "$PROBE_T_END" $PROBE_FLAGS
fi

rc=$?
echo "=== exit code: $rc ==="
exit $rc
