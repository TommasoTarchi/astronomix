#!/usr/bin/env bash
# Submit the astronomix FD-versus-FV dynamo ladder on JUPITER, one GH200 per run.
#
# Both of astronomix's MHD schemes on the same driven subsonic dynamo box
# (zero-net-flux seed at beta = 1e6, 40 crossing times, 81 spectral snapshots
# carrying the transfer spectra, double precision, Pallas backend):
#
#   fd   WENO-Z + constrained transport, SSP-RK4, C_cfl = 1.5, isothermal
#   vl2  VL2 + PLM + HLLD + GLM, C_cfl = 0.3, gamma = 1.0001
#   pp   the FD scheme with the positivity-preserving WENO, C_cfl = 0.75
#        (an extra row: what the positivity mechanism costs in Rm and Pm)
#
#   bash run_dynamo_ladder.sh                  # fd and vl2 at 64, 128, 256; pp at 64, 128
#   bash run_dynamo_ladder.sh fd 128 256       # one scheme, the resolutions given
#   bash run_dynamo_ladder.sh controls         # the 64^3 controls, one job
#
# The controls (data/fd_fv/controls/) each change one thing against the 64^3
# production runs: single precision (both schemes), the FD scheme at the
# positivity-preserving run's C_cfl = 0.75 without the limiter (separates the
# limiter from the smaller step), and two more forcing realisations per scheme
# (the run-to-run scatter of Gamma and Pm).
#
# The reduced runs land in data/fd_fv/ (a few hundred kB each); the job logs in
# /e/project1/astronomix/storcks1/runs/. When the queue has drained:
#
#   python make_mechanism_table.py --data data/fd_fv --summary
#   python make_mechanism_table.py --data data/fd_fv --audit
#   python make_fd_fv_reynolds_figure.py --data data/fd_fv
#
# The AthenaPK legs of the earlier study (athenapk_turb.py) ran on another
# cluster's A100 build and are not part of this ladder.
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HERE_RELATIVE=examples/scripts/forward/mhd/turbulence
OUTDIR="$REPO/$HERE_RELATIVE/data/fd_fv"
SETUP_ANY_PRECISION="--seed-field sin --beta 1e6 --tcross 40 --nsnap 81 --transfer"
SETUP="$SETUP_ANY_PRECISION --x64"

SCHEME="${1:-all}"
shift || true
if [[ $# -gt 0 ]]; then
    RESOLUTIONS=("$@")
else
    RESOLUTIONS=(64 128 256)
fi

# Walltime per (scheme, N): about three times the GH200 timing probes (FD in
# x64: 1 s, 10 s and 160 s per crossing time at 64^3, 128^3 and 256^3; VL2 about
# a third of that; the positivity-preserving FD at half the step, twice FD).
walltime() {
    case "$1_$2" in
        *_64) echo "01:00:00" ;;
        *_128) echo "02:00:00" ;;
        vl2_256) echo "04:00:00" ;;
        fd_256) echo "06:00:00" ;;
        *) echo "12:00:00" ;;
    esac
}

submit() {
    local scheme=$1 n=$2 flags
    case "$scheme" in
        fd) flags="--scheme fd" ;;
        vl2) flags="--scheme vl2" ;;
        pp) flags="--scheme fd --pp" ;;
        *) echo "unknown scheme $scheme (fd, vl2 or pp)" >&2; exit 1 ;;
    esac
    # shellcheck disable=SC2086  # the flag strings are meant to split
    ASTRONOMIX_REPO="$REPO" sbatch --job-name="fdfv_${scheme}_n${n}" \
        --time="$(walltime "$scheme" "$n")" \
        --export=ALL,RUN_DIR="$HERE_RELATIVE" \
        "$REPO/pytests/runners/gpu1.sh" \
        python dynamo_convergence.py $flags --n "$n" $SETUP --outdir "$OUTDIR"
}

submit_controls() {
    local out="$OUTDIR/controls" run="python dynamo_convergence.py --n 64"
    local commands=(
        "$run --scheme fd $SETUP_ANY_PRECISION --tag wenoz_x32_N64 --outdir $out"
        "$run --scheme vl2 $SETUP_ANY_PRECISION --tag vl2_x32_N64 --outdir $out"
        "$run --scheme fd --cfl 0.75 $SETUP --tag wenoz_cfl075_N64 --outdir $out"
    )
    for seed in 43 44; do
        commands+=("$run --scheme fd --seed $seed $SETUP --tag wenoz_seed${seed}_N64 --outdir $out")
        commands+=("$run --scheme vl2 --seed $seed $SETUP --tag vl2_seed${seed}_N64 --outdir $out")
    done
    local joined
    joined=$(printf "%s; " "${commands[@]}")
    ASTRONOMIX_REPO="$REPO" sbatch --job-name="fdfv_controls_n64" --time=01:00:00 \
        --export=ALL,RUN_DIR="$HERE_RELATIVE" \
        "$REPO/pytests/runners/gpu1.sh" bash -c "${joined%; }"
}

if [[ "$SCHEME" == "controls" ]]; then
    submit_controls
elif [[ "$SCHEME" == "all" ]]; then
    for N in 64 128 256; do
        submit fd "$N"
        submit vl2 "$N"
    done
    for N in 64 128; do
        submit pp "$N"
    done
else
    for N in "${RESOLUTIONS[@]}"; do
        submit "$SCHEME" "$N"
    done
fi

squeue --me --format="%i %j %T %M %l" | grep fdfv_ || true
