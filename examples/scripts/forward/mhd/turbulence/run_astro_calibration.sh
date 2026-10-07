#!/usr/bin/env bash
# The calibration ladder (explicit Laplacian viscosity / resistivity on top of
# the scheme's numerical one) for astronomix, mirroring the AthenaPK ladder in
# data/calibration/: same box, beta = 100, zero-net-flux seed, transfer spectra,
# 60 dumps over 40 crossing times. The imposed values are scaled to each
# scheme's numerical coefficients at each resolution (roughly 0.5x, 1x, 2x
# numerical for eta and 0.7x, 1.4x for nu), from the mechanism table.
#
#   bash run_astro_calibration.sh 64          # WENO5 (JS) and WENO-Z at 64^3
#   bash run_astro_calibration.sh 256         # the same at 256^3 (12 x ~3 h)
#   bash run_astro_calibration.sh 256 js      # only the Jiang-Shu ladder
#
# Analyse with make_calibration_figure.py / make_calibration_model.py
# --data data/calibration_astro (see DYNAMO_MECHANISM.md).
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HERE="$REPO/examples/scripts/forward/mhd/turbulence"
RUN="$REPO/examples/gallery/supernova_showcase/run.sh"
N="${1:-64}"
WEIGHTS="${2:-both}"
OUT="$HERE/data/calibration_astro"
COMMON="--n $N --beta 1e2 --seed-field sin --transfer --nsnap 60 --tcross 40 --outdir $OUT"

# imposed coefficients per (weights, N): "eta values" / "nu values"
declare -A ETA NU
ETA[js,64]="5e-4 1e-3 2e-3";      NU[js,64]="8e-4 1.6e-3"
ETA[js,256]="1e-4 2e-4 4e-4";     NU[js,256]="1.6e-4 3.2e-4"
ETA[z,64]="4e-4 8e-4 1.6e-3";     NU[z,64]="6e-4 1.2e-3"
ETA[z,256]="8e-5 1.6e-4 3.2e-4";  NU[z,256]="1.2e-4 2.4e-4"

submit() {   # submit <weights> <flag> <tag-prefix>
    local W=$1 FLAG=$2 PFX=$3
    pq sub -t a100 -n 1 --name "astx_calib${N}_${PFX}_eta0_nu0" -- bash -c \
        "cd $REPO && $RUN $HERE/dynamo_convergence.py $COMMON $FLAG --tag calib_${PFX}_eta0_nu0_N$N"
    for E in ${ETA[$W,$N]}; do
        T=$(echo "$E" | sed 's/\.//; s/-//')
        pq sub -t a100 -n 1 --name "astx_calib${N}_${PFX}_eta${T}" -- bash -c \
            "cd $REPO && $RUN $HERE/dynamo_convergence.py $COMMON $FLAG --ohm-diff $E --tag calib_${PFX}_eta${T}_nu0_N$N"
    done
    for V in ${NU[$W,$N]}; do
        T=$(echo "$V" | sed 's/\.//; s/-//')
        pq sub -t a100 -n 1 --name "astx_calib${N}_${PFX}_nu${T}" -- bash -c \
            "cd $REPO && $RUN $HERE/dynamo_convergence.py $COMMON $FLAG --mom-diff $V --tag calib_${PFX}_eta0_nu${T}_N$N"
    done
}

[[ "$WEIGHTS" == "both" || "$WEIGHTS" == "js" ]] && submit js "" js
[[ "$WEIGHTS" == "both" || "$WEIGHTS" == "z"  ]] && submit z "--weno-z" wenoz
pq stat
