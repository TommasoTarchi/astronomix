#!/usr/bin/env bash
# Two follow-ups to the mechanism study, one job per run on the A100 queue.
#
#  1. WENO-Z dissipation test (data/dissipation_wenoz/): the astronomix
#     dissipation-budget runs repeated with WENO-Z weights instead of Jiang-Shu,
#     everything else identical (smooth forcing, seed 42), so the change in
#     nu_eff / eta_eff is attributable to the weights alone.
#         python make_mechanism_table.py --data data/dissipation \
#                data/dissipation_mech data/dissipation_wenoz --summary
#
#  2. Matched-forcing ensemble (data/ensemble_matched/): astronomix (WENO5 and
#     WENO-Z) driven with AthenaPK's own 30-mode forcing statistics
#     (--forcing athenapk), and AthenaPK PLM, each with four forcing
#     realisations, in the kinematic-eigenmode setup of data/reynolds/
#     (zero-net-flux seed, beta = 1e12) so the growth rates can be averaged over
#     realisations. The run lengths follow data/reynolds/: 35 crossing times for
#     astronomix, 60 for PLM, which grows slower.
#
#   bash run_wenoz_and_ensemble.sh            # both tracks
#   bash run_wenoz_and_ensemble.sh wenoz      # only track 1
#   bash run_wenoz_and_ensemble.sh ensemble   # only track 2
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HERE="$REPO/examples/scripts/forward/mhd/turbulence"
RUN="$REPO/examples/gallery/supernova_showcase/run.sh"
PYTHON=/export/home/lstorcks/.local/share/mamba/envs/astx/bin/python
TRACK="${1:-all}"

if [[ "$TRACK" == "all" || "$TRACK" == "wenoz" ]]; then
    for N in 64 128 256; do
        pq sub -t a100 -n 1 --name "astx_wenoz_diss_n$N" -- bash -c \
            "cd $REPO && $RUN $HERE/dynamo_convergence.py --n $N --weno-z --transfer \
             --seed-field sin --beta 1e6 --tcross 40 --nsnap 81 \
             --tag wenoz_diss_N$N --outdir $HERE/data/dissipation_wenoz"
    done
fi

if [[ "$TRACK" == "all" || "$TRACK" == "ensemble" ]]; then
    OUT="$HERE/data/ensemble_matched"
    for N in 64 128; do
        for SEED in 42 43 44 45; do
            pq sub -t a100 -n 1 --name "astx_ens_js_s${SEED}_n$N" -- bash -c \
                "cd $REPO && $RUN $HERE/dynamo_convergence.py --n $N --forcing athenapk \
                 --seed-field sin --beta 1e12 --tcross 35 --nsnap 101 --seed $SEED \
                 --tag apkforce_seed${SEED}_N$N --outdir $OUT"
            pq sub -t a100 -n 1 --name "astx_ens_z_s${SEED}_n$N" -- bash -c \
                "cd $REPO && $RUN $HERE/dynamo_convergence.py --n $N --forcing athenapk --weno-z \
                 --seed-field sin --beta 1e12 --tcross 35 --nsnap 101 --seed $SEED \
                 --tag wenoz_apkforce_seed${SEED}_N$N --outdir $OUT"
        done
        for RSEED in 20190729 20190730 20190731 20190732; do
            pq sub -t a100 -n 1 --name "apk_ens_plm_s${RSEED}_n$N" -- bash -c \
                "$PYTHON $HERE/athenapk_turb.py --n $N --scheme plm \
                 --seed-field sin --beta 1e12 --tcross 60 --nsnap 61 --rseed $RSEED \
                 --tag plm_seed${RSEED}_N$N --outdir $OUT"
        done
    done
fi

pq stat
