#!/bin/bash
# Submit the pre-registered single- vs multiple-shooting campaign of kh_recon.py
# to Slurm on JUPITER.
#
# Every (method, configuration, horizon) is one chain of dependent gpu1.sh jobs
# running the same `kh_recon.py run` command: finished cells are skipped and an
# interrupted batch resumes from its checkpoint, so each link continues where
# the previous one hit the 12 h limit. Arms:
#
#   tuning       truths 1 2 x inits 0 1 (one batch of 4), the search grid of
#                every Adam method (TUNING_GRID=small: 3 learning rates each;
#                full: 6 configurations each)
#   reporting    truths 3-6 x inits 0-3 (two batches of 8), the configuration
#                each method selected on the tuning runs (SINGLE_T20,
#                SINGLE_T60, COARSE_T20, ... hold its `run` options), plus
#                Gauss-Newton
#   replication  truth 0 x inits 0-15, the original campaign's settings
#
# Usage (from the repository root):
#
#   export ASTRONOMIX_REPO=$PWD
#   ARM=tuning bash examples/scripts/differentiability/kh_shooting_campaign.sh
#   ARM=reporting SINGLE_T60="--learning-rate 1e-3 --schedule cosine" ... bash ...
#
# HORIZONS (default "20 60") restricts the horizons; DRY_RUN=1 prints the
# sbatch commands instead of submitting them.

set -u

REPOSITORY=${ASTRONOMIX_REPO:?export ASTRONOMIX_REPO (the checkout to run)}
OUTPUT=${OUTPUT:-/e/project1/astronomix/storcks1/revision_data/kh_shooting}
ARM=${ARM:?set ARM to tuning, reporting or replication}
LINK_TIME=${LINK_TIME:-11:55:00}

HORIZONS=${HORIZONS:-20 60}

# Chain lengths (12 h links) per horizon: a batch of 8 single-shooting runs
# of 1980 steps takes about 20 h at T = 20 and 63 h at T = 60 on one GH200
# (multiple shooting about two thirds of that); surplus links find their
# cells done and exit at once.
LINKS_T20=${LINKS_T20:-2}
LINKS_T60=${LINKS_T60:-6}

# Submit one chain of NUM_LINKS dependent jobs running `kh_recon.py run ARGS`.
submit_chain() {
    local name=$1
    local num_links=$2
    shift 2
    local previous=""
    for link in $(seq 1 "$num_links"); do
        local dependency=""
        if [ -n "$previous" ]; then
            dependency="--dependency=afterany:$previous"
        fi
        local command=(
            sbatch --parsable --job-name="$name" --time="$LINK_TIME" $dependency
            --export=ALL,RUN_DIR=examples/scripts/differentiability
            "$REPOSITORY/pytests/runners/gpu1.sh"
            python kh_recon.py run "$@"
        )
        if [ "${DRY_RUN:-0}" = 1 ]; then
            echo "${command[*]}"
            previous="DRY$link"
        else
            previous=$("${command[@]}")
            echo "$name link $link: job $previous"
        fi
    done
}

links_for() {
    if [ "$1" = 20 ]; then echo "$LINKS_T20"; else echo "$LINKS_T60"; fi
}

case "$ARM" in
tuning)
    # The small grid gives every Adam method the same three learning rates
    # (its other settings as in the original campaign); the full grid adds a
    # second value of each method's own knob (schedule, continuation stages,
    # initial penalty): six configurations per method.
    seeds=(--truths 1 2 --inits 0 1 --batch-size 4 --output-directory "$OUTPUT/tuning")
    if [ "${TUNING_GRID:-small}" = full ]; then
        schedules=(cosine constant)
        stage_sets=("2 4 8 16" "4 16")
        penalties=(1 10)
    else
        schedules=(cosine)
        stage_sets=("2 4 8 16")
        penalties=(1)
    fi
    for horizon in $HORIZONS; do
        links=$(links_for $horizon)
        for learning_rate in 3e-4 1e-3 3e-3; do
            for schedule in "${schedules[@]}"; do
                submit_chain "khs_tune_s_T${horizon}" "$links" --method single --horizon $horizon \
                    --learning-rate $learning_rate --schedule $schedule "${seeds[@]}"
            done
            for stages in "${stage_sets[@]}"; do
                submit_chain "khs_tune_c_T${horizon}" "$links" --method coarse_to_fine --horizon $horizon \
                    --learning-rate $learning_rate --cutoff-stages $stages "${seeds[@]}"
            done
            for penalty in "${penalties[@]}"; do
                submit_chain "khs_tune_m_T${horizon}" "$links" --method multiple --horizon $horizon \
                    --learning-rate $learning_rate --penalty-initial $penalty "${seeds[@]}"
            done
        done
    done
    ;;
reporting)
    # Two batches of 8 runs, each its own chain.
    for truths in "3 4" "5 6"; do
        seeds=(--truths $truths --inits 0 1 2 3 --batch-size 8 --output-directory "$OUTPUT/reporting")
        for horizon in $HORIZONS; do
            links=$(links_for $horizon)
            single_options_name="SINGLE_T$horizon"
            coarse_options_name="COARSE_T$horizon"
            multiple_options_name="MULTIPLE_T$horizon"
            # The selected options are split into words on purpose.
            submit_chain "khs_rep_s_T$horizon" "$links" --method single --horizon $horizon \
                ${!single_options_name:?} "${seeds[@]}"
            submit_chain "khs_rep_c_T$horizon" "$links" --method coarse_to_fine --horizon $horizon \
                ${!coarse_options_name:?} "${seeds[@]}"
            submit_chain "khs_rep_m_T$horizon" "$links" --method multiple --horizon $horizon \
                ${!multiple_options_name:?} "${seeds[@]}"
            submit_chain "khs_rep_g_T$horizon" 1 --method gauss_newton --horizon $horizon \
                --num-iterations 60 "${seeds[@]}"
        done
    done
    ;;
replication)
    for inits in "0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15"; do
        seeds=(--truths 0 --inits $inits --batch-size 8 --output-directory "$OUTPUT/replication")
        for horizon in $HORIZONS; do
            links=$(links_for $horizon)
            submit_chain "khs_repl_s_T$horizon" "$links" --method single --horizon $horizon "${seeds[@]}"
            submit_chain "khs_repl_m_T$horizon" "$links" --method multiple --horizon $horizon "${seeds[@]}"
        done
    done
    ;;
*)
    echo "unknown ARM $ARM" >&2
    exit 1
    ;;
esac
