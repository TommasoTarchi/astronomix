# Shared environment for astronomix Slurm runners on JUPITER (JSC).
#
# Source this by ABSOLUTE path from every runner (under Slurm, $0 is the
# spool copy, so relative includes break)::
#
#     source /e/home/jusers/storcks1/jupiter/astronomix/pytests/runners/_env.sh
#
# Provides: micromamba env activation, repo cwd, the JAX flags the sharded
# code path needs, and $RUN_LOG_DIR for job output.

# no `set -e`: module/micromamba shell functions misbehave under it
set -u

# A worktree of the repository is selected by exporting ASTRONOMIX_REPO before
# sbatch (with --export=ALL); its checkout is then imported, not the editable
# install's.
export ASTRONOMIX_REPO="${ASTRONOMIX_REPO:-/e/home/jusers/storcks1/jupiter/astronomix}"
export RUN_LOG_DIR="/e/project1/astronomix/storcks1/runs"
mkdir -p "$RUN_LOG_DIR"

# --- micromamba env (never `conda activate`: silently falls back to
# --- system python) ---
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate astronomix

# --- JAX flags for the sharded code path ---
# The repo uses integer mesh-axis names + with_sharding_constraint (GSPMD
# style); the Shardy partitioner rejects them.
export JAX_USE_SHARDY_PARTITIONER=false
# Preallocate one large contiguous pool per process.  JAX only initialises
# the GPU picked via local_device_ids, so even with --gpu-bind=none (all four
# node GPUs visible to every task) each process preallocates only its own
# device.  On-demand allocation (PREALLOCATE=false) fragments the BFC pool
# and made a 43 GB compiled temp buffer fail on a 96 GB GH200 with plenty of
# total memory free.  0.92 leaves ~8 GB for the CUDA context and NCCL.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

# --- optional communication optimizations (gated so existing runners are
# --- unaffected; Slurm --export eats commas, hence one variable per knob) ---
# Latency-hiding scheduler: overlap collectives (halo exchanges, all-to-all)
# with compute instead of serialising them.
if [ "${OPT_LHS:-0}" = "1" ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true"
fi
# Decompose collective-permutes into send/recv pairs and pipeline them so
# the halo exchange of one field overlaps the stencil work of another.
if [ "${OPT_P2P:-0}" = "1" ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_pipelined_p2p=true --xla_gpu_collective_permute_decomposer_threshold=0"
fi
# NCCL transport diagnostics (stderr): verify IB/GDRDMA and per-GPU NIC use.
if [ "${OPT_NCCL_DEBUG:-0}" = "1" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT
fi

export PYTHONPATH="$ASTRONOMIX_REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- process layout of the sharded scaling drivers (weak_*.sh,
# --- strong_scaling.sh, validate_sharded.sh) ---
# The job shape is --nodes=N --ntasks-per-node=T, T the GPUs used per node.
# PROCESS_LAYOUT=node (default) launches one process per node driving all GPUs
# of its node (on a single node: one process claiming T GPUs via autocvd);
# intra-node halo exchanges then stay inside one process, which makes the
# multi-node rungs ~30% faster on JUPITER.  PROCESS_LAYOUT=gpu launches one
# process per GPU.  Sets SRUN_LAYOUT (srun options) and DRIVER_GPU_ARGS.
sharded_layout() {
    # NCCL needs more than the 8% of device memory that the default XLA pool
    # leaves free once the pairwise halo exchange adds its per-pair
    # communicators on many nodes; NVLS buffers are not needed for halo
    # exchanges.
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${SHARDED_MEM_FRACTION:-0.90}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
    local gpus_per_node="${SLURM_NTASKS_PER_NODE:-4}"
    local nodes="${SLURM_JOB_NUM_NODES:-1}"
    DRIVER_GPU_ARGS=""
    if [ "${PROCESS_LAYOUT:-node}" = "node" ]; then
        SRUN_LAYOUT="--ntasks=$nodes --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=none"
        if [ "$nodes" = "1" ]; then
            DRIVER_GPU_ARGS="--gpus $gpus_per_node"
        fi
    else
        SRUN_LAYOUT="--gpu-bind=none"
        if [ "${SLURM_NTASKS:-1}" = "1" ]; then
            DRIVER_GPU_ARGS="--gpus 1"
        fi
    fi
}
cd "$ASTRONOMIX_REPO"
