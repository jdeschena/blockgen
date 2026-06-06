#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# CKPT_PATH may be a local .ckpt path or a HuggingFace spec of the form
# hf:jdeschena/blockgen/<subfolder>. Note: LM1B checkpoints are not part of the
# public release — train your own (see scripts/train/lm1b) and set CKPT_PATH.
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a local .ckpt (no public LM1B checkpoint is released)}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/lm1b/blockgen_uniform_block_ancestral}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TEMP="${TEMP:-1.0}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

cd "${REPO_ROOT}"

# Block-ancestral sampling of a uniform BlockGen model on LM1B.
# sampler.block_size must match the block size used during training.
python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    data=lm1b \
    data.cache_dir="${CACHE_DIR}" \
    model=small-block-dit \
    model.length=128 \
    model.attn_backend=sdpa \
    algo=blockgen-uniform \
    sampler=block-ancestral \
    sampler.block_size="${BLOCK_SIZE}" \
    sampler.noise_removal=ancestral \
    sampler.steps="${STEPS}" \
    sampler.temperature="${TEMP}" \
    sampler.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
    sampler.use_kv_cache=True \
    loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
    loader.num_workers=8 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
