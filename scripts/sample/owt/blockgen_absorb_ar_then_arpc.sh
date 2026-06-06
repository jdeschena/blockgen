#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# CKPT_PATH may be a local .ckpt path or a HuggingFace spec of the form
# hf:jdeschena/blockgen/<subfolder>.
CKPT_PATH="${CKPT_PATH:-hf:jdeschena/blockgen/blockgen/multi_block_size/16/owt/0.05_0.95/absorb}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/blockgen_absorb_ar_then_arpc}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_AR_TOKENS="${NUM_AR_TOKENS:-0}"
TEMP="${TEMP:-1.0}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

cd "${REPO_ROOT}"

# AR-then-ARPC (AR-informed predictor-corrector) sampling of a masked
# (absorbing) multi-block 1+16 BlockGen model. Multi-block models share the
# KV cache between the AR and diffusion paths, so they use model.x0_causal=True.
# sampler.block_size must match the block size used during training, and
# sampler.num_ar_tokens must be a multiple of the block size.
python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-block-dit \
    model.x0_causal=True \
    model.attn_backend=sdpa \
    model.length=1024 \
    algo=blockgen-absorb \
    sampler=ar-then-arpc \
    sampler.block_size="${BLOCK_SIZE}" \
    sampler.num_ar_tokens="${NUM_AR_TOKENS}" \
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
