#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# CKPT_PATH may be a local .ckpt path or a HuggingFace spec of the form
# hf:jdeschena/blockgen/<subfolder>.
CKPT_PATH="${CKPT_PATH:-hf:jdeschena/blockgen/blockgen/single_block_size/16/tinygsm/absorb}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/gsm8k_test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/tinygsm/blockgen_absorb_block_ancestral}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TEMP="${TEMP:-1.0}"

cd "${REPO_ROOT}"

# Block-ancestral sampling of a masked (absorbing) single-block-16 BlockGen model.
# sampler.block_size must match the block size used during training.
# TEMP sets the sampling temperature (the paper sweeps it; best TinyGSM is T=0.1).
python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${CACHE_DIR}" \
    data.data_path="${DATA_PATH}" \
    model=small-block-dit \
    model.x0_causal=False \
    model.attn_backend=sdpa \
    model.length=512 \
    algo=blockgen-absorb \
    sampler=block-ancestral \
    sampler.block_size="${BLOCK_SIZE}" \
    sampler.steps="${STEPS}" \
    sampler.temperature="${TEMP}" \
    sampler.early_stopping=True \
    loader.eval_batch_size=32 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
