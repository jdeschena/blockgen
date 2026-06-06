#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/owt/blockgen_absorb_single_b16}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-16}"

cd "${REPO_ROOT}"

# Masked (absorbing) BlockGen with a single block size of 16.
# Single-block-size models cache only once a block is generated and never
# share the KV cache between the AR and diffusion paths, so they MUST use
# model.x0_causal=False (only multi-block-size models use x0_causal=True).
python -u -m main \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-block-dit \
    model.x0_causal=False \
    algo=blockgen-absorb \
    algo.loss_type=ce \
    "algo.block_weights=0.0 0.0 0.0 0.0 1.0" \
    algo.block_size_per_gpu=same \
    loader.global_batch_size=512 \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=1_000_000 \
    trainer.val_check_interval=50_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    hydra.run.dir="${OUTPUT_DIR}"
