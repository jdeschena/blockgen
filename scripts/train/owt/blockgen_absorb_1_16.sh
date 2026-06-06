#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/owt/blockgen_absorb_1_16}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-16}"

cd "${REPO_ROOT}"

# Masked (absorbing) BlockGen with a 1+16 block-size mixture (mostly diffusion).
# Multi-block models share the KV cache between the AR and diffusion paths,
# so they require model.x0_causal=True.
python -u -m main \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-block-dit \
    model.x0_causal=True \
    algo=blockgen-absorb \
    algo.loss_type=ce \
    algo.loss_type_special_cases="[]" \
    algo.pure_noise_block_sizes=[1] \
    "algo.block_weights=0.05 0.0 0.0 0.0 0.95" \
    algo.block_size_per_gpu=u-stratified \
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
