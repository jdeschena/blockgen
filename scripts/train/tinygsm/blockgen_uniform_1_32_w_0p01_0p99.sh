#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinygsm/blockgen_uniform_1_32_w_0p01_0p99}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-8}"

cd "${REPO_ROOT}"

# Uniform BlockGen with a 1+32 block-size mixture (1% AR / 99% diffusion).
# ELBO loss everywhere, with cross-entropy at block size 1 (pure-noise inputs).
# Multi-block models share the KV cache between the AR and diffusion paths,
# so they require model.x0_causal=True.
python -u -m main \
    data=tiny-gsm \
    data.cache_dir="${CACHE_DIR}" \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.wrap=False \
    data.train_on_prompt=False \
    data.train_on_pad=True \
    data.filter_too_long=True \
    model=small-block-dit \
    model.length=512 \
    model.x0_causal=True \
    algo=blockgen-uniform \
    algo.loss_type=elbo \
    algo.loss_type_special_cases="[1,ce]" \
    algo.pure_noise_block_sizes=[1] \
    "algo.block_weights=0.01 0.0 0.0 0.0 0.0 0.99" \
    algo.block_size_per_gpu=u-stratified \
    loader.global_batch_size=512 \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=16 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=10_000 \
    trainer.max_steps=250_000 \
    trainer.limit_val_batches=500 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    hydra.run.dir="${OUTPUT_DIR}"
