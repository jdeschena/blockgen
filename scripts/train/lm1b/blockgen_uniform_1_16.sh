#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/lm1b/blockgen_uniform_1_16}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

cd "${REPO_ROOT}"

# Uniform BlockGen with a 1+16 block-size mixture (uniform weights).
python -u -m main \
    data=lm1b \
    data.cache_dir="${CACHE_DIR}" \
    model=small-block-dit \
    model.length=128 \
    algo=blockgen-uniform \
    algo.loss_type=ce-noisy \
    algo.pure_noise_block_sizes=[1] \
    "algo.block_weights=0.5 0.0 0.0 0.0 0.5" \
    algo.validation.mode='log-sum-exp-8' \
    loader.global_batch_size=512 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=50_000 \
    trainer.max_steps=250_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    hydra.run.dir="${OUTPUT_DIR}"
