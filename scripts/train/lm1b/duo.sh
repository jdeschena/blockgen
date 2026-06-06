#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/lm1b/duo}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

cd "${REPO_ROOT}"

python -u -m main \
    data=lm1b \
    data.cache_dir="${CACHE_DIR}" \
    model=small \
    model.length=128 \
    algo=duo-base \
    loader.global_batch_size=512 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.val_check_interval=50_000 \
    trainer.max_steps=250_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    hydra.run.dir="${OUTPUT_DIR}"
