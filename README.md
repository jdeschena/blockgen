# BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers

By [Justin Deschenaux](https://jdeschena.com) and [Caglar Gulcehre](https://www.caglar.ai).

[![arXiv](https://img.shields.io/badge/arXiv-2606.02241-red.svg)](https://arxiv.org/abs/2606.02241)
[![HuggingFace](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/jdeschena/blockgen)

**Abstract**: Is the uniform-state diffusion framework a more powerful paradigm for discrete diffusion? Recent studies indicate that this may be the case. In combination with predictor-corrector samplers, uniform-state diffusion models (USDMs) produce samples of higher-quality than masked diffusion models (MDMs), and USDMs equal or outperform MDMs in downstream tasks. Two issues remain unresolved. First, existing work compares uniform and masked diffusion with un-informed correctors that re-inject noise at random positions, rather than targeting tokens most likely to be wrong. Second, prior work compares full-sequence diffusion models, so we do not know whether the same conclusion holds when tokens are generated block by block. To address these issues, we introduce BlockGen, a blockwise sequence model that we instantiate with both masked and uniform diffusion. BlockGen trains on a mixture of block sizes and its likelihood interpolates between AR and pure diffusion more finely than models with a fixed block size. BlockGen enables AR-informed predictor-corrector sampling (ARPC), which combines AR and diffusion predictions to re-generate unlikely tokens without an auxiliary verifier.

This repository contains training and evaluation code for BlockGen along with the discrete-diffusion baselines we compare against (AR, MDLM, Duo). BlockGen comes in two variants:

- **BlockGen-absorb**: masked / absorbing-state diffusion within each block.
- **BlockGen-uniform**: uniform (token-substitution) diffusion within each block.

We release pretrained checkpoints for two settings:

- **TinyGSM**: math reasoning, SmolLM-135M tokenizer, 250k steps.
- **OpenWebText (OWT)**: general LM, GPT-2 tokenizer, 1M steps.

[Getting started](#getting-started) · [Data](#data) · [Checkpoints](#checkpoints) · [Training](#training) · [Sampling & evaluation](#sampling--evaluation) · [Citation](#citation)

# Getting started

Create a fresh environment and install the Python dependencies:

```bash
conda create -n blockgen python=3.12
conda activate blockgen
pip install -r requirements.txt
```

`requirements.txt` intentionally does **not** pin `torch` or `numpy`. We work inside the NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.02-py3`), which already ships matching CUDA / cuDNN / NCCL builds. If you are not using the container, install `torch` and `numpy` **before** running `pip install -r requirements.txt`.

The entry point is `main.py`, using [Hydra](https://hydra.cc/). The available config groups are:

# Data

Datasets are downloaded and tokenized on first use into `data.cache_dir` (defaults to `./data_cache` in the provided scripts):

- **TinyGSM** (`data=tiny-gsm`) is loaded from the [`TinyGSM/TinyGSM`](https://huggingface.co/datasets/TinyGSM/TinyGSM) HuggingFace dataset, tokenized with the SmolLM-135M tokenizer.
- **OpenWebText** (`data=openwebtext` / `data=openwebtext-split`) is downloaded from HuggingFace and tokenized with the GPT-2 tokenizer.
- **GSM8K test** (`data=gsm8k-test`) is read from the JSON file shipped in this repo at [`data/gsm8k_test.json`](data/gsm8k_test.json) (point at it with `data.data_path=...`).

# Checkpoints

All released checkpoints are in the HuggingFace repo [`jdeschena/blockgen`](https://huggingface.co/jdeschena/blockgen). Each checkpoint is a subfolder containing:

- `model.safetensors` — the **EMA** weights of the backbone.
- `config.json` — the model sub-config, used to rebuild the architecture.

To load the checkpoints directly for sampling, set `eval.checkpoint_path=hf:jdeschena/blockgen/<subfolder>`, which downloads the checkpoint directly, and creates the backbone from the config. For example:

```bash
python -m main mode=gsm8k_eval \
    eval.checkpoint_path=hf:jdeschena/blockgen/blockgen/single_block_size/16/tinygsm/absorb \
    ...
```

### Layout on Huggingface

```
baselines/
  owt/{ar,mdlm,duo} # OWT baselines, 1M steps
  tinygsm/{ar,mdlm,duo} # TinyGSM baselines, 250k steps

blockgen/
  single_block_size/ # one fixed block size
    16/owt/{absorb,uniform}
    16/tinygsm/{absorb,uniform}
    32/tinygsm/{absorb,uniform}
  multi_block_size/ # 2 block-size mixture (AR + diffusion)
    # <weights> in {0.05_0.95, uniform}
    16/owt/<weights>/{absorb,uniform}
    # <weights> in {0.01_0.99, 0.05_0.95, 0.1_0.9}
    16/tinygsm/<weights>/{absorb,uniform}
    # <weights> in {0.01_0.99, 0.05_0.95}
    32/tinygsm/<weights>/{absorb,uniform}
```

Here `<weights>` is the block-size mixture weight on the AR (block size 1) vs. diffusion block (e.g. `0.05_0.95` puts 5% mass on AR and 95% on the 16- or 32-token diffusion block). `absorb` / `uniform` correspond to `algo=blockgen-absorb` / `algo=blockgen-uniform`.

# Scripts

The training and sampling scripts are in `scripts/` and are organized by dataset (`owt` / `tinygsm` / `lm1b`) and method. The baselines are `ar` / `mdlm` / `duo`. The BlockGen scripts can use a **single-block-size** vs **multi-block-size** (train on mixture density). When training with a mixture over block sizes, one should select mixture weights (e.g. `0.05_0.95`, `0.1_0.9`, `0.01_0.99`, or uniform weights). 

One can sample with standard **ancestral sampling** (ar, block-by-block or full sequence ancestral), or **AR-Informed Predictor Corrector sampling** (ARPC), which uses AR predictions to detect and re-generate low-quality tokens. 

Example call to a training script:

```bash
DEVICES=8 NUM_NODES=1 \
OUTPUT_DIR=./outputs/tinygsm/blockgen_absorb_1_16 \
bash scripts/train/tinygsm/blockgen_absorb_1_16.sh
```

You can also call `main.py` directly. The BlockGen models use the block-DiT backbone (`model=small-block-dit`). Single-block-size models use `model.x0_causal=False`. The multi-block models use `model.x0_causal=True` to share the KV cache between the AR and diffusion modes.

```bash
python -m main mode=train \
    data=tiny-gsm \
    data.cache_dir=./data_cache \
    data.wrap=False data.train_on_prompt=False \
    data.train_on_pad=True data.filter_too_long=True \
    model=small-block-dit model.length=512 model.x0_causal=True \
    algo=blockgen-absorb \
    algo.loss_type=ce \
    algo.pure_noise_block_sizes=[1] \
    algo.block_weights="0.05 0.0 0.0 0.0 0.95" \
    algo.block_size_per_gpu=u-stratified \
    loader.global_batch_size=512 loader.batch_size=64 \
    trainer.devices=8 trainer.max_steps=250_000
```

`algo.block_weights` is the distribution over block sizes (powers of two). Using `algo.block_weights="0.05 0.0 0.0 0.0 0.95"` means training for 5% of the steps with block size 1 (AR) and 95% with block size 16.

# Acknowledgements

This codebase builds on a number of excellent open-source projects:

- [**Duo**](https://github.com/s-sahoo/duo): discrete diffusion baseline; our overall training/eval scaffolding is descended from theirs.
- [**MDLM**](https://github.com/kuleshov-group/mdlm): masked discrete diffusion language model baseline.
- [**PUMA**](https://github.com/JaeyeonKim01/PUMA): reference for the TinyGSM data preparation.
- [**PRISM**](https://github.com/JaeyeonKim01/PRISM): reference for the Sudoku data preparation.


# Citation

```
@misc{deschenaux2026blockgen,
      title={BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers},
      author={Justin Deschenaux and Caglar Gulcehre},
      year={2026},
      eprint={2606.02241},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.02241},
}
```
