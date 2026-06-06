"""Loading BlockGen / baseline checkpoints from safetensors + config.

Released checkpoints are published as a folder containing:
  - model.safetensors : the EMA weights of the backbone
  - config.json       : the model sub-config, used to rebuild the architecture

A checkpoint is referenced either by a local folder path or by a Hugging Face
spec of the form ``hf:<repo_id>/<subfolder>``, e.g.::

    eval.checkpoint_path=hf:jdeschena/blockgen/mdlm

which downloads ``mdlm/config.json`` and ``mdlm/model.safetensors`` from the
``jdeschena/blockgen`` repo, rebuilds the backbone from the saved model config,
and loads the EMA weights into it.
"""

import json
import os

import omegaconf
from safetensors.torch import load_file

HF_PREFIX = 'hf:'
_CONFIG_NAME = 'config.json'
_WEIGHTS_NAME = 'model.safetensors'


def is_safetensors_spec(checkpoint_path):
  """True if ``checkpoint_path`` points at a safetensors release checkpoint."""
  if checkpoint_path.startswith(HF_PREFIX):
    return True
  if checkpoint_path.endswith('.safetensors'):
    return True
  return (os.path.isdir(checkpoint_path)
          and os.path.exists(os.path.join(checkpoint_path, _WEIGHTS_NAME)))


def _resolve_spec(checkpoint_path):
  """Resolve a spec to local (config_path, weights_path), downloading if needed."""
  if checkpoint_path.startswith(HF_PREFIX):
    from huggingface_hub import hf_hub_download
    spec = checkpoint_path[len(HF_PREFIX):]
    parts = spec.split('/')
    if len(parts) < 2:
      raise ValueError(
        f"Expected 'hf:<org>/<repo>[/<subfolder>]', got '{checkpoint_path}'")
    repo_id = '/'.join(parts[:2])
    subfolder = '/'.join(parts[2:]) or None
    config_path = hf_hub_download(
      repo_id, filename=_CONFIG_NAME, subfolder=subfolder)
    weights_path = hf_hub_download(
      repo_id, filename=_WEIGHTS_NAME, subfolder=subfolder)
    return config_path, weights_path

  if os.path.isdir(checkpoint_path):
    return (os.path.join(checkpoint_path, _CONFIG_NAME),
            os.path.join(checkpoint_path, _WEIGHTS_NAME))

  # A direct path to a .safetensors file; config.yaml sits next to it.
  return (os.path.join(os.path.dirname(checkpoint_path), _CONFIG_NAME),
          checkpoint_path)


def load_from_safetensors(diffusion_model, config, tokenizer):
  """Rebuild the architecture from the saved config and load EMA weights.

  The published safetensors already contain the EMA weights, so they are loaded
  directly as the backbone parameters and ``model.ema`` is set to None (no EMA
  swap is applied downstream).
  """
  config_path, weights_path = _resolve_spec(config.eval.checkpoint_path)

  # Reconstruct the architecture from the checkpoint's saved model config.
  with open(config_path) as f:
    saved_model = omegaconf.OmegaConf.create(json.load(f))
  config.model = omegaconf.OmegaConf.merge(config.model, saved_model)

  model = diffusion_model(config, tokenizer=tokenizer)
  state_dict = load_file(weights_path)
  model.backbone.load_state_dict(
    state_dict, strict=config.eval.strict_loading)
  model.ema = None
  return model
