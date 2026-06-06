"""Implementations of the samplers.

Content:
  - Pure implementations of posteriors, and sampling utils
  - Sampler classes. Looping over step, after state init.
  - Dataclasses naming convention / use case:
      - State: sampler data persisted across steps.
      - Context: data passed to the denoising neural network.
      - View: data derived from state and used only within a
              single step to pass variables around.
"""

import abc
import torch
import torch.nn.functional as F
from dataclass_patch import dataclass
import utils


def sample_categorical(categorical_probs, gumbel_noise=None):
  """Sample from categorical using the Gumbel trick"""
  if gumbel_noise is None:
    gumbel_noise = (
      1e-10
      - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
  return (categorical_probs / gumbel_noise).argmax(dim=-1)


def _normalize_posterior_inputs(x0_probs, xt, alpha_s, 
                                alpha_t, use_float64=False):
  if use_float64:
    x0_probs = x0_probs.to(torch.float64)
  if alpha_s.ndim == 2:
    alpha_s = alpha_s.unsqueeze(-1)
  if alpha_t.ndim == 2:
    alpha_t = alpha_t.unsqueeze(-1)
  return x0_probs, xt, alpha_s, alpha_t


@torch.compile
def absorbing_posterior_probs(x0_probs, xt, alpha_s, alpha_t,
                              mask_index, vocab_size,
                              use_float64=False):
  """Posterior q(x_s | x_t, x_0) for absorbing diffusion"""
  x0_probs, xt, alpha_s, alpha_t = _normalize_posterior_inputs(
    x0_probs, xt, alpha_s, alpha_t, use_float64=use_float64)
  denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
  xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs)
  is_masked = (xt == mask_index).to(x0_probs.dtype).unsqueeze(-1)
  masked_posterior = (
    xt_one_hot * (1 - denoise_prob)
    + x0_probs * denoise_prob)
  return xt_one_hot * (1 - is_masked) + masked_posterior * is_masked


@torch.compile
def sample_absorbing_posterior(x0_probs, xt, alpha_s, alpha_t,
                               mask_index,
                               noise_removal_step=False):
  """Sample from absorbing posterior without materializing it"""
  sampled_x0 = sample_categorical(x0_probs)

  if noise_removal_step:
    should_denoise = torch.ones_like(sampled_x0, dtype=torch.bool)
  else:
    denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
    should_denoise = torch.rand_like(
      sampled_x0, dtype=torch.float64) < denoise_prob

  is_masked = (xt == mask_index)
  should_denoise_mask = is_masked & should_denoise
  return torch.where(should_denoise_mask, sampled_x0, xt)


def _expand_alpha_like(alpha, ref):
  """Expand alpha to match the dimension of ref. One of
     (B, 1), (B, L), (B, L, 1). Output should have ndim=2"""
  if alpha.ndim == 3:
    alpha = alpha.squeeze(-1)
  if alpha.ndim != 2:
    raise ValueError(f'Unexpected alpha shape: {alpha.shape}')
  if alpha.shape[1] == 1:
    return alpha.expand_as(ref)
  if alpha.shape[1] != ref.shape[1]:
    raise ValueError(
      f'Alpha shape {alpha.shape} not broadcastable to {ref.shape}')
  return alpha


@torch.compile
def uniform_posterior_probs(x0_probs, xt, alpha_s, alpha_t,
                            vocab_size, use_float64=False):
  """Posterior q(x_s | x_t, x_0) for uniform diffusion."""
  x0_probs, xt, alpha_s, alpha_t = _normalize_posterior_inputs(
    x0_probs, xt, alpha_s, alpha_t, use_float64=use_float64)
  alpha_ts = alpha_t / alpha_s
  d_alpha = alpha_s - alpha_t
  xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs)
  p_xt = torch.gather(x0_probs, -1, xt[..., None])
  numerator = (
    alpha_t * vocab_size * x0_probs * xt_one_hot
    + (alpha_ts - alpha_t) * xt_one_hot
    + d_alpha * x0_probs
    + (1 - alpha_ts) * (1 - alpha_s) / vocab_size)
  denominator = alpha_t * vocab_size * p_xt + (1 - alpha_t)
  return numerator / denominator


@torch.compile
def sample_uniform_posterior(x0_probs, xt, alpha_s, alpha_t,
  vocab_size, noise_removal_step=False):
  """Sample from uniform posterior without materializing it."""
  p_xt = torch.gather(x0_probs, -1, xt.unsqueeze(-1)).squeeze(-1)
  alpha_t = _expand_alpha_like(alpha_t, p_xt).to(x0_probs.dtype)
  alpha_s = _expand_alpha_like(alpha_s, p_xt).to(x0_probs.dtype)
  denominator = alpha_t * vocab_size * p_xt + (1 - alpha_t)

  sampled_x0 = sample_categorical(x0_probs)
  sample_threshold = torch.rand_like(p_xt)

  if noise_removal_step:
    keep_xt_prob = (
      alpha_t * vocab_size * p_xt / denominator).clamp(0.0, 1.0)
    return torch.where(sample_threshold < keep_xt_prob,
                       xt, sampled_x0)

  alpha_ts = alpha_t / alpha_s
  sample_uniform_prob = (
    (1 - alpha_ts) * (1 - alpha_s) / denominator).clamp(0.0, 1.0)
  keep_xt_prob = (
    (alpha_t * vocab_size * p_xt + alpha_ts - alpha_t)
    / denominator).clamp(0.0, 1.0)
  uniform_samples = torch.randint(0, vocab_size, xt.shape,
                                  device=xt.device)
  # priority: uniform > xt > x0
  keep_or_resample = torch.where(
    (sample_uniform_prob + keep_xt_prob).clamp(max=1.0)
    > sample_threshold,
    xt,
    sampled_x0)
  return torch.where(sample_uniform_prob > sample_threshold,
                     uniform_samples,
                     keep_or_resample)


def sample_posterior(process, x0_probs, xt, alpha_s, alpha_t, *,
  mask_index=None, vocab_size, noise_removal_step=False,
  posterior_sampler='fast', use_float64=False):
  if process == 'absorbing':
    return sample_absorbing_posterior(
      x0_probs=x0_probs, xt=xt,
      alpha_s=alpha_s, alpha_t=alpha_t,
      mask_index=mask_index,
      noise_removal_step=noise_removal_step)
  elif process == 'uniform':
    if posterior_sampler == 'fast':
      return sample_uniform_posterior(
        x0_probs=x0_probs, xt=xt,
        alpha_s=alpha_s, alpha_t=alpha_t,
        vocab_size=vocab_size,
        noise_removal_step=noise_removal_step)
    elif posterior_sampler == 'naive':
      posterior_probs = compute_posterior(
        process, x0_probs, xt, alpha_s, alpha_t,
        mask_index=mask_index,
        vocab_size=vocab_size,
        use_float64=use_float64)
      return sample_categorical(posterior_probs)
    else:
      raise ValueError(posterior_sampler)
  else:
    raise ValueError(process)


def compute_posterior(process, x0_probs, xt, alpha_s, alpha_t, *,
  mask_index=None, vocab_size, use_float64=False):
  if process == 'absorbing':
    return absorbing_posterior_probs(
      x0_probs=x0_probs,
      xt=xt,
      alpha_s=alpha_s,
      alpha_t=alpha_t,
      mask_index=mask_index,
      vocab_size=vocab_size,
      use_float64=use_float64)
  if process == 'uniform':
    return uniform_posterior_probs(
      x0_probs=x0_probs,
      xt=xt,
      alpha_s=alpha_s,
      alpha_t=alpha_t,
      vocab_size=vocab_size,
      use_float64=use_float64)
  raise ValueError(f'Unknown process: {process}')


def _maybe_cast_log_probs(log_probs, use_float64):
  if use_float64:
    return log_probs.to(torch.float64)
  return log_probs


def _model_mask_index(model):
  if model.diffusion_type != 'absorbing':
    return None
  if not hasattr(model, 'mask_index'):
    raise AttributeError(
      'Absorbing diffusion models must define model.mask_index')
  return model.mask_index


def _decode_direct_tokens(log_probs, greedy):
  if greedy:
    return log_probs.argmax(dim=-1)
  return sample_categorical(log_probs.exp())


def _decode_posterior_tokens(
    model, log_probs, current_tokens, alpha_s, alpha_t, *,
    greedy, noise_removal_step, use_float64):
  probs = log_probs.exp()
  mask_index = _model_mask_index(model)
  if greedy:
    posterior = compute_posterior(
      model.diffusion_type, probs, current_tokens,
      alpha_s, alpha_t,
      mask_index=mask_index,
      vocab_size=model.vocab_size,
      use_float64=use_float64)
    return posterior.argmax(dim=-1)
  return sample_posterior(
    model.diffusion_type, probs, current_tokens,
    alpha_s, alpha_t,
    mask_index=mask_index,
    vocab_size=model.vocab_size,
    noise_removal_step=noise_removal_step)


def _decode_ancestral_update(
    sampler, model, log_probs, current_tokens, alpha_s, alpha_t, *,
    is_last_step):
  """Generic ancestral transition kernel on an arbitrary token window.

  This is the shared black-box update used by the full-sequence ancestral
  sampler and the block samplers. Scheduling and state transitions stay
  outside this function; it only maps:
    current_tokens_t + model log p(x0 | xt) + (alpha_t, alpha_s)
  to:
    next_tokens_s
  """
  if is_last_step and sampler.noise_removal == 'greedy':
    return log_probs.argmax(dim=-1)
  return _decode_posterior_tokens(
    model, log_probs, current_tokens, alpha_s, alpha_t,
    greedy=False,
    noise_removal_step=is_last_step,
    use_float64=sampler.use_float64)


def _early_stop_token(state, xt, idx, tokenizer):
  """Early stopping for token-by-token generation.

  This is useful for tasks where generation should stop at 
  the first EOS. For example, for GSM8K, we only care about 
  generating one answer.

  Called after a single token has been written to xt[:, idx]. 
  Return true when all sequences are done.
  """
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id

  newly_finished = (~state.finished) & (xt[:, idx] == eos_id)
  if newly_finished.any():
    xt[newly_finished, idx + 1:] = pad_id
  state.finished = state.finished | newly_finished
  return bool(state.finished.all())


def _early_stop_block(state, block_start, block_end, tokenizer):
  """Early stopping for block-by-block generation.

  This is useful for tasks where generation should stop at 
  the first EOS. For example, for GSM8K, we only care about 
  generating one answer.

  Called after the current block has been finalized. Return 
  true when all sequences are done
  """
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id
  xt = state.xt
  # Detect newly-finished sequences (EOS in block, not already done).
  block = xt[:, block_start: block_end]  # [B, block_size]
  eos_mask = block == eos_id  # [B, block_size]
  eos_mask[state.finished] = False  # exclude already-done rows
  newly_done = eos_mask.any(dim=1)  # [B]

  if newly_done.any():
    eos_pos = eos_mask.int().argmax(dim=1)  # first EOS pos per row
    for b in newly_done.nonzero(as_tuple=False).view(-1):
      ep = eos_pos[b].item()
      xt[b, block_start + ep + 1:] = pad_id
    state.finished = state.finished | newly_done
  return bool(state.finished.all())


# ----------------------------------------------------------
# Sampler classes, actually used to generate
# ----------------------------------------------------------
@dataclass(kw_only=True)
class BaseState:
  prefix_tokens: torch.Tensor = None   # (B, max_prefix_len) or (B, max_prefix_len, V)
  prefix_lengths: torch.Tensor = None  # (B,) int64


class Sampler(abc.ABC):
  """API that all samplers must comply with."""

  @abc.abstractmethod
  def init_state(self, model, num_samples, *, num_steps,
                 eps, prefix_tokens, prefix_lengths):
    ...

  @abc.abstractmethod
  def step(self, model, state):
    ...

  def _validate_prefix_args(self, prefix_tokens, prefix_lengths):
    assert (prefix_tokens is None) == (prefix_lengths is None), \
      'prefix_tokens and prefix_lengths must both be set or both be None'

  def _project_prefix(self, xt, prefix_tokens, prefix_lengths):
    if prefix_tokens is None:
      return xt
    P = prefix_tokens.shape[1]
    positions = torch.arange(P, device=xt.device)
    mask = positions[None, :] < prefix_lengths[:, None]  # (B, P)
    if xt.ndim == 3:
      mask = mask.unsqueeze(-1)
    xt[:, :P] = torch.where(mask, prefix_tokens, xt[:, :P])
    return xt

  def metadata(self, state):
    return {'nfe': state.nfe}


@dataclass(kw_only=True)
class AncestralState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool


@dataclass
class AncestralContext:
  temperature: float = 1.0
  kv_cache: bool = False


class AncestralSampler(Sampler):
  """Full-sequence ancestral sampler for MDLM & DUO"""

  def __init__(self, use_float64=False,
               noise_removal='ancestral', steps_policy='full',
               temperature=1.0):
    self.use_float64 = use_float64
    self.noise_removal = noise_removal
    self.steps_policy = steps_policy
    self.temperature = temperature
    assert noise_removal in ('ancestral', 'greedy')
    assert steps_policy in ('full', 'proportional')

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = model.prior_sample(num_samples, model.num_tokens)
    if prefix_tokens is not None:
      start_idx = int(prefix_lengths.min())
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
    else:
      start_idx = 0

    if num_steps is None:
      num_steps = model.config.sampler.steps
    # Adapt number of sampling steps to the number of tokens
    #  left to sample.
    if self.steps_policy == 'proportional':
      gen_len = model.num_tokens - start_idx
      num_steps = int(round(num_steps * gen_len))

    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    ones = torch.ones(xt.shape[0], 1, dtype=model.dtype,
                      device=model.device)
    state = AncestralState(
      xt=xt, timesteps=timesteps, ones=ones, start_idx=start_idx,
      step_idx=0, nfe=0, done=False,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)
    return state

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones

    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)

    log_probs = _maybe_cast_log_probs(
      model.forward(xt=state.xt, sigma=sigma_t,
                    context=AncestralContext(temperature=self.temperature)),
      self.use_float64)

    current_tokens = state.xt[:, state.start_idx:]
    log_probs_window = log_probs[:, state.start_idx:]
    current_tokens[:] = _decode_ancestral_update(
      self, model, log_probs_window, current_tokens,
      alpha_s, alpha_t, is_last_step=is_last_step)

    self._project_prefix(state.xt, state.prefix_tokens, 
                         state.prefix_lengths)
    state.nfe += 1
    state.step_idx += 1
    if is_last_step:
      state.done = True
    return state

@dataclass(kw_only=True)
class ARState(BaseState):
  xt: torch.Tensor
  ones: torch.Tensor
  zeros: torch.Tensor
  start_idx: int
  token_idx: int
  cached_len: int   # number of tokens committed to the KV cache
  nfe: int
  done: bool
  # Whether each xt has already generated an EOS. Only used 
  #  when early_stopping is True.
  finished: torch.Tensor = None  


@dataclass
class ARContext:
  kv_cache: bool
  temperature: float = 1.0


class ARSampler(Sampler):
  """Autoregressive sampler, generate tokens L2R, feed prefix
     only (suggested to use with causal transformers only)"""

  def __init__(self, use_float64, kv_cache, greedy,
               early_stopping, temperature=1.0):
    self.use_float64 = use_float64
    self.kv_cache = kv_cache
    self.greedy = greedy
    self.early_stopping = early_stopping
    self.temperature = temperature

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    if prefix_lengths is not None and prefix_lengths.unique().numel() > 1:
      raise NotImplementedError(
        'ARSampler does not support variable-length prefixes')
    xt = torch.zeros(num_samples, model.num_tokens,
                     dtype=torch.long, device=model.device)
    if prefix_tokens is not None:
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
      start_idx = int(prefix_lengths.max())
    else:
      xt[:, 0] = model.tokenizer.bos_token_id
      start_idx = 1
    ones = torch.ones(num_samples, 1, dtype=model.dtype,
                      device=model.device)
    zeros = torch.zeros_like(ones)
    if self.kv_cache:
      model.reset_kv_cache()
    if self.early_stopping:
      finished = torch.zeros(num_samples, dtype=torch.bool,
                             device=model.device)
    else:
      finished = None
    gen_len = xt.shape[1] - start_idx
    state = ARState(xt=xt, ones=ones, zeros=zeros,
      start_idx=start_idx, token_idx=start_idx, cached_len=0,
      nfe=0, done=(start_idx >= xt.shape[1]), finished=finished)
    return state

  def step(self, model, state):
    x = state.xt
    if self.kv_cache:
      start_idx = state.cached_len
    else:
      start_idx = 0
    end_idx = state.token_idx - 1

    x_input = x[:, start_idx: end_idx + 1]
    log_p = model.forward(xt=x_input, sigma=state.zeros,
      context=ARContext(kv_cache=self.kv_cache,
                        temperature=self.temperature))
    log_probs = _maybe_cast_log_probs(log_p[:, -1], self.use_float64)
    new_tok = _decode_direct_tokens(log_probs, greedy=self.greedy)

    x[:, state.token_idx] = new_tok

    if self.early_stopping:
      _early_stop_token(state, x, state.token_idx, model.tokenizer)

    if self.kv_cache:
      state.cached_len = state.token_idx
    state.nfe += 1
    state.token_idx += 1
    state.done = state.token_idx >= x.shape[1]
    if self.early_stopping and bool(state.finished.all()):
      state.done = True
    return state


@dataclass(kw_only=True)
class BlockGenAncestralState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  block_idx: int
  step_idx: int  # step index INSIDE a block
  # num. prefix tokens INSIDE current block (should not be changed)
  block_prefix_len: int
  num_steps_in_block: int  # num steps per block
  nfe: int
  done: bool
  # Whether each xt has already generated an EOS. Only used 
  #  when early_stopping is True.
  finished: torch.Tensor = None


@dataclass
class BlockDiTContext:
  mode: str
  block_size: int = 1
  temperature: float = 1.0


@dataclass
class BlockStepView:
  block_start: int
  block_end: int
  xt_b: torch.Tensor
  x0: torch.Tensor
  is_last_step: bool
  alpha_t: torch.Tensor
  alpha_s: torch.Tensor
  sigma_t: torch.Tensor


def _compute_block_num_steps(sampler, block_prefix_len, total_num_steps):
  if sampler.steps_policy == 'full' or block_prefix_len == 0:
    return total_num_steps
  return max(1, round(
    total_num_steps * (sampler.block_size - block_prefix_len)
    / sampler.block_size))


def _prepare_block_view(sampler, model, state):
  block_start = state.block_idx * sampler.block_size
  block_end = block_start + sampler.block_size

  if sampler.kv_cache:
    # During the first sampling step for block i, we must
    #  pass the content of the PREVIOUS block, so that it
    #  can be cached. Otherwise, we just feed the current,
    #  noisy block.
    if state.step_idx == 0:
      x0_start = model.ctx_cached_len
    else:
      x0_start = block_start
  else:
    # If no cache, then just feed the whole sequence.
    x0_start = 0

  # With kv cache, empty when step_idx > 0 (nothing new
  #  to commit)
  x0 = state.xt[:, x0_start: block_start]
  xt_b = state.xt[:, block_start: block_end]

  is_last_step = (state.step_idx == state.num_steps_in_block - 1)
  # If using max number of steps -> 0
  # If there is a prefix, start at lower noise level
  time_start_idx = (len(state.timesteps) - 1
                    - state.num_steps_in_block)
  t = state.timesteps[time_start_idx + state.step_idx] * state.ones
  s = state.timesteps[time_start_idx + state.step_idx + 1] * state.ones
  _, alpha_t = model.noise(t)
  sigma_t = model._sigma_from_alphat(alpha_t)
  # target time s < t
  alpha_s = (torch.ones_like(alpha_t) if is_last_step
             else model.noise(s)[1])

  return BlockStepView(
    block_start=block_start,
    block_end=block_end,
    xt_b=xt_b,
    x0=x0,
    is_last_step=is_last_step,
    alpha_t=alpha_t,
    alpha_s=alpha_s,
    sigma_t=sigma_t)


def _restore_block_prefix(state, xs_b, block_start):
  if state.block_prefix_len > 0:
    xs_b[:, :state.block_prefix_len] = (
      state.xt[:, block_start: block_start + state.block_prefix_len])


def _run_block_model(sampler, model, view):
  context = BlockDiTContext(mode='generate',
                            block_size=sampler.block_size,
                            temperature=sampler.temperature)
  return _maybe_cast_log_probs(
    model.forward(x0=view.x0, xt=view.xt_b,
                  sigma=view.sigma_t, context=context),
    sampler.use_float64)


def _run_standard_block_step(sampler, model, state):
  view = _prepare_block_view(sampler, model, state)
  log_probs = _run_block_model(sampler, model, view)
  next_block_tokens = _decode_ancestral_update(
    sampler, model, log_probs, view.xt_b,
    view.alpha_s, view.alpha_t, is_last_step=view.is_last_step)
  return _finalize_block_step(
    sampler, model, state, view, next_block_tokens)


def _finalize_block_step(sampler, model, state, view, xs_b):
  """Commit the updated block and update the sampler state.

  Flow:
  - Write the new block back into the sequence.
  - Re-project variable-length prefix tokens (if any).
  - If this was not the last denoising step for the block, continue denoising
    it.
  - Otherwise, either stop early on EOS, finish generation, or move to the
    next block.
  """
  _restore_block_prefix(state, xs_b, view.block_start)
  state.xt[:, view.block_start: view.block_end] = xs_b
  # Re-project variable-length prefix: for samples whose prefix
  # extends into or past this block, restore original tokens.
  sampler._project_prefix(
    state.xt, state.prefix_tokens, state.prefix_lengths)
  state.nfe += 1
  state.step_idx += 1

  if not view.is_last_step:
    return state

  if (sampler.early_stopping
      and _early_stop_block(state, view.block_start, view.block_end,
                            model.tokenizer)):
    state.done = True
    return state

  if view.block_end >= state.xt.shape[1]:
    state.done = True
    return state

  state.block_idx += 1
  state.step_idx = 0
  # For variable-length prefixes, some samples may still have
  # prefix tokens in the next block.
  if state.prefix_lengths is not None:
    next_block_start = state.block_idx * sampler.block_size
    # Per-sample prefix overlap with the next block
    per_sample_overlap = (state.prefix_lengths
                          - next_block_start).clamp(min=0)
    # Use min overlap as the block_prefix_len (conservative:
    # _project_prefix handles the rest per-sample)
    state.block_prefix_len = int(per_sample_overlap.min().item())
  else:
    state.block_prefix_len = 0
  if sampler.steps_policy == 'proportional':
    state.num_steps_in_block = _compute_block_num_steps(
      sampler, state.block_prefix_len,
      total_num_steps=len(state.timesteps) - 1)
  return state


class BlockGenAncestralSampler(Sampler):
  """Block-by-block ancestral sampler for BlockDiT models."""

  def __init__(self, use_float64, num_steps, noise_removal,
               steps_policy, block_size, kv_cache, early_stopping,
               temperature):
    self.use_float64 = use_float64
    self.noise_removal = noise_removal
    self.steps_policy = steps_policy
    self.kv_cache = kv_cache
    self.block_size = block_size
    self.num_steps = num_steps
    self.early_stopping = early_stopping
    self.temperature = temperature
    assert noise_removal in ('ancestral', 'greedy')
    assert steps_policy in ('full', 'proportional')

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    if prefix_lengths is not None and prefix_lengths.unique().numel() > 1:
      raise NotImplementedError(
        'BlockGenAncestralSampler does not support '
        'variable-length prefixes')
    xt = model.prior_sample(num_samples, model.num_tokens)
    if num_steps is None:
      num_steps = self.num_steps
    if prefix_tokens is not None:
      xt[:, :prefix_tokens.shape[1]] = prefix_tokens
      block_idx = prefix_tokens.shape[1] // self.block_size
      block_prefix_len = prefix_tokens.shape[1] % self.block_size
    else:
      block_idx = 0
      block_prefix_len = 0
    # How many steps to use in the first block, if we give
    #  a prefix of already-generated tokens?
    num_steps_in_block = _compute_block_num_steps(
      self, block_prefix_len, total_num_steps=num_steps)

    ones = torch.ones(num_samples, 1, dtype=model.dtype,
                      device=model.device)
    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    if self.kv_cache:
      model.reset_kv_cache()
    if self.early_stopping:
      finished = torch.zeros(num_samples, dtype=torch.bool, 
                             device=model.device)
    else:
      finished = None
    return BlockGenAncestralState(
      xt=xt, timesteps=timesteps, ones=ones,
      block_idx=block_idx, step_idx=0,
      block_prefix_len=block_prefix_len,
      num_steps_in_block=num_steps_in_block, nfe=0, 
      done=False, finished=finished)

  def step(self, model, state):
    return _run_standard_block_step(self, model, state)


# ----------------------------------------------------------
# AR-then-block samplers
#
# Base class ARThenBlockSampler handles:
#   - init_state (shared AR + block init logic)
#   - step       (dispatch: AR phase or block phase)
#   - _ar_step   (concrete, shared)
#   - _block_step (raises NotImplementedError: subclasses define
#                  the block denoising strategy)
#
# Adding a new block strategy = subclassing ARThenBlockSampler
# and implementing _block_step.
# ----------------------------------------------------------
@dataclass(kw_only=True)
class ARThenBlockState(BaseState):
  xt: torch.Tensor
  ones: torch.Tensor
  zeros: torch.Tensor
  phase: str  # 'ar' / 'block'
  nfe: int
  done: bool

  # AR phase
  token_idx: int
  ar_end_idx: int

  # Block phase
  timesteps: torch.Tensor
  block_idx: int
  step_idx: int  # step within current block
  block_prefix_len: int  # num prefix tokens in the block
  num_steps_in_block: int
  finished: torch.Tensor = None  # [B] bool; only set when early_stopping=True

  # Variable-length prefix support
  prefix_tokens: torch.Tensor = None
  prefix_lengths: torch.Tensor = None


class ARThenBlockSampler(Sampler):
  """Base class: AR prefix generation followed by block-by-block
  generation. Subclasses implement _block_step to define the
  block denoising strategy."""

  def __init__(self,
    num_ar_tokens,
    block_size,
    num_steps,
    ar_commit_every_step,
    use_float64,
    noise_removal,
    steps_policy,
    kv_cache,
    early_stopping,
    temperature):
    assert num_ar_tokens % block_size == 0, (
      f'num_ar_tokens ({num_ar_tokens}) must be a multiple of '
      f'block_size ({block_size})')
    self.num_ar_tokens = num_ar_tokens
    self.block_size = block_size
    self.num_steps = num_steps
    self.ar_commit_every_step = ar_commit_every_step
    self.use_float64 = use_float64
    self.noise_removal = noise_removal
    self.steps_policy = steps_policy
    self.kv_cache = kv_cache
    self.early_stopping = early_stopping
    self.temperature = temperature
    assert noise_removal in ('ancestral', 'greedy')
    assert steps_policy in ('full', 'proportional')

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    xt = model.prior_sample(num_samples, model.num_tokens)
    if num_steps is None:
      num_steps = self.num_steps

    if prefix_tokens is not None:
      # For variable-length prefixes, use min length for
      # scheduling (AR/block phase boundary) and re-project
      # per-sample prefix tokens at every step.
      if prefix_lengths is None:
        prefix_lengths = torch.full(
          (num_samples,), prefix_tokens.shape[1],
          device=model.device, dtype=torch.long)
      P = prefix_tokens.shape[1]
      positions = torch.arange(P, device=model.device)
      mask = positions[None, :] < prefix_lengths[:, None]
      xt[:, :P] = torch.where(mask, prefix_tokens, xt[:, :P])
      prefix_len = int(prefix_lengths.min().item())
    else:
      prefix_len = 0

    ar_end_idx = self.num_ar_tokens
    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    ones = torch.ones(num_samples, 1, dtype=model.dtype,
                      device=model.device)
    zeros = torch.zeros_like(ones)
    model.reset_kv_cache()

    if prefix_len < ar_end_idx:
      # Tokens left to generate in AR mode
      phase = 'ar'
      block_idx = ar_end_idx // self.block_size
      block_prefix_len = 0
      num_steps_in_block = num_steps
    else:
      # Prefix covers the whole AR section; start in block phase
      phase = 'block'
      block_idx = prefix_len // self.block_size
      block_start = block_idx * self.block_size
      block_prefix_len = prefix_len - block_start
      num_steps_in_block = _compute_block_num_steps(
        self, block_prefix_len, total_num_steps=num_steps)
    if self.early_stopping:
      finished = torch.zeros(num_samples, dtype=torch.bool,
                             device=model.device)
    else:
      finished = None

    return ARThenBlockState(
      xt=xt, ones=ones, zeros=zeros,
      phase=phase, token_idx=prefix_len, ar_end_idx=ar_end_idx,
      timesteps=timesteps,
      block_idx=block_idx, step_idx=0,
      block_prefix_len=block_prefix_len,
      num_steps_in_block=num_steps_in_block,
      nfe=0, done=False, finished=finished,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)

  def step(self, model, state):
    if state.phase == 'ar':
      return self._ar_step(model, state)
    return self._block_step(model, state)

  def _ar_step(self, model, state):
    # Unlike standard AR sampling, the clean token at position i is only
    # determined after step i (sampled from log_p). It therefore enters the
    # KV cache as context at step i+1. x0_start = ctx_cached_len ensures
    # decoded-but-not-yet-cached tokens are passed as x0 and committed.
    if self.kv_cache:
      x0_start = model.ctx_cached_len
    else:
      x0_start = 0

    if self.ar_commit_every_step:
      x0 = state.xt[:, x0_start: state.token_idx]
      xt_cur = state.xt[:, state.token_idx: state.token_idx + 1]
      block_size = 1
      decode_pos = 0
    else:
      # Feed full block at every step, but generate tokens l2r.
      #  Previously generated tokens are NOT committed until
      #  the end of the block.
      block_start = (
        (state.token_idx // self.block_size) * self.block_size)
      x0 = state.xt[:, x0_start: block_start]
      xt_cur = state.xt[
        :, block_start: block_start + self.block_size]
      block_size = self.block_size
      decode_pos = state.token_idx - block_start

    log_p = model.forward(x0=x0, xt=xt_cur, sigma=state.zeros,
      context=BlockDiTContext(mode='generate',
                              block_size=block_size,
                              temperature=self.temperature))
    log_probs = _maybe_cast_log_probs(log_p[:, decode_pos],
                                      self.use_float64)

    # Sample new token, but preserve prefix for samples
    # where this position is still within their prefix.
    sampled = _decode_direct_tokens(log_probs, greedy=False)
    if state.prefix_lengths is not None:
      in_prefix = state.prefix_lengths > state.token_idx
      state.xt[:, state.token_idx] = torch.where(
        in_prefix, state.xt[:, state.token_idx], sampled)
    else:
      state.xt[:, state.token_idx] = sampled

    if self.early_stopping:
      _early_stop_token(state, state.xt, state.token_idx, model.tokenizer)

    state.nfe += 1
    state.token_idx += 1
    if self.early_stopping and bool(state.finished.all()):
      state.done = True
      return state
    if state.token_idx >= state.ar_end_idx:
      state.phase = 'block'
      state.step_idx = 0
      if state.ar_end_idx >= state.xt.shape[1]:
        state.done = True
    return state

  def _block_step(self, model, state):
    raise NotImplementedError


class ARThenBlockAncestralSampler(ARThenBlockSampler):
  """AR prefix generation followed by block-by-block
     ancestral diffusion."""

  def _block_step(self, model, state):
    return _run_standard_block_step(self, model, state)


class ARThenARPCSampler(ARThenBlockSampler):
  """AR prefix generation followed by block-by-block
     AR Predictor-Corrector (ARPC)."""

  def __init__(self,
    num_ar_tokens,
    block_size,
    num_steps,
    ar_commit_every_step,
    use_float64,
    noise_removal,
    steps_policy,
    kv_cache,
    early_stopping,
    temperature,
    # ARPC arguments
    warmup_steps,
    guide_every,
    corruption_mode,
    divergence_measure,
    diffusion_metric,
    ar_metric):
    super().__init__(
      num_ar_tokens=num_ar_tokens,
      block_size=block_size,
      num_steps=num_steps,
      ar_commit_every_step=ar_commit_every_step,
      use_float64=use_float64,
      noise_removal=noise_removal,
      steps_policy=steps_policy,
      kv_cache=kv_cache,
      early_stopping=early_stopping,
      temperature=temperature)
    # ARPC arguments
    self.arpc_warmup_steps = warmup_steps
    self.arpc_guide_every = guide_every
    self.arpc_corruption_mode = corruption_mode
    self.arpc_divergence_measure = divergence_measure
    self.arpc_diffusion_metric = diffusion_metric
    self.arpc_ar_metric = ar_metric
    # Validate arguments
    assert warmup_steps >= 0
    assert guide_every > 0
    assert corruption_mode in ('random', 'divergence',
                               'diffusion_metric',
                               'ar_metric')
    assert divergence_measure in ('kld', 'reverse_kld', 'tvd')
    assert diffusion_metric in ('entropy', 'confidence', 'margin')
    assert ar_metric in ('nll', 'gap_to_top1', 'entropy')

  def _block_step(self, model, state):
    return self._arpc_phase(model, state)

  def _divergence_scores(self, p_ar, p_diff):
    eps = 1e-12
    if self.arpc_divergence_measure == 'kld':
      return (p_ar * (
        (p_ar + eps).log() - (p_diff + eps).log())).sum(-1)
    elif self.arpc_divergence_measure == 'reverse_kld':
      return (p_diff * (
        (p_diff + eps).log() - (p_ar + eps).log())).sum(-1)
    elif self.arpc_divergence_measure == 'tvd':
      return 0.5 * (p_ar - p_diff).abs().sum(-1)
    else:
      raise ValueError(self.arpc_divergence_measure)

  def _diffusion_scores(self, log_p_x0):
    p_x0 = log_p_x0.exp()
    if self.arpc_diffusion_metric == 'entropy':
      return torch.special.entr(p_x0).sum(-1)
    elif self.arpc_diffusion_metric == 'confidence':
      return -p_x0.max(dim=-1).values
    elif self.arpc_diffusion_metric == 'margin':
      top2 = torch.topk(p_x0, 2, dim=-1).values
      return -(top2[..., 0] - top2[..., 1])
    else:
      raise ValueError(self.arpc_diffusion_metric)

  def _ar_scores(self, log_p_ar, x_current):
    if self.arpc_ar_metric == 'nll':
      return -log_p_ar.gather(
        -1, x_current.unsqueeze(-1)).squeeze(-1)
    elif self.arpc_ar_metric == 'gap_to_top1':
      log_top1 = log_p_ar.max(-1).values
      log_xhat = log_p_ar.gather(
        -1, x_current.unsqueeze(-1)).squeeze(-1)
      return log_top1 - log_xhat
    elif self.arpc_ar_metric == 'entropy':
      return torch.special.entr(log_p_ar.exp()).sum(-1)
    else:
      raise ValueError(self.arpc_ar_metric)

  def _compute_corruption_indices(
      self, log_p_x0, log_p_ar, num_to_corrupt, *,
      x_current, block_prefix_len):
    """Select non-prefix positions to re-noise in the corrector step.

    Prefix positions within the block (0..block_prefix_len-1) are
    excluded from selection by setting their scores to -inf.
    Returns indices [B, num_to_corrupt].
    """
    if self.arpc_corruption_mode == 'random':
      scores = torch.rand(
        log_p_x0.shape[0], self.block_size,
        device=log_p_x0.device)
    elif self.arpc_corruption_mode == 'divergence':
      scores = self._divergence_scores(log_p_ar.exp(),
                                       log_p_x0.exp())
    elif self.arpc_corruption_mode == 'diffusion_metric':
      scores = self._diffusion_scores(log_p_x0)
    elif self.arpc_corruption_mode == 'ar_metric':
      scores = self._ar_scores(log_p_ar, x_current)
    else:
      raise ValueError(self.arpc_corruption_mode)

    if block_prefix_len > 0:
      scores[:, :block_prefix_len] = -float('inf')
    return torch.topk(scores, num_to_corrupt, dim=-1).indices

  def _arpc_phase(self, model, state):
    view = _prepare_block_view(self, model, state)
    log_probs = _run_block_model(self, model, view)

    is_guided = (state.step_idx >= self.arpc_warmup_steps
                 and (state.step_idx - self.arpc_warmup_steps)
                      % self.arpc_guide_every == 0)

    if is_guided and not view.is_last_step:
      # Predictor: sample clean block from posterior at alpha_s=1
      xs_b = _decode_posterior_tokens(
        model, log_probs, view.xt_b.clone(),
        state.ones, view.alpha_t,
        greedy=False,
        noise_removal_step=True,
        use_float64=self.use_float64)
      # Restore prefix tokens overwritten by the posterior sample
      _restore_block_prefix(state, xs_b, view.block_start)
      state.xt[:, view.block_start: view.block_end] = xs_b
      self._project_prefix(
        state.xt, state.prefix_tokens, state.prefix_lengths)

      # AR verification (only for divergence/ar_metric modes)
      log_p_ar = None
      if self.arpc_corruption_mode in ('divergence', 'ar_metric'):
        if self.kv_cache:
          # Prefix is cached; pass only the block to verify
          x0_ar = state.xt[:, view.block_start: view.block_end]
        else:
          # No cache; pass full prefix + block to verify
          x0_ar = state.xt[:, :view.block_end]
        xt_ar = model.prior_sample(state.xt.shape[0], self.block_size)
        ctx_ar = BlockDiTContext(mode='ar_verify',
                                 block_size=self.block_size,
                                 temperature=self.temperature)
        log_p_ar = model.forward(x0=x0_ar, xt=xt_ar,
                                 sigma=state.ones, context=ctx_ar)
        if self.use_float64:
          log_p_ar = log_p_ar.to(torch.float64)
        state.nfe += 1

      # Corrector: re-noise uncertain positions back to alpha_s
      num_to_corrupt = max(1, round(
        (1.0 - view.alpha_s.mean().item()) * self.block_size))
      top_k_idxs = self._compute_corruption_indices(
        log_probs, log_p_ar, num_to_corrupt,
        x_current=state.xt[:, view.block_start: view.block_end],
        block_prefix_len=state.block_prefix_len)

      noisy_tokens = model.prior_sample(
        state.xt.shape[0], num_to_corrupt)
        
      state.xt[:, view.block_start: view.block_end].scatter_(
        1, top_k_idxs, noisy_tokens)
      self._project_prefix(state.xt, state.prefix_tokens, 
                           state.prefix_lengths)
      xs_b = state.xt[:, view.block_start: view.block_end]
    else:
      # Standard posterior sample (non-guided or last step)
      xs_b = _decode_ancestral_update(self, model, log_probs, 
        view.xt_b, view.alpha_s, view.alpha_t, 
        is_last_step=view.is_last_step)

    return _finalize_block_step(self, model, state, view, xs_b)


def run_sampler(sampler, model, num_samples, *,
               num_steps=None, eps=1e-5, prefix_tokens=None,
               prefix_lengths=None):
  state = sampler.init_state(model, num_samples,
                             num_steps=num_steps, eps=eps,
                             prefix_tokens=prefix_tokens,
                             prefix_lengths=prefix_lengths)
  while not state.done:
    state = sampler.step(model, state)
  return state.xt, sampler.metadata(state)


def get_sampler(config):
  s = config.sampler

  if s.predictor == 'ancestral':
    return AncestralSampler(
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      temperature=s.temperature)

  if s.predictor == 'ar':
    return ARSampler(
      use_float64=s.use_float64,
      kv_cache=s.use_kv_cache,
      greedy=s.greedy,
      early_stopping=s.early_stopping,
      temperature=s.temperature)

  if s.predictor == 'block_ancestral':
    return BlockGenAncestralSampler(
      block_size=s.block_size,
      num_steps=s.steps,
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      kv_cache=s.use_kv_cache,
      early_stopping=s.early_stopping,
      temperature=s.temperature)

  if s.predictor == 'ar_then_block':
    return ARThenBlockAncestralSampler(
      num_ar_tokens=s.num_ar_tokens,
      block_size=s.block_size,
      num_steps=s.steps,
      ar_commit_every_step=s.ar_commit_every_step,
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      kv_cache=s.use_kv_cache,
      early_stopping=s.early_stopping,
      temperature=s.temperature)

  if s.predictor == 'ar_then_arpc':
    return ARThenARPCSampler(
      num_ar_tokens=s.num_ar_tokens,
      block_size=s.block_size,
      num_steps=s.steps,
      ar_commit_every_step=s.ar_commit_every_step,
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      kv_cache=s.use_kv_cache,
      early_stopping=s.early_stopping,
      temperature=s.temperature,
      warmup_steps=s.arpc.warmup_steps,
      guide_every=s.arpc.guide_every,
      corruption_mode=s.arpc.corruption_mode,
      divergence_measure=s.arpc.divergence_measure,
      diffusion_metric=s.arpc.diffusion_metric,
      ar_metric=s.arpc.ar_metric)


  raise ValueError(f'Unknown sampler predictor: {s.predictor}')
