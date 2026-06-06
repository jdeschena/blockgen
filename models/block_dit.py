from functools import partial, cache

import flash_attn
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import (
  TimestepEmbedder,
  EmbeddingLayer,
  DDiTFinalLayer,
  DDiTBlock,
)

try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except (ImportError, RuntimeError):
  flex_attention = None  # type: ignore
  create_block_mask = None  # type: ignore
  FLEX_ATTN_AVAILABLE = False


if FLEX_ATTN_AVAILABLE:
  _flex_attn = torch.compile(
    flex_attention, mode='max-autotune-no-cudagraphs', dynamic=False)



import torch._inductor.config as inductor_cfg
inductor_cfg.triton.cudagraphs = False
inductor_cfg.coordinate_descent_tuning = True
torch._dynamo.config.cache_size_limit = 1024

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


#################################################################################
#                            Attention Mask Functions                           #
#################################################################################

def training_block_diff_mask(b, h, q_idx, kv_idx, *, 
                             block_size, seq_len, x0_causal):
  """Block diffusion attention mask for training.

  Sequence layout: [x0 (seq_len), xt (seq_len)]
    xt -> xt:  block-diagonal (each xt block attends to itself only)
    xt -> x0:  offset-block-causal (xt block i attends to x0 blocks 0..i-1)
    x0 -> x0:  token-causal if x0_causal else block-causal
  """
  del b, h
  x0_q  = q_idx  < seq_len
  x0_kv = kv_idx < seq_len
  block_q  = torch.where(x0_q,  q_idx  // block_size, (q_idx  - seq_len) // block_size)
  block_kv = torch.where(x0_kv, kv_idx // block_size, (kv_idx - seq_len) // block_size)

  xt_to_xt = (block_q == block_kv) & ~x0_q & ~x0_kv
  xt_to_x0 = (block_q >  block_kv) & ~x0_q &  x0_kv
  if x0_causal:
    x0_to_x0 = (q_idx >= kv_idx) & x0_q & x0_kv
  else:
    x0_to_x0 = (block_q >= block_kv) & x0_q & x0_kv
  return xt_to_xt | xt_to_x0 | x0_to_x0


def block_generation_mask(b, h, q_idx, kv_idx, *, ctx_len, 
                          xt_len, block_size, x0_causal):
  """Attention mask for block-by-block generation.

  Sequence layout: [x0_prefix (ctx_len), xt (xt_len)]
    xt -> x0:  all allowed (condition on clean prefix)
    xt -> xt:  block-diagonal
    x0 -> x0:  token-causal if x0_causal else block-causal
    x0 -> xt:  not allowed
  """
  del b, h, xt_len  # xt_len is used by callers to set Q_LEN/KV_LEN, not needed here
  x0_q  = q_idx  < ctx_len
  x0_kv = kv_idx < ctx_len

  xt_to_x0 = ~x0_q & x0_kv
  xt_to_xt = (((q_idx - ctx_len) // block_size) == ((kv_idx - ctx_len) // block_size)) & ~x0_q & ~x0_kv
  if x0_causal:
    x0_to_x0 = (q_idx >= kv_idx) & x0_q & x0_kv
  else:
    x0_to_x0 = (q_idx // block_size >= kv_idx // block_size) & x0_q & x0_kv
  return xt_to_x0 | xt_to_xt | x0_to_x0


def ar_verification_mask(b, h, q_idx, kv_idx, *, ctx_len, 
                         gen_len):
  """Attention mask for AR verification.

  Sequence layout: [x0_clean (ctx_len), x0_generated (gen_len), mask_tokens (gen_len)]
    clean -> clean:  token-causal
    gen   -> clean:  all allowed
    gen   -> gen:    token-causal
    mask  -> clean:  all allowed
    mask  -> gen:    offset-causal (mask[i] sees gen[0..i-1])
    mask  -> mask:   self-attention only (diagonal)
  """
  del b, h
  gen_start  = ctx_len
  mask_start = ctx_len + gen_len

  q_clean = q_idx  < ctx_len
  q_gen   = (q_idx  >= gen_start)  & (q_idx  < mask_start)
  q_mask  =  q_idx  >= mask_start
  kv_clean = kv_idx < ctx_len
  kv_gen   = (kv_idx >= gen_start)  & (kv_idx < mask_start)
  kv_mask  =  kv_idx >= mask_start

  clean_to_clean = (q_idx >= kv_idx)          & q_clean & kv_clean
  gen_to_clean   =                               q_gen   & kv_clean
  gen_to_gen     = (q_idx >= kv_idx)          & q_gen   & kv_gen
  mask_to_clean  =                               q_mask  & kv_clean
  mask_to_gen    = (kv_idx < q_idx - gen_len) & q_mask  & kv_gen
  mask_to_mask   = (q_idx == kv_idx)          & q_mask  & kv_mask

  return clean_to_clean | gen_to_clean | gen_to_gen | mask_to_clean | mask_to_gen | mask_to_mask


def _block_generation_mask_kv_cache(b, h, q_idx, kv_idx, *, 
  ctx_len, xt_len, block_size, x0_causal, q_start):
  """KV-cache variant of block_generation_mask: q_idx is 
     offset by q_start."""
  return block_generation_mask(b, h, q_idx + q_start, kv_idx,
                                ctx_len=ctx_len, xt_len=xt_len,
                                block_size=block_size, x0_causal=x0_causal)


def _ar_verification_mask_kv_cache(b, h, q_idx, kv_idx, *, ctx_len, gen_len):
  """KV-cache variant of ar_verification_mask: queries start 
     right after cached ctx."""
  return ar_verification_mask(b, h, q_idx + ctx_len, kv_idx,
                               ctx_len=ctx_len, gen_len=gen_len)


def _training_flex_mask(seq_len: int, block_size: int, x0_causal: bool):
  assert FLEX_ATTN_AVAILABLE, 'flex_attention not available'
  return create_block_mask(
    partial(training_block_diff_mask,
            block_size=block_size, seq_len=seq_len, x0_causal=x0_causal),
    B=None, H=None, Q_LEN=seq_len * 2, KV_LEN=seq_len * 2)


def _training_sdpa_mask(seq_len: int, block_size: int, 
                        x0_causal: bool):
  L = seq_len * 2
  return training_block_diff_mask(b=None, h=None,
    q_idx=torch.arange(L)[:, None], 
    kv_idx=torch.arange(L)[None, :], block_size=block_size, 
    seq_len=seq_len, x0_causal=x0_causal)


def _ar_verify_flex_mask(ctx_len: int, gen_len: int):
  assert FLEX_ATTN_AVAILABLE, 'flex_attention not available'
  L = ctx_len + 2 * gen_len
  return create_block_mask(
    partial(ar_verification_mask, ctx_len=ctx_len, 
    gen_len=gen_len), B=None, H=None, Q_LEN=L, KV_LEN=L)


def _ar_verify_sdpa_mask(ctx_len: int, gen_len: int):
  L = ctx_len + 2 * gen_len
  return ar_verification_mask(b=None, h=None,
    q_idx=torch.arange(L)[:, None], 
    kv_idx=torch.arange(L)[None, :], ctx_len=ctx_len, 
    gen_len=gen_len)


# Build the actual (q, k, v) -> out callable.
def _sdpa_with_mask(q, k, v, *, mask):
  return F.scaled_dot_product_attention(q, k, v, 
    attn_mask=mask.to(q.device))

@cache
def _training_kernel(backend: str, seq_len: int, 
                     block_size: int, x0_causal: bool):
  if backend == 'flex':
    return partial(
      _flex_attn, block_mask=_training_flex_mask(seq_len, 
        block_size, x0_causal))
  else:
    return partial(_sdpa_with_mask,
    mask=_training_sdpa_mask(seq_len, block_size, x0_causal))


@cache
def _ar_verify_kernel(backend: str, ctx_len: int, 
                      gen_len: int):
  if backend == 'flex':
    return partial(
      _flex_attn, block_mask=_ar_verify_flex_mask(ctx_len, 
        gen_len))
  else:
    return partial(
      _sdpa_with_mask, mask=_ar_verify_sdpa_mask(ctx_len, 
        gen_len))

@cache
def _generate_kv_cache_kernel(backend: str, ctx_len: int, 
  xt_len: int, block_size: int, x0_causal: bool, 
  q_start: int):
  """Cached kernel for KV-cache generation.

  ctx_len = total clean prefix length (cached + new).
  q_start = ctx_cached_len (absolute position where queries begin).
  Queries cover [x0_new (ctx_len - q_start), xt (xt_len)]; KV covers the full sequence.
  When q_start == 0 this is equivalent to the full-sequence generation kernel.
  """
  assert FLEX_ATTN_AVAILABLE or backend != 'flex'
  q_len  = ctx_len - q_start + xt_len
  kv_len = ctx_len + xt_len
  fn = partial(_block_generation_mask_kv_cache,
    ctx_len=ctx_len, xt_len=xt_len, block_size=block_size, 
    x0_causal=x0_causal, q_start=q_start)
  if backend == 'flex':
    return partial(
      _flex_attn, block_mask=create_block_mask(fn, B=None, 
        H=None, Q_LEN=q_len, KV_LEN=kv_len))
  else:
    mask = fn(b=None, h=None,
              q_idx=torch.arange(q_len)[:, None],
              kv_idx=torch.arange(kv_len)[None, :])
    return partial(_sdpa_with_mask, mask=mask)

@cache
def _ar_verify_kv_cache_kernel(backend: str, ctx_len: int, gen_len: int):
  """Cached kernel for KV-cache AR verification.

  Queries are [x0_generated, mask_tokens] (2*gen_len tokens) starting at ctx_len.
  KV covers [cached_ctx (ctx_len), x0_generated, mask_tokens].
  """
  assert FLEX_ATTN_AVAILABLE or backend != 'flex'
  q_len  = 2 * gen_len
  kv_len = ctx_len + 2 * gen_len
  fn = partial(_ar_verification_mask_kv_cache, 
    ctx_len=ctx_len, gen_len=gen_len)
  if backend == 'flex':
    return partial(
      _flex_attn, block_mask=create_block_mask(
      fn, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len))
  else:
    mask = fn(b=None, h=None,
              q_idx=torch.arange(q_len)[:, None],
              kv_idx=torch.arange(kv_len)[None, :])
    return partial(_sdpa_with_mask, mask=mask)


def flash_attention_causal(q, k, v):
  """Full-sequence causal attention via flash_attn. q/k/v: [B, H, S, D]."""
  out = flash_attn.flash_attn_func(
    q.transpose(1, 2).contiguous(),
    k.transpose(1, 2).contiguous(),
    v.transpose(1, 2).contiguous(),
    causal=True)
  return out.transpose(1, 2)


class Rotary(torch.nn.Module):
  def __init__(self, dim, max_seq_len: int, base=10_000):
    super().__init__()
    self.max_seq_len = max_seq_len
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    cos, sin = self._init_freq()
    self.register_buffer('cos_cached', cos, persistent=False)
    self.register_buffer('sin_cached', sin, persistent=False)

  def _init_freq(self):
    t = torch.arange(self.max_seq_len, 
      device=self.inv_freq.device).type_as(self.inv_freq)
    freqs = torch.einsum('i,j->ij', t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    cos[:, :, 2, :, :].fill_(1.)
    sin[:, :, 2, :, :].fill_(0.)
    # Doubled to cover the [x0, xt] concatenation used during training.
    cos = torch.cat([cos, cos], dim=1)
    sin = torch.cat([sin, sin], dim=1)
    return cos, sin

  def forward(self, seq_len: int):
    """Return (cos, sin) for the first seq_len positions."""
    if seq_len > 2 * self.max_seq_len:
      raise ValueError(
        f'seq_len={seq_len} exceeds Rotary capacity {2 * self.max_seq_len}')
    return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]

  def forward_with_positions(self, positions: torch.Tensor):
    """Return (cos, sin) for explicit position indices."""
    return self.cos_cached[:, positions], self.sin_cached[:, positions]


class BlockDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.vocab_size = vocab_size
    self.length = config.model.length
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.attn_backend = config.model.attn_backend
    self.x0_causal = config.model.x0_causal
    self.adaLN = config.algo.adaLN

    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    self.rotary_emb = Rotary(dim // config.model.n_heads, 
                             max_seq_len=self.length)
    if self.adaLN:
      self.sigma_map = TimestepEmbedder(cond_dim)

    self.blocks = nn.ModuleList([
      DDiTBlock(
        dim=dim,
        n_heads=config.model.n_heads,
        cond_dim=cond_dim,
        adaLN=self.adaLN,
        dropout=config.model.dropout)
      for _ in range(config.model.n_blocks)])

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN)
    # Number of clean tokens currently committed to the KV cache.
    # Cached tensors live inside each DDiTBlock.
    self.ctx_cached_len = 0
    self.use_kv_cache = getattr(config.sampler, 
                                'use_kv_cache', True)

  def _cond(self, sigma):
    if self.adaLN:
      return F.silu(self.sigma_map(sigma))
    return None

  def _fwd_blocks(self, x, rotary_cos_sin, c, attn_kernel,
                  cache_commit_len=None):
    for block in self.blocks:
      x = block(x, rotary_cos_sin, c=c, attn_kernel=attn_kernel,
                cache_commit_len=cache_commit_len)
    return x

  def reset_kv_cache(self):
    self.ctx_cached_len = 0
    for block in self.blocks:
      block.reset_kv_cache()

  def forward(self, x0, xt, sigma, context):
    """
    train:     x0, xt [B, L] token ids -> logits [B, L, V]
    generate:  x0 = new clean prefix [B, n_ctx_new], 
               xt = noisy block [B, xt_len] 
                  -> logits [B, xt_len, V]
    ar_verify: x0 = context/generated tokens, 
               xt = mask tokens [B, n_gen]
                  -> logits [B, n_gen, V]
    """
    if context.mode == 'train':
      return self.forward_train(sigma, x0, xt, 
        context.block_size)
    x_cat = torch.cat([x0, xt], dim=1)
    if context.mode == 'generate':
      return self.forward_generate(sigma, x_cat, 
        n_ctx_new=x0.shape[1], block_size=context.block_size)
    if context.mode == 'ar_verify':
      return self.forward_ar_verify(sigma, x_cat, 
        n_gen=xt.shape[1])
    raise ValueError(f'Unknown mode: {context.mode!r}')

  def forward_train(self, sigma, x0, xt, block_size):
    """Training forward.

    x0/xt: clean/noisy token ids [B, L]. 
           Returns logits for xt, shape [B, L, V].
    """
    assert x0.shape == xt.shape
    seq_len = x0.shape[1]
    x_cat = torch.cat([self.vocab_embed(x0), 
                       self.vocab_embed(xt)], dim=1)
    rotary_cos_sin = self.rotary_emb(2 * seq_len)
    attn_kernel = _training_kernel(self.attn_backend, seq_len, 
                                   block_size, self.x0_causal)
    c = self._cond(sigma)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      x_cat = self._fwd_blocks(x_cat, rotary_cos_sin, c, 
                               attn_kernel)
      return self.output_layer(x_cat[:, seq_len:], c)

  def forward_generate(self, sigma, x_cat, n_ctx_new, block_size):
    """x_cat = [x0_new (n_ctx_new), xt]. Returns logits [B, xt_len, V].

    If use_kv_cache: commits n_ctx_new tokens to cache and uses
    cached prefix for attention (incremental mode).
    Otherwise: full self-attention over x_cat, no cache interaction.
    Both modes produce identical outputs.
    """
    xt_len = x_cat.shape[1] - n_ctx_new

    if self.use_kv_cache:
      ctx_len = self.ctx_cached_len + n_ctx_new
      q_start = self.ctx_cached_len
    else:
      ctx_len = n_ctx_new
      q_start = 0

    if n_ctx_new == 0 and block_size == 1:
      attn_kernel = flash_attention_causal
    else:
      attn_kernel = _generate_kv_cache_kernel(
        self.attn_backend, ctx_len, xt_len, block_size,
        self.x0_causal, q_start=q_start)

    x_cat = self.vocab_embed(x_cat)
    positions = torch.arange(q_start, q_start + n_ctx_new + xt_len,
                             device=x_cat.device)
    rotary_cos_sin = self.rotary_emb.forward_with_positions(positions)
    c = self._cond(sigma)

    if self.use_kv_cache:
      commit_len = n_ctx_new if (n_ctx_new > 0 or self.ctx_cached_len > 0) else None
    else:
      commit_len = None  # plain _attn, no cache interaction

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      x_cat = self._fwd_blocks(x_cat, rotary_cos_sin, c,
                               attn_kernel, cache_commit_len=commit_len)
      logits = self.output_layer(x_cat[:, n_ctx_new:], c)
    if self.use_kv_cache and n_ctx_new > 0:
      self.ctx_cached_len += n_ctx_new
    return logits

  def forward_ar_verify(self, sigma, x_cat, n_gen):
    """x_cat = [clean, gen, mask] (no cache) or [gen, mask]
       (cache hot). Returns logits [B, n_gen, V]."""
    if self.ctx_cached_len > 0:
      ctx_len = self.ctx_cached_len
      attn_kernel = _ar_verify_kv_cache_kernel(
        self.attn_backend, ctx_len, n_gen)
      positions = torch.arange(ctx_len, ctx_len + n_gen,
                               device=x_cat.device)
      rotary_cos_sin = self.rotary_emb.forward_with_positions(
        torch.cat([positions, positions]))
      cache_commit_len = 0
      logit_start = n_gen
    else:
      ctx_len = x_cat.shape[1] - 2 * n_gen
      attn_kernel = _ar_verify_kernel(self.attn_backend,
                                      ctx_len, n_gen)
      positions = torch.arange(ctx_len + n_gen,
                               device=x_cat.device)
      rotary_cos_sin = self.rotary_emb.forward_with_positions(
        torch.cat([positions, positions[ctx_len:]]))
      cache_commit_len = None
      logit_start = ctx_len + n_gen

    x_cat = self.vocab_embed(x_cat)
    c = self._cond(sigma)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      x_cat = self._fwd_blocks(x_cat, rotary_cos_sin, c,
        attn_kernel, cache_commit_len=cache_commit_len)
      return self.output_layer(x_cat[:, logit_start:], c)


# ──────────────────────────────────────────────────────────────────
# KV-cache correctness tests
#
# Run with:  python -m models.block_dit
#
# Each test compares forward_generate / forward_ar_verify outputs
# produced by the incremental KV-cache path against the no-cache
# full-prefix path.  They must be bit-exact (or near-exact in bf16).
# ──────────────────────────────────────────────────────────────────
def _kv_cache_tests():
  import sys
  import omegaconf

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cpu':
    print('SKIP: KV-cache tests require CUDA (model uses autocast cuda).')
    return

  torch.manual_seed(0)
  B          = 2        # batch
  block_size = 4
  vocab_size = 50
  atol       = 5e-2     # bf16 tolerance; bit-exact expected but allow tiny rounding

  cfg = omegaconf.OmegaConf.create({
    'model': {
      'hidden_size':  64,
      'cond_dim':     64,
      'n_heads':      4,
      'n_blocks':     2,
      'dropout':      0.0,   # deterministic
      'attn_backend': 'sdpa',
      'x0_causal':    True,
      'length':       64,
    },
    'algo':    {'adaLN': False},
    'sampler': {'use_kv_cache': True},
  })

  model = BlockDiT(cfg, vocab_size=vocab_size).to(device).eval()
  sigma = torch.full((B,), 0.5, device=device)

  # Fixed random tokens (reused across tests)
  x0_b0  = torch.randint(0, vocab_size, (B, block_size), device=device)
  x0_b1  = torch.randint(0, vocab_size, (B, block_size), device=device)
  xt_b0  = torch.randint(0, vocab_size, (B, block_size), device=device)
  xt_b1a = torch.randint(0, vocab_size, (B, block_size), device=device)  # step-0 noisy
  xt_b1b = torch.randint(0, vocab_size, (B, block_size), device=device)  # step-1+ noisy (posterior-updated)
  xt_b2  = torch.randint(0, vocab_size, (B, block_size), device=device)
  mask   = torch.randint(0, vocab_size, (B, block_size), device=device)  # for ar_verify

  failures = []
  def check(name, a, b):
    diff = (a.float() - b.float()).abs().max().item()
    ok   = diff < atol
    tag  = 'PASS' if ok else 'FAIL'
    print(f'  {tag}  {name}  (max_diff={diff:.2e})')
    if not ok:
      failures.append(name)

  # ── Test 1 ──────────────────────────────────────────────────────
  # Block 0, no prefix: n_ctx_new=0, cache empty.
  # Both paths use _attn (commit_len=None) → must be identical.
  print('\nTest 1 – block 0, no prefix (n_ctx_new=0, empty cache)')
  model.use_kv_cache = False;  model.reset_kv_cache()
  out_nc = model.forward_generate(sigma, xt_b0, n_ctx_new=0, block_size=block_size)

  model.use_kv_cache = True;   model.reset_kv_cache()
  out_c  = model.forward_generate(sigma, xt_b0, n_ctx_new=0, block_size=block_size)
  check('no_cache vs cache', out_nc, out_c)

  # ── Test 2 ──────────────────────────────────────────────────────
  # Block 1, step 0: same x_cat passed to both; cache is empty so
  # _attn_kv_cache (with empty cache) == _attn.  Commits x0_b0.
  print('\nTest 2 – block 1, step 0 (n_ctx_new=block_size, empty cache)')
  x_cat_s0 = torch.cat([x0_b0, xt_b1a], dim=1)

  model.use_kv_cache = False;  model.reset_kv_cache()
  out_nc = model.forward_generate(sigma, x_cat_s0, n_ctx_new=block_size, block_size=block_size)

  model.use_kv_cache = True;   model.reset_kv_cache()
  out_c  = model.forward_generate(sigma, x_cat_s0, n_ctx_new=block_size, block_size=block_size)
  check('no_cache vs cache', out_nc, out_c)
  assert model.ctx_cached_len == block_size, f'Expected ctx_cached_len={block_size}'

  # ── Test 3 ──────────────────────────────────────────────────────
  # Block 1, step 1+: THE CRITICAL TEST.
  # After step 0 committed x0_b0, the posterior updates the noisy tokens
  # to xt_b1b.  Next model call:
  #   cache path  → x_cat = xt_b1b only  (n_ctx_new=0, ctx_cached_len=block_size)
  #   no-cache    → x_cat = [x0_b0, xt_b1b]  (n_ctx_new=block_size)
  # Logits for xt_b1b must agree.
  print('\nTest 3 – block 1, step 1+ (incremental cache vs full-prefix no-cache)')

  model.use_kv_cache = False;  model.reset_kv_cache()
  x_cat_full = torch.cat([x0_b0, xt_b1b], dim=1)
  out_nc = model.forward_generate(sigma, x_cat_full, n_ctx_new=block_size, block_size=block_size)

  # Build cache: run step 0 (commits x0_b0)
  model.use_kv_cache = True;   model.reset_kv_cache()
  _  = model.forward_generate(sigma, x_cat_s0, n_ctx_new=block_size, block_size=block_size)
  assert model.ctx_cached_len == block_size
  # Step 1+: incremental – only the (updated) noisy block
  out_c = model.forward_generate(sigma, xt_b1b, n_ctx_new=0, block_size=block_size)
  check('no_cache vs incremental cache', out_nc, out_c)

  # ── Test 4 ──────────────────────────────────────────────────────
  # Three-block chain: block 2, step 1+ (two blocks cached).
  print('\nTest 4 – block 2, step 1+ (two-block cached prefix vs full no-cache)')

  x0_full_2 = torch.cat([x0_b0, x0_b1], dim=1)

  model.use_kv_cache = False;  model.reset_kv_cache()
  x_cat_b2  = torch.cat([x0_full_2, xt_b2], dim=1)
  out_nc = model.forward_generate(sigma, x_cat_b2, n_ctx_new=2*block_size, block_size=block_size)

  # Build cache incrementally
  model.use_kv_cache = True;   model.reset_kv_cache()
  # Block 0 step (no commit – empty x0)
  _ = model.forward_generate(sigma, xt_b0, n_ctx_new=0, block_size=block_size)
  assert model.ctx_cached_len == 0
  # Block 1 step 0: commits x0_b0
  _ = model.forward_generate(sigma, torch.cat([x0_b0, xt_b1a], dim=1),
                              n_ctx_new=block_size, block_size=block_size)
  assert model.ctx_cached_len == block_size
  # Block 2 step 0: commits x0_b1
  _ = model.forward_generate(sigma, torch.cat([x0_b1, xt_b2], dim=1),
                              n_ctx_new=block_size, block_size=block_size)
  assert model.ctx_cached_len == 2 * block_size
  # Block 2 step 1+: incremental
  out_c = model.forward_generate(sigma, xt_b2, n_ctx_new=0, block_size=block_size)
  check('no_cache vs incremental cache', out_nc, out_c)

  # ── Test 5 ──────────────────────────────────────────────────────
  # forward_ar_verify: cached prefix (ctx_cached_len=block_size) vs
  # full prefix in x_cat (ctx_cached_len=0).
  # Scenario: AR prefix = x0_b0 (already cached); verify x0_b1 with
  # mask tokens.
  print('\nTest 5 – forward_ar_verify: cached prefix vs full-prefix in x_cat')
  sigma_ones = torch.ones(B, device=device)

  # No-cache: x_cat = [x0_b0 (ctx), x0_b1 (gen), mask]
  model.use_kv_cache = False;  model.reset_kv_cache()
  x_cat_verify = torch.cat([x0_b0, x0_b1, mask], dim=1)
  out_nc = model.forward_ar_verify(sigma_ones, x_cat_verify, n_gen=block_size)

  # Cached: commit x0_b0 first via a generate call, then ar_verify
  # with x_cat = [x0_b1, mask] only.
  model.use_kv_cache = True;   model.reset_kv_cache()
  # Commit x0_b0 by running a generate step (n_ctx_new=block_size, empty xt)
  dummy_xt = torch.randint(0, vocab_size, (B, block_size), device=device)
  _ = model.forward_generate(sigma, torch.cat([x0_b0, dummy_xt], dim=1),
                              n_ctx_new=block_size, block_size=block_size)
  assert model.ctx_cached_len == block_size
  x_cat_verify_c = torch.cat([x0_b1, mask], dim=1)
  out_c = model.forward_ar_verify(sigma_ones, x_cat_verify_c, n_gen=block_size)
  check('no_cache vs cached prefix', out_nc, out_c)

  # ── Summary ─────────────────────────────────────────────────────
  print()
  if failures:
    print(f'FAILED: {failures}')
    sys.exit(1)
  else:
    print('All KV-cache tests passed.')


if __name__ == '__main__':
  _kv_cache_tests()

