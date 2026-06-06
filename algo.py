import torch
import collections
from dataclass_patch import dataclass

import trainer_base


@dataclass
class BlockGenTrainingContext(trainer_base.TrainingContext):
  mode: str = 'train'
  block_size: int | None = None
  # Training and validation share the default post-processing path.
  # The special post-processing modes are only used during sampling.
  post_process_mode: str = 'default'


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()
    
  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del xt, sigma, context
    model_output[:, :, self.mask_index] = self.neg_infinity
    return torch.log_softmax(model_output, dim=-1)

  def _process_sigma(self, sigma, context=None):
    return None

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step, train_mode,
          valid_tokens=None):
    del train_mode, current_accumulation_step, valid_tokens, context

    x0 = input_tokens
    output = self.forward(xt=x0, sigma=None)
    per_token_nll =  - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]
    return per_token_nll, None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.post_process_mode = config.algo.post_process_mode
    self._validate_configuration()

  def _process_model_output_log_probs(
      self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def _process_model_output_logits(self, model_output, xt, sigma):
    del sigma
    model_output[..., self.mask_index] = self.neg_infinity
    xt_unsq = xt.unsqueeze(-1)
    unmasked = (xt != self.mask_index)[..., None]
    deterministic = model_output.new_full(
      model_output.shape, self.neg_infinity)
    deterministic.scatter_(-1, xt_unsq, 0.0)
    # For masked positions, return raw logits (unnormalized).
    return torch.where(unmasked, deterministic, model_output)

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.post_process_mode not in {
      'log_probs', 'logits'}:
      raise ValueError(self.post_process_mode)

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del context
    if self.post_process_mode == 'log_probs':
      return self._process_model_output_log_probs(
        model_output, xt, sigma)
    elif self.post_process_mode == 'logits':
      return self._process_model_output_logits(
        model_output, xt, sigma)
    else:
      raise ValueError(self.post_process_mode)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del context, xt, train_mode
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    if low_var:
      loss_coefficient = -1
    else:
      loss_coefficient = dalpha_t / (1 - alpha_t)
    return loss_coefficient * log_p_theta

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    if self.post_process_mode == 'logits':
      raise ValueError(
        'post_process_mode=logits is sampling-only. '
        'Use log_probs mode for training or evaluation.')
    return super().nll(
      x0=x0,
      output_tokens=output_tokens,
      context=context,
      current_accumulation_step=current_accumulation_step,
      train_mode=train_mode,
      valid_tokens=valid_tokens)


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del xt, sigma, context
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del context, train_mode
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    if low_var:
      term1 = 0
    else:
      term1 = self.vocab_size * (1 / xbar_xt
                                 - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss


class BlockGenBase:
  def __init__(self, config, tokenizer):
    self.block_size_weights = torch.tensor([
      float(x) for x in config.algo.block_weights.split(' ')])

    if self.block_size_weights.sum().item() != 1.0:
      diff = 1 - self.block_size_weights.sum().item()
      self.block_size_weights[0] += diff

    print('Block weights:',
      [f'{x:.4f}' for x in self.block_size_weights.tolist()])
    
    # Setup special loss per block size (instead of default in loss_type)
    special = config.algo.loss_type_special_cases
    assert len(special) % 2 == 0
    self.loss_per_block_size = dict(
      zip(special[::2], special[1::2]))
    # Block sizes where we always make the input fully noise
    # Examples: at block sizes 1,2,4, we might decide to 
    #  feed pure noise.
    self.pure_noise_block_sizes = set(
      config.algo.pure_noise_block_sizes)

    self.validation_mode = config.algo.validation.mode
    self.valid_lse = 'log-sum-exp' in self.validation_mode 
    self.valid_jensen = 'jensen' in self.validation_mode
    if any((self.valid_lse, self.valid_jensen)):
      self.validation_mode_k = int(self.validation_mode.split('-')[-1])
    else:
      self.validation_mode_k = None

    self.block_size_per_gpu = config.algo.block_size_per_gpu
    self.block_size_generator = None
    if self.block_size_per_gpu == 'u-stratified':
      block_size_cumprobs = torch.cumsum(
        self.block_size_weights, dim=0)
      block_size_cumprobs = torch.cat(
        [torch.tensor([0]), block_size_cumprobs], dim=0)
      self.block_cumprobs_lo = block_size_cumprobs[:-1].reshape(-1, 1)
      self.block_cumprobs_hi = block_size_cumprobs[1:].reshape(-1, 1)
      self.u_rv = None

  def _nll_per_token_pure_diffusion(self,
    log_x_theta, xt, x0, alpha_t, dalpha_t):
    raise NotImplementedError

  def _validate_configuration(self):
    assert torch.isclose(self.block_size_weights.sum(), torch.tensor(1.0))
    assert not self.time_conditioning
    assert any((self.valid_lse, self.valid_jensen,
      self.validation_mode == 'mc1')), self.validation_mode
    if self.validation_mode_k is not None:
      assert self.validation_mode_k >= 1

  def _reset_kv_cache(self):
    self.backbone.reset_kv_cache()

  def get_block_size(self, current_accumulation_step=None):
    # Note: shortcut to avoid triggering the rand generator
    if len(self.block_size_weights) == 1:
      return 1
    # Each GPU uses the same block size
    if self.block_size_per_gpu == 'same':
      log_block_size = torch.multinomial(
      self.block_size_weights, 1).item()
    elif self.block_size_per_gpu == 'random':
      # Initialize generator if first step
      if (self.block_size_generator is None 
          and self._trainer is not None):
        self.block_size_generator = torch.Generator('cpu')
        self.block_size_generator.manual_seed(
          self.config.seed + self.trainer.global_rank)
      # Sample with generator (different per GPU)
      log_block_size = torch.multinomial(
        self.block_size_weights, 1, 
        generator=self.block_size_generator).item()
    elif self.block_size_per_gpu == 'u-stratified':
      self.config.trainer.accumulate_grad_batches
      # In case of gradient accumulation, make sure to 
      #  stratify as if there was no accumulation.
      if current_accumulation_step is not None:
        if current_accumulation_step == 0:
          self.u_rv = torch.rand(1, 1)
        u_rv = self.u_rv
      else:
        u_rv = torch.rand(1, 1)   
      lin = torch.linspace(0, 1, 
        self.config.trainer.accumulate_grad_batches 
        * self.trainer.world_size + 1)[:-1]
      u_stratified = (lin + u_rv) % 1  # (1, world_size)
      in_bounds = (u_stratified >= self.block_cumprobs_lo) \
                & (u_stratified < self.block_cumprobs_hi)
      idx_per_device = in_bounds.long().argmax(dim=0)  # (world_size,)
      idx_per_device = idx_per_device[
        torch.randperm(len(idx_per_device),
                       generator=self.block_size_generator)]
      if current_accumulation_step is None:
        idx = self.trainer.global_rank
      else:
        idx = self.trainer.world_size * current_accumulation_step \
              + self.trainer.global_rank
      log_block_size = idx_per_device[idx].item()
    else:
      raise NotImplementedError
    return 2 ** log_block_size

  def _use_pure_noise(self, train_mode, context=None):
    if context is None:
      raise ValueError('BlockGenBase requires context.')
    if train_mode:
      return context.block_size in self.pure_noise_block_sizes
    return context.block_size == 1

  def _get_loss_type(self, block_size, train_mode):
    """
    During training, we allow specifying a different objective,
      such as cross-entropy.

    During evaluation, we use the ELBO, except when the block 
      size is 1. In that case, if we trained with cross-entropy,
      which implies an autoregressive decomposition, we evaluate
      with the true NLL.
    """
    loss_type = self.loss_per_block_size.get(
        block_size, self.loss_type)
    
    if not train_mode:
      if block_size > 1:
        loss_type = 'elbo'
      else:  # block size == 1
        loss_type = 'ce'
    return loss_type
      
  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del low_var
    if context is None:
      raise ValueError('BlockGenBase.nll_per_token requires context.')
    loss_type = self._get_loss_type(context.block_size, train_mode)

    if loss_type == 'elbo':
      return self._nll_per_token_pure_diffusion(
        log_x_theta, xt, x0, alpha_t, dalpha_t)
    elif loss_type == 'ce':
      logits_clean = - torch.gather(log_x_theta, dim=-1,
                                index=x0[..., None])[..., 0]
      return logits_clean
    elif loss_type == 'ce-noisy':
      logits_clean = - torch.gather(log_x_theta, dim=-1,
                                index=x0[..., None])[..., 0]
      logits_clean = torch.where(xt == x0, 0.0,
                                 logits_clean)
      return logits_clean
    else:
      raise ValueError(self.loss_type)

  def training_step(self, batch, batch_idx):
    current_accumulation_step = (
      batch_idx % self.trainer.accumulate_grad_batches)
    block_size = self.get_block_size(current_accumulation_step)
    losses = self._loss(
      batch['input_ids'],
      batch['attention_mask'],
      BlockGenTrainingContext(block_size=block_size),
      current_accumulation_step=current_accumulation_step,
      train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def _validation_step_mc_k(self, batch, block_size, k):
    loss_sum = 0
    nlls_sum = 0

    for _ in range(k):
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          BlockGenTrainingContext(
                            block_size=block_size))
      loss_sum = loss_sum + losses.loss
      nlls_sum = nlls_sum + losses.nlls
    
    return trainer_base.Loss(
      loss=loss_sum / k,
      nlls=nlls_sum / k,
      prior_loss=0,
      num_tokens=losses.num_tokens)

  def validation_step(self, batch, batch_idx):
    del batch_idx
    if self.validation_mode == 'mc1':
      block_size = block_size = self.get_block_size()
      losses = self._validation_step_mc_k(
        batch, block_size=block_size, k=1)
    elif any((self.valid_lse, self.valid_jensen)):
      all_losses = []
      positive_weights = []
      for i, gamma_i in enumerate(self.block_size_weights):
        if gamma_i > 0:
          block_size = 2 ** i
          losses = self._validation_step_mc_k(
            batch, block_size, k=self.validation_mode_k)
          all_losses.append(losses)
          positive_weights.append(gamma_i)
      
      # Combine block-wise ELBOs
      positive_gammas = torch.tensor(positive_weights, 
                                     device=self.device)
      all_loglihoods = torch.stack([-l.nlls for l in all_losses])
      if self.valid_lse:  # tighter bound
        nll_bound = - torch.logsumexp(
          all_loglihoods + positive_gammas.log(), dim=0)
      elif self.valid_jensen:  # looser / training bound
        nll_bound = - (all_loglihoods * positive_gammas).sum(-1)
      else:
        raise ValueError('?????')

      losses = trainer_base.Loss(
        loss=nll_bound / all_losses[0].num_tokens,
        nlls=nll_bound,
        prior_loss=0,
        num_tokens=all_losses[0].num_tokens)
    else:
      raise ValueError(self.validation_mode)

    self.metrics.update_valid(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    return losses.loss


class BlockGenUniform(BlockGenBase, DUO_BASE):
  _nll_per_token_pure_diffusion = DUO_BASE.nll_per_token

  def __init__(self, config, tokenizer):
    BlockGenBase.__init__(self, config, tokenizer)
    DUO_BASE.__init__(self, config, tokenizer)


class BlockGenAbsorb(BlockGenBase, MDLM):
  _nll_per_token_pure_diffusion = MDLM.nll_per_token

  def __init__(self, config, tokenizer):
    BlockGenBase.__init__(self, config, tokenizer)
    MDLM.__init__(self, config, tokenizer)

