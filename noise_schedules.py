"""Noise schedule definitions for discrete diffusion models."""

import abc
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


class NoiseSchedule(torch.nn.Module, abc.ABC):
  def forward(self, t):
    return self.alpha_prime_t(t), self.alpha_t(t)

  @abc.abstractmethod
  def alpha_t(self, t):
    pass

  @abc.abstractmethod
  def alpha_prime_t(self, t):
    pass


class Cosine(NoiseSchedule):
  """alpha_t = 1 - cos(pi/2 * (1 - t))"""
  def __init__(self, eps):
    super().__init__()
    self.eps = eps
    self.half_pi = torch.pi / 2

  def alpha_t(self, t):
    base_alpha = 1 - torch.cos(self.half_pi * (1 - t))
    return self.eps + (1 - self.eps) * base_alpha

  def alpha_prime_t(self, t):
    angle = self.half_pi * (1 - t)
    return -(1 - self.eps) * torch.sin(angle) * self.half_pi


class LogLinear(NoiseSchedule):
  """alpha_t = 1 - t"""
  def __init__(self, eps):
    super().__init__()
    self.eps = eps

  def alpha_t(self, t):
    base_alpha = 1 - t
    return self.eps + (1 - self.eps) * base_alpha

  def alpha_prime_t(self, t):
    return -(1 - self.eps) * torch.ones_like(t)


def get_noise(config):
  noise_config = config.noise
  if noise_config.type == 'log-linear':
    noise = LogLinear(noise_config.eps)
  elif noise_config.type == 'cosine':
    noise = Cosine(noise_config.eps)
  else:
    raise ValueError(f'Unknown noise type: {noise_config.type}')

  return noise
