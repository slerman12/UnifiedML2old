# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from collections import namedtuple

import torch
import torch.distributions as pyd
from torch.distributions.utils import _standard_normal

import Utils


# A Gaussian Normal distribution with its standard deviation clipped
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=None, high=None, eps=1e-6, stddev_clip=None, one_hot=False):
        super().__init__(loc, scale)
        self.low, self.high = low, high
        self.eps = eps
        self.stddev_clip = stddev_clip
        self.one_hot = one_hot

    # No grad, defaults to no clip
    def sample(self, sample_shape=torch.Size([1]), to_clip=False):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, to_clip=to_clip)

    def rsample(self, sample_shape=torch.Size([1]), to_clip=True, ):
        shape = self._extended_shape(sample_shape)  # Draw multiple samples

        rand = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)  # Explore
        dev = rand * self.scale.expand(shape)  # Deviate

        if to_clip:
            dev = Utils.rclamp(dev, -self.stddev_clip, self.stddev_clip)  # Don't explore /too/ much
        x = self.loc.expand(shape) + dev

        x = x.transpose(0, len(sample_shape))  # Batch dim first

        if self.low is not None and self.high is not None:
            # Differentiable truncation
            return Utils.rclamp(x, self.low + self.eps, self.high - self.eps)

        return x


# Stochastic gradient ascent (SGA) on a uniform sample w.r.t. a network
class SGAUniform(pyd.Uniform):
    def __init__(self, module, low=0, high=1, optim_lr=0.01, steps=1, descent=False):
        super().__init__(low, high)

        self.module = module
        self.low, self.high = low, high

        self.Sampler = namedtuple('Sampler', 'optim')

        self.optim_lr = optim_lr
        self.steps = steps
        self.descent = descent

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)

        z = torch.nn.Parameter(self.low + rand * (self.high - self.low))

        utility = self.module(z)
        optim = torch.optim.Adam(z, lr=self.optim_lr)

        if self.descent:
            utility *= -1

        for _ in range(self.steps):
            Utils.optimize(-utility.mean(), self.Sampler(optim))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.module(value)
