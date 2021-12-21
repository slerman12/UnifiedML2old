# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn

import Utils

from Blocks.Architectures.Residual import ResidualBlock, Residual


class CNNEncoder(nn.Module):
    """
    Basic CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645).
    """

    def __init__(self, obs_shape, out_channels=32, depth=3, pixels=True,
                 optim_lr=None, target_tau=None):

        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        in_channels = obs_shape[0]
        self.out_channels = out_channels

        self.obs_shape = obs_shape
        self.pixels = pixels

        # CNN
        self.CNN = nn.Sequential(*sum([(nn.Conv2d(in_channels if i == 0 else out_channels,
                                                  out_channels, 3, stride=2 if i == 0 else 1),
                                        nn.ReLU())
                                       for i in range(depth + 1)], ()))

        # Initialize model
        self.init(optim_lr, target_tau)

    def init(self, optim_lr=None, target_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if target_tau is not None:
            self.target = copy.deepcopy(self)
            self.target_tau = target_tau

        # Dimensions
        _, height, width = self.obs_shape
        height, width = Utils.cnn_output_shape(height, width, self.CNN)

        self.repr_shape = (self.out_channels, height, width)  # Feature map shape
        self.flattened_dim = math.prod(self.repr_shape)  # Flattened features dim

    def update_target_params(self):
        assert hasattr(self, 'target')
        Utils.soft_update_params(self, self.target, self.target_tau)

    # Encodes
    def forward(self, obs, *context, flatten=True):
        obs_shape = obs.shape  # Preserve leading dims
        assert obs_shape[-3:] == self.obs_shape
        obs = obs.flatten(0, -4)  # Encode last 3 dims

        # Normalizes pixels
        if self.pixels:
            obs = obs / 255.0 - 0.5

        # Optionally append context to channels assuming dimensions allow
        context = [c.reshape(obs.shape[0], c.shape[-1], 1, 1).expand(-1, -1, *self.obs_shape[1:])
                   for c in context]
        obs = torch.cat([obs, *context], 1)

        # CNN encode
        h = self.CNN(obs)

        h = h.view(*obs_shape[:-3], *h.shape[-3:])
        if tuple(h.shape[-3:]) != self.repr_shape:
            print(tuple(h.shape[-3:]), self.repr_shape)
        assert tuple(h.shape[-3:]) == self.repr_shape

        if flatten:
            return h.flatten(-3)
        return h


class ResidualBlockEncoder(CNNEncoder):
    """
    Residual block-based CNN encoder,
    Isotropic means no bottleneck / dimensionality conserving
    e.g., Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf).
    """

    def __init__(self, obs_shape, context_dim=0, out_channels=64, num_blocks=1, pixels=True, pre_residual=False, isotropic=False,
                 optim_lr=None, target_tau=None):

        super().__init__(obs_shape, out_channels, 0, pixels)

        # Dimensions
        in_channels = obs_shape[0] + context_dim
        out_channels = in_channels if isotropic else out_channels

        pre = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                      kernel_size=3, stride=1,
                                      # padding='same' if isotropic else 1, bias=False),
                                      padding=1, bias=False),
                            nn.BatchNorm2d(out_channels))

        if pre_residual:
            pre = Residual(pre)

        # CNN
        self.CNN = nn.Sequential(pre,
                                 nn.ReLU(),
                                 *[ResidualBlock(out_channels, out_channels)
                                   for _ in range(num_blocks)])

        self.init(optim_lr, target_tau)

        # Isotropic
        if isotropic:
            assert obs_shape[-2] == self.repr_shape[1]
            assert obs_shape[-1] == self.repr_shape[2]
