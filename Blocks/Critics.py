# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import statistics

import torch
from torch import nn
from torch.distributions import Normal

import Utils

from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.Residual import ResidualBlock


class EnsembleQCritic(nn.Module):
    """
    MLP-based Critic network, employs ensemble Q learning,
    e.g. DrQV2 (https://arxiv.org/abs/2107.09645).
    Outputs a Normal distribution over the ensemble.
    """

    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2, critic_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):
        super().__init__()

        repr_dim = math.prod(repr_shape)

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        in_dim = feature_dim if discrete else feature_dim + action_dim
        Q_dim = action_dim if discrete else 1

        # MLP
        self.Q_head = nn.ModuleList([MLP(in_dim=in_dim,
                                         hidden_dim=hidden_dim,
                                         out_dim=Q_dim,
                                         depth=1,
                                         l2_norm=critic_norm)
                                     for _ in range(ensemble_size)])

        self.__post__(optim_lr=optim_lr, target_tau=target_tau, repr_shape=repr_shape,
                      feature_dim=feature_dim, hidden_dim=hidden_dim, action_dim=action_dim,
                      ensemble_size=ensemble_size, critic_norm=critic_norm, discrete=discrete)

    def __post__(self, action_dim, discrete, optim_lr=None, target_tau=None, **kwargs):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if target_tau is not None:
            self.target_tau = target_tau
            target = self.__class__(action_dim=action_dim, discrete=discrete, **kwargs)
            target.load_state_dict(self.state_dict())
            self.target = target

        self.discrete = discrete
        self.action_dim = action_dim

    def update_target_params(self):
        assert self.target_tau is not None
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, action=None, context=None):
        h = self.trunk(obs)

        if context is None:
            context = torch.empty(0, device=h.device)

        # Ensemble
        if self.discrete:
            # All actions' Q-values
            Qs = tuple(Q_net(h, context) for Q_net in self.Q_head)  # [b, n]

            if action is None:
                action = torch.arange(self.action_dim).expand_as(Qs[0])  # [b, n]
            else:
                # Q values for a discrete action
                Qs = tuple(Utils.gather_index(Q, action) for Q in Qs)  # [b, 1]
        else:
            assert action is not None and \
                   action.shape[-1] == self.action_dim, f'action with dim={self.action_dim} needed for continuous space'

            action = action.view(obs.shape[0], -1, self.action_dim)  # [b, n, d]

            shape = action.shape[:-1]  # Preserve leading dims
            h = h.unsqueeze(1).expand(*shape, -1).flatten(end_dim=1)
            action = action.flatten(end_dim=1)

            # Q-values for continuous action(s)
            Qs = tuple(Q_net(h, action, context).view(*shape) for Q_net in self.Q_head)  # [b, n]

        # Dist
        Q = Normal(statistics.mean(*Qs), statistics.stdev(*Qs))
        Q.__dict__.update({'Qs': Qs,
                           'action': action})

        return Q


class CNNEnsembleQCritic(EnsembleQCritic):
    """
    CNN-based Critic network, employs ensemble Q learning,
    e.g. Efficient-Zero (https://arxiv.org/pdf/2111.00210.pdf) (except with ensembling).
    """

    def __init__(self, repr_shape, hidden_channels, out_channels, num_blocks,
                 hidden_dim, action_dim, ensemble_size=2, critic_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):
        super().__init__((1,), 1, 1, 0, 0, target_tau=None, optim_lr=None)  # Unused parent MLP

        in_channels, height, width = repr_shape

        # CNN
        self.trunk = nn.Sequential(*[ResidualBlock(in_channels, hidden_channels)
                                     for _ in range(num_blocks)],
                                   nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Flatten())

        # CNN dimensions
        trunk_h, trunk_w = Utils.cnn_output_shape(height, width, self.trunk)
        feature_dim = out_channels * trunk_h * trunk_w

        # MLP dimensions
        in_dim = feature_dim if discrete else feature_dim + action_dim
        Q_dim = action_dim if discrete else 1

        # MLP
        self.Q_head = nn.ModuleList([MLP(in_dim, Q_dim, hidden_dim, 2)
                                     for _ in range(ensemble_size)])

        self.__post__(optim_lr=optim_lr, target_tau=target_tau, repr_shape=repr_shape,
                      hidden_channels=hidden_channels, out_channels=out_channels, num_blocks=num_blocks,
                      hidden_dim=hidden_dim, action_dim=action_dim, ensemble_size=ensemble_size,
                      critic_norm=critic_norm, discrete=discrete)
