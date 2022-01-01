# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn
from torch.distributions import Normal

import Utils

from Blocks.Architectures.MLP import MLP


class EnsembleQCritic(nn.Module):
    """
    MLP-based Critic network, employs ensemble Q learning,
    returns a Normal distribution over the ensemble.
    """
    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2, l2_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):
        super().__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        repr_dim = math.prod(repr_shape)

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        in_dim = feature_dim if discrete else feature_dim + action_dim
        out_dim = action_dim if discrete else 1

        self.Q_head = Utils.Ensemble([MLP(in_dim, out_dim, hidden_dim, 2, l2_norm=l2_norm)
                                     for _ in range(ensemble_size)], 0)

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

    def update_target_params(self):
        assert hasattr(self, 'target')
        Utils.soft_update_params(self, self.target, self.target_tau)

    def forward(self, obs, action=None, context=None):
        h = self.trunk(obs)

        if context is None:
            context = torch.empty(0, device=h.device)

        # Ensemble

        if self.discrete:
            # All actions' Q-values
            Qs = self.Q_head(h, context)  # [e, b, n]

            if action is None:
                action = torch.arange(self.action_dim, device=obs.device).expand_as(Qs[0])  # [b, n]
            else:
                # Q values for a discrete action
                Qs = Utils.gather_indices(Qs, action)  # [e, b, 1]

        else:
            assert action is not None and \
                   action.shape[-1] == self.action_dim, f'action with dim={self.action_dim} needed for continuous space'

            action = action.reshape(obs.shape[0], -1, self.action_dim)  # [b, n, d]

            h = h.unsqueeze(1).expand(*action.shape[:-1], -1)

            # Q-values for continuous action(s)
            Qs = self.Q_head(h, action, context).squeeze(-1)  # [e, b, n]

        # Dist
        stddev, mean = torch.std_mean(Qs, dim=0)
        Q = Normal(mean, stddev + 1e-3)
        Q.__dict__.update({'Qs': Qs,
                           'action': action})

        return Q


class PolicyRatioCritic(EnsembleQCritic):
    """
    PRO critic, employs ensemble Q learning via policy ratio, A.K.A Proportionality,
    returns a Normal distribution over the ensemble.
    """
    def __init__(self, actor, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2, l2_norm=False,
                 discrete=False, target_tau=None, optim_lr=None):
        super().__init__(repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size * 2, l2_norm, discrete)

        PRO_heads = []

        for i in range(ensemble_size * 2, step=2):
            class PRO(nn.Module):  # TODO discrete actions
                def forward(self, obs, action, context=None):
                    M, B = super().Q_head(obs, action, context)
                    return M.abs() * actor(obs).log_prob(action) + B
            PRO_heads.append(PRO())

        self.Q_head = PRO_heads

        self.init(optim_lr, target_tau)


# class DuelingEnsembleQCritic(EnsembleQCritic):
#     """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""
#
#     def __init__(self,
#                  input_channels,
#                  output_size,
#                  pixels=30,
#                  n_atoms=51,
#                  hidden_size=256,
#                  grad_scale=2 ** (-1 / 2),
#                  noisy=0,
#                  std_init=0.1):
#         super().__init__()
#         if noisy:
#             self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
#                             NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
#                             NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
#                             NoisyLinear(hidden_size, n_atoms, std_init=std_init)
#                             ]
#         else:
#             self.linears = [nn.Linear(pixels * input_channels, hidden_size),
#                             nn.Linear(hidden_size, output_size * n_atoms),
#                             nn.Linear(pixels * input_channels, hidden_size),
#                             nn.Linear(hidden_size, n_atoms)
#                             ]
#         self.advantage_layers = [nn.Flatten(-3, -1),
#                                  self.linears[0],
#                                  nn.ReLU(),
#                                  self.linears[1]]
#         self.value_layers = [nn.Flatten(-3, -1),
#                              self.linears[2],
#                              nn.ReLU(),
#                              self.linears[3]]
#         self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
#         self.advantage_out = self.advantage_layers[3]
#         self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
#         self.value = nn.Sequential(*self.value_layers)
#         self.network = self.advantage_hidden
#         self._grad_scale = grad_scale
#         self._output_size = output_size
#         self._n_atoms = n_atoms
#
#     def forward(self, input):
#         x = scale_grad(input, self._grad_scale)
#         advantage = self.advantage(x)
#         value = self.value(x).view(-1, 1, self._n_atoms)
#         return value + (advantage - advantage.mean(dim=1, keepdim=True))
#
#     def advantage(self, input):
#         x = self.advantage_hidden(input)
#         x = self.advantage_out(x)
#         x = x.view(-1, self._output_size, self._n_atoms)
#         return x + self.advantage_bias
#
#     def reset_noise(self):
#         for module in self.linears:
#             module.reset_noise()
#
#     def set_sampling(self, sampling):
#         for module in self.linears:
#             module.sampling = sampling
