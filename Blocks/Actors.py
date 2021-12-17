# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn
from torch.distributions import Categorical

from Distributions import TruncatedNormal, SGAUniform

import Utils

from Blocks.Architectures.MLP import MLP


class TruncatedGaussianActor(nn.Module):
    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, l2_norm=False,
                 discrete=False, stddev_schedule=None,  stddev_clip=None,
                 target_tau=None, optim_lr=None):
        super().__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        repr_dim = math.prod(repr_shape)

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        out_dim = action_dim * 2 if stddev_schedule is None else action_dim

        self.Pi_head = MLP(feature_dim, out_dim, hidden_dim, 2, l2_norm=l2_norm)

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

    def forward(self, obs, step=None):
        h = self.trunk(obs)

        if self.stddev_schedule is None or step is None:
            raw_mean, log_stddev = self.Pi_head(h).chunk(2, dim=-1)
            stddev = torch.exp(log_stddev)
        else:
            raw_mean = self.Pi_head(h)
            stddev = torch.full_like(raw_mean,
                                     Utils.schedule(self.stddev_schedule, step))

        self.raw_mean = raw_mean
        mean = torch.tanh(raw_mean)

        Pi = TruncatedNormal(mean, stddev, low=-1, high=1, stddev_clip=self.stddev_clip)

        return Pi


class CategoricalCriticActor(nn.Module):  # "Creator" for short
    def __init__(self, exploit_schedule=1):
        super().__init__()

        self.exploit_schedule = exploit_schedule

    def forward(self, Q, step=None, temp=1, sample_q=True):
        # Sample q or mean
        q = Q.rsample() if sample_q else Q.mean

        exploit_factor = Utils.schedule(self.exploit_schedule, step)
        u = exploit_factor * q + (1 - exploit_factor) * Q.stddev

        logits = (u - u.max(dim=-1, keepdim=True)[0])
        Q_Pi = Categorical(logits=logits / temp)

        best_eps, best_ind = torch.max(u, -1)
        best_action = Utils.gather_indices(Q.action, best_ind, 1)

        def action_sampler(sample_shape):
            i = Q_Pi.sample(sample_shape)
            return Utils.gather_indices(Q.action, i, 1)

        Q_Pi.__dict__.update({'best': best_action,
                              'best_u': best_eps,
                              'sample_ind': Q_Pi.sample,
                              'sample': action_sampler,
                              'Q': Q,
                              'q': q,
                              'actions': Q.action,
                              'u': u})
        return Q_Pi


class GaussianActorEnsemble(TruncatedGaussianActor):
    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2,
                 l2_norm=False, discrete=False, stddev_schedule=None,  stddev_clip=None,
                 target_tau=None, optim_lr=None):
        super().__init__(repr_shape, feature_dim, hidden_dim, action_dim, l2_norm,
                         discrete, stddev_schedule, stddev_clip, target_tau, optim_lr)

        out_dim = action_dim * 2 if stddev_schedule is None else action_dim

        self.Pi_head = Utils.Ensemble([MLP(feature_dim, out_dim, hidden_dim, 2, l2_norm=l2_norm)
                                       for _ in range(ensemble_size)])

        self.init(optim_lr, target_tau)


class SGDActor(nn.Module):
    def __init__(self, critic, low=-1, high=1, lr=0.01, steps=1):
        super().__init__()

        self.critic = critic
        self.low, self.high = low, high
        self.optim_lr = lr
        self.steps = steps

        self.Q_reduction = lambda Qs: torch.min(Qs, 0)[0]

    def forward(self, obs, start_action=None):
        if start_action is None:
            low, high = [torch.full((obs.shape[0], self.action_dim), i)
                         for i in (self.low, self.high)]
        else:
            low = high = start_action
            assert obs.shape[0] == start_action.shape[0]

        Pi = SGAUniform(module=lambda action: self.Q_reduction(self.critic(obs, action)),
                        low=low, high=high, optim_lr=self.optim_lr, steps=self.steps)
        return Pi
