# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy
from functools import partial

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
            mean_tanh, log_stddev = self.Pi_head(h).chunk(2, dim=-1)
            stddev = torch.exp(log_stddev)
        else:
            mean_tanh = self.Pi_head(h)
            stddev = torch.full_like(mean_tanh,
                                     Utils.schedule(self.stddev_schedule, step))

        self.mean_tanh = mean_tanh  # Pre-Tanh mean can be regularized (https://openreview.net/pdf?id=9xhgmsNVHu)
        mean = torch.tanh(self.mean_tanh)

        Pi = TruncatedNormal(mean, stddev, low=-1, high=1, stddev_clip=self.stddev_clip)

        return Pi


class CategoricalCriticActor(nn.Module):  # "Creator" for short
    def __init__(self, exploit_schedule=1):
        super().__init__()

        self.exploit_schedule = exploit_schedule

    def forward(self, Q, step=None, temp=1, sample_q=True, actions_log_prob=0):
        # Sample q or mean
        q = Q.rsample() if sample_q else Q.mean

        exploit_factor = Utils.schedule(self.exploit_schedule, step)
        u = exploit_factor * q + (1 - exploit_factor) * Q.stddev
        u_logits = u - u.max(dim=-1, keepdim=True)[0]
        Q_Pi = Categorical(logits=u_logits / temp + actions_log_prob)

        best_eps, best_ind = torch.max(u, -1)
        best_action = Utils.gather_indices(Q.action, best_ind, 1)

        sample = Q_Pi.sample

        def action_sampler(sample_shape=torch.Size()):
            i = sample(sample_shape)
            return Utils.gather_indices(Q.action, i, 1)

        Q_Pi.__dict__.update({'best': best_action,
                              'best_u': best_eps,
                              'sample_ind': sample,
                              'sample': action_sampler,
                              'Q': Q,
                              'q': q,
                              'actions': Q.action,
                              'u': u})
        return Q_Pi


class GaussianActorEnsemble(TruncatedGaussianActor):
    """"Ensembles actions output by Gaussian actors,
    returns all actor outputs unaltered, simply grouped"""
    def __init__(self, repr_shape, feature_dim, hidden_dim, action_dim, ensemble_size=2,
                 l2_norm=False, discrete=False, stddev_schedule=None,  stddev_clip=None,
                 target_tau=None, optim_lr=None):
        super().__init__(repr_shape, feature_dim, hidden_dim, action_dim, l2_norm,
                         discrete, stddev_schedule, stddev_clip)

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

    def forward(self, obs, start_action=None):
        if start_action is None:
            low, high = [torch.full((obs.shape[0], self.action_dim), i)
                         for i in (self.low, self.high)]
        else:
            low = high = start_action
            assert obs.shape[0] == start_action.shape[0]

        Pi = SGAUniform(module=lambda action: torch.min(self.critic(obs, action).Qs, 0)[0],
                        low=low, high=high, optim_lr=self.optim_lr, steps=self.steps, descent=True)
        return Pi


class MetaActor(nn.Module):
    """A simple 'Meta Actor' who contains meta parameters like temps, etc.
    and optionally returns a Gaussian"""
    def __init__(self, stddev_schedule, optim_lr=None, min=-1, max=1, **metas):
        super().__init__()

        self.stddev_schedule = stddev_schedule
        self.min, self.max = min, max

        self.meta = Utils.Meta(optim_lr, **metas)
        if optim_lr is not None:
            self.optim = self.meta.optim

    def forward(self, step=None, *names):
        stddev = None if step is None or self.stddev_schedule is None \
            else Utils.schedule(self.stddev_schedule, step)
        return self.meta(*names) if stddev is None \
            else TruncatedNormal(self.meta(*names), stddev, self.min, self.max)
