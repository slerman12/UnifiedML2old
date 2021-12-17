# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch


def deepPolicyGradient(actor, critic, obs, step, num_actions=5, sample_q=True, exploit_temp=1, logs=None):
    Pi = actor(obs, step)

    actions = Pi.rsample(num_actions)

    Q = critic(obs, actions)

    # Sample q or mean
    q = Q.rsample() if sample_q else Q.mean

    # Exploitation-exploration tradeoff
    u = exploit_temp * q + (1 - exploit_temp) * Q.stddev

    exploit_explore_loss = -torch.mean(u)

    # "Entropy maximization"
    # Future-action uncertainty maximization in reward
    # Entropy in future decisions means exploring the uncertain, the lesser-explored

    # "Entropy maximization"
    # Entropy - 'aleatory' - uncertainty - randomness in decision-making
    # - keeps exploration active, gradients tractable

    # "Trust region optimization"
    # Policies that change too rapidly per batch are unstable, so we try to bound their temperament a little
    # ...within a "trust region", ideally one that keeps large gradients from propelling params beyond local optima

    if logs is not None:
        assert isinstance(logs, dict)
        logs['exploit_explore_loss'] = exploit_explore_loss.item()
        logs['avg_Q_stddev'] = Q.stddev.mean().item()
        logs['avg_Pi_prob'] = Pi.log_prob(actions).exp().mean().item()
        logs['avg_u'] = u.mean().item()

    return exploit_explore_loss
