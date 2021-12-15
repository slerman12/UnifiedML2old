# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils


def deepPolicyGradient(actor, critic, obs, step, num_actions=5, sample_q=True, exploit_schedule=1, Pi=None, logs=None):
    if Pi is None:
        Pi = actor(obs, step)

    actions = Pi.rsample(num_actions)
    Pi_log_probs = Pi.log_prob(actions)

    Q = critic(obs, actions)

    # Sample q or mean
    q = Q.rsample() if sample_q else Q.mean

    # Exploitation-exploration tradeoff
    exploit_factor = 1 - Utils.schedule(actor.stddev_schedule if exploit_schedule is None
                                        else exploit_schedule, step)
    u = exploit_factor * q + (1 - exploit_factor) * Q.stddev

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
        logs['avg_Pi_probs'] = Pi_log_probs.exp().mean().item()
        logs['avg_u'] = u.mean().item()

    return exploit_explore_loss
