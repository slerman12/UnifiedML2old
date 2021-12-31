# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


def ensembleQLearning(critic, actor, obs, action, reward, discount, next_obs, step,
                      num_actions=1, priority_temp=0, Q_reduction='min', one_hot=False, exploit_schedule=1, logs=None):
    with torch.no_grad():
        has_future = ~torch.isnan(next_obs.flatten(1).sum(1))
        next_obs = next_obs[has_future]

        next_v = torch.zeros_like(discount)

        if critic.discrete:
            # All actions
            next_actions_log_probs = 0
            next_actions = None
        else:
            if actor.discrete and one_hot:
                # One-hots
                action = Utils.one_hot(action, critic.action_dim)
                next_actions = torch.eye(critic.action_dim, device=obs.device).expand(has_future.nansum(), -1, -1)
                next_actions_log_probs = 0
            else:
                # Sample actions
                if has_future.any():
                    next_Pi = actor(next_obs, step)
                    next_actions = next_Pi.rsample(num_actions)
                    next_actions_log_probs = next_Pi.log_prob(next_actions).sum(-1).flatten(1)

        if has_future.any():
            next_Q = critic.target(next_obs, next_actions)

            # How to reduce Q ensembles
            if Q_reduction == 'min':
                next_q, _ = torch.min(next_Q.Qs, 0)
            elif Q_reduction == 'sample_reduce':
                next_q = next_Q.sample() - next_Q.stddev
            elif Q_reduction == 'mean_reduce':
                next_q = next_Q.mean - next_Q.stddev
            elif Q_reduction == 'mean':
                next_q = next_Q.mean  # e.g., https://openreview.net/pdf?id=9xhgmsNVHu
            elif Q_reduction == 'sample':
                next_q = next_Q.sample()

            # Uncertainty shouldn't sway uncertainty! Confidence shouldn't compound!
            # The confidence of the explorer should be curtailed by the cynicism of the objective observer/oracle
            # u = torch.softmax((1 - temp) * next_q + temp * next_Q.stddev, -1)
            temp = 1
            exploit_factor = Utils.schedule(exploit_schedule, step)
            # next_u = exploit_factor * next_Q.mean + (1 - exploit_factor) * next_Q.stddev
            next_u = exploit_factor * next_q + (1 - exploit_factor) * next_Q.stddev  # dpg_Q_reduction, I think
            # u = exploit_factor * next_Q.sample() + (1 - exploit_factor) * next_Q.stddev
            next_u_logits = next_u - next_u.max(dim=-1, keepdim=True)[0]
            next_probs = torch.softmax(next_u_logits / temp + next_actions_log_probs, -1)
            next_v[has_future] = torch.sum(next_q * next_probs, -1, keepdim=True)

        target_q = reward + (discount * next_v)

    Q = critic(obs, action)

    # Temporal difference (TD) error (via MSE, but could also use Huber)
    td_error = F.mse_loss(Q.Qs, target_q.expand_as(Q.Qs), reduction='none')
    # td_error = Q.Qs.shape[0] * F.mse_loss(Q.Qs, target_q.expand_as(Q.Qs))  # Scales with ensemble size
    # td_error = F.mse_loss(Q.mean, target_q)  # Better since consistent with entropy? Capacity for covariance

    # Re-prioritize based on certainty e.g., https://arxiv.org/pdf/2007.04938.pdf
    td_error *= torch.sigmoid(-Q.stddev * priority_temp) + 0.5

    td_error = td_error.mean()

    if logs is not None:
        assert isinstance(logs, dict)
        logs['q_mean'] = Q.mean.mean().item()
        logs['q_stddev'] = Q.stddev.mean().item()
        logs.update({f'q{i}': q.mean().item() for i, q in enumerate(Q.Qs)})
        logs['target_q'] = target_q.mean().item()
        logs['temporal_difference_error'] = td_error.item()

    return td_error
