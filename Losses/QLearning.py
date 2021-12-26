# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


def ensembleQLearning(actor, critic, obs, action, reward, discount, next_obs, step,
                      num_actions=1, Q_reduction='min', one_hot=False, exploit_schedule=1, entropy_temp=0, logs=None):
    with torch.no_grad():
        if critic.discrete:
            # All actions
            next_Q = critic.target(next_obs)
            next_actions_log_probs = 0
        else:
            if actor.discrete and one_hot:
                # One-hots
                action = Utils.one_hot(action, actor.action_dim)
                next_actions = torch.eye(actor.action_dim, device=obs.device).expand(obs.shape[0], -1, -1)
                next_actions_log_probs = 0
            else:
                # Sample actions  Note: original DDPG used EMA target for this
                # next_Pi = actor.target(next_obs, step)
                next_Pi = actor(next_obs, step)
                next_actions = next_Pi.rsample(num_actions)
                sampled_next_actions = next_actions.transpose(0, 1)
                next_actions_log_probs = next_Pi.log_prob(sampled_next_actions).sum(-1).transpose(0, 1).flatten(1)
            next_Q = critic.target(next_obs, next_actions)

        # How to reduce Q ensembles
        # if Q_reduction == 'min':
        #     next_q, _ = torch.min(next_Q.Qs, 0)
        # elif Q_reduction == 'sample_reduce':
        #     next_q = next_Q.sample() - next_Q.stddev
        # elif Q_reduction == 'mean_reduce':
        #     next_q = next_Q.mean - next_Q.stddev
        # elif Q_reduction == 'mean':
        #     next_q = next_Q.mean  # e.g., https://openreview.net/pdf?id=9xhgmsNVHu
        # elif Q_reduction == 'sample':
        #     next_q = next_Q.sample()
        next_q, _ = torch.min(next_Q.Qs, 0)

        # Value V = expected Q
        # next_probs = torch.softmax(next_Pi_log_probs, -1)

        # prob = next_Pi_log_probs.exp()
        # Overconfident on what we know -- Q-value bias, which needs correction via, e.g., min-reducing,
        # but fail to evaluate the potential of the unknown, e.g., when to explore confidently into the unknown, bravely
        # In Q learning, future uncertainty is over-confidence -- confirmation bias
        # An independent judge, such as actor entropy, is better -- can be optimistic about uncertainty
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
        next_v = torch.sum(next_q * next_probs, -1, keepdim=True)

        # "Entropy maximization"
        # Future-action uncertainty maximization in reward
        # Entropy in future decisions means exploring the uncertain, the lesser-explored
        # Anxiety vs. comfort

        target_q = reward + (discount * next_v)

        # "Munchausen reward":
        # Current-action certainty maximization in reward, thereby increasing so-called "action-gap" w.r.t. above
        # Furthermore, off-policy sampling of outdated rewards might be mitigated to a degree by on-policy estimate
        # Another salient heuristic: "optimism in the face of uncertainty" (Brafman & Tennenholtz, 2002) literally
        # Equivalence / policy consistency

    Q = critic(obs, action)

    # Temporal difference (TD) error (via MSE, but could also use Huber)
    td_error = F.mse_loss(Q.Qs, target_q.expand_as(Q.Qs))
    # td_error = Q.Qs.shape[0] * F.mse_loss(Q.Qs, target_q.expand_as(Q.Qs))  # Scales with ensemble size
    # td_error = F.mse_loss(Q.mean, target_q)  # Better since consistent with entropy? Capacity for covariance

    # Judgement/humility - Pi, Q, Q_Pi entropy, and log_prob (the latter might help discovery by reducing past prob)
    # entropy = entropy_temp * Q.stddev.mean()
    entropy = entropy_temp * Q.entropy().mean()  # Can also use this in deepPolicyGradient and Creator

    if logs is not None:
        assert isinstance(logs, dict)
        logs['q_mean'] = Q.mean.mean().item()
        logs['q_stddev'] = Q.stddev.mean().item()
        logs.update({f'q{i}': q.mean().item() for i, q in enumerate(Q.Qs)})
        logs['target_q'] = target_q.mean().item()
        logs['td_error'] = td_error.item()

    return td_error - entropy
