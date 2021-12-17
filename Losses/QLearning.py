# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


def ensembleQLearning(actor, critic, obs, action, reward, discount, next_obs, step,
                      num_actions=5, Q_reduction='min', entropy_temp=0, logs=None):
    """
    Notes:
        If num_actions = 1, would be great to test mean vs sample

        For Q_reduction, can also try mean minus uncertainty,
        where uncertainty is next_Q.stddev or convolved L2 distances (https://arxiv.org/pdf/2110.03375.pdf)

        Should next_probs be re-multiplied by next_Q.mean?
            ! It should be next_Q_Pi.logits
            Or rather, it can be mitigated by a temp

        Entropy can be per policy (if temp trainable), per categorical policy (Q_Pi_dist), or per Q stddev

        Consider N-step entropy... +actor(traj_o).entropy(traj_a) ... or should these be added a la Munchausen?
    """
    with torch.no_grad():
        if critic.discrete:
            # All actions
            next_Q = critic(next_obs)
            next_Pi_log_probs = 1
        else:
            if actor.discrete:
                # One-hots
                action = Utils.one_hot(action, actor.action_dim)
                next_actions = torch.eye(actor.action_dim, device=obs.device).expand(obs.shape[0], -1, -1)
                next_Pi_log_probs = 1
            else:
                # Sample actions
                next_Pi = actor(next_obs, step)
                next_actions = next_Pi.rsample(num_actions)
                next_Pi_log_probs = next_Pi.log_prob(next_actions)
            next_Q = critic(next_obs, next_actions)

        # How to reduce Q ensembles
        if Q_reduction == 'min':
            next_q, _ = torch.min(next_Q.Qs, 0)
        elif Q_reduction == 'mean':
            next_q = next_Q.mean  # e.g., https://openreview.net/pdf?id=9xhgmsNVHu

        # Value V = expected Q
        print(next_Pi_log_probs.shape, next_Q.mean.shape)
        next_probs = torch.softmax(next_Q.mean * next_Pi_log_probs, -1)
        print(next_probs.mean())
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
    td_error = 2*F.mse_loss(Q.Qs, target_q.expand_as(Q.Qs))
    # td_error = F.mse_loss(Q.mean, target_q)  # Better since consistent with entropy? Capacity for covariance

    # Judgement/humility
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
