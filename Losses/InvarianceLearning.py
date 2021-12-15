import torch.nn.functional as F


# Q-value augmentation invariance  TODO use Koila (https://github.com/rentruewang/koila/issues/17)
def rQdia(
        critic, obs, obs_orig, action,
        distribution='replay',  # TODO 'policy', pass in actor, equalize expected Value; can also try 'uniform' random
        n_scaling=1, m_scaling=1  # ∈ (0, 1], lower = more efficient
):
    # rQdia (Regularizing Q-Value Distributions With Image Augmentation)

    batch_size = action.shape[0]  # |B|

    num_states = max(1, round(batch_size * n_scaling))  # n
    num_actions = max(1, round(batch_size * m_scaling))  # m

    obs = obs[:num_states]
    obs_orig = obs_orig[:num_states]
    action = action[:num_actions]

    if critic.discrete:
        # Q dists
        obs_Q1_dist, obs_Q2_dist = critic(obs)
        obs_orig_Q1_dist, obs_orig_Q2_dist = critic(obs_orig)
    else:
        obs_dim = obs.shape[1]
        action_dim = action.shape[1]

        obs_orig_pairs = obs_orig.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, obs_dim)  # s^(i)
        obs_pairs = obs.unsqueeze(1).expand(-1, num_actions, -1).reshape(obs_orig_pairs.shape)  # aug(s^(i))
        action_pairs = action.unsqueeze(0).expand(num_states, -1, -1).reshape(-1, action_dim)  # a^(j)

        # Q dists
        obs_orig_Q1_dist, obs_orig_Q2_dist = critic(obs_orig_pairs, action_pairs)  # Q(s^(i), a^(j))
        # TODO use critic.target / no_grad
        obs_Q1_dist, obs_Q2_dist = critic(obs_pairs, action_pairs)  # Q(aug(s^(i)), a^(j))

    # TODO Weigh by detached Q-value distributions
    rQdia_loss = F.mse_loss(obs_orig_Q1_dist, obs_Q1_dist) + F.mse_loss(obs_orig_Q2_dist, obs_Q2_dist)

    return rQdia_loss
