# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


def bootstrapYourOwnLatent(obs, positive, encoder, projector, predictor, logs=None):
    """
    Bootstrap Your Own Latent (https://arxiv.org/abs/2006.07733),
    self-supervision via EMA target
    """
    with torch.no_grad():
        positive = encoder.target(positive)
        positive = projector.target(positive)

    # Assumes obs already encoded
    anchor = predictor(projector(obs))

    self_supervised_loss = -F.cosine_similarity(anchor, positive, -1).mean()

    if logs is not None:
        logs['byol_loss'] = self_supervised_loss

    return self_supervised_loss


def dynamicsLearning(obs, traj_o, traj_a, traj_r,
                     encoder, dynamics, projector, obs_predictor=None, reward_predictor=None,
                     depth=1, logs=None):
    assert depth < traj_o.shape[1], f"depth {depth} exceeds future trajectory size of {traj_o.shape[1] - 1} steps"

    # Predict future
    forecast = [dynamics(obs, traj_a[:, 0], flatten=False)]
    for k in range(1, depth):
        forecast.append(dynamics(forecast[-1], traj_a[:, k], flatten=False))
        # "Re-normalization", as in SPR (https://arxiv.org/abs/2007.05929), or at least in their code
        # shape = forecast[-1].shape
        # forecast[-1] = forecast[-1].flatten(-3)
        # forecast[-1] -= forecast[-1].min(-1, True)[0]
        # forecast[-1] /= forecast[-1].max(-1, True)[0]
        # forecast[-1] = forecast[-1].view(*shape)
    forecast = torch.stack(forecast, 1).flatten(-3)

    # Self supervision
    dynamics_loss = 0
    future = traj_o[:, 1:depth + 1]
    if obs_predictor is not None:
        dynamics_loss -= bootstrapYourOwnLatent(forecast, future, encoder, projector, obs_predictor, logs)

    if reward_predictor is not None:  # TODO redundant call to projector
        reward_prediction = reward_predictor(projector(forecast)).squeeze(-1)
        dynamics_loss -= F.mse_loss(reward_prediction, traj_r[:, :depth])

    return dynamics_loss
