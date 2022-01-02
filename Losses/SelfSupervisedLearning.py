# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


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
                     depth=1, one_hot=False, logs=None):
    assert depth < traj_o.shape[1], f"depth {depth} exceeds future trajectory size of {traj_o.shape[1] - 1} steps"

    if traj_a.shape[-1] == 1:
        # Assumes actions are discrete, converts to one-hot
        traj_a = Utils.one_hot(traj_a, num_classes=dynamics.in_channels - obs.shape[-3])

    if one_hot:
        # Differentiable continuous to one-hot
        traj_a = Utils.rone_hot(traj_a)

    # Predict future
    forecast = [dynamics(obs, traj_a[:, 0], flatten=False, renormalize=True)]
    for k in range(1, depth):
        forecast.append(dynamics(forecast[-1], traj_a[:, k], flatten=False, renormalize=True))
    forecast = torch.stack(forecast, 1).flatten(-3)

    # Self supervision
    dynamics_loss = 0
    future = traj_o[:, 1:depth + 1]
    if obs_predictor is not None:
        dynamics_loss -= bootstrapYourOwnLatent(forecast, future, encoder, projector, obs_predictor, logs)

    if reward_predictor is not None:  # TODO redundant call to projector, maybe just use predictor
        # reward_prediction = reward_predictor(projector(forecast))
        reward_prediction = reward_predictor(forecast)
        dynamics_loss -= F.mse_loss(reward_prediction, traj_r[:, :depth])

    return dynamics_loss
