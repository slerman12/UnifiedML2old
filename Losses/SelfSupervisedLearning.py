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
        positive_norm = F.normalize(positive)

    # Assumes obs already encoded
    anchor = predictor(projector(obs))

    # Why not yield cosine similarity of anchor / positive - rounding error?
    # return -F.cosine_similarity(anchor, positive, -1).mean()

    anchor_norm = F.normalize(anchor)
    self_supervised_loss = - (anchor_norm * positive_norm)
    self_supervised_loss = self_supervised_loss.sum(dim=-1).mean()

    return self_supervised_loss


def dynamicsLearning(obs, traj_o, traj_a, traj_r,
                     encoder, dynamics, projector, obs_predictor=None, reward_predictor=None,
                     depth=1, logs=None):
    assert depth < traj_o.shape[1], f"depth {depth} exceeds future trajectory size of {traj_o.shape[1] - 1} steps"

    forecast = [dynamics(obs, traj_a[:, 0])]
    for k in range(1, depth):
        forecast.append(dynamics(forecast[-1], traj_a[:, k], flatten=False))
    forecast = torch.stack(forecast, 1).flatten(-3)

    dynamics_loss = 0
    for predictor, predicting in zip([obs_predictor, reward_predictor], [traj_o[:, 1:depth + 1], traj_r]):
        if predictor is not None:
            dynamics_loss -= bootstrapYourOwnLatent(forecast, predicting, encoder, projector, predictor, logs)

    if logs is not None:
        logs['dynamics_loss'] = dynamics_loss

    return dynamics_loss
