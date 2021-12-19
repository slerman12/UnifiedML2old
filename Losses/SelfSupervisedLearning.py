# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


def bootstrapYourOwnLatent(encoder, critic, predictor, anchor, positive, contrastive=False):
    """
    Bootstrap Your Own Latent (https://arxiv.org/abs/2006.07733),
    self-supervision via EMA target
    """
    with torch.no_grad():
        positive = encoder.target(positive)
        positive = F.normalize(critic.target.trunk[0](positive))  # Kind of presumptive to use Critic as projection

    anchor = encoder(anchor)  # Redundant, can just pass in obs/concept
    anchor = F.normalize(predictor(critic.trunk[0](anchor)))  # Can just yield cosine similarity of anchor / positive

    if contrastive:
        # Contrastive predictive coding (https://arxiv.org/pdf/1807.03748.pdf)
        self_supervised_loss = 0
        pass  # TODO use negative samples via uncorrelated batch samples
    else:
        # Bootstrap Your Own Latent (https://arxiv.org/abs/2006.07733)
        self_supervised_loss = - (anchor * positive.detach())
        self_supervised_loss = self_supervised_loss.sum(dim=-1).mean()

    return self_supervised_loss


def dynamicsLearning(dynamics, projection_g, prediction_q, encoder, traj_o, traj_a, depth=1, cheaper=False, logs=None):
    assert depth < traj_o.shape[1], f"depth {depth} exceeds future trajectory size of {traj_o.shape[1] - 1} time steps"

    with torch.no_grad():
        if cheaper:
            traj_o_target = encoder.target(traj_o[:, 1:depth + 1])
            projections = projection_g.target(traj_o_target.flatten(-3))
        else:
            traj_o_target = encoder.target(traj_o[:, 1:])
            projections = projection_g.target(traj_o_target.flatten(-3))

    forecasts = encoder(traj_o[:, 0]) if cheaper else encoder(traj_o)
    dynamics_loss = 0
    for k in range(depth):
        if cheaper:
            forecasts = dynamics(forecasts, traj_a[:, k])
        else:
            forecasts = dynamics(forecasts[:, :-1], traj_a[:, k:])
        projected_forecasts = projection_g(forecasts.flatten(-3))
        predictions = prediction_q(projected_forecasts)

        if cheaper:
            dynamics_loss -= F.cosine_similarity(predictions, projections[:, k], -1).mean()
        else:
            dynamics_loss -= F.cosine_similarity(predictions, projections[:, k:], -1).mean()

    if logs is not None:
        logs['dynamics_loss'] = dynamics_loss

    return dynamics_loss


class DynamicsLoss(torch.nn.Module):
    def __init__(self, encoder, dynamics, projector, state_predictor=None, reward_predictor=None, aug=None):
        super().__init__()

        self.encoder = encoder
        self.dynamics = dynamics
        self.projector = projector
        self.state_predictor = state_predictor
        self.reward_predictor = reward_predictor
        self.aug = aug

    def forward(self, obs, traj_o, traj_a=None, traj_r=None, depth=1, logs=None):
        assert depth < traj_o.shape[1], f"depth {depth} exceeds future trajectory size of {traj_o.shape[1] - 1} steps"

        with torch.no_grad():
            traj_o_next = traj_o[:, 1:depth + 1]
            traj_o_next = self.projector.target(self.encoder.target(traj_o_next))

        forecast = [self.dynamics(obs, traj_a[:, 0])]
        for k in range(1, depth):
            forecast.append(self.dynamics(forecast[-1], traj_a[:, k]))
        forecast = torch.stack(forecast, 1).flatten(-3)
        forecast = self.projector(forecast)

        dynamics_loss = 0
        if self.state_predictor is not None:
            dynamics_loss -= F.cosine_similarity(self.state_predictor(forecast),
                                                 traj_o_next, -1).mean()
        if self.reward_predictor is not None:
            dynamics_loss -= F.cosine_similarity(self.reward_predictor(forecast),
                                                 traj_r, -1).mean()

        if logs is not None:
            logs['dynamics_loss'] = dynamics_loss

        return dynamics_loss
