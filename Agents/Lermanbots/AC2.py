# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder, ResidualBlockEncoder
from Blocks.Actors import GaussianActorEnsemble, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic
from Blocks.Architectures.MLP import MLPBlock

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class AC2Agent(torch.nn.Module):
    # TODO CREATOR! This is DPG-Creator + Dyno
    """Actor Critic Creator (AC2) Agent"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, device, log,  # On-boarding
                 num_critics=3, num_actors=2  # AC2
                 ):
        super().__init__()

        self.discrete = discrete
        self.RL = RL
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr, target_tau=target_tau).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      num_critics, discrete=False,  # False for now
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.actor = CategoricalCriticActor(stddev_schedule) if discrete \
            else GaussianActorEnsemble(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                       num_actors, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       optim_lr=lr).to(device)

        self.dynamics = ResidualBlockEncoder(self.encoder.repr_shape, action_shape[-1],
                                             pixels=False, isotropic=True,
                                             optim_lr=lr).to(device)

        self.projector = MLPBlock(self.encoder.flattened_dim, hidden_dim, hidden_dim, hidden_dim,
                                  depth=2, layer_norm=True,
                                  target_tau=target_tau, optim_lr=lr).to(device)

        self.obs_predictor = MLPBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                      depth=2, layer_norm=True,
                                      optim_lr=lr).to(device)
        self.reward_predictor = MLPBlock(hidden_dim, 1, hidden_dim, hidden_dim,
                                         depth=2, layer_norm=True,
                                         optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)

            Pi = self.actor(obs, self.step)

            action = Pi.sample() if self.training \
                else Pi.mean

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = action.uniform_(-1, 1)

            if self.discrete:
                action = torch.argmax(action, -1)  # Since discrete is using vector representations

            return action

    # "Dream"
    def update(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *traj, step = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r, traj_l = traj

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # "Journal teachings"

        logs = {'episode': self.episode, 'step': self.step} if self.log else None

        instruction = ~torch.isnan(label.flatten(-1).sum(1))

        # "Acquire Wisdom"

        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # Infer
            action = self.actor(obs[instruction], self.step)

            mistake = cross_entropy(action, label[instruction], reduction='none')

            # Supervised loss
            supervised_loss = mistake.mean()

            if self.log:
                logs.update({'supervised_loss': supervised_loss.item()})

            # Update actor
            Utils.optimize(supervised_loss,
                           self.encoder,
                           self.actor)

            if self.RL:
                # Auxiliary reinforcement
                reward[instruction] = -mistake

        if self.RL:
            # "Predict" / "Discern" / "Learn" / "Grow"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, logs=logs)

            # Convert discrete action trajectories to one-hot
            if self.discrete:
                traj_a = Utils.one_hot(traj_a, num_classes=self.actorSAURUS.action_dim)

            future = -torch.isnan(next_obs.flatten(1).sum(1))

            # Dynamics loss
            dynamics_loss = SelfSupervisedLearning.dynamicsLearning(
                obs[future], traj_o[future], traj_a[future], traj_r[future], self.encoder, self.dynamics,
                self.projector, self.obs_predictor, self.reward_predictor, self.depth, logs
            )

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss,
                           self.encoder,
                           self.critic,
                           self.dynamics, self.projector, self.obs_predictor, self.reward_predictor)

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor)
        return logs
