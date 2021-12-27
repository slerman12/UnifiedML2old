# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder, ResidualBlockEncoder, SPRCNNEncoder
from Blocks.Actors import CategoricalCriticActor, GaussianActorEnsemble
from Blocks.Critics import EnsembleQCritic
from Blocks.Architectures.MLP import MLPBlock

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class SPRAgent(torch.nn.Module):
    """Self-Predictive Representations (https://arxiv.org/abs/2007.05929)
    Modifies generalized-DQN for continuous/classification support"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, device, log,  # On-boarding
                 depth=5, num_actors=2, num_actions=2  # SPR
                 ):
        super().__init__()

        self.discrete = discrete  # Continuous supported!
        self.RL = RL  # And classification...
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        self.depth, self.num_actions = depth, num_actions

        # Models
        # self.encoder = SPRCNNEncoder(obs_shape, optim_lr=lr, target_tau=target_tau).to(device)
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr, target_tau=target_tau).to(device)

        # Continuous actions creator
        self.creator = None if self.discrete \
            else GaussianActorEnsemble(self.encoder.repr_shape, feature_dim, hidden_dim, self.action_dim, num_actors,
                                       stddev_schedule=stddev_schedule, stddev_clip=stddev_clip, optim_lr=lr)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.dynamics = ResidualBlockEncoder(self.encoder.repr_shape, action_shape[-1],
                                             pixels=False, isotropic=True,
                                             optim_lr=lr).to(device)

        self.projector = MLPBlock(self.encoder.flattened_dim, hidden_dim, hidden_dim, hidden_dim,
                                  depth=2, layer_norm=True,
                                  target_tau=target_tau, optim_lr=lr).to(device)

        self.predictor = MLPBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                  depth=2, layer_norm=True,
                                  optim_lr=lr).to(device)

        self.actor = CategoricalCriticActor(stddev_schedule).to(device)

        # Data augmentation
        self.aug = torch.nn.Sequential(RandomShiftsAug(pad=4), IntensityAug(0.05))

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)

            # "Candidate actions"
            creations = None if self.discrete \
                else self.creator(obs).sample(self.num_actions)

            # DQN actor is based on critic
            Pi = self.actor(self.critic(obs, creations), self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

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
        obs = self.encoder(obs, flatten=False)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # "Journal teachings"

        logs = {'episode': self.episode, 'step': self.step} if self.log else None

        instruction = ~torch.isnan(label.flatten(1).sum(1))

        # "Acquire Wisdom"

        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # "Candidate classifications"
            creations = self.creator(obs[instruction].flatten(-3)).sample(self.num_actions)

            # Infer
            action = self.actor(self.critic(obs[instruction].flatten(-3), creations), self.step)

            mistake = cross_entropy(action, label[instruction], reduction='none')

            # Supervised loss
            supervised_loss = mistake.mean()

            if self.log:
                logs.update({'supervised_loss': supervised_loss.item()})

            # Update actor
            Utils.optimize(supervised_loss,
                           self.encoder,
                           self.creator,
                           self.critic)

            if self.RL:
                # Auxiliary reinforcement
                reward[instruction] = -mistake

        if self.RL:
            # "Predict" / "Discern" / "Plan" / "Learn" / "Grow"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.creator,
                                                      obs.flatten(-3), action, reward, discount, next_obs, self.step,
                                                      logs=logs)

            # Convert discrete action trajectories to one-hot
            if self.discrete:
                traj_a = Utils.one_hot(traj_a, num_classes=self.actor.action_dim)

            # Dynamics loss
            dynamics_loss = SelfSupervisedLearning.dynamicsLearning(obs, traj_o, traj_a, traj_r,
                                                                    self.encoder, self.dynamics, self.projector,
                                                                    self.predictor, depth=5, logs=logs)

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss,
                           self.encoder,
                           self.critic,
                           self.dynamics, self.projector, self.predictor)

            self.critic.update_target_params()

            if not self.discrete:
                # Creator loss
                actor_loss = PolicyLearning.deepPolicyGradient(self.creator, self.critic, obs.detach(),
                                                               self.step, self.num_actions, logs=logs)

                # Update creator
                Utils.optimize(actor_loss,
                               self.creator)

        return logs
