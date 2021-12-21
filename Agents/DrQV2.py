# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import TruncatedGaussianActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DrQV2Agent(torch.nn.Module):
    """Data-Regularized Q-Network V2 (https://arxiv.org/abs/2107.09645)"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, device, log,  # On-boarding
                 ):
        super().__init__()

        self.discrete = discrete  # Discrete supported!
        self.RL = RL  # And classification...
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        if discrete:
            print('Original DrQV2 does not support discrete action spaces. Instantiating generalized DPG...')

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      l2_norm=False,  # Disabled
                                      ensemble_size=2, discrete=False,  # False for now
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.actor = CategoricalCriticActor(stddev_schedule) if discrete \
            else TruncatedGaussianActor(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                        l2_norm=False,  # Disabled
                                        stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
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

            if self.discrete:
                action = torch.eye(self.actor.action_dim, device=self.device).expand(obs.shape[0], -1, -1)
                Q = self.critic(obs, action)

                Pi = self.actor(Q)
                action = torch.argmax(Pi.sample() if self.training
                                      else Pi.best, -1)
            else:
                Pi = self.actor(obs, self.step)
                action = Pi.sample()

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

        logs = {'episode': self.episode, 'step': self.step} if self.log \
            else None

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # "Predict" / "Discern" / "Learn" / "Grow"

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                  obs, action, reward, discount, next_obs,
                                                  self.step, logs=logs)

        # Update critic
        Utils.optimize(critic_loss,
                       self.encoder,
                       self.critic)

        self.critic.update_target_params()

        # Actor loss
        actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                       self.step, logs=logs)

        # Update actor
        Utils.optimize(actor_loss,
                       self.actor)

        return logs
