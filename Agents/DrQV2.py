# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import TruncatedGaussianActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DrQV2Agent(torch.nn.Module):
    """Data-Regularized Q-Network V2 (https://arxiv.org/abs/2107.09645)
    Generalizes to DQNDPGAgent in the Discrete case"""
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

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.actor = TruncatedGaussianActor(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                            discrete=discrete, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                            optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # "See"
            obs = self.encoder(obs)

            Pi = self.actor(obs, self.step)

            action = Pi.sample() if self.training \
                else Pi.mean

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps:
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

        # "Envision" / "Imagine"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # "Journal teachings"

        logs = {'time': time.time() - self.birthday,
                'step': self.step, 'episode': self.episode} if self.log \
            else None

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        # Supervised learning
        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # Supervised data
            x = self.encoder(obs)

            # Infer
            y_predicted = self.actor(x[instruction], self.step).mean

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            # Supervised loss
            supervised_loss = mistake.mean()

            if self.log:
                logs.update({'supervised_loss': supervised_loss.item()})
                logs.update({'accuracy': (torch.argmax(y_predicted, -1)
                                          == label[instruction]).float().mean().item()})

            # Update supervised
            Utils.optimize(supervised_loss,
                           self.encoder,
                           self.actor)

            # Auxiliary reinforcement
            if self.RL:
                action[instruction] = y_predicted.detach()
                reward[instruction] = -mistake[:, None].detach()
                next_obs[instruction, :] = float('nan')

        # Reinforcement learning
        if self.RL:
            # "Perceive"

            # Encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            # "Predict" / "Discern" / "Learn" / "Grow"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
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
