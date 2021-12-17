# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import TruncatedGaussianActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DPGAgent(torch.nn.Module):
    """Deep Policy Gradient"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log, one_hot=False  # On-boarding
                 ):
        super().__init__()

        self.discrete = discrete  # Discrete supported!
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      ensemble_size=2,
                                      target_tau=target_tau, optim_lr=lr).to(device)

        self.actor = TruncatedGaussianActor(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                            l2_norm=True, discrete=discrete,
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
            dist = self.actor(obs, self.step)

            action = dist.sample() if self.training \
                else dist.mean

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = action.uniform_(-1, 1)

            if self.discrete:
                action = torch.argmax(action, -1)

            return action

    # "Dream"
    def update(self, replay):
        logs = {'episode': self.episode, 'step': self.step} if self.log \
            else None

        # "Recollect"

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.log:
            logs['batch_reward'] = reward.mean().item()

        # "Predict" / "Discern" / "Learn" / "Grow"

        # Critic loss
        self.num_actions = 1
        self.Q_reduction = 'min'
        self.entropy_temp = 0
        critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                  obs, action, reward, discount, next_obs,
                                                  self.step, self.num_actions, self.Q_reduction,
                                                  self.entropy_temp, logs=logs)

        # Update critic
        Utils.optimize(critic_loss,
                       self.encoder,
                       self.critic)

        self.critic.update_target_params()

        # Actor loss
        self.sample_q = False
        self.exploit_temp = 1
        # self.exploit_temp = 1 - Utils.schedule(self.actor.stddev_schedule, self.step)
        actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                       self.step, self.num_actions,
                                                       self.sample_q, self.exploit_temp,
                                                       logs=logs)

        # Update actor
        Utils.optimize(actor_loss,
                       self.actor)

        return logs
