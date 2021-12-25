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


class DQNDPGAgent(torch.nn.Module):
    """Deep Q Network, Deep Policy Gradient (DQN-DPG) Agent"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, device, log,  # On-boarding
                 num_actions=2, one_hot=True, bellman_Q_reduction='min', dpg_Q_reduction='min'  # DQNDPG
                 ):
        super().__init__()

        self.discrete = discrete
        self.RL = RL
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        self.num_actions, self.one_hot, self.bm_Q_reduction, self.dpg_Q_reduction = \
            num_actions, one_hot, bellman_Q_reduction, dpg_Q_reduction

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr, target_tau=target_tau).to(device)

        self.creator = GaussianActorEnsemble(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                             stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                             optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.actor = CategoricalCriticActor(stddev_schedule)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)

            # DPG "Candidate actions"
            creation = self.creator(obs, self.step).sample(self.num_actions)

            # DQN component is based on critic
            Pi = self.actor(self.critic(obs, creation), self.step)

            action = Pi.sample() if self.training else Pi.best

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

            # "Candidate actions"
            creations = self.creator(obs[instruction]).sample(self.num_actions)

            # Infer
            action = self.actor(self.critic(creations), self.step).sample()

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
                                                      self.step, Q_reduction=self.bm_Q_reduction,
                                                      one_hot=self.one_hot and self.discrete,
                                                      logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.encoder,
                           self.critic)

            self.critic.update_target_params()

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, Q_reduction=self.dpg_Q_reduction,
                                                           one_hot=self.one_hot and self.discrete,
                                                           logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.creator)
        return logs
