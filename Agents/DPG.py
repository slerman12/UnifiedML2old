# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.distributions import Categorical

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import TruncatedGaussianActor, CategoricalCriticActor
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

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      l2_norm=False,
                                      ensemble_size=2,
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.actor = TruncatedGaussianActor(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                            l2_norm=False, discrete=discrete,
                                            stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                            optim_lr=lr).to(device)

        self.creator = CategoricalCriticActor(exploit_schedule=stddev_schedule)

        self.num_actions = 1
        self.sample_q = False

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)

            if self.discrete:
                actions = torch.eye(self.actor.action_dim, device=self.device).expand(obs.shape[0], -1, -1)
            else:
                Pi = self.actor(obs, self.step)
                # actions = Pi.sample(self.num_actions) if self.num_actions > 1 \
                #     else Pi.mean
                actions = Pi.sample(self.num_actions)

            Q = self.critic(obs, actions)

            # temp = Utils.schedule(self.actor.stddev_schedule, self.step)

            # log_prob = Pi.log_prob(actions).sum(-1, keepdim=True)
            # q = Q.sample() if self.sample_q else Q.mean
            # u = (1 - temp) * q + temp * Q.stddev
            # Q_Pi = Categorical(u * log_prob / temp, -1)

            # prob = Pi.log_prob(Q.actions).sum(-1, True).exp()
            # u = torch.softmax((1 - temp) * Q.sample() + temp * Q.stddev, -1)
            # Q_Pi = Categorical((u + prob) / 2 / temp, -1)

            # def creator(Q, Pi, meta):
            #     prob = Pi.log_prob(Q.actions).sum(-1, True).exp()
            #     u = torch.softmax(meta.opportunism * Q.sample() + (1 - meta.opportunism) * Q.stddev, -1)
            #     Q_Pi = Categorical((u + prob) / 2 / meta.discovery, -1)
            #     return Q_Pi

            actions_log_prob = Pi.log_prob(Q.actions).sum(-1, True)
            temp = 1
            exploit_factor = Utils.schedule(self.exploit_schedule, self.step)
            q = Q.sample() if self.sample_q else Q.mean
            u = exploit_factor * q + (1 - exploit_factor) * Q.stddev
            u_logits = u - u.max(dim=-1, keepdim=True)[0]
            Q_Pi = Categorical(logits=u_logits / temp + actions_log_prob)

            # actions_log_prob = Pi.log_prob(Q.actions).sum(-1, True)
            # u = meta.exploit_factor * Q.sample() + (1 - meta.exploit_factor) * Q.stddev  # Q.mean
            # u_logits = u - u.max(dim=-1, keepdim=True)[0]
            # Q_Pi = Categorical(logits=u_logits / meta.temp + actions_log_prob)

            action = Utils.gather_indices(Q.action, Q_Pi.sample() if self.training else torch.argmax(u, -1), 1)

            # Q_Pi = self.creator(self.critic(obs, actions), self.step, temp, sample_q, actions_log_prob)
            # action = Q_Pi.sample() if self.training else Q_Pi.best

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = action.uniform_(-1, 1)

            if self.discrete:
                action = torch.argmax(action, -1)

            return action.squeeze(1)

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
        # self.exploit_temp = 1 - Utils.schedule(self.actor.stddev_schedule, self.step)
        actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                       self.step, self.num_actions, self.Q_reduction,
                                                       self.exploit_temp, logs=logs)

        # Update actor
        Utils.optimize(actor_loss,
                       self.actor)

        return logs
