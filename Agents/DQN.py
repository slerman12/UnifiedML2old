# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This SOURCE is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import CategoricalCriticActor, GaussianActorEnsemble
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DQNAgent(torch.nn.Module):
    """Deep Q Network
    Generalized to continuous action spaces and classification"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, device, log,  # On-boarding
                 num_actors=2, num_actions=1):  # DQN continuous
        super().__init__()

        self.discrete = discrete  # Continuous supported
        self.RL = RL  # And classification too...
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps
        self.action_dim = action_shape[-1]

        self.num_actions = num_actions  # Num actions per actor

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr)

        # Continuous actions creator
        self.creator = None if self.discrete \
            else GaussianActorEnsemble(self.encoder.repr_shape, feature_dim, hidden_dim, self.action_dim, num_actors,
                                       stddev_schedule=stddev_schedule, stddev_clip=stddev_clip, optim_lr=lr)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, self.action_dim,
                                      discrete=discrete, optim_lr=lr, target_tau=target_tau)

        self.actor = CategoricalCriticActor(stddev_schedule)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # "See"
            obs = self.encoder(obs)

            # "Candidate actions"
            creations = None if self.discrete \
                else self.creator(obs, self.step).sample(self.num_actions)

            # DQN actor is based on critic
            Pi = self.actor(self.critic(obs, creations), self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps:
                    action = torch.randint(self.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def update(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *traj, step = Utils.to_torch(
            batch, self.device)

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # "Journal teachings"

        logs = {'time': time.time() - self.birthday,
                'step': self.step, 'episode': self.episode} if self.log \
            else None

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # Supervised data
            x = self.encoder(obs)

            # "Candidate classifications"
            creations = self.creator(x[instruction], self.step).mean

            # Infer
            y_predicted = self.actor(self.critic(x[instruction], creations), self.step).best

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            # Supervised loss
            supervised_loss = mistake.mean()

            if self.log:
                logs.update({'supervised_loss': supervised_loss.item()})
                logs.update({'accuracy': (torch.argmax(y_predicted, -1)
                                          == label[instruction]).float().mean().item()})

            # Update actor
            Utils.optimize(supervised_loss,
                           self.encoder,
                           self.creator)

            # Auxiliary reinforcement
            if self.RL:
                action[instruction] = y_predicted.detach()
                reward[instruction] = -mistake[:, None].detach()
                next_obs[instruction, :] = float('nan')

        if self.RL:
            # "Visualize"

            # Encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            # "Predict" / "Discern" / "Learn" / "Grow"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.creator,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.encoder,
                           self.critic)

            self.critic.update_target_params()

            if not self.discrete:
                # Creator loss
                actor_loss = PolicyLearning.deepPolicyGradient(self.creator, self.critic, obs.detach(),
                                                               self.step, self.num_actions, logs=logs)

                # Update creator
                Utils.optimize(actor_loss,
                               self.creator)

        return logs
