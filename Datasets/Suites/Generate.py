# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.


class Generative:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self.time_step = self.env.reset()

        return self.time_step

    def step(self, action):
        self.time_step = self.env.step(action)

        return self.time_step

    def observation_spec(self):
        return self.env.action_spec().replace(name='observation')  # TODO

    def action_spec(self):
        return self.env.observation_spec().replace(name='action')


# Simple Actor-Critic to Generator-Discriminator conversion
def to_generative(time_step):
    return time_step._replace(action=time_step.action,
                              observation=time_step.observation._uniform(),
                              reward=0)
