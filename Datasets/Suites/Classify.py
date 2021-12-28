# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from dm_env import specs, StepType

from Datasets.Suites._Wrappers import ActionSpecWrapper, AugmentAttributesWrapper, ExtendedTimeStep

print("Seeding replay... training has not begun yet.")


class ClassificationEnvironment:
    def __init__(self, experiences, batch_size, num_workers, train):

        def worker_init_fn(worker_id):
            seed = np.random.get_state()[1][0] + worker_id
            np.random.seed(seed)
            random.seed(seed)

        self.num_classes = len(experiences.classes)
        self.action_repeat = 1
        self.train = train

        self.batches = torch.utils.data.DataLoader(dataset=experiences,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

        self.count = 0
        self.length = len(self.batches)
        self._batches = iter(self.batches)

    @property
    def batch(self):
        self.count += 1
        try:
            batch = next(self._batches)
        except StopIteration:
            self._batches = iter(self.batches)
            batch = next(self._batches)
        if self.depleted and self.train:
            print('All data loaded; classification training env successfully "depleted."')
        return batch

    @property
    def depleted(self):
        return self.count >= self.length

    def reset(self):
        x, y = [np.array(batch, dtype='float32') for batch in self.batch]
        self.time_step = ExtendedTimeStep(observation=x, label=np.expand_dims(y, 1),
                                          step_type=StepType.FIRST, reward=0)
        return self.time_step

    def step(self, action):
        # TODO redundantly calls Agent
        # ExperienceReplay expects at least a reset state and 'next obs', with 'reward' with 'next obs'
        assert self.time_step.observation.shape[0] == action.shape[0], 'Agent must produce actions for each obs'
        self.copied = getattr(self, 'copied', False)
        if self.copied:
            self.time_step = self.time_step._replace(step_type=StepType.LAST, reward=self.reward)
        else:
            self.time_step = self.time_step._replace(step_type=StepType.MID, action=action, reward=0)
            self.reward = np.sum(self.time_step.label == np.argmax(action, -1)) / len(action)
        self.copied = not self.copied
        return self.time_step

    def observation_spec(self):
        if not hasattr(self, 'observation'):
            self.observation = np.array(self.batch[0])
        return specs.BoundedArray(self.observation.shape[1:], self.observation.dtype, 0, 255, 'observation')

    def action_spec(self):
        return specs.BoundedArray((self.num_classes,), 'float32', 0, self.num_classes - 1, 'action')


def make(task, frame_stack=4, action_repeat=4, max_episode_frames=None, truncate_episode_frames=None,
         train=True, seed=1, batch_size=1, num_workers=1):

    assert task in torchvision.datasets.__all__

    # Options:
    # torchvision.datasets.__all__ =
    # ('LSUN', 'LSUNClass',
    #  'ImageFolder', 'DatasetFolder', 'FakeData',
    #  'CocoCaptions', 'CocoDetection',
    #  'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
    #  'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
    #  'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
    #  'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
    #  'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
    #  'VisionDataset', 'USPS', 'Kinetics400', 'HMDB51', 'UCF101',
    #  'Places365')

    dataset = getattr(torchvision.datasets, task)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    experiences = dataset(root=f'./Datasets/ReplayBuffer/Classify/{task}_{"Train" if train else "Eval"}',
                          train=train,
                          download=True,
                          transform=transform)

    env = ClassificationEnvironment(experiences, batch_size if train else len(experiences), num_workers, train)

    env = ActionSpecWrapper(env, env.action_spec().dtype, discrete=False)
    env = AugmentAttributesWrapper(env, unsqueeze_batch_dim=False)

    return env
