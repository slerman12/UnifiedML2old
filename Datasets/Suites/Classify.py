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


class ClassificationEnvironment:
    def __init__(self, experiences, batch_size, num_workers):

        def worker_init_fn(worker_id):
            seed = np.random.get_state()[1][0] + worker_id
            np.random.seed(seed)
            random.seed(seed)

        self.num_classes = len(experiences.classes)
        self.action_repeat = 1

        self.batches = iter(torch.utils.data.DataLoader(dataset=experiences,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        worker_init_fn=worker_init_fn))

    @property
    def batch(self):
        try:
            batch = next(self.batches)
        except StopIteration:
            batch = next(iter(self.batches))
        return batch

    def reset(self):
        x, y = [np.array(batch) for batch in self.batch]
        time_step = ExtendedTimeStep(observation=x.squeeze(0), label=y)  # Squeezes if batch size 1
        return time_step

    def step(self, action):
        x, y = [np.array(batch) for batch in self.batch]
        time_step = ExtendedTimeStep(step_type=StepType.LAST, observation=x.squeeze(0), action=action, label=y,
                                     reward=int(y == np.argmax(action, -1)))  # Squeezes if batch size 1
        return time_step

    def observation_spec(self):
        if not hasattr(self, 'observation'):
            self.observation = np.array(self.batch[0])
        return specs.BoundedArray(self.observation.shape[1:], self.observation.dtype, 0, 255, 'observation')

    def action_spec(self):
        if not hasattr(self, 'action'):
            self.action = np.array(self.batch[1])
        return specs.BoundedArray((self.num_classes,), self.action.dtype, 0, self.num_classes - 1, 'action')


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

    env = ClassificationEnvironment(experiences, 1, num_workers)

    env = ActionSpecWrapper(env, env.action_spec().dtype, discrete=False)
    env = AugmentAttributesWrapper(env)

    return env
