# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from Datasets.Experiences.ExperienceReplay import ExperienceReplay


class CIFAR10(ExperienceReplay):
    def __init__(self, loader, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)

        self.loader = loader

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.specs = dict(classes=classes)

        self.merging_enabled = False

    def add(self, experiences=None, store=False):
        raise Exception("add not implemented for this replay")

    def store_episode(self):
        raise Exception("store not implemented for this replay")

    def __len__(self):
        return len(self.loader)


def Cifar_10(batch_size, num_workers, **kwargs):

    def worker_init_fn(worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loading = torchvision.datasets.CIFAR10(root='./Raw',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_loading,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    test_loading = torchvision.datasets.CIFAR10(root='./Raw',
                                                train=False,
                                                download=True,
                                                transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_loading,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)

    return tuple([CIFAR10(loader, batch_size, num_workers, **kwargs)
                  for loader in [train_loader, test_loader]])
