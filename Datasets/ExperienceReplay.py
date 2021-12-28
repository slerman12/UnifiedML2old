# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
import datetime
import io
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np

import torch
from torch.utils.data import IterableDataset


class ExperienceReplay:
    def __init__(self, batch_size, num_workers, capacity, action_spec, save, path='.',
                 obs_spec=None, nstep=0, discount=1):

        # Episode storage

        self.store_path = Path(path + '_' + str(datetime.datetime.now()))
        self.store_path.mkdir(exist_ok=True, parents=True)

        self.num_episodes = 0
        self.num_experiences_stored = 0

        if obs_spec is None:
            obs_spec = {'name': 'obs', 'shape': (1,), 'dtype': 'float32'},

        self.specs = (obs_spec, action_spec,
                      {'name': 'label', 'shape': (1,), 'dtype': 'float32'},
                      {'name': 'reward', 'shape': (1,), 'dtype': 'float32'},
                      {'name': 'discount', 'shape': (1,), 'dtype': 'float32'},
                      {'name': 'step', 'shape': (1,), 'dtype': 'float32'},)

        self.episode = {spec['name']: [] for spec in self.specs}
        self.episode_len = 0

        self.merging_enabled = True

        # Parallelized experience loading

        self.experiences = Experiences(load_path=self.store_path,
                                       capacity=capacity // max(1, num_workers),
                                       num_workers=num_workers,
                                       fetch_every=1000,
                                       save=save,
                                       nstep=nstep,
                                       discount=discount)

        # Batch loading

        self._replay = None

        def worker_init_fn(worker_id):
            seed = np.random.get_state()[1][0] + worker_id
            np.random.seed(seed)
            random.seed(seed)

        self.batches = torch.utils.data.DataLoader(dataset=self.experiences,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    # Returns a batch of experiences
    def sample(self):
        return next(self)  # Can iterate via next

    # Allows iteration
    def __next__(self):
        return self.replay.__next__()

    # Allows iteration
    def __iter__(self):
        return self.replay.__iter__()

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)
        return self._replay

    # Tracks single episode in memory buffer
    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # Enable merging of different Experience Replays - maybe not stable
        if isinstance(experiences, ExperienceReplay):
            assert experiences.merging_enabled, 'added replay not compatible with merging'
            if self.episode_len > 0:
                assert set([spec.name for spec in experiences.specs]) == \
                       set([name for name in self.episode]), 'make sure to store before merging a disjoint replay'
            self.experiences.load_paths.append(experiences.store_path)
            self.num_episodes += experiences.num_episodes
            self.num_experiences_stored += experiences.num_experiences_stored
            experiences = [{name: experiences.episode[name][i] for name in experiences.episode}
                           for i in range(experiences.episode_len)]

        # An "episode" of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for spec in self.specs:
                # Make sure everything is a numpy batch
                if np.isscalar(exp[spec['name']]) or exp[spec['name']] is None:
                    exp[spec['name']] = np.full((1,) + spec['shape'], exp[spec['name']], spec['dtype'])
                if len(exp[spec['name']].shape) == len(spec['shape']):
                    exp[spec['name']] = np.expand_dims(exp[spec['name']], 0)

                # Validate consistency
                assert spec['shape'] == exp[spec['name']].shape[1:], 'Unexpected shape for ' + spec['name']
                assert spec['dtype'] == exp[spec['name']].dtype.name, 'Unexpected dtype for ' + spec['name']

                # Adds the experiences
                self.episode[spec['name']].append(exp[spec['name']])

        self.episode_len += len(experiences)

        if store:
            self.store_episode()  # Stores them in file system

    # Stores episode (to file in system)
    def store_episode(self):
        for spec in self.specs:
            # Concatenate into one big episode batch
            self.episode[spec['name']] = np.concatenate(self.episode[spec['name']], axis=0)

        # Expands 'step' since it has no batch length in classification
        episode_len = self.episode['observation'].shape[0]
        if self.episode['step'].shape[0] == 1:
            self.episode['step'] = np.repeat(self.episode['step'], episode_len, axis=0)

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        episode_name = f'{timestamp}_{self.num_episodes}_{self.episode_len}.npz'

        # Save episode
        save_path = self.store_path / episode_name
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **self.episode)
            buffer.seek(0)
            with save_path.open('wb') as f:
                f.write(buffer.read())

        self.num_episodes += 1
        self.num_experiences_stored += self.episode_len
        self.episode = {spec['name']: [] for spec in self.specs}
        self.episode_len = 0


# Multi-cpu workers iteratively and efficiently build batches of experience in parallel (from files)
class Experiences(IterableDataset):
    def __init__(self, load_path, capacity, num_workers, fetch_every, save=False, nstep=0, discount=1):

        # Dataset construction via parallel workers

        self.load_paths = [load_path]

        self.episode_names = []
        self.episodes = dict()

        self.num_experiences_loaded = 0
        self.capacity = capacity

        self.num_workers = max(1, num_workers)

        self.fetch_every = fetch_every
        self.samples_since_last_fetch = fetch_every

        self.save = save

        self.nstep = nstep
        self.discount = discount

    def load_episode(self, episode_name):
        try:
            with episode_name.open('rb') as episode_file:
                episode = np.load(episode_file)
                episode = {key: episode[key] for key in episode.keys()}
        except:
            return False

        episode_len = next(iter(episode.values())).shape[0] - 1

        while episode_len + self.num_experiences_loaded > self.capacity:
            early_episode_name = self.episode_names.pop(0)
            early_episode = self.episodes.pop(early_episode_name)
            early_episode_len = next(iter(early_episode.values())).shape[0] - 1
            self.num_experiences_loaded -= early_episode_len
            # Deletes early episode file
            early_episode_name.unlink(missing_ok=True)
        self.episode_names.append(episode_name)
        self.episode_names.sort()
        self.episodes[episode_name] = episode
        self.num_experiences_loaded += episode_len

        if not self.save:
            episode_name.unlink(missing_ok=True)  # Deletes file

        return True

    # Populates workers with up-to-date data
    def worker_fetch_episodes(self):
        if self.samples_since_last_fetch < self.fetch_every:
            return

        self.samples_since_last_fetch = 0

        try:
            worker = torch.utils.data.get_worker_info().id
        except:
            worker = 0

        # In case multiple Experience Replays merged
        load_path = random.choice(self.load_paths)

        episode_names = sorted(load_path.glob('*.npz'), reverse=True)  # Episodes
        num_fetched = 0
        # Find one new episode
        for episode_name in episode_names:
            episode_idx, episode_len = [int(x) for x in episode_name.stem.split('_')[1:]]
            if episode_idx % self.num_workers != worker:  # Each worker stores their own dedicated data
                continue
            if episode_name in self.episodes.keys():  # Don't store redundantly
                break
            if num_fetched + episode_len > self.capacity:  # Don't overfill
                break
            num_fetched += episode_len
            if not self.load_episode(episode_name):
                break  # Resolve conflicts

    def sample(self, episode_names, metrics=None):
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    # N-step cumulative discounted rewards
    def process(self, episode):
        episode_len = len(episode['observation'])
        idx = np.random.randint(episode_len - self.nstep)

        # Transition
        obs = episode['observation'][idx]
        action = episode['action'][idx + 1]
        next_obs = episode['observation'][idx + self.nstep]
        reward = np.full_like(episode['reward'][idx + 1], np.NaN)
        discount = np.ones_like(episode['discount'][idx + 1])
        label = episode['label'][idx].squeeze()
        step = episode['step'][idx]

        # Trajectory
        traj_o = episode['observation'][idx:idx + self.nstep + 1]
        traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
        traj_r = episode['reward'][idx + 1:idx + self.nstep + 1]
        traj_l = episode['label'][idx:idx + self.nstep + 1]

        # Compute cumulative discounted reward
        for i in range(1, self.nstep + 1):
            if episode['reward'][idx + i] != np.NaN:
                step_reward = episode['reward'][idx + i]
                if np.isnan(reward):
                    reward = np.zeros(1)
                reward += discount * step_reward
                discount *= episode['discount'][idx + i] * self.discount

        return obs, action, reward, discount, next_obs, label, traj_o, traj_a, traj_r, traj_l, step

    def fetch_sample_process(self):
        try:
            self.worker_fetch_episodes()  # Populate workers with up-to-date data
        except:
            traceback.print_exc()

        self.samples_since_last_fetch += 1

        if len(self.episode_names) > 0:
            episode_name = self.sample(self.episode_names)  # Sample an episode

            episode = self.episodes[episode_name]

            return self.process(episode)  # Process episode into a compact experience

    def __iter__(self):
        # Keep fetching, sampling, and building batches
        while True:
            yield self.fetch_sample_process()  # Yields a single experience
