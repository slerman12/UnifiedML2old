# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import sys
import warnings
from typing import MutableSequence
import glob
from pathlib import Path

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def plot(path, plot_experiments=None, plot_agents=None, plot_suites=None, plot_tasks=None,
         include_train=False):  # TODO
    include_train = False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Make sure non empty and lists, and gather names
    empty = True
    specs = [plot_experiments, plot_agents, plot_suites, plot_tasks]
    plot_name = ''
    for i, spec in enumerate(specs):
        if spec is not None:
            empty = False
            if not isinstance(spec, MutableSequence):
                specs[i] = [spec]
            # Plot name
            plot_name += "_".join(specs[i]) + '_'
    if empty:
        return

    # Style
    plt.style.use('bmh')
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.loc'] = 'lower right'

    # All CSVs from path, recursive
    csv_names = glob.glob('./Benchmarking/**/*.csv', recursive=True)

    csv_list = []
    found_suite_tasks = set()
    found_suites = set()

    # Data recollection/parsing
    for csv_name in csv_names:
        # Parse files
        experiment, agent, suite, task_seed_eval = csv_name.split('/')[2:]
        task_seed = task_seed_eval.split('_')
        task, seed, eval = '_'.join(task_seed[:-2]), task_seed[-2], task_seed[-1].replace('.csv', '')

        # Map suite names to properly-cased names
        suite = {k.lower(): k for k in ['Atari', 'DMC', 'Classify']}[suite.lower()]

        # Whether to include this CSV
        include = True

        if not include_train and eval.lower() != 'eval':
            include = False

        datums = [experiment, suite.lower(), task, agent]
        for i, spec in enumerate(specs):
            if spec is not None and datums[i]not in spec:
                include = False

        if not include:
            continue

        # Add CSV
        csv = pd.read_csv(csv_name)
        found_suite_task = task + ' (' + suite + ')'
        csv['Agent'] = agent + ' (' + experiment + ')'
        csv['Suite'] = suite
        csv['Task'] = found_suite_task

        csv_list.append(csv)
        found_suite_tasks.update({found_suite_task})
        found_suites.update({suite})

    # Non-empty check
    if len(csv_list) == 0:
        return

    df = pd.concat(csv_list, ignore_index=True)
    found_suite_tasks = np.sort(list(found_suite_tasks))

    # PLOTTING (tasks)

    # Dynamically compute num columns/rows
    num_cols = int(np.floor(np.sqrt(len(found_suite_tasks))))
    while len(found_suite_tasks) % num_cols != 0:
        num_cols -= 1
    num_rows = len(found_suite_tasks) // num_cols

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

    # Plot tasks
    for i, task in enumerate(found_suite_tasks):
        task_data = df[df['Task'] == task]

        # Capitalize column names
        task_data.columns = [' '.join([c_name.capitalize() for c_name in col_name.split('_')])
                             for col_name in task_data.columns]

        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 and num_cols > 1 else axs[col] if num_cols > 1 \
            else axs[row] if num_rows > 1 else axs
        hue_order = np.sort(task_data.Agent.unique())

        # Format title
        title = ' '.join([task_name[0].upper() + task_name[1:] for task_name in task.split('_')])

        y_axis = 'Accuracy' if 'classify' in task.lower() else 'Reward'

        sns.lineplot(x='Step', y=y_axis, data=task_data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
        ax.set_title(f'{title}')

        if 'classify' in task.lower():
            ax.set_ybound(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Eval Accuracy')

    plt.tight_layout()
    plt.savefig(path / (plot_name + 'Tasks.png'))

    plt.close()

    # PLOTTING (suites)

    num_cols = len(found_suites)

    # Create subplots
    fig, axs = plt.subplots(1, num_cols, figsize=(4 * num_cols, 3))

    # Sort suites
    found_suites = [found for s in ['Atari', 'DMC', 'Classify'] for found in found_suites if s in found]

    # Plot suites
    for col, suite in enumerate(found_suites):
        task_data = df[df['Suite'] == suite]

        # Capitalize column names
        task_data.columns = [' '.join([c_name.capitalize() for c_name in col_name.split('_')])
                             for col_name in task_data.columns]

        # Human-normalize Atari
        if suite.lower() == 'atari':
            for task in task_data.Task.unique():
                for game in random:
                    if game.lower() in task.lower():
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=SettingWithCopyWarning)

                            task_data.loc[task_data['Task'] == task, 'Reward'] -= random[game]
                            task_data.loc[task_data['Task'] == task, 'Reward'] /= human[game] - random[game]

        ax = axs[col] if num_cols > 1 else axs
        hue_order = np.sort(task_data.Agent.unique())

        y_axis = 'Accuracy' if 'classify' in suite.lower() else 'Reward'

        sns.lineplot(x='Step', y=y_axis, data=task_data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
        ax.set_title(f'{suite}')

        if suite.lower() == 'classify':
            ax.set_ybound(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Eval Accuracy')
        elif suite.lower() == 'atari':
            ax.yaxis.set_major_formatter(FuncFormatter('{:.0%}'.format))
            ax.set_ylabel('Human-Normalized Score')
        elif suite.lower() == 'dmc':
            ax.set_ybound(0, 1000)

    plt.tight_layout()
    plt.savefig(path / (plot_name + 'Suites.png'))

    plt.close()


# Atari data
random = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
}
human = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}


if __name__ == "__main__":
    # Experiments to plot
    plot_experiments = sys.argv[1:] if len(sys.argv) > 1 else 'Exp'

    path = f"./Benchmarking/{'_'.join(plot_experiments)}/Plots"

    plot(path, plot_experiments=plot_experiments)
