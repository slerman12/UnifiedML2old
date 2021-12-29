# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import sys
from typing import MutableSequence
import glob
from pathlib import Path

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def plot(path='./', experiments=None, suites=None, tasks=None, agents=None):

    path_dirs = path.split('/')
    path_dir = Path('/'.join(path_dirs[:-1]).replace('Agents.', ''))
    path_dir.mkdir(parents=True, exist_ok=True)
    path = path_dir / path_dirs[-1]

    if experiments is None and suites is None and tasks is None and agents is None:
        return
    if experiments is not None and not isinstance(experiments, MutableSequence):
        experiments = [experiments]
    if suites is not None and not isinstance(suites, MutableSequence):
        suites = [suites]
    if tasks is not None and not isinstance(tasks, MutableSequence):
        tasks = [tasks]
    if agents is not None and not isinstance(agents, MutableSequence):
        agents = [agents]

    # Style
    plt.style.use('bmh')
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.loc'] = 'lower right'

    files = glob.glob('./**/*.csv', recursive=True)

    df_list = []
    suite_tasks = set()

    for file in files:
        agent_experiment, suite, task_seed_eval = file.split('/')[2:]

        # Parse files
        task_seed = task_seed_eval.split('_')
        task, seed, eval = '_'.join(task_seed[:-2]), task_seed[-2], task_seed[-1].replace('.csv', '')
        agent_experiment = agent_experiment.split('_')
        agent, experiment = agent_experiment[0], '_'.join(agent_experiment[1:])

        if 'Eval' not in eval:
            continue
        if experiments is not None and experiment not in experiments:
            continue
        if suites is not None and suite not in suites:
            continue
        if tasks is not None and task not in tasks:
            continue
        if agents is not None and agent not in agents:
            continue

        csv = pd.read_csv(file)

        suite_task = task + ' (' + suite.upper() + ')'

        csv['Agent'] = agent + ' (' + experiment + ')'
        csv['Task'] = suite_task

        df_list.append(csv)
        suite_tasks.update({suite_task})

    if len(df_list) == 0:
        return

    df = pd.concat(df_list, ignore_index=True)
    suite_tasks = np.sort(list(suite_tasks))

    # Dynamically compute num columns
    num_cols = int(np.floor(np.sqrt(len(suite_tasks))))
    while len(suite_tasks) % num_cols != 0:
        num_cols -= 1
    assert len(suite_tasks) % num_cols == 0, f'{suite_tasks.shape[0]} tasks, {num_cols} columns invalid'

    num_rows = len(suite_tasks) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

    # Plot
    for i, task in enumerate(suite_tasks):
        data = df[df['Task'] == task]
        task = ' '.join([task_name.capitalize() for task_name in task.split('_')])
        data.columns = [' '.join([name.capitalize() for name in col_name.split('_')]) for col_name in data.columns]

        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 and num_cols > 1 else axs[col] if num_cols > 1 \
            else axs[row] if num_rows > 1 else axs
        hue_order = np.sort(data.Agent.unique())

        if 'classify' in task.lower():
            y_axis = 'Eval Accuracy (%)'
            data[y_axis] = data['Reward'].copy()
            ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
        else:
            y_axis = 'Reward'

        sns.lineplot(x='Step', y=y_axis, data=data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
        ax.set_title(f'{task}')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # Experiments to plot
    experiments = sys.argv[1:] if len(sys.argv) > 1 \
        else ['Exp']

    path = f'{"_".join(experiments)}_Plot.png'

    plot(path, experiments)
