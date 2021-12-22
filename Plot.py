import glob
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Pass in experiments to plot
try:
    experiments = sys.argv[1:]
except:
    experiments = []

# Style
plt.style.use('bmh')
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['legend.loc'] = 'lower right'

files = glob.glob('./**/*.csv', recursive=True)

df_list = []
envs = []
tasks = []

for file in files:
    agent_experiment, env, task_seed = file.split('/')[1:]

    # Parse files
    task_seed = task_seed.split('_')
    task, seed = '_'.join(task_seed[:-1]), task_seed[-1]
    agent_experiment = agent_experiment.split('_')
    agent, experiment = agent_experiment[0], '_'.join(agent_experiment[1:])
    
    if experiment not in experiments:
        continue

    print('Plotting', agent, experiment, env, task, seed)

    csv = pd.read_csv(file)

    csv['Agent'] = agent + '_' + experiment
    csv['Environment'] = env
    csv['Task'] = task

    df_list.append(csv)
    envs.append(env)
    tasks.append(task)
    
df = pd.concat(df_list, ignore_index=True)
tasks = np.sort(tasks)

# Dynamically compute num columns
num_cols = int(np.floor(np.sqrt(len(tasks))))
while len(tasks) % num_cols != 0:
    num_cols -= 1
assert len(tasks) % num_cols == 0, f'{tasks.shape[0]} tasks, {num_cols} columns invalid'

num_rows = len(tasks) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

# Plot
for i, task in enumerate(tasks):
    data = df[df['task'] == task]
    task = ' '.join([task_name.capitalize() for task_name in task.split('_')])
    data.columns = [' '.join([name.capitalize() for name in col_name.split('_')]) for col_name in data.columns]

    row = i // num_cols
    col = i % num_cols
    ax = axs[row, col] if num_rows > 1 else axs[col] if num_cols > 1 else axs
    hue_order = np.sort(data.Agent.unique())

    sns.lineplot(x='Step', y='Reward', data=data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
    ax.set_title(f'{task}')

plt.tight_layout()
plt.savefig(f'{"_".join(experiments)}_Plot.png')
