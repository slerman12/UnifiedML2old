# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import argparse
import json
import subprocess

from Hyperparams.task.atari.generate_atari import atari_tasks
from Hyperparams.task.dmc.generate_dmc import easy, medium, hard
agents = ['DQN', 'DrQV2', 'SPR', 'DQNDPG',
          # 'DynoSOAR', 'Ascend', 'AC2'
          ]


common_sweeps = {'atari': [f'task=atari/{task} Agent=Agents.{agent}Agent' for task in atari_tasks for agent in agents],
                 'dmc': [f'task=dmc/{task} Agent=Agents.{agent}Agent' for task in easy for agent in agents],
                 'classify': [f'task=classify/{task} Agent=Agents.{agent}Agent' for task in ['mnist', 'cifar10'] for agent in agents]}


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="job",
                    help='job name')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='uses CPUs only, not GPUs')
parser.add_argument('--lab', action='store_true', default=False,
                    help='uses csxu')
parser.add_argument('--K80', action='store_true', default=False,
                    help='uses K80 GPU')
parser.add_argument('--V100', action='store_true', default=False,
                    help='uses V100 GPU')
parser.add_argument('--A100', action='store_true', default=False,
                    help='uses A100 GPU')
parser.add_argument('--num-cpus', type=int, default=4,
                    help='how many CPUs to use')
parser.add_argument('--mem', type=int, default=25,
                    help='memory to request')
parser.add_argument('--file', type=str, default="Run.py",
                    help='file to run')
parser.add_argument('--params', type=str, default="",
                    help='params to pass into file')
parser.add_argument('--sweep_name', type=str, default="",
                    help='a common sweep to run')
args = parser.parse_args()

if len(args.sweep_name) > 0 and args.sweep_name in common_sweeps:
    args.params = common_sweeps[args.sweep_name]
elif args.params[0] == '[':
    args.params = json.loads(args.params)
else:
    args.params = [args.params]

# Sweep
for param in args.params:
    slurm_script = f"""#!/bin/bash
    #SBATCH {"-c {}".format(args.num_cpus) if args.cpu else "-p gpu -c {}".format(args.num_cpus)}
    {"" if args.cpu else "#SBATCH --gres=gpu"}
    {"#SBATCH -p csxu -A cxu22_lab" if args.cpu and args.lab else "#SBATCH -p csxu -A cxu22_lab --gres=gpu" if args.lab else ""}
    #SBATCH -t {"15-00:00:00" if args.lab else "5-00:00:00"} -o ./{args.name}.log -J {args.name}
    #SBATCH --mem={args.mem}gb 
    {"#SBATCH -C K80" if args.K80 else "#SBATCH -C V100" if args.V100 else "#SBATCH -C A100" if args.A100 else ""}
    source /scratch/slerman/miniconda/bin/activate agi
    python3 {args.file} {param}
    """

    # Write script
    with open("sbatch_script", "w") as file:
        file.write(slurm_script)

    # Launch script (with error checking / re-launching)
    success = "error"
    while "error" in success:
        try:
            success = str(subprocess.check_output(['sbatch {}'.format("sbatch_script")], shell=True))
            print(success[2:][:-3])
            if "error" in success:
                print("Errored... trying again")
        except:
            success = "error"
            if "error" in success:
                print("Errored... trying again")
    print("Success!")
