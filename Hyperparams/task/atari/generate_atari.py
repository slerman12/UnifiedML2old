ATARI_ENVS = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]

out = ""
for task in ATARI_ENVS:
    f = open(f"./{task.lower()}.yaml", "w")
    f.write(r"""defaults:
  - 100K
  - _self_

suite: atari
action_repeat: 4
frame_stack: 3
task_name: {}""".format(task))
    f.close()
    out += ' "' + task.lower() + '"'
print(out)
