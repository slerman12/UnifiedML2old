import glob
import os

files = glob.glob(os.getcwd() + "/*")

for file in files:
    if 'generate_dmc' not in file:
        print(file.split('/')[-2:])
        f = open(file, "a")
        f.write(r"""
hydra:
  job:
    env_set:
      # Environment variables for MuJoCo
      MKL_SERVICE_FORCE_INTEL: '1'
      MUJOCO_GL: 'egl'""")