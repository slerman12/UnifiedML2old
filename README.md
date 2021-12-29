

# UnifiedML

[comment]: <> (This is an original PyTorch implementation of DrQ-v2 from)

[comment]: <> ([[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning]]&#40;https://arxiv.org/abs/2107.09645&#41; by)

[comment]: <> ([Denis Yarats]&#40;https://cs.nyu.edu/~dy1042/&#41;, [Rob Fergus]&#40;https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php&#41;, [Alessandro Lazaric]&#40;http://chercheurs.lille.inria.fr/~lazaric/Webpage/Home/Home.html&#41;, and [Lerrel Pinto]&#40;https://www.lerrelpinto.com&#41;.)

[comment]: <> (<p align="center">)

[comment]: <> (  <img width="19.5%" src="https://i.imgur.com/NzY7Pyv.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/O5Va3NY.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/PCOR9Mm.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/H0ab6tz.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/sDGgRos.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/gj3qo1X.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/FFzRwFt.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/W5BKyRL.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/qwOGfRQ.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/Uubf00R.gif">)

[comment]: <> ( </p>)

[comment]: <> (## Method)

[comment]: <> (DrQ-v2 is a model-free off-policy algorithm for image-based continuous control. DrQ-v2 builds on [DrQ]&#40;https://github.com/denisyarats/drq&#41;, an actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements including:)

[comment]: <> (- Switch the base RL learner from SAC to DDPG.)

[comment]: <> (- Incorporate n-step returns to estimate TD error.)

[comment]: <> (- Introduce a decaying schedule for exploration noise.)

[comment]: <> (- Make implementation 3.5 times faster.)

[comment]: <> (- Find better hyper-parameters.)

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="https://i.imgur.com/SemY10G.png" width="100%"/>)

[comment]: <> (</p>)

[comment]: <> (These changes allow us to significantly improve sample efficiency and wall-clock training time on a set of challenging tasks from the [DeepMind Control Suite]&#40;https://github.com/deepmind/dm_control&#41; compared to prior methods. Furthermore, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL.)

[comment]: <> (<p align="center">)

[comment]: <> (  <img width="100%" src="https://imgur.com/mrS4fFA.png">)

[comment]: <> (  <img width="100%" src="https://imgur.com/pPd1ks6.png">)

[comment]: <> ( </p>)

[comment]: <> (## Citation)

[comment]: <> (If you use this repo in your research, please consider citing the paper as follows:)

[comment]: <> (```)

[comment]: <> (@article{yarats2021drqv2,)

[comment]: <> (  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},)

[comment]: <> (  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},)

[comment]: <> (  journal={arXiv preprint arXiv:2107.09645},)

[comment]: <> (  year={2021})

[comment]: <> (})

[comment]: <> (```)

[comment]: <> (Please also cite our original paper:)

[comment]: <> (```)

[comment]: <> (@inproceedings{yarats2021image,)

[comment]: <> (  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},)

[comment]: <> (  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},)

[comment]: <> (  booktitle={International Conference on Learning Representations},)

[comment]: <> (  year={2021},)

[comment]: <> (  url={https://openreview.net/forum?id=GY6-6sTvGaf})

[comment]: <> (})

[comment]: <> (```)

[comment]: <> (## Instructions)

[comment]: <> (Install [MuJoCo]&#40;http://www.mujoco.org/&#41; if it is not already the case:)

[comment]: <> (* Obtain a license on the [MuJoCo website]&#40;https://www.roboti.us/license.html&#41;.)

[comment]: <> (* Download MuJoCo binaries [here]&#40;https://www.roboti.us/index.html&#41;.)

[comment]: <> (* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.)

[comment]: <> (* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.)

[comment]: <> (* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.)

[comment]: <> (Install the following libraries:)

[comment]: <> (```sh)

[comment]: <> (sudo apt update)

[comment]: <> (sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3)

[comment]: <> (```)

[comment]: <> (Install dependencies:)

[comment]: <> (```sh)

[comment]: <> (conda env create -f Conda.yml)

[comment]: <> (conda activate drqv2)

[comment]: <> (```)

[comment]: <> (Train the agent:)

[comment]: <> (```sh)

[comment]: <> (python train.py task=quadruped_walk)

[comment]: <> (```)

[comment]: <> (Monitor results:)

[comment]: <> (```sh)

[comment]: <> (tensorboard --logdir exp_local)

[comment]: <> (```)

[comment]: <> (If you are only interested in the RL portion, check out our UnifiedRL. It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.)


[comment]: <> (If you want to run a state-of-art RL algorithm using only code meant to spoon-feed a baby, use python RunSimpleExample.py)

[comment]: <> (If you want to run classification tasks as well as RL in a unified comprehensive framework that is also joyously simple, use python Run.py)

[comment]: <> (And if you want all of the above, but also faster, distributed training -- on either multiple GPUs/CPUs OR one GPU and multiple CPUs &#40;or even just multiple CPUs... or I guess even just one CPU if you're really testy; the world's your oyster with this one&#41;, then use python RunParallel.py)

[comment]: <> (Each of the above commands has the same interface)

[comment]: <> (## License)

[comment]: <> (The majority of DrQ-v2 is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.)

[comment]: <> (## Financing)

[comment]: <> (If you have not yet, please consider donating:)

[comment]: <> ([comment]: <> &#40;[![Donate]&#40;https://img.shields.io/badge/Donate-PayPal-green.svg?style=social&#41;]&#40;https://www.paypal.com/cgi-bin/&#41;&#41;)

[comment]: <> ([![Donate]&#40;https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat&#41;]&#40;https://www.paypal.com/cgi-bin/&#41;)

[comment]: <> ([comment]: <> &#40;[![Donate]&#40;https://img.shields.io/badge/Donate-PayPal-green.svg?style=for-the-badge&#41;]&#40;https://www.paypal.com/cgi-bin/&#41;&#41;)

[comment]: <> ([comment]: <> &#40;[![Donate]&#40;https://img.shields.io/badge/PayPal-Donate-green.svg?style=for-the-badge&#41;]&#40;https://www.paypal.com/cgi-bin/&#41;&#41;)

[comment]: <> ([comment]: <> &#40;[![Donate]&#40;https://img.shields.io/badge/Give_money-yasss-green.svg?style=for-the-badge&#41;]&#40;https://www.paypal.com/cgi-bin/&#41;&#41;)

[comment]: <> ([comment]: <> &#40;[![Donate]&#40;https://img.shields.io/badge/paypal-green.svg?style=for-the-badge&#41;]&#40;https://www.paypal.com/cgi-bin/&#41;&#41;)

[comment]: <> (We are a nonprofit, single-PhD student team whose bank account is quickly hemmoraging.)

[comment]: <> (If you are an investor wishing to invest more seriously, [please contact **agi.\_\_init\_\_**]&#40;mailto:agi.init@gmail.com&#41;.)

[comment]: <> (Mark Zuckerburg, if you're looking for an heir... &#40;not joking&#41;.)