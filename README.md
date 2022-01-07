[comment]: <> (# Unified ML, unified in one place, ml, it's so unified )
# Unified ML

## :open_umbrella: Unified Learning?
All agents support discrete RL, continuous control, classification, and generative modeling.

## :runner: Running The Code

To start a train session, once installed:

```
python Run.py
```

[comment]: <> (The default agent and task are DQN and Pong respectively.)

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, and videos are automatically stored in: ```./Benchmarking```.


![alt text](evolve.gif)

## Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, a la carte!

[comment]: <> (**Drink up!** :beers:)

## :wrench: Setting Up The Environment 

[comment]: <> (Pretty simple:)
Let's get to business:

### 1. Clone The Repo

```
git clone github.com/agi-init/UnifiedML
cd UnifiedML
```

### 2. Gemme Some Dependencies

```
confa env -f create Conda.yml
```

[comment]: <> (# Installing Suites)

### 3. Make sure your conda env is activated.

```
conda activate ML
```

[comment]: <> (*zip zap bippidy bap!* ~ &#40;don't run that&#41;)

[comment]: <> (### *THERE, HAPPY!??*)

## :stadium: Installing The Suites 

### 1. Classify

[comment]: <> (Comes preinstalled.  :smirk:)
Comes preinstalled. 

### 2. Atari
```
pip install autorom
AutoROM --accept-license
```
Then:
```
mkdir Atari_ROMS
AutoROM --install-dir ./Atari_ROMS
ale-import-roms ./ATARI_ROMS
```
### 3. MuJoCo
Download MuJoCo from here: https://mujoco.org/download.

Make a ```.mujoco``` folder in your home directory:

```
mkdir ~/.mujoco
```

Unrar, unzip, and move (```unrar```, ```unzip```, and ```mv```) downloaded MuJoCo version folder into ```~/.mujoco```. 

And run:
```
pip install git+https://github.com/deepmind/dm_control.git
```
***Voila.***

## :point_up: Examples

### Experiments

[comment]: <> (Plots are automatically generated during training and stored in:)

[comment]: <> (```./Benchmarking/<experiment>/Plots/```.)
The ```experiment=``` flag can differentiate a distinct experiment; you can optionally control which experiment data is automatically plotted with ```plotting.plot_experiments=```.

```
python Run.py experiment=ExpName1 "plotting.plot_experiments=['ExpName1']"
```


[comment]: <> (### *Atari example:*)

[comment]: <> (```)

[comment]: <> (python Run.py task=atari/breakout)

[comment]: <> (```)

[comment]: <> (All agents support all suites, discrete and continuous control.)

### RL

Humanoid example: 
```
python Run.py task=dmc/humanoid_run
```

DrQV2 Agent in Atari:
```
python Run.py Agent=Agents.DrQV2Agent task=atari/battlezone
```

SPR Agent in MuJoCo:
```
python Run.py Agent=Agents.SPRAgent task=dmc/humanoid_walk
```

### Classification

DQN Agent in CIFAR-10:
```
python Run.py Agent=Agents.DQNAgent task=classify/cifar10 RL=false
```

*Note:* without ```RL=false```, additional RL would augment the supervised learning by treating reward as negative error. ```RL=false``` sets training to standard supervised-only classification.

### Generative Modeling

Via the ```generate=true``` flag:
```
python Run.py task=classify/mnist generate=true
```


[comment]: <> (Also, manual plotting via ```Plot.py```:)

[comment]: <> (```)

[comment]: <> (python Plot.py <experiment1> <experiment2> <...>)

[comment]: <> (```)

[comment]: <> (```)

[comment]: <> (python Run.py task=atari/breakout generate=true)

[comment]: <> (```)


[comment]: <> (And if you use any part of this code — even look at it, or think about it — **be sure to cite the above!**)

[comment]: <> (And if you use any part of this code, **be sure to cite the above!**)


[comment]: <> (## :thinking: Details)

## Paper & Citing

For detailed documentation, check out our [[**Papér**](https://arxiv.com)].

[comment]: <> (Please see [paper]&#40;https://arxiv.com&#41; for more details.)

[comment]: <> (### How is this possible?)

[comment]: <> (**RL**: All agents implement our "Creator"/"DPG" framework to support both continuous and discrete action spaces.)

[comment]: <> (**Classification**: treated as a reinforcement learning suite called "Classify" akin to Atari or DMC, with datasets re-framed as tasks that yield labels rather than rewards.)

[comment]: <> (**Generative** modeling reframes the Actor-Critic as a Generator-Discriminator, a surprisingly simple RL-GAN unification.)

[comment]: <> (## Citing The Hard Worker Who Labored For You Day And Mostly Day)

[comment]: <> (## Citing)

```
@inproceedings{yarats2021image,
  title={bla},
  author={Sam Lerman and Chenliang Xu},
  booktitle={bla},
  year={2022},
  url={https://openreview.net}
}
```
If you use any part of this code, **be sure to cite the above!**

### Also

An acknowledgment to [[Denis Yarats](https://cs.nyu.edu/~dy1042/)], whose excellent [**DrQV2 repo**](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

```
@inproceedings{yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```

**Feel free to cite also the above!**



[comment]: <> (## Repository Structure)

[comment]: <> (Agents are self-contained in their respective ```./Agents``` file.)

[comment]: <> (```Run.py``` handles all training, evaluation, and logging.)

[comment]: <> (```./Datasets``` includes ```Environment.py```, which handles the environment "roll out," and ```ExperienceReplay.py``` which stores and retrieves data using parallel CPU workers.)

[comment]: <> (Hyper-param configurations in ```.\Hyperparams```.)

[comment]: <> (Architectures, losses, probability distributions, and simple helpers defined in ```./Blocks```, ```./Losses```, ```Distributions.py``` and ```Utils.py``` respectively.)

[comment]: <> (Files are succinct, intuitive, and try to be self-explanatory.)

## Desideratum, Pedagogy, and Research

All files are designed to be useful for educational purposes in their simplicity and structure, and research advancements/prototyping in their transparency and minimalism.

## Note

### If you are only interested in the RL portion, 

Check out our [**UnifiedRL**](https:github.com/agi-init/UnifiedRL) library. 

[comment]: <> (It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.~)

[comment]: <> (# License)

<hr class="solid">

[comment]: <> (## License)

[*MIT License Included.*](https://github.com/agi-init/UnifiedML/MIT_LICENSE)

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