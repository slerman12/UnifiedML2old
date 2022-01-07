[comment]: <> (# Unified ML, unified in one place, ml, it's so unified )
# Unified ML

![alt text](evolve.gif)

## Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, a la carte!

**Drink up!** :beers:

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

Let's get to business:

## :wrench: Setting Up The Environment 

[comment]: <> (Pretty simple:)

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

## :point_up: Exampling The Examples 

[comment]: <> (### *Atari example:*)

[comment]: <> (```)

[comment]: <> (python Run.py task=atari/breakout)

[comment]: <> (```)

### *MuJoCo example:* 
```
python Run.py task=dmc/humanoid_run
```

### *Classify example:* 
```
python Run.py task=classify/mnist 
```

## :thinking: More Examples

### Plotting
Plots are automatically generated during training and stored in: 
```./Benchmarking/<experiment>/Plots/```.

```
python Run.py plot_per_steps=5000 experiment=ExpName1 "plotting.plot_experiments=['ExpName1']"
```

The ```plot_per_steps=``` flag can be used to configure the training step frequency for plotting; the ```experiment=``` flag can differentiate a distinct experiment; you can optionally control which experiment data is automatically plotted with ```plotting.plot_experiments=```.

Manual plotting via ```Plot.py```:

```
python Plot.py <experiment1> <experiment2> <...>
```


### *DQN Agent in Atari*
```
python Run.py Agent=Agents.DQNAgent task=atari/boxing
```

### *DrQV2 Agent in Atari*
```
python Run.py Agent=Agents.DrQV2Agent task=atari/battlezone
```

### *SPR Agent in MuJoCo*
```
python Run.py Agent=Agents.SPRAgent task=dmc/humanoid_walk
```

### *DQN Agent in Classification*
```
python Run.py Agent=Agents.DQNAgent task=classify/cifar10
```

### *Generative modeling using MNIST*
```
python Run.py task=classify/mnist generate=true
```

[comment]: <> (## Citing The Hard Worker Who Labored For You Day And Mostly Day)
## Citing 

For detailed documentation, check out our [[**Papér**](https://arxiv.com)].

```
@inproceedings{yarats2021image,
  title={bla},
  author={Sam Lerman and Chenliang Xu},
  booktitle={bla},
  year={2022},
  url={https://openreview.net}
}
```

And if you use any part of this code — even look at it, or think about it — **be sure to cite the above!**

## Also

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