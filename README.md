# Unified ML, unified in one place, ml, it's so unified (draft)

![alt text](evolve.gif)

## Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, a la carte!

**Drink up!** :beers:

## :runner: Running The Code 

To run, once you've got everything set up and installed (:grimacing:):

```
python Run.py
```

Let's get to business:

## :wrench: Setting Up The Environment 

[comment]: <> (Pretty simple:)

### 1. Clone Repo

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

*zip zap bippidy bap!* ~ (don't run that)

[comment]: <> (### *THERE, HAPPY!??*)

## :stadium: Installing The Suites 

### 1. Classify
Comes preinstalled.  :smirk:

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

### *Atari example:*
```
python Run.py task=atari/pong
```

### *MuJoCo example:* 
```
python Run.py task=dmc/humanoid_run
```

### *Classify example:* 
```
python Run.py task=classify/mnist
```

## Citing The Hard Worker(s) Who Labored For You Day And Mostly Day

For detailed documentation, check out our [**Papér**](https://arxiv.com):

```
@inproceedings{yarats2021image,
  title={bla},
  author={Sam Lerman and Chenliang Xu},
  booktitle={bla},
  year={2022},
  url={https://openreview.net}
}
```

And if you use any part of this code — even look at it, or think about it — be sure to cite!

Thank you to [Denis Yarats](https://cs.nyu.edu/~dy1042/), et. al., whose beautifully written [**DrQV2**](https://github.com/facebookresearch/drqv2) repo inspired much of this library!

Their work:

```
@inproceedings{yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```

## Note

### If you are only interested in the RL portion, 

Check out our [UnifiedRL](https:github.com/agi-init/UnifiedRL) library. 

It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.~


[comment]: <> (## License)

[comment]: <> (MIT license.)

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