# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Sets all Torch and Numpy random seeds
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Saves Torch objects to root
def save(path, **to_save):
    save_path = path.replace('Agents.', '')
    with save_path.open('wb') as f:
        torch.save(to_save, f)


# Loads Torch objects from root
def load(path, *keys):
    save_path = path.replace('Agents.', '')
    print(f'resuming: {save_path}')
    with save_path.open('rb') as f:
        loaded = torch.load(f)
    return tuple(loaded[k] for k in keys)


# Initializes model weights according to common distributions
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# Copies parameters from one model to another, with optional EMA
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


# Compute the output shape of a CNN layer
def conv_output_shape(in_height, in_width, kernel_size=1, stride=1, padding=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(padding) is not tuple:
        padding = (padding, padding)
    out_height = math.floor(((in_height + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    out_width = math.floor(((in_width + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
    return out_height, out_width


# Compute the output shape of a whole CNN
def cnn_output_shape(height, width, block):
    if isinstance(block, (nn.Conv2d, nn.AvgPool2d)):
        height, width = conv_output_shape(height, width,
                                          kernel_size=block.kernel_size,
                                          stride=block.stride,
                                          padding=block.padding)
    elif hasattr(block, 'output_shape'):
        height, width = block.output_shape(height, width)
    elif hasattr(block, 'modules'):
        for module in block.children():
            height, width = cnn_output_shape(height, width, module)

    output_shape = (height, width)  # TODO should probably do (width, height) universally

    return output_shape


# "Ensembles" (stacks) multiple modules' outputs
class Ensemble(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *x):
        return torch.stack([module(*x) for module in self.ensemble],
                           self.dim)


# Merges multiple critics into one if so desired (ensembles of ensembles)
class MergeCritics(nn.Module):
    def __init__(self, *critics):
        super().__init__()
        self.critics = critics

    def forward(self, obs, action=None, context=None):
        Q = [critic(obs, action, context) for critic in self.critics]
        Qs = torch.cat([Q_.Qs for Q_ in Q], 0)
        # Dist
        stddev, mean = torch.std_mean(Qs, dim=0)
        merged_Q = Normal(mean, stddev + 1e-12)
        merged_Q.__dict__.update({'Qs': Qs,
                                  'action': Q[0].action})
        return merged_Q


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes):
    assert x.shape[-1] == 1
    # x = x.squeeze(-1).unsqueeze(-1)  # Or this
    x = x.long()
    shape = x.shape[:-1]
    zeros = torch.zeros(*shape, num_classes, dtype=x.dtype, device=x.device)
    return zeros.scatter(len(shape), x, 1).float()


# Differentiable one_hot
def rone_hot(x):
    return x - (x - one_hot(torch.argmax(x, -1, keepdim=True), x.shape[-1]))


# Differentiable clamp
def rclamp(x, min, max):
    return x - (x - torch.clamp(x, min, max))


# (Multi-dim) indexing
def gather_indices(item, ind, dim=-1):
    ind = ind.long().expand(*item.shape[:dim], ind.shape[-1])  # Assumes ind.shape[-1] is desired num indices
    if -1 < dim < len(item.shape) - 1:
        trail_shape = item.shape[dim + 1:]
        ind = ind.reshape(ind.shape + (1,)*len(trail_shape))
        ind = ind.expand(*ind.shape[:dim + 1], *trail_shape)
    return torch.gather(item, dim, ind)


# Basic L2 normalization
class L2Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps)


# Context manager that temporarily switches on eval() mode for specified models; then resets them
class act_mode:
    def __init__(self, *models):
        super().__init__()
        self.models = models

    def __enter__(self):
        self.start_modes = []
        for model in self.models:
            self.start_modes.append(model.training)
            model.eval()

    def __exit__(self, *args):
        for model, mode in zip(self.models, self.start_modes):
            model.train(mode)
        return False


# Converts data to Torch Tensors and moves them to the specified device as floats
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


# Backward pass on a loss; clear the grads of models; step their optimizers
def optimize(loss=None, *models, clear_grads=True, backward=True, step_optim=True):
    # Clear grads
    if clear_grads:
        for model in models:
            model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        loss.backward()

    # Optimize
    if step_optim:
        for model in models:
            model.optim.step()


# Increment/decrement a value in proportion to a step count based on a string-formatted schedule (only supports linear)
def schedule(schedule, step):
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        if match:
            start, stop, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * start + mix * stop


# Helps contain learnable meta coefficients like temperatures, etc.
class Meta(nn.Module):
    def __init__(self, optim_lr=None, **metas):
        super().__init__()

        self.metas = list(metas.keys())
        for name, meta in metas.items():
            meta = nn.Parameter(torch.full((1,), meta))
            setattr(self, name, meta)  # {name: init_value}

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

    def forward(self, *names):
        if len(names) == 0:
            names = self.metas
        return torch.cat([getattr(self, meta) for meta in names])


class ReNormalize(nn.Module):
    def forward(self, x):
        # "Re-normalization", as in SPR (https://arxiv.org/abs/2007.05929), or at least in their code
        y = x - x.min(-1, keepdim=True)[0]
        y /= y.max(-1, keepdim=True)[0]
        return y.view(*x.shape)
