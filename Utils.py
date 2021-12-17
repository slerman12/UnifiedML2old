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


# Sets all Torch and Numpy random seeds
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Saves Torch objects to root
def save(root_path, **to_save):
    save_path = root_path / 'Saved.pt'
    with save_path.open('wb') as f:
        torch.save(to_save, f)


# Loads Torch objects from root
def load(root_path, *keys):
    save_path = root_path / 'Saved.pt'
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


# Copies parameters from one model to another
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
        super(Ensemble, self).__init__()

        self.ensemble = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *x):
        return torch.stack([module(*x) for module in self.ensemble],
                           self.dim)


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
def gather_index(item, ind):
    ind = ind.long().view(*item.shape[:-1], -1)
    return torch.gather(item, -1, ind)


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


# Increment/decrement a value in proportion to a step count based on a string-formatted schedule
def schedule(sched, step):
    try:
        return float(sched)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', sched)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', sched)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(sched)


# Helps contain learnable meta coefficients like temperatures, etc.
class Meta(nn.Module):
    def __init__(self, optim_lr=None, **metas):
        super().__init__()

        for name, meta in metas.items():
            meta = nn.Parameter(torch.full((1,), meta))
            setattr(self, name, meta)

        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)


# Converts an agent to a classifier
def to_classifier(agent, regression=False):
    def update(replay):

        if agent.training:
            agent.step += 1

        # "Recollect"

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, y_label = to_torch(batch, agent.device)

        # "Imagine" / "Envision"

        # Augment
        if agent.training and hasattr(agent, 'aug'):
            obs = agent.aug(obs)

        # Encode
        obs = agent.encoder(obs)

        # "Predict" / "Learn" / "Grow"

        dist = agent.actor(obs)

        y_pred = dist.logits if hasattr(dist, 'logits') \
            else dist.mean

        # Regression or classification
        loss = nn.MSELoss()(y_pred, y_label) if regression \
            else nn.CrossEntropyLoss()(y_pred, y_label)

        # Update
        if agent.training:
            optimize(loss, agent.encoder, agent.actor)

        logs = {
            'step': agent.step,
            'loss': loss.item()
        }

        if not regression:
            logs.update({
                'accuracy': torch.sum(torch.argmax(y_pred, -1)
                                      == y_label, -1) / y_pred.shape[0]
            })

        return logs

    setattr(agent, 'original_update', agent.update)
    setattr(agent, 'update', update)

    return agent


# Converts a classifier to an agent
def to_agent(classifier):

    if hasattr(classifier, 'original_update'):
        update = getattr(classifier, 'original_update')
        setattr(classifier, 'update', update)

    return classifier
