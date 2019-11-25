import torch
import torch.nn as nn
import numpy as np
import os

def cat_lists(x):
    return torch.cat([torch.Tensor(xitem) for xitem in x])

def make_deterministic(env, seed = 0):
    env.seed(seed)
    # torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return env

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias

def load_empty_policy(classname, f, hidden = 64):
    assert(os.path.exists(f))
    [obs_space, action_space] = torch.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return classname(obs_space.shape, action_space, hidden_size = hidden).to(device)