import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class ReTanh(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(torch.tanh(input), torch.zeros_like(input))


def get_activation_fn(act_fn):
    if isinstance(act_fn, str):
        if act_fn == "ReTanh":
            act_fn = lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
        else:
            act_fn = getattr(nn, act_fn)()
    return act_fn
