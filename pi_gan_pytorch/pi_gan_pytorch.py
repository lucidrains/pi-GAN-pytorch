import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum

class piGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x