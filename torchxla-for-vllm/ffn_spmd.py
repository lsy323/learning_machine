# XLA_FLAGS='--xla_dump_to=./ffn_spmd' python ffn_spmd.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr

xr.use_spmd()

num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices, 1)
device_ids = np.array(range(num_devices))
mesh = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))

bs = 128
seq_len = 512
input_dim = 4096
hidden_dim = 14336


class FeedForward(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x):
        w1_proj = self.w1(x)
        w3_proj = self.w3(x)
        act = F.silu(w1_proj * w3_proj)
        res = self.w2(act)
        return res


# Suppose this is the full weight tensor
w1_full = torch.rand(hidden_dim, input_dim)
w2_full = torch.rand(input_dim, hidden_dim)
w3_full = torch.rand(hidden_dim, input_dim)


def load_weights(model):
    model.w1.weight.data = w1_full
    model.w2.weight.data = w2_full
    model.w3.weight.data = w3_full


tp_size = xr.world_size()
ffn = FeedForward(input_dim, hidden_dim)
load_weights(ffn)

input = torch.rand(bs, seq_len, input_dim)
input = input.to('xla')
ffn = ffn.to('xla')
xs.mark_sharding(input, mesh, ('x', None, None))
xs.mark_sharding(ffn.w1.weight, mesh, ('x', None))
xs.mark_sharding(ffn.w2.weight, mesh, (None, 'x'))
xs.mark_sharding(ffn.w3.weight, mesh, ('x', None))

with torch.no_grad():
    ffn_out = ffn(input)
    ffn_out.cpu()
