# XLA_FLAGS='--xla_dump_to=./ffn_manual_sharding' python ffn_manual_sharding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr

bs = 8
seq_len = 128
input_dim = 4096
hidden_dim = 14336


def all_reduce(tensor):
    return xm.all_reduce(xm.REDUCE_SUM, tensor)


# ParallelLinear
# Suppose 2 devices
# Base case: A [m, n] @ B[n, k] => C[m, k], A is FFN input, B, C are weights
#            C [m, k] @ D[k, n] => O[m, n]
# Column parallel B = [B1, B2], Sharded along column
#                 A @ B' => C' = [m, k // 2] (Output has partial shape)

# Row paralle D = [ D1
#                   D2 ]
# C' [m, k // 2] @ D' [k // 2, n] => C' [m, n] (Output has full shape but partial sum)
# all-reduce(C) for full result

class FeedForward(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, n_chips):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim // n_chips, bias=False)
        self.w2 = nn.Linear(hidden_dim // n_chips, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim // n_chips, bias=False)

    def forward(self, x):
        # Out shape: [bs, seq_len, hidden_dim // n_chips]
        w1_proj = self.w1(x)
        w3_proj = self.w3(x)
        act = F.silu(w1_proj * w3_proj)
        # Out shape: [bs, seq_len, input_dim]
        res = self.w2(act)
        res = all_reduce(res)
        return res


# Suppose this is the full weight tensor
w1_full = torch.rand(hidden_dim, input_dim)
w2_full = torch.rand(input_dim, hidden_dim)
w3_full = torch.rand(hidden_dim, input_dim)


def load_weights(model, index, tp_size):
    per_chip_col_size = hidden_dim // tp_size
    per_chip_row_size = hidden_dim // tp_size

    col_parallel_start_idx = index * per_chip_col_size
    col_parallel_end_idx = col_parallel_start_idx + per_chip_col_size
    row_parallel_start_idx = index * per_chip_row_size
    row_parallel_end_idx = row_parallel_start_idx + per_chip_row_size
    model.w1.weight.data = w1_full[
        col_parallel_start_idx:col_parallel_end_idx, :]
    model.w2.weight.data = w2_full[:,
                                   row_parallel_start_idx:row_parallel_end_idx]
    model.w3.weight.data = w3_full[
        col_parallel_start_idx:col_parallel_end_idx, :]


def _mp_fn(index):
    tp_size = xr.world_size()
    print(f"n_chips: {tp_size}, current ordinal: {index}")
    ffn = FeedForward(input_dim, hidden_dim, tp_size)
    load_weights(ffn, index, tp_size)
    
    if index == 0:
        print(f"w1/w3 full shape: {w1_full.shape}")
        print(f"w2 full shape: {w2_full.shape}")
        print(f"w1/w3 sharded shape: {ffn.w1.weight.shape}")
        print(f"w2 sharded shape: {ffn.w2.weight.shape}")
    
    input = torch.rand(bs, seq_len, input_dim)
    input = input.to('xla')
    ffn = ffn.to('xla')

    with torch.no_grad():
        ffn_out = ffn(input)
        ffn_out.cpu()


if __name__ == '__main__':
    torch_xla.launch(_mp_fn, args=())
