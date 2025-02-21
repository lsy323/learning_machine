import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.debug.profiler as xp


class FeedForward(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        w1_proj = self.w1(x)
        w3_proj = self.w3(x)
        act = F.silu(w1_proj * w3_proj)
        res = self.w2(act)
        return res


# class M(torch.nn.Module):

#     def __init__(self, input_dim, hidden_dim, n_layer):
#         super(M, self).__init__()
#         self.layers = []
#         for _ in range(n_layer):
#             self.layers.append(FeedForward(input_dim, hidden_dim))

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

bs = 8
seq_len = 128
input_dim = 4096
intermediate_size = 14336
# input_dim = 256
# intermediate_size = 512

input = torch.rand(bs, seq_len, input_dim)
ffn = FeedForward(input_dim, intermediate_size)

input = input.to('xla')
ffn = ffn.to('xla')

with torch.no_grad():
    ffn_out = ffn(input)

print(torch_xla._XLAC._get_xla_tensors_text([ffn_out]))
print(torch_xla._XLAC._get_xla_tensors_hlo([ffn_out]))

compiled = torch.compile(ffn, backend='openxla', fullgraph=True)
output_dynamo = compiled(input)

server = xp.start_server(9012)
# profile_logdir = '/tmp/torchxla/compare_dynamo'
# xp.trace_detached(
#       'localhost:9012',
#       profile_logdir,
#       duration_ms=10000)

with xp.Trace("torch.compile"):
    # Warm up
    res = compiled(input)
    res.cpu()
    start_time = time.time()
    for _ in range(100):
        res = compiled(input)
        res.cpu()
    end_time = time.time()
    elapse_time = (end_time - start_time) / 1e+6
    print(f"elapse time {elapse_time}")

with xp.Trace("LTC"):
    # Warm up
    res = ffn(input)
    res.cpu()
    start_time = time.time()
    for _ in range(100):
        res = ffn(input)
        res.cpu()
    end_time = time.time()
    elapse_time = (end_time - start_time) / 1e+6
    print(f"elapse time {elapse_time}")
