# XLA_DYNAMO_DEBUG=1 python torch_xla_example.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla


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


bs = 8
seq_len = 128
input_dim = 4096
intermediate_size = 14336

input = torch.rand(bs, seq_len, input_dim)
ffn = FeedForward(input_dim, intermediate_size)

input = input.to('xla')
ffn = ffn.to('xla')

with torch.no_grad():
    ffn_out = ffn(input)

print(f"lazy IR {torch_xla._XLAC._get_xla_tensors_text([ffn_out])}")
print(f"HLO: {torch_xla._XLAC._get_xla_tensors_hlo([ffn_out])}")

compiled = torch.compile(ffn, backend='openxla', fullgraph=True)
output_dynamo = compiled(input)
