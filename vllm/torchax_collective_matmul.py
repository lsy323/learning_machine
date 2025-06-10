# XLA_FLAGS='--xla_dump_to=./hlo_ffn_spmd' python torchax_collective_matmul.py
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchax

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

from torchax.interop import jax_jit, extract_all_buffers
import time

class FeedForwardLinear(torch.nn.Module):
    """FeedForward using nn.Linear layers (original implementation)."""

    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardLinear, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        w1_proj = self.w1(x)
        w3_proj = self.w3(x)
        act = F.silu(w1_proj * w3_proj)
        res = self.w2(act)
        return res


class FeedForwardMatmul(torch.nn.Module):
    """FeedForward using torch.matmul for matrix multiplications."""

    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardMatmul, self).__init__()
        # Raw weight parameters in shape [input_dim, output_dim]
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.bfloat16))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, input_dim, dtype=torch.bfloat16))
        self.w3 = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.bfloat16))

    def forward(self, x):
        # Manual matrix multiplication: x @ weight
        w1_proj = torch.matmul(x, self.w1)  # [..., input_dim] @ [input_dim, hidden_dim]
        w3_proj = torch.matmul(x, self.w3)  # [..., input_dim] @ [input_dim, hidden_dim]
        act = F.silu(w1_proj * w3_proj)
        res = torch.matmul(act, self.w2)    # [..., hidden_dim] @ [hidden_dim, input_dim]
        return res


class FeedForwardEinsum(torch.nn.Module):
    """FeedForward using torch.einsum for matrix multiplications."""

    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardEinsum, self).__init__()
        # Raw weight parameters in shape [input_dim, output_dim]
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.bfloat16))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, input_dim, dtype=torch.bfloat16))
        self.w3 = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.bfloat16))

    def forward(self, x):
        # Using einsum for matrix multiplication with flexible batch dimensions
        w1_proj = torch.einsum('...i,ih->...h', x, self.w1)  # [..., input_dim] @ [input_dim, hidden_dim] -> [..., hidden_dim]
        w3_proj = torch.einsum('...i,ih->...h', x, self.w3)  # [..., input_dim] @ [input_dim, hidden_dim] -> [..., hidden_dim]
        act = F.silu(w1_proj * w3_proj)
        res = torch.einsum('...h,hi->...i', act, self.w2)    # [..., hidden_dim] @ [hidden_dim, input_dim] -> [..., input_dim]
        return res


# Use the einsum version by default
FeedForward = FeedForwardEinsum


# Parse command line arguments
parser = argparse.ArgumentParser(description='Torchax collective matmul benchmark')
parser.add_argument('--profile', action='store_true',
                    help='Enable profiling mode (reduces iterations to 3)')
parser.add_argument('--profile-dir', type=str, 
                    default='gs://lsiyuan-public/vllm-torchax/torchax_collective_matmul/baseline',
                    help='Directory for profiler output (default: gs://lsiyuan-public/vllm-torchax/torchax_collective_matmul/baseline)')
args = parser.parse_args()

# Run with torchax
num_tokens = 512 * 512
model_dim = 8192
hidden_dim = 28672
# num_tokens = 16
# model_dim = 32
# hidden_dim = 64

with torch.no_grad():
    ffn = FeedForward(model_dim, hidden_dim)

# Create a 1D device mesh
devices = jax.devices()
mesh = Mesh(devices, axis_names=('x',))

torchax.enable_globally()

x = torch.zeros((num_tokens, model_dim)).to(torch.bfloat16).to('jax')
ffn = ffn.to('jax')

# Create sharding spec for the mesh
replication_sharding = NamedSharding(mesh, P())
row_sharding = NamedSharding(mesh, P('x', None))
col_sharding = NamedSharding(mesh, P(None, 'x'))

# Place x on the device mesh with sharding
x = x.apply_jax(jax.device_put, row_sharding)
# x = x.apply_jax(jax.device_put, replication_sharding)

# Apply sharding to FFN
ffn.w1.data = ffn.w1.data.apply_jax(jax.device_put, row_sharding)
ffn.w3.data = ffn.w3.data.apply_jax(jax.device_put, row_sharding)
ffn.w2.data = ffn.w2.data.apply_jax(jax.device_put, col_sharding)

params, buffers = extract_all_buffers(ffn)
params_and_buffers = {**params, **buffers}

@jax_jit
def wrap_model_fwd(params_and_buffers, x):
    return torch.func.functional_call(ffn, params_and_buffers, kwargs={'x': x})

# warm up
out = wrap_model_fwd(params_and_buffers, x)
out._elem.block_until_ready()

# Set number of iterations based on profile flag
num_iterations = 3 if args.profile else 100

if args.profile:
    print(f"Profiling mode enabled - running {num_iterations} iterations")
    print(f"Profiler output directory: {args.profile_dir}")
else:
    print(f"Benchmark mode - running {num_iterations} iterations")

if args.profile:
    jax.profiler.start_trace(args.profile_dir)

start_time = time.time()
for _ in range(num_iterations):
    out = wrap_model_fwd(params_and_buffers, x)
    out._elem.block_until_ready()
    
end_time = time.time()
print(f"Average time per iteration: {(end_time - start_time) / num_iterations:.4f} seconds")

if args.profile:
    jax.profiler.stop_trace()

# Input [512x512=262144, 8192]
# w1 [28672, 8192]
# w2 [8192, 28672]

# w1 on each chip [28672 / 8 = 3584, 8192]
