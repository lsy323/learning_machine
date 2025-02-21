import torch
import torch_xla
import torch_xla.experimental.custom_kernel
from torch.utils import _pytree as pytree


batch_size = 8
num_heads = 32
head_size = 128
num_kv_heads = 32
context_len = 512
num_blocks = 2048
block_size = 128
pages_per_compute_block = 32
MAX_NUM_BLOCKS_PER_SEQ = 512

query = torch.randn((batch_size, 1, num_heads, head_size), dtype=torch.bfloat16)
k_cache = torch.randn((num_kv_heads, num_blocks * block_size, head_size), dtype=torch.bfloat16)
v_cache = torch.randn((num_kv_heads, num_blocks * block_size, head_size), dtype=torch.bfloat16)

block_tables = torch.randint(0, num_blocks, (batch_size, MAX_NUM_BLOCKS_PER_SEQ), dtype=torch.int32)
context_lens = torch.tensor([context_len] * batch_size, dtype=torch.int32)

sm_scale = head_size ** -0.5
query = query.squeeze(1)
query = query * sm_scale
head_size = query.shape[-1]
num_slots = k_cache.shape[-2]
k_cache = k_cache.reshape(-1, num_slots // block_size, block_size, head_size)
v_cache = v_cache.reshape(-1, num_slots // block_size, block_size, head_size)

args = (query, k_cache, v_cache, context_lens, block_tables, pages_per_compute_block)
args = pytree.tree_map_only(torch.Tensor, lambda x: x.to('xla'), args)

output = torch.ops.xla.paged_attention(*args)

print(torch_xla._XLAC._get_xla_tensors_hlo([output]))
