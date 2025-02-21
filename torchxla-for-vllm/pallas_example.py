import jax
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs
import torch
import torch_xla
from torch_xla.experimental.custom_kernel import flash_attention


q = torch.randn(4, 2, 128, 4).to("xla")
k = torch.randn(4, 2, 128, 4).to("xla")
v = torch.randn(4, 2, 128, 4).to("xla")

o = flash_attention(q, k, v)

# print(torch_xla._XLAC._get_xla_sharding_spec(o))
    # f"{{devices=[{n_devices},1,1,1]0,1,2,3}}")
# print(o.cpu())

