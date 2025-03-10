# XLA_FLAGS='--xla_dump_to=./torch_xla_local_spmd' python torchxla_local_spmd.py

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np

xr.use_spmd()

print(f"local process id {xr.process_index()}, number of local devices: {xr.addressable_runtime_device_count()}")

device = xm.xla_device()


process_id = xr.process_index()
num_local_devices = xr.addressable_runtime_device_count()

mesh_shape = (num_local_devices, 1)
device_id_start = process_id * num_local_devices
device_ids = np.arange(device_id_start, device_id_start + num_local_devices)

mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
print(str(mesh.get_op_sharding(('x', None))))

x = torch.randn(16, 128).to(device)
# y = torch.randn(16, 128).to(device)
x = xs.mark_sharding(x, mesh, ('x', None))
# y = xs.mark_sharding(y, mesh, ('x', None))

def f(x, y):
    return x + y

with torch.no_grad():
    # z = f(x, y)
    z = x + 1

xm.mark_step()
xm.wait_device_ops()
z = z.cpu()
print(z)
print("finish computing z")
