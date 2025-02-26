import functools

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.experimental.custom_kernel
from torch_xla.experimental.custom_kernel import trace_pallas

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental import shard_map
from mesh_utils import get_mesh

n_device = 4

# print("init mesh")
# P = jax.sharding.PartitionSpec
# partition = P(None, 'x')
# mesh = jax.make_mesh((n_device,), ('x',))

def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem, my_id: int):
  # right_neighbor = jax.lax.rem(my_id + 1, n_device)
  remote_copy_op = pltpu.make_async_remote_copy(
      src_ref=input_ref,
      dst_ref=output_ref,
      send_sem=send_sem,
      recv_sem=recv_sem,
      device_id=my_id, # Should rename
      device_id_type=pltpu.DeviceIdType.LOGICAL,
  )
  remote_copy_op.start()
  remote_copy_op.wait()

def jax_function(x, my_id):
    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # TPUMemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        scratch_shapes=(
            # We allocate DMA semaphores in scratch memory.
            [pltpu.SemaphoreType.DMA] * 2
        ),
    )
    right_permute = pl.pallas_call(
        functools.partial(
          right_permute_kernel,
          my_id=my_id
        ),
        out_shape=out_shape,
        grid_spec=grid_spec,
        )
    return right_permute(x)


def torchxla_wrapper(
    x: torch.Tensor,
    global_trace_tensor: torch.Tensor,
    my_id,
) -> torch.Tensor:
  mesh = get_mesh()
  P = jax.sharding.PartitionSpec
  partition = P(None, 'x')
  next_neighbor = (my_id + 1) % n_device
  shmapped = shard_map.shard_map(
        functools.partial(jax_function, my_id=next_neighbor),
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_rep=False,
    )
  # jitted = jax.jit(shmapped, static_argnames=["my_id"])
  payload, _ = trace_pallas(
      # jax_function,
      shmapped,
      global_trace_tensor,
      extract_from_shmap=True)
  print(f"check payload {payload}")
  return torch_xla._XLAC._xla_tpu_custom_call([x], payload, [list(x.shape)], [x.dtype])

def _mp_fn(index):
    tp_size = xr.world_size()
    print(f"n_chips: {tp_size}, current ordinal: {index}")
    input_arr = torch.rand((8, 128)).to('xla')
    global_trace_tensor = torch.rand((8, 512)).to('xla')
    out_arr = torchxla_wrapper(
      input_arr, global_trace_tensor, index)
    print(out_arr)

if __name__ == '__main__':
    torch_xla.launch(_mp_fn, args=())
