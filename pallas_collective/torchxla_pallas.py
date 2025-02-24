import functools

import torch
import torch_xla
import torch_xla.runtime as xr
from torch_xla.experimental.custom_kernel import trace_pallas

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


n_device = 4

def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem, my_id: int):
  right_neighbor = jax.lax.rem(my_id + 1, n_device)
  remote_copy_op = pltpu.make_async_remote_copy(
      src_ref=input_ref,
      dst_ref=output_ref,
      send_sem=send_sem,
      recv_sem=recv_sem,
      device_id=(right_neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
  )
  remote_copy_op.start()
  remote_copy_op.wait()

@functools.partial(
    jax.jit,
    static_argnames=[
        "my_id",
    ],
)
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
    my_id,
) -> torch.Tensor:
  payload, _ = trace_pallas(
      jax_function,
      x,
      my_id=my_id,
      static_argnames=[
          "my_id"
      ])
  return torch_xla._XLAC._xla_tpu_custom_call([x], payload, x.shape, [x.dtype])

def _mp_fn(index):
    tp_size = xr.world_size()
    print(f"n_chips: {tp_size}, current ordinal: {index}")
    input_arr = torch.rand((8, 128))
    out_arr = torchxla_wrapper(input_arr, index)
    print(out_arr)

if __name__ == '__main__':
    torch_xla.launch(_mp_fn, args=())

