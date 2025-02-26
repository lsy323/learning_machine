import jax
from jax.experimental import shard_map
import functools
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem, right_neighbor):
#   right_neighbor = 1
  remote_copy_op = pltpu.make_async_remote_copy(
      src_ref=input_ref,
      dst_ref=output_ref,
      send_sem=send_sem,
      recv_sem=recv_sem,
      device_id=right_neighbor,
      device_id_type=pltpu.DeviceIdType.LOGICAL,
  )
  remote_copy_op.start()
  remote_copy_op.wait()

def f(x, const):
    return x + const

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

def wrapper(x, right_neighbor):
    right_permute = pl.pallas_call(
        functools.partial(right_permute_kernel, right_neighbor=right_neighbor),
        out_shape=out_shape,
        grid_spec=grid_spec,
    )
    return right_permute(x)

n_device = 4
mesh = jax.make_mesh((n_device,), ('x',))
P = jax.sharding.PartitionSpec
partition = P(None, 'x')

const_val = 1
# shmapped = shard_map.shard_map(
#     functools.partial(f, const=const_val),
#     mesh=mesh,
#     in_specs=partition,
#     out_specs=partition,
#     check_rep=False,
# )
curr_neighbor = 1
shmapped = shard_map.shard_map(
    functools.partial(wrapper, right_neighbor=curr_neighbor),
    mesh=mesh,
    in_specs=partition,
    out_specs=partition,
    check_rep=False,
)

jitted = jax.jit(shmapped)

num_devices = jax.local_device_count()
input_arr = jax.random.uniform(jax.random.key(0), (8, 128 * num_devices))
# partition = P(None, 'x')
# mesh = jax.make_mesh((num_devices,), ('x',))
# sharding = jax.sharding.NamedSharding(mesh, partition)
# input_arr = jax.device_put(input_arr, sharding)

ir = jitted.lower(input_arr).compiler_ir()
print(ir)

def _extract_backend_config(
    module: "jaxlib.mlir._mlir_libs._mlir.ir.Module",
    extract_from_shmap: bool = False):
    for operation in module.body.operations:
        if extract_from_shmap and ("shmap_body" not in str(operation.name)):
            continue
        assert len(
            operation.body.blocks) == 1, "The passing module is not compatible."
        for op in operation.body.blocks[0].operations:
            if op.name == "stablehlo.custom_call":
                return op.backend_config.value
    return None

payload = _extract_backend_config(ir, extract_from_shmap=True)
print(payload)