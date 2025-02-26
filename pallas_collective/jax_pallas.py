import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import functools

P = jax.sharding.PartitionSpec

num_devices = jax.local_device_count()
assert num_devices > 1, "Please run this notebook with more than one device."
assert "TPU" in jax.devices()[0].device_kind, "Please run this notebook with TPU devices."
print(f"Running with {num_devices} {jax.devices()[0].device_kind} devices.")

partition = P(None, 'x')
mesh = jax.make_mesh((num_devices,), ('x',))
sharding = jax.sharding.NamedSharding(mesh, partition)

# Create an input array that shards the last dimension across
# all devices.
input_arr = jax.random.uniform(jax.random.key(0), (8, 128 * num_devices))
input_arr = jax.device_put(input_arr, sharding)


def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem):
#   my_id = jax.lax.axis_index('x')
#   my_id = 0
#   right_neighbor = jax.lax.rem(my_id + 1, num_devices)
  right_neighbor = 1
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
    right_permute_kernel,
    out_shape=out_shape,
    grid_spec=grid_spec,
)

my_id = 0
right_neighbor = (my_id + 1) % num_devices
def right_permute_wrapper(x, right_neighbor):
    return pl.pallas_call(
        functools.partial(right_permute_kernel, right_neighbor),
        out_shape=out_shape,
        grid_spec=grid_spec,
    )(x)

# Wrap the kernel within a shard_map to call.
# jitted = jax.jit(
#     shard_map.shard_map(
#         right_permute,
#         mesh=mesh,
#         in_specs=partition,
#         out_specs=partition,
#         check_rep=False,
#     )
# )
# breakpoint()
jitted2 = jax.jit(shard_map.shard_map(
    # functools.partial(right_permute_wrapper, right_neighbor),
    right_permute,
    mesh=mesh,
    in_specs=partition,
    out_specs=partition,
    check_rep=False))
# jitted2 = jax.jit(right_permute)
ir = jitted2.lower(input_arr).compiler_ir()
print(ir)
breakpoint()

# ir = jitted.lower(input_arr).compiler_ir()
# print(ir)
# pallas_result = jitted(input_arr)

# Compare Pallas result to XLA shard_map result.
perm = tuple((src, (src + 1) % num_devices) for src in range(num_devices))

xla_result = jax.jit(
    shard_map.shard_map(
        lambda x: jax.lax.ppermute(x, 'x', perm),
        mesh=mesh, in_specs=partition, out_specs=partition)
)(input_arr)

print('Input = ', input_arr[0, ::128])
print('Pallas Result = ', pallas_result[0, ::128])
print('jax.lax.ppermute Result = ', xla_result[0, ::128])
print(
    'Difference |Pallas - jax.lax.ppermute| = ',
    jnp.mean(jnp.abs(pallas_result - xla_result)),
)