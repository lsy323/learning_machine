# Utilities for training
import functools
import time
import torch
import torch_xla2
from torch_xla2 import interop
import optax
import jax
from jax.sharding import PartitionSpec as P, NamedSharding

SEQLEN = 2048

## Interface for optimizer_fn:
# def optimizer_fn(optimizer_state, weights, gradients) -> new_optimizer_state, new_weights

## Interface for loss_fn:
## loss_fn(model_out, label) -> scalar

class TraininableLlama:

  def __init__(self, model):
    self.orig_model = model

  # Args is what dataloader gives
  def call(self, weights, buffers, args, kwargs):
    weights_and_buffers = copy.copy(weights)
    weights_and_buffers.update(buffers)
    return torch.func.call_functional(
      self.orig_model, weights_and_buffers, args, kwargs)




def fake_dataloader(size, seqlen):
  for _ in range(size):
    x = torch.randint(0, 32000, (8, seqlen), device='cpu')
    yield x, (x + 1) % 32000

def group_data(dataloader, block_size):
    """yields tuple of inputs, label with seqlen == block_size"""

    tally = 0
    inputs = []
    labels = []

    for line in dataloader:
        x, y = line #line['input_ids'], line['labels']
        inputs.append(x)
        labels.append(y)
        seqlen = x.shape[1]
        tally += seqlen
        if tally >= block_size:
            inputs_stacked = torch.concat(inputs, dim=-1)
            labels_stacked = torch.concat(labels, dim=-1)
            yield inputs_stacked, labels_stacked
            tally = 0
            inputs = []
            labels = []


def sharded_device_put(tensor, sharding):
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)
    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    shape = tensor.shape
    x_split = [jax.device_put(tensor[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


# NOTE: this line makes jax.remat able to take torch functions
remat = interop.torch_view(jax.remat)

def make_train_step(model, loss_fn, optax_optimizer):

  env = torch_xla2.default_env()

  @functools.partial(
    remat, 
    policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
  def loss(weights, args, label): # inputs are XLATensor
    with env:
      res = torch.func.functional_call(model, weights,  args)
      num_tokens = res.shape[-1]
      flattened = res.reshape(-1, num_tokens)
      label = label.reshape(-1)
      l = loss_fn(flattened, label)
      return l

  jloss = interop.jax_view(loss)
  grad_fn = jax.value_and_grad(jloss)

  @functools.partial(
    jax.jit,
    donate_argnums=(0, 1)
  )
  def step(weights, opt_state, args, label): #inputs are array
    with jax.named_scope('compute_gradient'):
        loss, gradient = grad_fn(weights, args, label)

    with jax.named_scope("optimizer_updates"):
        updates, opt_state = optax_optimizer.update(
            gradient, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
    return loss, weights, opt_state

  return step


def train_loop(mesh, model, weights, data_loader, input_freqs_cis, lr, seqlen):
  print('start training')
  min_loop_time = 10000

  env = torch_xla2.default_env()

  jax_params = env.t2j_iso(weights)
  jax_optimizer = optax.adamw(lr)
  opt_state = jax_optimizer.init(jax_params)
  train_step = make_train_step(model, 
    loss_fn=torch.nn.CrossEntropyLoss(), 
    optax_optimizer=jax_optimizer,
  )

  def _expand_input(input_seq):
    seqlen = input_seq.shape[1]
    freqs_cis = env.t2j_iso(input_freqs_cis[:seqlen])
    mask = torch.full((seqlen, seqlen), float("-inf"), device='cpu')
    mask = torch.triu(mask, diagonal=1)
    return (input_seq, 0, freqs_cis, mask)

  replicated_sharding = NamedSharding(mesh, P())
  fsdp_sharding = NamedSharding(mesh, P('fsdp'))
  def _shard_first_dim(x):
    with jax.default_device(jax.devices('cpu')[0]):
      xj = env.to_xla(x).jax()
    xj = jax.make_array_from_callback(
      xj.shape, fsdp_sharding, lambda a: xj[a]
    )
    return xj

  def _replicate(x):
    with jax.default_device(jax.devices('cpu')[0]):
      xj = env.to_xla(x).jax()
    xj = jax.make_array_from_callback(
      xj.shape, replicated_sharding, lambda a: xj
    )
    return xj

  data_iter = group_data(fake_dataloader(1000, seqlen), seqlen)



  for i, item in enumerate(data_iter):
    inputs, labels = item

    input_seq, pos, freqs_cis, mask = _expand_input(inputs)

    input_seq = _shard_first_dim(input_seq)
    freqs_cis = freqs_cis
    mask = _replicate(mask)
    labels = _shard_first_dim(labels)

    print('INPUT shape', inputs.shape)

    if i == 5:
      jax.profiler.start_trace('/tmp/llama3')
    step_start = time.perf_counter()
    loss, jax_params, opt_state = train_step(
        jax_params, opt_state, (input_seq, pos, freqs_cis, mask), labels)
    jax.block_until_ready((loss, jax_params))
    step_end = time.perf_counter()
    if i == 6:
      jax.profiler.stop_trace()

    print(i, 'loss', loss, 'step latency: ', step_end - step_start)
    min_loop_time =  min(min_loop_time, step_end - step_start)
    print('======')
    if i >= 6:
        break
  
  return min_loop_time


