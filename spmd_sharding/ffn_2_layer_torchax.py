import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from torch import nn
import torch
import time

import torchax as tx
import torchax.interop
import torchax.train
import optax

tx.enable_globally()




class RandomTensorDataset:
    def __init__(self, tensor_shape, element_count):
        self.tensor_shape = tensor_shape
        self.element_count = element_count

    def __iter__(self):
        for _ in range(self.element_count):
            yield torch.randn(self.tensor_shape)

num_devices = len(jax.devices())
model_axis = 4
mesh_shape = (num_devices // model_axis, model_axis)
device_ids = np.array(range(num_devices))
print(f"running SPMD with num_devices: {num_devices} mesh: {mesh_shape}", flush=True)

batch_size = num_devices // model_axis
mesh = jax.make_mesh((batch_size, model_axis), ('data', 'model'))

num_layers = 16
dim_out = dim = 4096
inner_dim = dim * 4
out_channels = 128

tokens_count = 2048
steps_count = 10

class FFN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(dim, inner_dim, bias=False)
    self.layer2 = nn.Linear(inner_dim, dim_out, bias=False)
    self.dropout = nn.Dropout(0.0)

  def forward(self, x):
    x = self.layer1(x)
    x = self.dropout(x)
    x = self.layer2(x)
    return x



class Model(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.m= torch.nn.ModuleList(
      [FFN() for _ in range(num_layers)]
    )
    self.output = nn.Linear(dim_out, out_channels, bias=False)

  def forward(self, x):
    for layer in self.m:
      x = layer(x)
    x = self.output(x)
    return x

with jax.default_device('cpu'):
  model = Model().to('jax')

data_sharding = NamedSharding(mesh, P('data'))

sharded_weights = {}
for name, weights in model.state_dict().items():
  print(name, weights.shape)
  if 'layer1' in name:
    sharding_spec = P('model', 'data')
  if 'layer2' in name:
    sharding_spec = P('data', 'model')
  if 'output' in name:
    sharding_spec = P('model', 'data')
  sharded_weights[name] = weights.apply_jax(jax.device_put, NamedSharding(mesh, sharding_spec))

mse_loss = nn.MSELoss(reduction='sum')


def call_model(weights, buffer, args):
  args[0].shard_(data_sharding)
  res = torch.func.functional_call(model, weights, (sample, ))
  res.shard_(data_sharding)
  return res

optimizer = optax.adamw(0.03)

opt_state = tx.interop.call_jax(optimizer.init, sharded_weights)

train_step = tx.train.make_train_step(
    call_model, 
    mse_loss,
    optimizer,
)

train_step = tx.interop.jax_jit(train_step, kwargs_for_jax_jit={'donate_argnums': (0, 2)})


target = torch.zeros(batch_size, tokens_count, out_channels, device='jax')
dataloader = RandomTensorDataset(tensor_shape=(batch_size, tokens_count, dim), element_count=steps_count)

start = time.time()
logdir = '/tmp/pytorch-tpu/'

for sample_index, sample in enumerate(dataloader):
    print("step {}/{}".format(sample_index, steps_count), flush=True)
    sample = sample.to('jax').apply_jax(jax.device_put, data_sharding)
    sample.apply_jax(jax.block_until_ready) # wait data sharding to complete

    start = time.perf_counter()
    loss, sharded_weights, opt_state = train_step(sharded_weights, {}, opt_state, sample, target)

    # wait until ready to take accurate time
    jax.block_until_ready(loss.jax())
    end = time.perf_counter()
    print('step {} used {}s'.format(sample_index, end - start))


print(f"sec / step is {(time.time() - start) / steps_count}")