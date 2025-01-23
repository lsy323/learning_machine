import torch_xla.runtime as xr
import numpy as np
from torch import nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
import torch
import time
from torch.optim import AdamW
import torch_xla.distributed.parallel_loader as pl


def log_tensor_sharding(t: torch.Tensor, log_prefix: str):
    xm.mark_step()
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    shape = t.shape
    device = t.device.type
    print(f"{log_prefix} sharding: {sharding} shape: {shape} device: {device}", flush=True)


class RandomTensorDataset:
    def __init__(self, tensor_shape, element_count):
        self.tensor_shape = tensor_shape
        self.element_count = element_count

    def __iter__(self):
        for _ in range(self.element_count):
            yield torch.randn(self.tensor_shape)

xr.use_spmd()
num_devices = xr.global_runtime_device_count()
model_axis = 2 
mesh_shape = (num_devices // model_axis, model_axis, 1, 1)
device_ids = np.array(range(num_devices))
print(f"running SPMD with num_devices: {num_devices} mesh: {mesh_shape}", flush=True)
mesh = xs.Mesh(device_ids, mesh_shape, ("data", "model", "sequence", "dcn"))

batch_size = num_devices // model_axis

num_layers = 4
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

model = Model().to(xm.xla_device())

for name, weights in model.state_dict().items():
  print(name, weights.shape)
  if 'layer1' in name:
    xs.mark_sharding(weights, mesh, ('model', 'data'))
  if 'layer2' in name:
    xs.mark_sharding(weights, mesh, ('data', 'model'))
  if 'output' in name:
    xs.mark_sharding(weights, mesh, ('model', 'data'))

mse_loss = nn.MSELoss(reduction='sum')
optimizer = AdamW(params=model.parameters())
target = torch.zeros(batch_size, tokens_count, out_channels).to(device=xm.xla_device())

dataloader = RandomTensorDataset(tensor_shape=(batch_size, tokens_count, dim), element_count=steps_count)
dataloader_wrapper = pl.MpDeviceLoader(
    dataloader,
    device=xm.xla_device(),
    input_sharding=xs.ShardingSpec(mesh, partition_spec=("data", None, None), minibatch=False)
)

def mark_sharding_grads(weights):
  for name, weights in model.state_dict().items():
    print(name, weights.shape)
    if 'layer1' in name:
      xs.mark_sharding(weights.grad, mesh, ('model', 'data'))
    if 'layer2' in name:
      xs.mark_sharding(weights.grad, mesh, ('data', 'model'))
    if 'output' in name:
      xs.mark_sharding(weights.grad, mesh, ('model', 'data'))

xm.mark_step()
start = time.time()
for sample_index, sample in enumerate(dataloader_wrapper):
    print("step {}/{}".format(sample_index, steps_count), flush=True)

    xs.mark_sharding(sample, mesh, ("data", None, None)) # ("model", "data")
    output = model(sample)
    xs.mark_sharding(output, mesh, ("data", None, None))
    loss = mse_loss(output, target)
    log_tensor_sharding(loss, 'loss')
    loss.backward()
    optimizer.step()
    xm.mark_step()

    log_tensor_sharding(model.m[0].layer1.weight, 'layer1')
    log_tensor_sharding(model.m[0].layer1.weight.grad, 'layer1 grad')
    optimizer.zero_grad()

print(f"Step / sec is {(time.time() - start) / steps_count}")