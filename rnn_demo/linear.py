import torch
import torchax
torchax.enable_globally()


linear = torch.nn.Linear(200, 1)

inputs = torch.randn(25, 2000, 200, device='jax')

print('shape1', inputs.transpose_(0, 1).shape)
print('shape2', inputs.shape)

inputs = torch.randn(25, 2000, 200, device='cuda')


exp = torch.export.export(linear, (torch.randn(25, 2000, 200), ))
exp.run_decompositions()
print(exp.graph_module.code)
print(inputs.transpose_(0, 1).shape)
print(inputs.shape)



