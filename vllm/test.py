import torch
import torch_xla

t = torch.rand(3).to('xla')
print(t)
