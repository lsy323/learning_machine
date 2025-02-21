# XLA_FLAGS='--xla_dump_to=./eager_vs_ltc' python eager_vs_ltc.py
import torch
import torch_xla
import torch_xla.core.xla_model as xm

t = torch.rand(3)
# Actual execution happens in each line.
t2 = torch.sin(t)
t3 = torch.abs(t2)
t4 = torch.cos(t3)

t = t.to('xla')
# Record computation instead of executing as program is interpreted
t2 = torch.sin(t)
t3 = torch.abs(t2)
# Stop the current graph recording
xm.mark_step()
# Execute pending execution
xm.wait_device_ops()
t4 = torch.cos(t3)
xm.mark_step()
xm.wait_device_ops()
