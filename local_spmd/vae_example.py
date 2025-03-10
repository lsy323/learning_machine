import torch
import torch.nn as nn
import os
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np

# Define a VAE Encoder that works with any resolution
class VAEEncoderFlexible(nn.Module):
    def __init__(self, input_channels=3, latent_dim=16):
        super(VAEEncoderFlexible, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Make feature map 1x1
        self.fc_mu = nn.Linear(128, latent_dim)  # Reduce to latent vector
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.global_avg_pool(x)  # Ensure fixed feature size
        x = torch.flatten(x, 1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class FFN(nn.Module):
    def __init__(self, embed_dim=64, ff_dim=256, dropout=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

xr.use_spmd()

#### Local SPMD with DDP

os.environ['XLA_USE_LOCAL_SPMD'] = '1'

device = xm.xla_device()
process_id = xr.process_index()
num_local_devices = xr.addressable_runtime_device_count()

mesh_shape = (num_local_devices, 1)
device_id_start = process_id * num_local_devices
device_ids = np.arange(device_id_start, device_id_start + num_local_devices)

mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

encoder = VAEEncoderFlexible().to(device)
spatial_dim = 128 * (process_id + 1)
img_tensor = torch.randn(16, 3, spatial_dim, spatial_dim).to(device)
xs.mark_sharding(img_tensor, mesh, ('x', None, None, None))


with torch.no_grad():
    mu, logvar = encoder(img_tensor)
    xm.mark_step()
    xm.wait_device_ops()
    print(mu)
    print(logvar)

os.environ['XLA_USE_LOCAL_SPMD'] = '0'


# Global SPMD
num_global_devices = xr.global_runtime_device_count()
mesh_shape = (num_global_devices, 1)
global_device_ids = np.arange(num_global_devices)
global_mesh = Mesh(global_device_ids, mesh_shape, ('x', 'y'))

ffn = FFN().to(device)

batch_size = 16
seq_length = 32
embed_dim = 64
x = torch.randn(batch_size, seq_length, embed_dim).to(device)
xs.mark_sharding(x, global_mesh, ('x', None, None))
ffn_out = ffn(x)
xm.mark_step()
xm.wait_device_ops()
print(x.shape)
print(x)
