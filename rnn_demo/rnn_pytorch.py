import os
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass

@dataclass
class SeqModelConfig:
    """ Parameters are LLM generated """
    n_features: int = 100
    n_hidden: int = 200
    n_targets: int = 1

class SeqModel(nn.Module):
    """ Architecture is LLM generated """
    def __init__(self, config: SeqModelConfig):
        super().__init__()
        self.gru = nn.GRU(config.n_features, config.n_hidden, batch_first=True)  #batch_first added
        self.linear = nn.Linear(config.n_hidden, config.n_targets)

    def forward(self, x: torch.Tensor):
        output, _ = self.gru(x)  #output, hidden state
        output = self.linear(output)
        return output

# generate data. 100MB per table. 2000 context. 100 features
def generate_data(n):
    """ context and feature count is LLM generated """
    return [{
        'x': torch.randn(250, 2000, 100).float(),
        'y': torch.randn(250, 2000, 1).float(),
        'w': torch.ones(250, 2000, 1).float(),
    } for _ in range(n)]


def train_step(model, optimizer, x, y, w):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model.train()
    optimizer.zero_grad()
    yhat = model(x)
    ydiff = yhat - y
    loss = (ydiff ** 2 * w).mean()
    loss.backward()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    print('step time is', start.elapsed_time(end))
    return loss.item() # Return a standard Python number

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check for GPU
    print('device is ', device)
    blocks = generate_data(1)
    model = SeqModel(SeqModelConfig()).to(device) # Move to device
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Use PyTorch optimizer

    for epoch in range(5):
        t0 = time.time()
        total_samples = 0
        nbytes = 0
        for X in blocks:
            x, y, w = X['x'].to(device), X['y'].to(device), X['w'].to(device) # Move data to device
            n = x.shape[0]
            nbytes += x.nelement() * x.element_size() # Correct byte calculation
            n_batches = 10
            batch_size, rem = divmod(n, n_batches)
            print(f"{batch_size=}")
            sample_b = 0
            for _ in range(n_batches):
                sample_e = sample_b + batch_size
                last_loss = train_step(model, optimizer, x[sample_b:sample_e], y[sample_b:sample_e], w[sample_b:sample_e])
                sample_b = sample_e
            total_samples += y.shape[0]

        t1 = time.time()
        dt = t1 - t0
        print(dt, "s", dt / total_samples * 1e6, "us/sample", nbytes / dt * 1e-6, "MB/s", f"Loss: {last_loss}") #added loss print

if __name__ == '__main__':
  main()
