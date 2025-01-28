import os
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import jax
import optax
import torchax as tx 
from torchax import train
tx.enable_globally()

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


def main():
    device = 'jax'
    print('device is ', device)
    blocks = generate_data(1)
    model = SeqModel(SeqModelConfig()).to('jax') # Move to device
    env = tx.default_env()
    #env.config.debug_accuracy_for_each_op = True

    jmodel = tx.interop.JittableModule(model)
    import pdb; pdb.set_trace()

    # Split the model parameters to weights and buffers
    # because buffers is the non-training params
    def model_fn(weights, buffers, args):
        return jmodel.functional_call('forward', weights, buffers, args)

    optimizer = optax.adam(0.1) 
    loss_fn = torch.nn.MSELoss()

    train_step = train.make_train_step(
        model_fn,
        loss_fn,
        optimizer,
    )
    train_step = tx.interop.jax_jit(train_step, {'donate_argnums': (0, 2)})

    opt_state = tx.interop.call_jax(optimizer.init, jmodel.params)
    weights = jmodel.params

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
                loss, weights, opt_state = train_step(weights, {}, opt_state, 
                                                      x[sample_b:sample_e], y[sample_b:sample_e])
                sample_b = sample_e
            total_samples += y.shape[0]

        loss.jax().block_until_ready()
        t1 = time.time()
        dt = t1 - t0
        print(dt, "s", dt / total_samples * 1e6, "us/sample", nbytes / dt * 1e-6, "MB/s", f"Loss: {last_loss}") #added loss print

if __name__ == '__main__':
  main()