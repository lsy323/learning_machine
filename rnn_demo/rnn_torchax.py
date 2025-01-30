import flax
import flax.nnx
import os
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import jax
import jax._src.prng
from jax import numpy as jnp
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

    
class FromFlax(nn.Module):
    
    def __init__(self, flax_module, env):
        super().__init__()
        state = flax.nnx.state(flax_module)
        graphdef, state = flax.nnx.split(flax_module)
        flattened_state, self._tree_spec = jax.tree_util.tree_flatten(state)

        cond = lambda a: isinstance(a, jax.Array) and a.dtype in (jnp.float32.dtype, jnp.bfloat16.dtype)
        self.params = torch.nn.ParameterList(
            [env.j2t_iso(a) for a in flattened_state if cond(a)]
        )
        self._other = [a for a in flattened_state if not cond(a)]
        self._flax_mod_graph = graphdef

    def forward(self, x):
        unflattened_state = jax.tree_unflatten(
            self._tree_spec,
            tx.interop.jax_view(list(self.params)) + self._other
        )
        flax_mod = flax.nnx.merge(self._flax_mod_graph, unflattened_state)
        res = tx.interop.call_jax(
            flax_mod,
            x
        )
        return res


class SeqModel(nn.Module):
    """ Architecture is LLM generated """
    def __init__(self, config: SeqModelConfig):
        super().__init__()
        #self.gru = nn.GRU(config.n_features, config.n_hidden, batch_first=True)  #batch_first added
        gru = flax.nnx.nn.recurrent.RNN(
            flax.nnx.nn.recurrent.GRUCell(
            in_features=config.n_features, hidden_features=config.n_hidden, rngs = flax.nnx.Rngs(0)))
        self.gru = FromFlax(gru, tx.default_env())
        self.linear = nn.Linear(config.n_hidden, config.n_targets)

    def forward(self, x: torch.Tensor):
        output = self.gru(x)  #output, hidden state
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
    env.config.debug_print_each_op = False

    jmodel = tx.interop.JittableModule(model)

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
    #train_step = tx.interop.jax_jit(train_step, {'donate_argnums': (0, 2)})


    

    opt_state = tx.interop.call_jax(optimizer.init, jmodel.params)
    weights = jmodel.params
    print(
        jax.jit(tx.interop.jax_view(train_step)
                ).lower(
                    tx.interop.jax_view(jmodel.params),
                    tx.interop.jax_view(jmodel.buffers),
                    tx.interop.jax_view(opt_state),
                    jax.ShapeDtypeStruct((25, 2000, 100), jnp.float32.dtype),
                    jax.ShapeDtypeStruct((25, 2000, 1), jnp.float32.dtype)
                ).as_text()
    )

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
        print(dt, "s", dt / total_samples * 1e6, "us/sample", nbytes / dt * 1e-6, "MB/s", f"Loss: {loss}") #added loss print

if __name__ == '__main__':
  main()