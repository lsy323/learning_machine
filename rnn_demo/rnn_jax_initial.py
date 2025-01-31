import os

# os.environ["OZEKI_HAS_TPU"] = "true"

import jax

import jax.numpy as jnp

import time

from collections import deque

import pyarrow as pa

from pathlib import Path

from dataclasses import dataclass

import concurrent.futures

import flax.nnx as nnx

import jax.random as jrandom

import optax

 

from dataclasses import dataclass

@dataclass

class SeqModelConfig:

    """ Parameters are LLM generated """

    n_features: int = 100

    n_hidden: int = 200

    n_targets: int = 1

 

class SeqModel(nnx.Module):

    """ Architecture is LLM generated """

    def __init__(self, config: SeqModelConfig, *, rngs: nnx.Rngs):

        gru = nnx.nn.recurrent.RNN(nnx.nn.recurrent.GRUCell(

            in_features=config.n_features, hidden_features=config.n_hidden, rngs = rngs))

        layers = [gru, nnx.Linear(config.n_hidden, config.n_targets, rngs=rngs)]

        self.layers = nnx.Sequential(*layers)

 

    def __call__(self, x: jax.Array):

        return self.layers(x)

       

def block_to_device(X):

    return {k: jax.device_put(v) for k, v in X.items()}

 

# generate data. 100MB per table. 2000 context. 100 features

def generate_data(n, key):

    """ context and feature count is LLM generated """

    return [{

        'x': jrandom.uniform(key, (250, 2000, 100), dtype=jnp.bfloat16),

        'y': jrandom.uniform(key, (250, 2000, 1), dtype=jnp.float32),

        'w': jnp.ones((250, 2000, 1), dtype=jnp.float32),

    } for _ in range(n)]

 

from contextlib import contextmanager

 

@contextmanager

def do_nothing():

    try:

        yield

    finally:

        pass

 

@nnx.jit

def train_step(model, optimizer, x, y, w):

    """ Industry standard """

    def loss_fn(model):

        yhat = model(x)

        ydiff = yhat - y

        return (ydiff ** 2 * w).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    optimizer.update(grads)

    return loss

 

def test():

    rng = nnx.Rngs(0)

    blocks = [block_to_device(X) for X in generate_data(1, rng())]

    model = SeqModel(SeqModelConfig(), rngs=rng)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3)) # Parameter from Google example code. Adam is industry standard

    with jax.transfer_guard("log"):

        for epoch in range(2):

            t0 = time.time()

            total_samples = 0

            with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False) if epoch == 1 else do_nothing():

                nbytes = 0

                for X in blocks:

                    n = X['x'].shape[0]

                    nbytes += X['x'].nbytes

                    n_batches = 10

                    batch_size, rem = divmod(n, n_batches)

                    print(f"{batch_size=}")

                    x, y, w = [X[k] for k in ('x', 'y', 'w')]

                    sample_b = 0

                    for _ in range(n_batches):

                        sample_e = sample_b + batch_size

                        last_loss = train_step(model, optimizer, x[sample_b:sample_e], y[sample_b:sample_e], w[sample_b:sample_e])

                        sample_b = sample_e

                    total_samples += y.shape[0]

                last_loss.block_until_ready()

                t1 = time.time()

            dt = t1-t0

            print(dt,"s", dt/total_samples*1e6, "us/sample", nbytes/dt*1e-6, "MB/s")

test()