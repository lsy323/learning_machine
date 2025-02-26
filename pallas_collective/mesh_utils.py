import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental import shard_map

mesh = None

n_device = 4

def get_mesh():
    global mesh
    if mesh is None:        
        print("init mesh")
        mesh = jax.make_mesh((n_device,), ('x',))
    return mesh