import jax
import jax.numpy as jnp
import torch
from torch.utils import _pytree as pytree
import torchax
from torchax.interop import jax_jit

torchax.enable_globally()

@jax_jit
def process_dict(data_dict):
  return {
      'sum_a_b': data_dict['a'] + data_dict['b'],
      'scaled_c': data_dict['c'] * 2.0
  }

# Create a dictionary of JAX arrays
# my_data = {
#     'a': jnp.array([1.0, 2.0, 3.0]),
#     'b': jnp.array([4.0, 5.0, 6.0]),
#     'c': jnp.array([0.1, 0.2, 0.3])
# }
my_data = {
    'a': torch.tensor([1.0, 2.0, 3.0]),
    'b': torch.tensor([4.0, 5.0, 6.0]),
    'c': torch.tensor([0.1, 0.2, 0.3]),
}
my_data = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'), my_data)

result_dict = process_dict(my_data)
print(result_dict)
# Expected output:
# {
#   'sum_a_b': DeviceArray([5., 7., 9.], dtype=float32),
#   'scaled_c': DeviceArray([0.2, 0.4, 0.6], dtype=float32)
# }
