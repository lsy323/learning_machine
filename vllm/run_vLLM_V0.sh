#/bin/bash
pip uninstall -y torch torchvision torch_xla jax jaxlib libtpu
git clone https://github.com/lsy323/vllm.git
cd vllm
git checkout lsiyuan/try-disable-dynamo-guard
git checkout HEAD~1
git checkout .
pip install -r requirements-tpu.txt
VLLM_TARGET_DEVICE="tpu" python setup.py develop
export HF_TOKEN=xxx # chaneg to your HF_token

# PyTorch/XLA simple test case
export PJRT_DEVICE=TPU
# python -c "import torch; import torch_xla; import torch_xla.runtime as xr; print(xr.device_type())"
# python -c "import torch; import torch_xla; import torch_xla.runtime as xr; print(torch_xla._XLAC._xla_get_devices())"
# python examples/offline_inference/tpu.py # failed with error: `RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed: `
vllm serve meta-llama/Meta-Llama-3.1-8B --swap-space 8 --disable-log-requests --tensor_parallel_size=4 --max-model-len=2048 --num-scheduler-steps=1 --port 6089 # hang

# # JAX matched function
# export JAX_PLATFORMS='tpu'
# python -c "import jax; print(jax.devices()); print(jax.devices()[0]); print(jax.devices()[0].memory_stats)"
