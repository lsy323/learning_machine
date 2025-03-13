#/bin/bash
pip uninstall -y torch torchvision torch_xla jax jaxlib libtpu
git clone https://github.com/lsy323/vllm.git
cd vllm
git checkout lsiyuan/try-disable-dynamo-guard
git checkout HEAD~1
git checkout .
pip install -r requirements-tpu.txt
# VLLM_TARGET_DEVICE="tpu" python setup.py develop
# python examples/offline_inference/tpu.py
python test.py
