# Run locally in your local TPUVM with vLLM V0:
Note: please modify `HF_TOKEN` with your HF_TOKEN
```
sudo docker run --privileged --net host --shm-size=16G --name vllmv0test -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash

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
python -c "import torch; import torch_xla; import torch_xla.runtime as xr; print(xr.device_type())"
python -c "import torch; import torch_xla; import torch_xla.runtime as xr; print(torch_xla._XLAC._xla_get_devices())"
python examples/offline_inference/tpu.py # failed with error: `RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed: `

# setup vLLM server
vllm serve meta-llama/Meta-Llama-3.1-8B --swap-space 8 --disable-log-requests --tensor_parallel_size=4 --max-model-len=2048 --num-scheduler-steps=1 --port 6089 # hang

# inference benchmark script
wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
pip install --upgrade google-cloud-storage
rm -rf inference-benchmark
git clone https://github.com/AI-Hypercomputer/inference-benchmark
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - 
apt-get update
apt-get install -y google-cloud-sdk
apt-get -y install jq
export HF_TOKEN=xxx
python inference-benchmark/benchmark_serving.py \
        --host localhost \
        --port 6089 \
        --num-prompts 1000 \
        --max-input-length 1024 \
        --max-output-length 1024 \
        --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
        --save-json-results \
        --model meta-llama/Meta-Llama-3.1-8B \
        --tokenizer meta-llama/Meta-Llama-3.1-8B \
        --request-rate 1 \
        --stream-request

# JAX matched function
export JAX_PLATFORMS='tpu'
python -c "import jax; print(jax.devices()); print(jax.devices()[0]); print(jax.devices()[0].memory_stats)"
```

# Run locally in your local TPUVM with vLLM V1 according to GKE used commands:
Note: please modify `HF_TOKEN` with your HF_TOKEN
```
sudo docker run --privileged  --shm-size 16G --net host --name testvllmar141557pm -it -d docker.io/vllm/vllm-tpu:d374f04a337dbd4aab31484b6fa2d4a5f20c2116 /bin/bash

export HF_TOKEN=xxx

VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --disable-log-requests --max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 --max-model-len=8192 --port 8000
```
and create a new termial(with the same docker container), SSH-ed to the existing lcoal TPUVM, and attach to the above created docker container:
```
sudo docker run --privileged  --shm-size 16G --net host --name testvllmar141557pm -it -d docker.io/vllm/vllm-tpu:d374f04a337dbd4aab31484b6fa2d4a5f20c2116 /bin/bash

sudo docker exec --privileged -it testvllmar141557pm /bin/bash

python inference-benchmark/benchmark_serving.py --save-json-results --port=6089 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --request-rate=1 --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B '--output-bucket=gs://manfeipublic'
```

# Run on XPK

NOTE: Modify HF_TOKEN to your HF_TOKEN in `run_xpk_with_vLLM_V0.sh` or `run_xpk_with_vLLM_V1.sh`
then

### run XPK workload with vLLM V0:
```
bash run_xpk_with_vLLM_V0.sh
```

### run XPK workload with vLLM V1:
```
bash run_xpk_with_vLLM_V1.sh
```
