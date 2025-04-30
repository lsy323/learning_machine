#!/usr/bin/env bash

set -eo

##### Runs SPMD training as an xpk job on a GKE cluster #####
#
# To run this, a _source_ install of xpk is required to access the latest TPU.
#
# Example: pip install git+https://github.com/AI-Hypercomputer/xpk.git@main
#

# # Always build a new image. This is fast when cached.
# ./buildpush.sh

# You can override these by setting corresponding environment variables.
: "${CLUSTER_NAME:=xpk-test-vllm-3-v6e-4}"
# piz-v6e-4-debug}"
# xpk-test-vllm-3-v6e-4}"
# : "${CLUSTER_NAME:=mlperf-v5p-128}"
# : "${CLUSTER_NAME:=lizhiyu-moe-v5p-512}"
: "${DOCKER_URL:=us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm}"
: "${NUM_SLICES:=1}"
# : "${TPU_TYPE:=v5p-128}"
: "${TPU_TYPE:=v6e-4}"
# : "${TPU_TYPE:=v5p-512}"
: "${ZONE:=us-east5-b}"
#: "${PROJECT_ID:=tpu-prod-env-automated}"
#: "${PROJECT_ID:=tpu-prod-env-one-vm}"
: "${PROJECT_ID:=cloud-ml-benchmarking}"

DATETIMESTR=$(date +%Y%m%d-%H%M%S)
# Notice: please update HF_TOKEN=xxx with your own HF_TOKEN
COMMAND="pip uninstall -y torch torchvision torch_xla jax jaxlib libtpu && \
rm -rf vllm && \
git clone -b gkefix https://github.com/richardsliu/vllm.git && \
cd vllm && \
pip install -r requirements/tpu.txt && \
VLLM_TARGET_DEVICE='tpu' python setup.py develop && \
export PJRT_DEVICE=TPU && \
export HF_TOKEN=xxx && \
VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --disable-log-requests --max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 --max-model-len=8192 --port 8003"

python /home/manfei/xpk/xpk.py workload create \
    --cluster ${CLUSTER_NAME} \
    --docker-image ${DOCKER_URL} \
    --workload "${USER}-$TPU_TYPE-${DATETIMESTR}" \
    --tpu-type=${TPU_TYPE} \
    --num-slices=${NUM_SLICES} \
    --zone $ZONE \
    --project $PROJECT_ID \
    --enable-debug-logs \
    --command "$COMMAND"
    #$--on-demand \
