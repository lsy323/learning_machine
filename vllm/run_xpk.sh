#!/usr/bin/env bash

set -eo

##### Runs SPMD training as an xpk job on a GKE cluster #####
#
# To run this, a _source_ install of xpk is required to access the latest TPU.
#
# Example: pip install git+https://github.com/AI-Hypercomputer/xpk.git@main
#

# Always build a new image. This is fast when cached.
./buildpush.sh

# You can override these by setting corresponding environment variables.
: "${CLUSTER_NAME:=xpk-test-vllm-3-v6e-4}"
# : "${CLUSTER_NAME:=mlperf-v5p-128}"
# : "${CLUSTER_NAME:=lizhiyu-moe-v5p-512}"
: "${DOCKER_URL:=gcr.io/cloud-ml-benchmarking/vllm_lsiyuan:latest}"
: "${NUM_SLICES:=1}"
# : "${TPU_TYPE:=v5p-128}"
: "${TPU_TYPE:=v6e-4}"
# : "${TPU_TYPE:=v5p-512}"
: "${ZONE:=us-east5-b}"
#: "${PROJECT_ID:=tpu-prod-env-automated}"
#: "${PROJECT_ID:=tpu-prod-env-one-vm}"
: "${PROJECT_ID:=cloud-ml-benchmarking}"

DATETIMESTR=$(date +%Y%m%d-%H%M%S)
COMMAND="bash run.sh" #python test.py"

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
