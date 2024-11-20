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
: "${CLUSTER_NAME:=bodaborg-v6e-256}"
: "${DOCKER_URL:=gcr.io/tpu-pytorch/llama3:latest}"
: "${NUM_SLICES:=1}"
: "${TPU_TYPE:=v6e-256}"
: "${ZONE:=us-east5-c}"
: "${PROJECT_ID:=tpu-prod-env-automated}"

DATETIMESTR=$(date +%Y%m%d-%H%M%S)
COMMAND="python run_xpk.py --batch_size=64 --model_type=70B --seqlen=8192 --use_custom_offload=False --model_impl=scan_manual --tp=4"

xpk workload create \
    --cluster ${CLUSTER_NAME} \
    --docker-image ${DOCKER_URL} \
    --workload "${USER}-xpk-v6e-256-$NUM_SLICES-${DATETIMESTR}" \
    --tpu-type=${TPU_TYPE} \
    --num-slices=${NUM_SLICES} \
    --on-demand \
    --zone $ZONE \
    --project $PROJECT_ID \
    --enable-debug-logs \
    --command "$COMMAND"
