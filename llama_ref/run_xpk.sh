#!/usr/bin/env bash

##### Runs SPMD training as an xpk job on a GKE cluster #####
#
# Example: ./run_xpk.sh --lr=0.001 --model_type=70B --batch_size=128 --use_custom_mesh
#
# To run this, a _source_ install of xpk is required to accesss the latest TPU.
#
# Example: pip install git+https://github.com/AI-Hypercomputer/xpk.git@main
#

CLUSTER_NAME=bodaborg-v6e-256
DOCKER_URL=gcr.io/tpu-pytorch/llama3:latest
NUM_SLICES=2
TPU_TYPE=v6e-256
ZONE=us-east5-c
PROJECT_ID=tpu-prod-env-automated

DATETIMESTR=$(date +%Y%m%d-%H%M%S)
COMMAND="python run.py $@"

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
