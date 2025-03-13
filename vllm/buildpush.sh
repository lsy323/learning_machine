#!/usr/bin/env bash

if groups "$USER" | grep -qw "docker"; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "build"
$SUDO docker build --network=host -t vllm_lsiyuan .
echo "tag"
$SUDO docker tag vllm_lsiyuan gcr.io/tpu-pytorch/vllm_lsiyuan:latest
echo "upload"
$SUDO docker push gcr.io/tpu-pytorch/vllm_lsiyuan:latest
