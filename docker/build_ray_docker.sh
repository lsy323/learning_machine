#!/bin/bash
set -x
set -e
SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Defines the base image for building ray support ontop of
#k: ${BASEIMAGE:="gcr.io/tpu-prod-env-multipod/lizhiyu-maxtext_base_image-2024_02_04"}
: ${BASEIMAGE:="gcr.io/supercomputer-testing/megatron-lm"}
: ${BASE_LABEL:="latest"}
BASE_TAG="${BASEIMAGE}:${BASE_LABEL}"

# Defines where to push the ray image
# by default, push to same repo with label `latest-ray` and a user-timestamp tag
: ${RAY_IMAGE:=${BASEIMAGE}}
: ${RAY_LABEL:="${BASE_LABEL}-ray"}
: ${RAY_TAG:="${RAY_IMAGE}:${RAY_LABEL}"}

# Add an additional build tag "ray-$USER-$TIMESTAMP"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BUILD_LABEL="ray-${USER}-${TIMESTAMP}"
BUILD_TAG="${RAY_IMAGE}:${BUILD_LABEL}"

pushd "${SCRIPTS_DIR}/.."

docker pull "$BASEIMAGE:$BASE_LABEL"
BASE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$BASEIMAGE:$BASE_LABEL")

docker build -f docker/ray.Dockerfile -t "$BUILD_TAG" \
    --build-arg "BASEIMAGE=${BASEIMAGE}:${BASE_LABEL}" \
    --build-arg "BASE_DIGEST=${BASE_DIGEST}" \
    --build-arg "MY_RAY_WORKER_DOCKER_IMAGE=${RAY_TAG}" \
    .

docker push "$BUILD_TAG"
docker tag "$BUILD_TAG" "$RAY_TAG"
docker push "$RAY_TAG"

popd

echo "Successfully built and pushed MantaRay docker to ${BUILD_TAG}"
echo "Also to ${RAY_TAG}"
