import argparse
import subprocess
import sys
import os
import re
import random
import json
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager


def main():
   mantaray_docker_cmd = [
      "/bin/bash", "-c", f"./docker/build_ray_docker.sh"
   ]

   # us-docker.pkg.dev/supercomputer-testing/dlsl/hf-transformers-llama-tpu-nightly-mantaray:nightly
   # us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-tpu-diffusers:v4

   # mantaray_docker_url = f"gcr.io/{args.project}/mantaray-pytorch-xla/transformers/mantaray"
   # cloud-ml-benchmarking
   mantaray_docker_url = f"gcr.io/cloud-ml-benchmarking/mantaray-pytorch-xla/transformers/mantaray"

   subprocess.run(mantaray_docker_cmd,
                  check=True,
                  env=dict(os.environ) | {
                     "BASEIMAGE": f"docker.io/vllm/vllm-tpu",
                     "BASE_LABEL": f"d374f04a337dbd4aab31484b6fa2d4a5f20c2116",
                     "RAY_IMAGE": mantaray_docker_url,
                     "RAY_LABEL": f"vllm-wrapped",
                     })
   
   print("Mantaray docker image build completed")


if __name__ == "__main__":
  main()

