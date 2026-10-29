#!/bin/bash
# A Python for the route's steps that runs inside the route image (route.Dockerfile): the scene VM sets
# SIM2CHIP_SCENE_PYTHON, SIM2CHIP_MAPANYTHING_PYTHON and SIM2CHIP_GSPLAT_PYTHON to this script (wsroute/vm.py).
# The folders in SIM2CHIP_ROUTE_MOUNTS (colon separated) are mounted at the same paths, so every path a step
# reads or writes means the same inside; the Docker socket goes along for the Fixer step, which starts its own
# container. The route's and Hugging Face's variables go in too.
set -euo pipefail
: "${SIM2CHIP_ROUTE_IMAGE:?set SIM2CHIP_ROUTE_IMAGE}"
mounts=()
IFS=: read -ra dirs <<< "${SIM2CHIP_ROUTE_MOUNTS:-$PWD}"
for d in "${dirs[@]}"; do mounts+=(-v "$d:$d"); done
envs=()
for v in $(compgen -e | grep -E '^(SIM2CHIP_|HF_|MOGE_|TOKEN_FACTORY|NEBIUS_API|PYTORCH_)'); do envs+=(-e "$v"); done
exec docker run --rm -i --gpus all --ipc=host --network host -u "$(id -u):$(id -g)" \
  --group-add "$(stat -c %g /var/run/docker.sock)" -v /var/run/docker.sock:/var/run/docker.sock \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 "${envs[@]}" "${mounts[@]}" -w "$PWD" \
  --entrypoint python "$SIM2CHIP_ROUTE_IMAGE" "$@"
