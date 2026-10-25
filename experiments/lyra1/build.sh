#!/usr/bin/env bash
# Build the Lyra 1 job image (experiments/lyra1/Dockerfile) and push it to the project's Nebius registry.
#   experiments/lyra1/build.sh          # build and push
#   experiments/lyra1/build.sh --local  # build only
# The tag is the hash of the Dockerfile: same file, same tag. Needs `nebius registry configure-helper` once and
# SIM2CHIP_NEBIUS_REGISTRY in .env, like images/build.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && . ./.env; set +a
: "${SIM2CHIP_NEBIUS_REGISTRY:?set SIM2CHIP_NEBIUS_REGISTRY in .env}"
tag=$(grep -v '^#' experiments/lyra1/Dockerfile | sha256sum | cut -c1-12)
image="$SIM2CHIP_NEBIUS_REGISTRY/eyeforge-lyra:$tag"
docker build --platform linux/amd64 -t "$image" experiments/lyra1
[ "${1:-}" = --local ] && { echo "built $image"; exit 0; }
docker push "$image"
echo "pushed $image"
