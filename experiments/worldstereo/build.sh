#!/usr/bin/env bash
# Build the WorldStereo job image (experiments/worldstereo/Dockerfile) and push it to the project's Nebius registry.
#   experiments/worldstereo/build.sh          # build and push
#   experiments/worldstereo/build.sh --local  # build only
#   experiments/worldstereo/build.sh route    # the route image on top of it (route.Dockerfile, scenevm.py), pushed
# The tag is the hash of the Dockerfile: same file, same tag (the route image's: route.Dockerfile and the
# WorldStereo image it starts from, as scenevm.route_tag computes it). Needs `nebius registry configure-helper`
# once and SIM2CHIP_NEBIUS_REGISTRY in .env, like images/build.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
{ set +x; } 2>/dev/null; set -a; [ -f .env ] && . ./.env; set +a
: "${SIM2CHIP_NEBIUS_REGISTRY:?set SIM2CHIP_NEBIUS_REGISTRY in .env}"
tag=$(grep -v '^#' experiments/worldstereo/Dockerfile | sha256sum | cut -c1-12)
image="$SIM2CHIP_NEBIUS_REGISTRY/eyeforge-worldstereo:$tag"
if [ "${1:-}" = route ]; then
  rtag=$({ cat experiments/worldstereo/route.Dockerfile; printf %s "$image"; } | sha256sum | cut -c1-12)
  route="$SIM2CHIP_NEBIUS_REGISTRY/eyeforge-route:$rtag"
  docker build --platform linux/amd64 --build-arg BASE="$image" -t "$route" -f experiments/worldstereo/route.Dockerfile \
    experiments/worldstereo
  docker push "$route"
  echo "pushed $route"
  exit 0
fi
docker build --platform linux/amd64 -t "$image" experiments/worldstereo
[ "${1:-}" = --local ] && { echo "built $image"; exit 0; }
docker push "$image"
echo "pushed $image"
