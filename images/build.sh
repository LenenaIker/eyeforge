#!/usr/bin/env bash
# Build the worker's two images (images/Dockerfile) and push them to the project's Nebius registry.
#   images/build.sh                # both
#   images/build.sh light          # only one
# Needs `nebius registry configure-helper` once, and SIM2CHIP_NEBIUS_REGISTRY (cr.<region>.nebius.cloud/<id>).
# The tag is the hash of the Dockerfile and the pip lists: same inputs, same tag, nothing to push again.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${SIM2CHIP_NEBIUS_REGISTRY:?set SIM2CHIP_NEBIUS_REGISTRY in .env}"

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
python3 - "$ctx" <<'EOF'
import ast, pathlib, sys
src = pathlib.Path("sim2chip/backends/nebius.py").read_text()
deps = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("WORKER_DEPS", "TEACHER_DEPS")}
out = pathlib.Path(sys.argv[1])
(out / "worker.txt").write_text("\n".join(deps["WORKER_DEPS"]) + "\n")
(out / "teacher.txt").write_text("\n".join(deps["TEACHER_DEPS"]) + "\n")
EOF
cp images/Dockerfile "$ctx/"
tag=$(cat "$ctx/Dockerfile" "$ctx/worker.txt" "$ctx/teacher.txt" | sha256sum | cut -c1-12)

for target in "${@:-light isaac}"; do
  for t in $target; do
    image="$SIM2CHIP_NEBIUS_REGISTRY/eyeforge-$t:$tag"
    docker build --platform linux/amd64 --target "$t" -t "$image" "$ctx"
    docker push "$image"
    echo "pushed $image"
  done
done
echo "set SIM2CHIP_IMAGE_TAG=$tag in .env (here and on the demo server)"
