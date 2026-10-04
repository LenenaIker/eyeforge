#!/usr/bin/env bash
# Build the worker's Python kits and two images (images/Dockerfile): the kits go to the cache bucket
# (envs/<tag>/), the images to the project's Nebius registry.
#   images/build.sh                # kits and both images
#   images/build.sh kits light     # only some of them
# Needs `nebius registry configure-helper` once, SIM2CHIP_NEBIUS_REGISTRY (cr.<region>.nebius.cloud/<id>), the
# Object Storage key (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) and a Python with boto3 ($PYTHON, default python3).
# The tag is the hash of the Dockerfile (without comments) and the pip lists: same inputs, same tag, nothing to
# push again.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${SIM2CHIP_NEBIUS_REGISTRY:?set SIM2CHIP_NEBIUS_REGISTRY in .env}"
PYTHON=${PYTHON:-python3}

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
python3 - "$ctx" <<'EOF'
import ast, pathlib, sys
src = pathlib.Path("sim2chip/backends/nebius.py").read_text()
deps = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in
        ("WORKER_DEPS", "WORKER_NODEPS", "TEACHER_DEPS", "PICODET_DEPS", "PICODET_NODEPS",
         "PADDLEDET_REF")}
out = pathlib.Path(sys.argv[1])
(out / "worker.txt").write_text("\n".join(deps["WORKER_DEPS"]) + "\n")
(out / "worker-nodeps.txt").write_text("\n".join(deps["WORKER_NODEPS"]) + "\n")
(out / "teacher.txt").write_text("\n".join(deps["TEACHER_DEPS"]) + "\n")
(out / "picodet.txt").write_text("\n".join(deps["PICODET_DEPS"]) + "\n")
(out / "picodet-nodeps.txt").write_text("\n".join(deps["PICODET_NODEPS"]) + "\n")
(out / "paddledet.txt").write_text(deps["PADDLEDET_REF"] + "\n")
EOF
cp images/Dockerfile "$ctx/"
tag=$( (grep -v '^#' "$ctx/Dockerfile"; cat "$ctx/worker.txt" "$ctx/worker-nodeps.txt" "$ctx/teacher.txt" "$ctx/picodet.txt" "$ctx/picodet-nodeps.txt" \
  "$ctx/paddledet.txt") | sha256sum | cut -c1-12)

for target in ${@:-kits light isaac}; do
  if [ "$target" = kits ]; then
    docker build --platform linux/amd64 --target kits --output "type=local,dest=$ctx/kits" "$ctx"
    PYTHONPATH=. "$PYTHON" - "$ctx/kits" "$tag" <<'EOF'
import pathlib, sys
from sim2chip.backends.nebius import ENVS_PREFIX, KITS
from sim2chip.cloud_sync import S3Store
from sim2chip import config
bucket, store = f"{config.env('SIM2CHIP_BUCKET_PREFIX', 'sim2chip')}-cache", S3Store()
for kit in KITS:
    f = pathlib.Path(sys.argv[1]) / f"{kit}.tar.zst"
    store.s3.upload_file(str(f), bucket, f"{ENVS_PREFIX}/{sys.argv[2]}/{f.name}")
    print(f"uploaded {f.name} ({f.stat().st_size / 1e9:.2f} GB) to s3://{bucket}/{ENVS_PREFIX}/{sys.argv[2]}/")
EOF
    continue
  fi
  image="$SIM2CHIP_NEBIUS_REGISTRY/eyeforge-$target:$tag"
  docker build --platform linux/amd64 --target "$target" -t "$image" "$ctx"
  docker push "$image"
  echo "pushed $image"
done
echo "set SIM2CHIP_IMAGE_TAG=$tag in .env (here and on the demo server)"
