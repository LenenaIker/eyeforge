#!/usr/bin/env bash
# Our photo-to-scene pipeline on any one photo, planned by the VLM (bg_plan.py): support, object phrases, --solid.
# Usage: [LAYOUT=photo] bash run_photo.sh NAME IMAGE    e.g. 003_home_office .../example_cases/images/003_home_office.png
# Output in data/scene3d_test/ours/NAME (sheet_isaac.jpg to look at, isaac/render_NN.png, review.json).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
HERE=experiments/scene3d
CASES=data/scene3d_test/photo_cases
OURS=data/scene3d_test/ours
VIEWS=${VIEWS:-20}
LAYOUT=${LAYOUT:-shuffle}  # photo: objects stay where the photo has them, only the camera moves
MIRA=/media/lenena-iker/Crucial/tools/Mira-Scene
TOOLS=/media/lenena-iker/Crucial/tools
SAM3=$HOME/miniforge3/envs/teacher-sam3/bin/python
TRELLIS=$HOME/miniforge3/envs/trellis2/bin/python
GEO=$TOOLS/envs/mira-geometry/bin/python
CCM=$TOOLS/envs/mira-ccm/bin/python
MESH=/media/lenena-iker/Crucial/envs/eyeforge-sam3d/bin/python
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
export PYTHONPATH=$MIRA/infer_scripts:$MIRA/Mira-CCM/src:$MIRA/UniDataset/src
export MIRA_MOGE2_ROOT=$TOOLS/MoGe MIRA_MOGE2_CHECKPOINT=$TOOLS/mira_ckpt/moge-2-vitl/model.pt

gpu_temp() {
  local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
  if [ "$t" -ge 87 ]; then echo "GPU at ${t} C, stopping"; exit 1; fi
}

name=$1
img=$2
out=$OURS/$name
mkdir -p "$out" "$CASES/logs"
echo "== $name $(date +%T)"
{
  rc=0
  [ -f "$out/plan.json" ] || $SAM3 $HERE/bg_plan.py --image "$img" --out "$out" > "$out/plan.log" 2>&1 || rc=$?
  [ "$rc" -eq 1 ] && { echo "no plan"; exit 1; }
  read -r fits sup objects solid fixed < <($SAM3 -c "import json; p = json.load(open('$out/plan.json'));
print(int(p['one_flat_support']), p['support'].replace(' ', '_'), ','.join(p['objects']).replace(' ', '_'),
    ','.join(p['solid']).replace(' ', '_') or '-', ','.join(p.get('fixed', [])).replace(' ', '_') or '-')")
  sup=${sup//_/ }; objects=${objects//_/ }; solid=${solid//_/ }; [ "$solid" = "-" ] && solid=""
  fixed=${fixed//_/ }; [ "$fixed" = "-" ] && fixed=""
  if [ "$fits" = 0 ]; then echo "   skipped: no single flat support ($(grep -o '"reason": "[^"]*' "$out/plan.json" | cut -d'"' -f4))"; exit 0; fi
  echo "   plan: $sup | $objects | solid $solid | fixed $fixed"
}
if [ -f "$out/isaac/fit.json" ]; then echo "done already"; exit 0; fi
gpu_temp
[ -f "$out/background.png" ] || $SAM3 $HERE/bg_prepare.py --image "$img" --objects "$objects" --support "$sup" \
    --out "$out" > "$out/prepare.log" 2>&1
[ -f "$out/depth_background.npz" ] || $GEO $HERE/bg_depth.py --dir "$out" > "$out/depth.log" 2>&1
gpu_temp
# Mira stages on a one-case directory, so they never touch the other scenes.
cdir=$CASES/$name
if ! ls "$cdir"/$name/mesh/sam3d/*.glb >/dev/null 2>&1; then
  mkdir -p "$cdir"
  printf '{"%s": {"image": "%s", "floor": "%s", "objects": [%s]}}\n' "$name" "$img" "$sup" \
    "$(echo "$objects" | tr ',' '\n' | sed 's/.*/{"phrase": "&", "on": "floor"}/' | paste -sd,)" > "$cdir/cases.json"
  $SAM3 $HERE/mira_prepare.py --cases "$cdir/cases.json" --out "$cdir" --max-objects 20 > "$cdir/prepare.log" 2>&1
  abs=$ROOT/$cdir
  (cd $MIRA/infer_scripts &&
    $GEO 1_estimation_depth.py --data_dir "$abs" --method moge2 --enable_metric > "$abs/01_depth.log" 2>&1 &&
    $CCM 2_inference_CCM.py --demo_dir "$abs" --output_dir "$abs" \
      --ckpt_dir $TOOLS/mira_ckpt/mira-scene/pipeline --use_cropped_condition > "$abs/02_ccm.log" 2>&1 &&
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $MESH 3_inference_mesh.py --demo_dir "$abs" --output_dir "$abs" \
      --ckpt_dir $TOOLS/sam-3d-objects/checkpoints/hf/pipeline.yaml > "$abs/03_mesh.log" 2>&1)
fi
# Each mesh's orientation from Mira's CCM fit: the silhouette fit in bg_build cannot tell front from back.
[ -f "$cdir/$name/ccm_pose.json" ] || $GEO $HERE/bg_ccm_pose.py --case "$cdir/$name" > "$out/ccm_pose.log" 2>&1
gpu_temp
PYOPENGL_PLATFORM=egl $TRELLIS $HERE/bg_build.py --dir "$out" --meshes "$cdir/$name" --views "$VIEWS" \
    --solid "$solid" --stack 0.2 --layout "$LAYOUT" --fixed "$fixed" > "$out/build.log" 2>&1
gpu_temp
$ISAAC $HERE/bg_isaac.py --dir "$out" > "$out/isaac.log" 2>&1
PYOPENGL_PLATFORM=egl $TRELLIS $HERE/view_labels.py --dir "$out" > "$out/labels.log" 2>&1
# The VLM rates the views; rejected scenes keep their files but stay out of training (review.json "keep").
$SAM3 $HERE/bg_review.py --dir "$out" > "$out/review.log" 2>&1 || true
echo "   ok $(date +%T), review $(grep -o '"score": [^,]*' "$out/review.json" | cut -d' ' -f2) $(grep -o '"keep": [a-z]*' "$out/review.json" | cut -d' ' -f2)"
