#!/usr/bin/env bash
# Our photo-to-scene pipeline on the HOPE train scenes (first photo of each), for the A/B/C/D student test.
# Usage: bash run_hope_train.sh [scene ...]   e.g. val/000001 test/000006 (default: every train scene but test/000021)
# Supports are written by hand per scene. With PLAN=1 a VLM plans every scene instead (bg_plan.py: support, object
# phrases, --solid) and scenes without one flat support are skipped; outputs then go to <name>_plan.
# Every finished scene is rated by a VLM (bg_review.py); hope_student.py's arm R trains on the kept ones.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
HERE=experiments/scene3d
CASES=data/scene3d_test/hope_train_cases
OURS=data/scene3d_test/ours
OBJECTS="can,box,bottle,carton"
VIEWS=${VIEWS:-20}
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

declare -A SUPPORT=(
  [val/000001]=sofa [val/000006]=table [test/000001]=floor [test/000006]=table [test/000011]=table
  [test/000016]=countertop [test/000021]=table [test/000026]=shelf [test/000031]=table [test/000036]=shelf
)
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(val/000001 val/000006 test/000001 test/000006 test/000011 test/000016 test/000026 test/000031 test/000036)

gpu_temp() {
  local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
  if [ "$t" -ge 87 ]; then echo "GPU at ${t} C, stopping"; exit 1; fi
}

for s in "${SCENES[@]}"; do
  name=hope_${s/\//_}${PLAN:+_plan}
  out=$OURS/$name
  img=data/hope/$s/rgb/000000.png
  sup=${SUPPORT[$s]}
  objects=$OBJECTS
  solid=$OBJECTS
  mkdir -p "$out" "$CASES/logs"
  echo "== $name $(date +%T)"
  if [ -n "${PLAN:-}" ]; then
    rc=0
    [ -f "$out/plan.json" ] || $SAM3 $HERE/bg_plan.py --image "$img" --out "$out" > "$out/plan.log" 2>&1 || rc=$?
    [ "$rc" -eq 1 ] && { echo "no plan, skipping"; continue; }
    read -r fits sup objects solid < <($SAM3 -c "import json; p = json.load(open('$out/plan.json'));
print(int(p['one_flat_support']), p['support'].replace(' ', '_'), ','.join(p['objects']).replace(' ', '_'),
      ','.join(p['solid']).replace(' ', '_') or '-')")
    sup=${sup//_/ }; objects=${objects//_/ }; solid=${solid//_/ }; [ "$solid" = "-" ] && solid=""
    if [ "$fits" = 0 ]; then echo "   skipped: no single flat support ($(grep -o '"reason": "[^"]*' "$out/plan.json" | cut -d'"' -f4))"; continue; fi
    echo "   plan: $sup | $objects | solid $solid"
  fi
  if [ -f "$out/isaac/fit.json" ]; then echo "done, skipping"; continue; fi
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
  gpu_temp
  PYOPENGL_PLATFORM=egl $TRELLIS $HERE/bg_build.py --dir "$out" --meshes "$cdir/$name" --views "$VIEWS" \
      --solid "$solid" --stack 0.2 > "$out/build.log" 2>&1
  gpu_temp
  $ISAAC $HERE/bg_isaac.py --dir "$out" > "$out/isaac.log" 2>&1
  PYOPENGL_PLATFORM=egl $TRELLIS $HERE/view_labels.py --dir "$out" > "$out/labels.log" 2>&1
  # The VLM rates the views; rejected scenes keep their files but stay out of training (review.json "keep").
  $SAM3 $HERE/bg_review.py --dir "$out" > "$out/review.log" 2>&1 || true
  echo "   ok $(date +%T), review $(grep -o '"score": [^,]*' "$out/review.json" | cut -d' ' -f2) $(grep -o '"keep": [a-z]*' "$out/review.json" | cut -d' ' -f2)"
done
