#!/bin/bash
# The walk test's kept arms on the given places (3, 5, 8, 13 m): A (the proxy as it is), B_layers (our back layer)
# and B_layers_lama (LaMa per view on what B_layers leaves), then metrics and sheets. Matrix-3D (D) runs apart.
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=~/miniforge3/envs/edge-sdg/bin/python
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
L=experiments/sphere_walk/logs
mkdir -p $L
for p in "$@"; do
  d=data/sphere_test/$p
  echo "== $p $(date +%T)"
  $PY experiments/sphere_walk/prep.py --dir $d
  $ISAAC experiments/sphere_walk/render_walk.py --dir $d 2>&1 | grep -E "views in|Error|Traceback"
  $PY experiments/sphere_walk/post.py --dir $d
  $SIM2CHIP_SAM3_PYTHON experiments/sphere_walk/layers_ours.py --dir $d 2>$L/layers_$p.err | tail -1 || echo "layers failed"
  $ISAAC experiments/sphere_walk/render_walk.py --dir $d --scene $d/walk/BL_scene --raw $d/walk/B_layers_raw 2>&1 | grep -E "views in|Error|Traceback"
  $PY experiments/sphere_walk/post.py --dir $d --arm B_layers
  $SIM2CHIP_SAM3_PYTHON experiments/sphere_walk/fill.py --dir $d --model lama --src B_layers > /dev/null 2>$L/lama_bl_$p.err || echo "lama failed"
  rm -rf $d/walk/raw $d/walk/B_layers_raw  # intermediate: post.py keeps the views and depths
  $PY experiments/sphere_walk/metrics.py --dir $d --arm A --arm B_layers --arm B_layers_lama 2>/dev/null
  $PY experiments/sphere_walk/sheet.py --dir $d --arm A --arm B_layers --arm B_layers_lama
done
