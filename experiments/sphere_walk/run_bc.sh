#!/bin/bash
# Arms B (LaMa, Qwen, our layers) and C (WorldGen layers) of the walk test after run_a.sh, then metrics and sheets.
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=~/miniforge3/envs/edge-sdg/bin/python
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
WG=/media/lenena-iker/Crucial/envs/worldgen/bin/python
L=experiments/sphere_walk/logs
mkdir -p $L
for p in "$@"; do
  d=data/sphere_test/$p
  echo "== $p $(date +%T)"
  $SIM2CHIP_SAM3_PYTHON experiments/sphere_walk/fill.py --dir $d --model lama > $L/lama_$p.log 2>&1 || echo "lama failed"
  echo "lama done $(date +%T)"
  $SIM2CHIP_ERASE_PYTHON experiments/sphere_walk/fill.py --dir $d --model qwen > $L/qwen_$p.log 2>&1 || echo "qwen failed"
  echo "qwen done $(date +%T)"
  HF_HOME=/media/lenena-iker/Crucial/tools/hf_cache $WG experiments/sphere_walk/worldgen_layers.py --dir $d > $L/worldgen_$p.log 2>&1 || echo "worldgen failed"
  tail -1 $L/worldgen_$p.log
  $ISAAC experiments/sphere_walk/render_walk.py --dir $d --scene $d/walk/C_scene --raw $d/walk/C_raw 2>&1 | grep -E "views in|Error|Traceback"
  $PY experiments/sphere_walk/post.py --dir $d --arm C
  $WG experiments/sphere_walk/layers_ours.py --dir $d > $L/layers_$p.log 2>&1 || echo "layers failed"
  tail -1 $L/layers_$p.log
  $ISAAC experiments/sphere_walk/render_walk.py --dir $d --scene $d/walk/BL_scene --raw $d/walk/B_layers_raw 2>&1 | grep -E "views in|Error|Traceback"
  $PY experiments/sphere_walk/post.py --dir $d --arm B_layers
  $PY experiments/sphere_walk/metrics.py --dir $d --arm A --arm B_lama --arm B_qwen --arm B_layers --arm C 2>/dev/null
  $PY experiments/sphere_walk/sheet.py --dir $d --arm A --arm B_lama --arm B_qwen --arm B_layers --arm C
done
