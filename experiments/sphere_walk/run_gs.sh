#!/bin/bash
# The splat test (docs/plan_gs_gen3c_2026-10-20.md) for one site after gs_train.py: each GS arm to USDZ, every arm
# rendered in Isaac on the walk views (post.py, metrics.py) and on GS1's held-out GEN3C frames (gs_eval.py).
#   experiments/sphere_walk/run_gs.sh data/sphere_test/<place>
set -euo pipefail
D=$1
W=$D/walk
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
EDGE=~/miniforge3/envs/edge-sdg/bin/python
GS=/media/lenena-iker/Crucial/tools/envs/gsplat/bin/python
GRUT=/media/lenena-iker/Crucial/tools/3dgrut
GS_ARMS=${GS_ARMS:-"GS1 GS0"}
MESH_ARMS=${MESH_ARMS:-"BL_scene N01_scene"}

for a in $GS_ARMS; do
  (cd $GRUT && source .venv/bin/activate && python "$ROOT/experiments/sphere_walk/gs_usd.py" "$ROOT/$W/$a/splat.ply")
  $ISAAC experiments/sphere_walk/render_walk.py --dir $D --scene $W/$a --raw $W/${a}_raw 2>&1 | grep "views in"
  for f in $W/$a/depth/*.npy; do cp "$f" "$W/${a}_raw/depth_$(basename "$f")"; done  # NuRec gives no depth
  $EDGE experiments/sphere_walk/post.py --dir $D --arm $a
  $EDGE experiments/sphere_walk/metrics.py --dir $D --arm $a || true  # exits 1 after writing its json
  rm -rf $W/${a}_raw
done
for a in $GS_ARMS $MESH_ARMS; do
  [ -d $W/${a}_hold_raw ] && [[ " $MESH_ARMS " == *" $a "* ]] && continue  # a mesh arm does not change
  $ISAAC experiments/sphere_walk/render_walk.py --dir $D --scene $W/$a --views $W/GS1/holdout.json \
    --raw $W/${a}_hold_raw 2>&1 | grep "views in"
done
(cd experiments/sphere_walk && $GS gs_eval.py --dir "$ROOT/$D" --holdout GS1 $(for a in $MESH_ARMS $GS_ARMS; do echo --arm $a; done))
