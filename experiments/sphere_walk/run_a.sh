#!/bin/bash
# Arm A of the walk test on the given places: prep, Isaac render, post.
set -e
cd "$(dirname "$0")/../.."
PY=~/miniforge3/envs/edge-sdg/bin/python
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
for p in "$@"; do
  d=data/sphere_test/$p
  $PY experiments/sphere_walk/prep.py --dir $d
  $ISAAC experiments/sphere_walk/render_walk.py --dir $d 2>&1 | grep -E "views in|Error|Traceback" || true
  $PY experiments/sphere_walk/post.py --dir $d
done
