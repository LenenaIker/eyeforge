#!/bin/bash
# GEN3C batch, after its jobs end: fetch the videos, fuse them into the site's scene, render it from every
# reachable view up to 5 m, then metrics and sheets against B_layers. TS (GEN3C's geometry and colour, a TSDF
# fusion) is the kept arm; GL (its colour on our layers) painted 1-2 % of the hidden texels (2026-10-19).
S=${2:-TS}
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=~/miniforge3/envs/edge-sdg/bin/python
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
p=$1
d=data/sphere_test/$p
[ -n "$SKIP_FETCH" ] || $PY experiments/sphere_walk/gen3c_batch.py fetch --place $p
if [ $S = TS ]; then
  $SIM2CHIP_SCENE_PYTHON experiments/sphere_walk/tsdf_gen3c.py --dir $d 2>&1 | tail -1
else
  $SIM2CHIP_SCENE_PYTHON experiments/sphere_walk/fuse_gen3c.py --dir $d 2>&1 | tail -1
fi
only=$($PY -c "
import json; v = json.load(open('$d/walk/views.json'))['views']
print(','.join(x['name'] for x in v if x['reachable'] and (x['move'] == '0' or float(x['move'][1:]) <= 5)))")
$ISAAC experiments/sphere_walk/render_walk.py --dir $d --scene $d/walk/${S}_scene --raw $d/walk/${S}_raw --only $only 2>&1 | grep -E "views in|Error|Traceback"
$PY experiments/sphere_walk/post.py --dir $d --arm $S
rm -rf $d/walk/${S}_raw
$PY experiments/sphere_walk/metrics.py --dir $d --arm B_layers --arm $S 2>/dev/null
$PY experiments/sphere_walk/sheet.py --dir $d --arm B_layers --arm $S --cols 0,f3,f5,b5,r3,r5,l5
# The scene as a .usd to walk in with Isaac's free camera.
$ISAAC experiments/sphere_walk/render_walk.py --dir $d --scene $d/walk/${S}_scene --save $d/walk/${S}_scene/scene.usd 2>&1 | grep -E "saved|Error|Traceback"
