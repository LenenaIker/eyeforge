#!/bin/bash
# Waits for the GPU to be free, then renders the slowed orbit clip and the student camera's shots with their masks.
set -e
cd /home/lenena-iker/work/eyeforge
R=/tmp/claude-1000/-home-lenena-iker-work-eyeforge/9e904f42-68bb-4fc4-95f4-bd7448ae89e4/scratchpad/overlay/replicator
ISAAC=/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh
PY=$(grep -E "^SIM2CHIP_GSPLAT_PYTHON=" .env | cut -d= -f2-)
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "gpu free $(date +%T)"
PYTHONPATH=$PWD $PY $R/bg_near.py --dir $R/run 2>&1 | grep -c sim2chip
PYTHONPATH=$PWD $ISAAC -m sim2chip.walkscene.isaac --dir $R/run > $R/isaac.log 2>&1
echo "orbit done $(date +%T)"
python3 $R/compose.py $R/run $R/replicator.mp4
rm -rf $R/shots_run; python3 $R/make_shots.py $R/run $R/shots_run
PYTHONPATH=$PWD $PY -m sim2chip.walkscene.background --dir $R/shots_run 2>&1 | grep -c sim2chip
PYTHONPATH=$PWD $ISAAC $R/isaac_idx.py --dir $R/shots_run > $R/isaac_shots.log 2>&1
rm -rf $R/thumbs; python3 $R/seg_thumbs.py $R/shots_run $R/thumbs
echo "all done $(date +%T)"
