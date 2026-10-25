#!/bin/bash
# Runs run_walk.sh on each sunny place as soon as its extraction (s3_sheet.py) has written its sheet.
cd "$(dirname "$0")/../.."
for f in data/real/photospheres-sunny/*.PHOTOSPHERE.jpg; do
  p=$(basename $f .PHOTOSPHERE.jpg)
  until [ -f data/sphere_test/sheet_s3_$p.jpg ]; do sleep 30; done
  experiments/sphere_walk/run_walk.sh $p
done
echo ALL DONE
