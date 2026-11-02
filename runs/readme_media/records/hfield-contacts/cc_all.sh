#!/bin/bash
# cc_all.sh <tag> <mjw copy> [shapes...]: MuJoCo Warp contacts (CPU device) for each shape, then compare
S=runs/readme_media/scratch/hfield-contacts; T=$1; W=$2; shift 2
for s in "${@:-sphere capsule box cylinder ellipsoid mesh}"; do
  for x in $s; do
    PYTHONPATH=$W .venv/bin/python $S/cc2.py mjw $x $S/cc/$x $T 2>&1 | grep "^mjw" | sed "s|^|[$x] |"
  done
done
