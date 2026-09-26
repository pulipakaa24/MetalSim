#!/bin/bash
# elliptic_check.sh MJW_PATH LABEL [REF_LABEL]: elliptic_check.py for go2 and g1 (CONE=elliptic) with MuJoCo Warp from
# MJW_PATH; writes runs/competitors/ellip_check_<robot>_<LABEL>.npz and compares with <REF_LABEL>'s files if given.
set -u
MJW=$1; LABEL=$2; REF=${3:-}
ROOT=/Users/aditya/robosim
export PYTHONPATH=$MJW${PYTHONPATH:+:$PYTHONPATH}
export MENAGERIE=${MENAGERIE:-$ROOT/upstream/mujoco_menagerie}
export CONE=${CONE:-elliptic}
for robot in ${MODELS:-go2 g1}; do
  out=$ROOT/runs/competitors/ellip_check_${robot}_${LABEL}.npz
  refarg=""; [ -n "$REF" ] && refarg="--ref $ROOT/runs/competitors/ellip_check_${robot}_${REF}.npz"
  $ROOT/.venv/bin/python $ROOT/scripts/diagnostics/competitors/elliptic_check.py $robot ${N:-512} ${STEPS:-200} --out $out $refarg 2>&1 | grep -v "^Warp\|^   Devices\|^   CUDA\|^   Kernel\|^     \"\|^     /"
done
