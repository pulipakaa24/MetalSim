#!/bin/bash
# scripts/diagnostics/competitors/elliptic_bench.sh MJW_PATH LABEL [MODELS...]
# Elliptic vs pyramidal cone throughput (metalsim_step.py, 4096 envs, matched solver budget) plus the per-kernel
# profile of the elliptic solve, with MuJoCo Warp imported from MJW_PATH (a fork worktree). Extra env vars
# (MJW_*) pass through. Run through the timing queue:
#   scripts/gpu_run.sh ellip_base timing 10 -- scripts/diagnostics/competitors/elliptic_bench.sh \
#       /Users/aditya/robosim/upstream/mujoco_warp-ellip-base base > runs/competitors/elliptic_base.log
set -u
MJW=$1; LABEL=$2; shift 2
MODELS=${*:-"go2 g1"}
ROOT=/Users/aditya/robosim
export PYTHONPATH=$MJW${PYTHONPATH:+:$PYTHONPATH}
export MENAGERIE=${MENAGERIE:-$ROOT/upstream/mujoco_menagerie}
PY=$ROOT/.venv/bin/python
D=$ROOT/scripts/diagnostics/competitors
N=${N:-4096}
echo "LABEL $LABEL mjw=$($PY -c 'import mujoco_warp;print(mujoco_warp.__file__)') head=$(git -C $MJW rev-parse --short HEAD) N=$N env=$(env | grep "^MJW_\|^JAC=\|^NJMAX=" | tr "\n" " ")"
for robot in $MODELS; do
  [ "${BENCH:-1}" = 1 ] || break
  for cone in ${CONES:-pyramidal elliptic}; do
    echo "== $robot $cone"; CONE=$cone $PY $D/metalsim_step.py $robot $N matched 2>&1 | grep -v "^Warp\|^   \|^Module"
  done
done
if [ "${PROFILE:-1}" = 1 ]; then
  for robot in $MODELS; do
    echo "== profile $robot ${PROFILE_CONE:-elliptic}"; CONE=${PROFILE_CONE:-elliptic} WP_METAL_PROFILE=1 $PY $D/metalsim_profile.py $robot $N kernels 2>&1 | grep -v "^Module\|^Warp\|^   Devices\|^   CUDA\|^   Kernel\|^     \"\|^     /" | head -14
  done
fi
