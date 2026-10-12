#!/bin/zsh
# Newton 1.5.2 (the version Isaac Lab 3.0 EA pins) in its own environment (.venv-newton152): MPM on Metal
# (metalsim/physics/mpm, tests/test_mpm.py), Isaac Lab's Franka-Pour at scale, and the VBD / coupled-solver reference
# runs. Newton is the fork's metalsim-1.5.2 branch (v1.5.2 + MetalSim's Metal changes, METALSIM_CHANGES.md there),
# pinned to a commit; Warp is the shared fork checkout of .venv (scripts/setup_warp.sh: upstream/warp-innate on the
# original machine, otherwise upstream/warp-metalsim). Newton 1.5.2's SolverMuJoCo needs mujoco / mujoco-warp ~=3.11
# (the MetalSim MuJoCo Warp fork is 3.14 and its contact_force_fn signature differs), so this environment uses stock
# mujoco-warp 3.11.0. Third-party versions: scripts/requirements-newton152.txt.
# Usage: scripts/setup_newton152.sh [venv]   (run after the .venv install and scripts/setup_warp.sh)
# Env: METALSIM_NEWTON152_REPO (default github.com/pulipakaa24/newton), METALSIM_NEWTON152_REF (branch, tag or
#      commit; default the pinned commit below), METALSIM_NEWTON152_DIR (default upstream/newton-metalsim-1.5.2),
#      METALSIM_WARP_DIR. Stock Newton 1.5.2 (MPM.md's "oracle", no Metal changes):
#      METALSIM_NEWTON152_REPO=https://github.com/newton-physics/newton METALSIM_NEWTON152_REF=v1.5.2 \
#      METALSIM_NEWTON152_DIR=upstream/newton-1.5.2 scripts/setup_newton152.sh .venv-newton152-stock
set -e
VENV=${1:-.venv-newton152}
REPO=${METALSIM_NEWTON152_REPO:-https://github.com/pulipakaa24/newton}     # fork of newton-physics/newton, branch metalsim-1.5.2
PIN=35791a077dbc74d8e5b4cb9d320faa72af2f3492                               # metalsim-1.5.2 head, 2026-10-11
REF=${METALSIM_NEWTON152_REF:-$PIN}
DIR=${METALSIM_NEWTON152_DIR:-upstream/newton-metalsim-1.5.2}
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "$DIR" ]; then
  { git clone -q "$REPO" "$DIR" && git -C "$DIR" checkout -q "$REF"; } || {
    echo "fork not reachable; upstream newton v1.5.2 plus patches/newton/metalsim-1.5.2-35791a07.diff (the same tree)"
    rm -rf "$DIR"; git clone -q https://github.com/newton-physics/newton "$DIR"
    git -C "$DIR" checkout -q cf5378db6002b5c19b86344737cdd40cb8e08f0c     # v1.5.2
    git -C "$DIR" apply "$HERE/../patches/newton/metalsim-1.5.2-35791a07.diff"
  }
elif [ -e "$DIR/.git" ] && [ -z "$METALSIM_NEWTON152_REF" ] && [ "$(git -C "$DIR" rev-parse HEAD)" != "$PIN" ]; then
  echo "note: $DIR is at $(git -C "$DIR" rev-parse --short HEAD), not the pinned ${PIN:0:8} (left as is)"
fi
if [ -d upstream/warp-innate ]; then WARP=upstream/warp-innate; else WARP=upstream/warp-metalsim; fi
WARP=${METALSIM_WARP_DIR:-$WARP}
[ -f "$WARP/warp/bin/libwarp.dylib" ] || { echo "no built Warp fork at $WARP: run scripts/setup_warp.sh first (in .venv)"; exit 1; }
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12
uv pip install -q --python "$VENV/bin/python" -r "$HERE/requirements-newton152.txt"
uv pip install -q --python "$VENV/bin/python" --no-deps "mujoco-warp==3.11.0" "warp-nn==0.3.0"   # warp-nn: example_mpm_anymal's ONNX policy
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$WARP"
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$DIR"
"$VENV/bin/python" -c "import warp as wp; wp.config.quiet=True; import newton, mujoco_warp, importlib.metadata as m; print('newton', m.version('newton'), newton.__file__, 'mujoco-warp', m.version('mujoco-warp'), 'warp', wp.__version__, wp.__file__)"
