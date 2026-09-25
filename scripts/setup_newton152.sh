#!/bin/zsh
# Newton 1.5.2 (the version Isaac Lab 3.0 EA pins, pyproject.toml override-dependencies) in its own environment
# (.venv-newton152) for the VBD / coupled-solver reference runs on Metal. Newton 1.5.2's SolverMuJoCo needs
# mujoco / mujoco-warp ~=3.11 (the MetalSim MuJoCo Warp fork is 3.14 and its contact_force_fn signature differs),
# so this environment uses stock mujoco-warp 3.11.0 with the shared Warp fork (upstream/warp-innate, which carries
# the Metal nextafterf fix Newton's VBD needs). Usage: scripts/setup_newton152.sh [venv]
set -e
VENV=${1:-.venv-newton152}
[ -d upstream/newton-1.5.2 ] || git -C upstream/newton worktree add ../newton-1.5.2 v1.5.2 \
  || git clone --branch v1.5.2 --depth 1 https://github.com/newton-physics/newton upstream/newton-1.5.2
if [ -d upstream/warp-innate ]; then WARP=upstream/warp-innate; else WARP=upstream/warp-metalsim; fi
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12
uv pip install -q --python "$VENV/bin/python" numpy scipy pytest usd-core "mujoco==3.11.0"
uv pip install -q --python "$VENV/bin/python" --no-deps "mujoco-warp==3.11.0"
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$WARP"
uv pip install -q --python "$VENV/bin/python" --no-deps -e upstream/newton-1.5.2
"$VENV/bin/python" -c "import warp as wp; wp.config.quiet=True; import newton, mujoco_warp, importlib.metadata as m; print('newton', m.version('newton'), 'mujoco-warp', m.version('mujoco-warp'), 'warp', wp.__version__)"
