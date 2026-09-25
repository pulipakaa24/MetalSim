#!/bin/zsh
# Installs the MetalSim fork of Newton (solver-level XPBD fixes: revolute angle wrap, joint relaxation,
# lambda-accumulating drives, per-body contact forces) editable, into a SEPARATE environment
# (.venv-newtonfork) so that .venv keeps the pinned upstream Newton (newton-physics/newton@45458023,
# 1.7.0.dev0) that the validation runs use. The Warp fork (scripts/setup_warp.sh, upstream/warp-innate)
# is shared, editable, by both environments; the other packages mirror .venv's pinned versions.
# Usage: scripts/setup_newton.sh [newton_dir] [venv]   (defaults: upstream/newton, .venv-newtonfork)
set -e
DIR=${1:-upstream/newton}
VENV=${2:-.venv-newtonfork}
REPO=${METALSIM_NEWTON_REPO:-https://github.com/pulipakaa24/newton}     # fork of newton-physics/newton, branch metalsim
REF=${METALSIM_NEWTON_REF:-metalsim}
WARP=${METALSIM_WARP_DIR:-upstream/warp-innate}
if [ ! -d "$DIR" ]; then
  git clone --branch "$REF" "$REPO" "$DIR" || {
    echo "fork not reachable; using upstream newton at the validated commit (no MetalSim solver fixes)"
    git clone https://github.com/newton-physics/newton "$DIR"
    (cd "$DIR" && git checkout 45458023f24629c1feffb1f336594808f537db14)
  }
fi
[ -d "$WARP" ] || scripts/setup_warp.sh "$WARP"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q "$VENV" --python 3.12
  if [ -x .venv/bin/python ]; then            # same pinned third-party versions as .venv
    .venv/bin/python -m pip freeze | grep -v '^-e\|^newton @\|^newton-physics\|^warp' > "$VENV/requirements.txt"
    uv pip install -q --python "$VENV/bin/python" -r "$VENV/requirements.txt"
  else
    uv pip install -q --python "$VENV/bin/python" numpy "mujoco>=3.14" "torch>=2.14" usd-core newton-usd-schemas
  fi
  uv pip install -q --python "$VENV/bin/python" --no-deps -e "$WARP"
  [ -d upstream/mujoco_warp ] && uv pip install -q --python "$VENV/bin/python" --no-deps -e upstream/mujoco_warp
  uv pip install -q --python "$VENV/bin/python" --no-deps -e ".[test]"
fi
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$DIR"
"$VENV/bin/python" -c "import warp as wp; wp.config.quiet=True; import newton, importlib.metadata as m; print('newton', m.version('newton'), newton.__file__)"
