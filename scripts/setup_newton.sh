#!/bin/zsh
# Installs the MetalSim fork of Newton (solver-level XPBD fixes: revolute angle wrap, joint relaxation,
# lambda-accumulating drives, per-body contact forces) editable, into a SEPARATE environment
# (.venv-newtonfork) so that .venv keeps the Newton of its `newton` extra (upstream newton-physics/newton@45458023,
# 1.7.0.dev0, the commit the validation runs used, plus only the Metal HashTable fix: fork branch
# metalsim-hashtable-metal, pinned in pyproject.toml). The Warp fork (scripts/setup_warp.sh, upstream/warp-innate)
# is shared, editable, by both environments; the other packages mirror .venv's pinned versions.
# Usage: scripts/setup_newton.sh [newton_dir] [venv]   (defaults: upstream/newton, .venv-newtonfork)
# Env: METALSIM_NEWTON_REPO, METALSIM_NEWTON_REF (default: the pinned commit below, head of the fork's metalsim branch:
# the XPBD fixes plus the Metal HashTable / 64-bit atomics fixes that the Warp fork's metalsim branch needs).
set -e
DIR=${1:-upstream/newton}
VENV=${2:-.venv-newtonfork}
REPO=${METALSIM_NEWTON_REPO:-https://github.com/pulipakaa24/newton}     # fork of newton-physics/newton, branch metalsim
PIN=fe7f5bd48010fcab428eac6373e85ba3317f7a42                           # metalsim head, 2026-10-11
REF=${METALSIM_NEWTON_REF:-$PIN}
PATCHES="$(cd "$(dirname "$0")/../patches" && pwd)"
# the Warp fork checkout shared with .venv: upstream/warp-innate on the original machine, otherwise
# setup_warp.sh's default (upstream/warp-metalsim)
if [ -d upstream/warp-innate ]; then WARP_DEFAULT=upstream/warp-innate; else WARP_DEFAULT=upstream/warp-metalsim; fi
WARP=${METALSIM_WARP_DIR:-$WARP_DEFAULT}
if [ ! -d "$DIR" ]; then
  { git clone "$REPO" "$DIR" && git -C "$DIR" checkout -q "$REF"; } || {     # REF: branch, tag or commit
    echo "fork not reachable; upstream newton at 45458023 plus patches/newton/metalsim-fe7f5bd4.diff (the same tree)"
    rm -rf "$DIR"; git clone https://github.com/newton-physics/newton "$DIR"
    git -C "$DIR" checkout -q 45458023f24629c1feffb1f336594808f537db14
    git -C "$DIR" apply "$PATCHES/newton/metalsim-fe7f5bd4.diff"
  }
fi
[ -d "$WARP" ] || scripts/setup_warp.sh "$WARP"
# $VENV/.metalsim_deps marks a completed dependency install: an environment left half-built by a failed run (e.g. the
# Warp checkout missing) is completed on the next run instead of being skipped because its python exists
if [ ! -f "$VENV/.metalsim_deps" ]; then
  [ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12
  if [ -x .venv/bin/python ]; then            # same pinned third-party versions as .venv
    .venv/bin/python -m pip freeze | grep -v '^-e\|^newton @\|^newton-physics\|^warp' > "$VENV/requirements.txt"
    uv pip install -q --python "$VENV/bin/python" -r "$VENV/requirements.txt"
  else
    uv pip install -q --python "$VENV/bin/python" numpy "mujoco>=3.14,<3.15" "torch>=2.14" ninja usd-core newton-usd-schemas
  fi
  uv pip install -q --python "$VENV/bin/python" --no-deps -e "$WARP"
  # the MuJoCo Warp fork checkout .venv uses (upstream/mujoco_warp on the original machine; README's pip install puts
  # it in .venv/src/mujoco-warp)
  MJW=upstream/mujoco_warp
  [ -d "$MJW" ] || MJW=$(.venv/bin/python -c "import mujoco_warp, os; print(os.path.dirname(os.path.dirname(mujoco_warp.__file__)))" 2>/dev/null || true)
  if [ -n "$MJW" ] && [ -d "$MJW" ]; then uv pip install -q --python "$VENV/bin/python" --no-deps -e "$MJW"
  else echo "warning: no MuJoCo Warp fork checkout found; $VENV has no mujoco_warp"; fi
  uv pip install -q --python "$VENV/bin/python" --no-deps -e ".[test]"
  touch "$VENV/.metalsim_deps"
fi
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$DIR"
"$VENV/bin/python" -c "import warp as wp; wp.config.quiet=True; import newton, importlib.metadata as m; print('newton', m.version('newton'), newton.__file__)"
