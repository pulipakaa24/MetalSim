#!/bin/zsh
# Deformables environment (.venv-flex): the MuJoCo Warp fork's metalsim-flex branch (MuJoCo C's flex contact model,
# deterministic flex rows, device-side sorts on Metal) and the Warp fork's metalsim-flex branch (segmented sort inside a
# Metal graph capture), both as git worktrees installed editable, so .venv and the other agents' checkouts are untouched.
# Usage: scripts/setup_flex.sh   (see docs/research/deformables_2026-09-25.md)
set -e
VENV=${1:-.venv-flex}
MJW=upstream/mujoco_warp-flex
WARP=upstream/warp-innate-flex
MJW_PIN=fc1d5eef8f9045309087bc822f36c1c62a7214a2   # head of the MuJoCo Warp fork's metalsim-flex, pinned
[ -d "$MJW" ] || (cd upstream/mujoco_warp && git fetch -q fork metalsim-flex && git worktree add "../mujoco_warp-flex" "$MJW_PIN" -b metalsim-flex 2>/dev/null || git worktree add "../mujoco_warp-flex" "$MJW_PIN")
if [ ! -d "$WARP" ]; then
  (cd upstream/warp-innate && git fetch -q fork metalsim-flex && (git worktree add "../warp-innate-flex" fork/metalsim-flex -b metalsim-flex 2>/dev/null || git worktree add "../warp-innate-flex" metalsim-flex))
  # the native library is unchanged on this branch: reuse the built one (or: cd $WARP && python build_lib.py --no-cuda)
  cp -a upstream/warp-innate/warp/bin/. "$WARP/warp/bin/"
fi
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q "$VENV" --python 3.12
  .venv/bin/python -m pip freeze | grep -v '^-e\|^newton\|^warp\|^mujoco-warp\|^mujoco_warp' > "$VENV/requirements.txt"
  uv pip install -q --python "$VENV/bin/python" -r "$VENV/requirements.txt"
fi
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$WARP" -e "$MJW" -e ".[test]"
"$VENV/bin/python" -c "import warp, mujoco_warp; print(warp.__file__); print(mujoco_warp.__file__)"
