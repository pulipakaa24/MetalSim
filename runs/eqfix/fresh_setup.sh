#!/bin/bash
# Fresh-setup check: a new venv, then the README's MuJoCo Warp install line exactly as written; report the installed commit.
set -e
WT=.
D=$WT/scratch/fresh
rm -rf "$D" && mkdir -p "$D" && cd "$D"
"$(uv python find 3.12)" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
LINE=$(grep -m1 '^pip install -e "git+https://github.com/pulipakaa24/mujoco_warp@' "$WT/README.md")
echo "README line: $LINE"
eval "$LINE"
SRC=$(python -c "import os, mujoco_warp; print(os.path.dirname(os.path.dirname(mujoco_warp.__file__)))")
echo "mujoco_warp imported from $SRC"
echo "installed commit: $(git -C "$SRC" rev-parse HEAD)"
git -C "$SRC" log --oneline -3
python -c "import mujoco; print('mujoco', mujoco.__version__)"
grep -n "efcid + 3 > njmax_in\|efcid + 6 > njmax_in" "$SRC/mujoco_warp/_src/constraint.py"
