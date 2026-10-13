#!/bin/bash
# Byte identity against the branch base (532577fe): every renderer and sensor on the identity scenes, with this branch's
# defaults and with its new switches off; plus the one tier-2 test that failed in the suite, run on the base tree.
set -u
WT=/Users/aditya/robosim/.claude/worktrees/agent-a5314e27bee02c142
PY=/Users/aditya/robosim/.venv/bin/python
D=$WT/runs/render_gaps2
export IDENTITY_EXTRAS=1
$PY $WT/scripts/diagnostics/render_identity_check.py render --root /Users/aditya/robosim/scratch/rg2_trees/base --out $D/identity_base.npz 2>&1 | grep rendered
$PY $WT/scripts/diagnostics/render_identity_check.py render --root $WT --out $D/identity_cur.npz 2>&1 | grep rendered
$PY $WT/scripts/diagnostics/render_identity_check.py render --root $WT --off --out $D/identity_cur_off.npz 2>&1 | grep rendered
echo "== base vs defaults"; $PY $WT/scripts/diagnostics/render_identity_check.py compare $D/identity_base.npz $D/identity_cur.npz
echo "== base vs switches off"; $PY $WT/scripts/diagnostics/render_identity_check.py compare $D/identity_base.npz $D/identity_cur_off.npz
echo "== base tree: test_env_map_usd_dome_import"
cd /Users/aditya/robosim/scratch/rg2_trees/base && $PY -m pytest -q -p no:cacheprovider tests/test_render_tier2.py -k usd_dome 2>&1 | tail -3
