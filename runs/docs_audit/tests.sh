#!/bin/bash
# Docs audit (2026-10-13): re-measure the test-suite counts the docs quote, on this tree (main 3670f4b8 + docs only).
# GPU job (render class: exclusive, so the bit-for-bit GPU tests do not share the device). Run through the queue:
#   scripts/gpu_run.sh docs_audit_tests render 25 -- bash <this file>
# Stages: versions, rigid suite (.venv: tests/ minus test_mpm.py, render tests included), MPM suite (.venv-newton152).
# -rs lists every skip with its reason. The untracked inputs the tests look for (Isaac's g1_minimal.usd, MuJoCo
# Menagerie) are linked in from the main checkout for the run and removed afterwards.
set -u
WT=.
MAIN=.
L=$WT/runs/docs_audit
cd "$WT"
export PYTHONPATH=$WT

mkdir -p assets/isaac/G1 upstream
LINKS=()
[ -e "$MAIN/assets/isaac/G1/g1_minimal.usd" ] && [ ! -e assets/isaac/G1/g1_minimal.usd ] && \
  ln -s "$MAIN/assets/isaac/G1/g1_minimal.usd" assets/isaac/G1/g1_minimal.usd && LINKS+=(assets/isaac/G1/g1_minimal.usd)
[ -e "$MAIN/upstream/mujoco_menagerie" ] && [ ! -e upstream/mujoco_menagerie ] && \
  ln -s "$MAIN/upstream/mujoco_menagerie" upstream/mujoco_menagerie && LINKS+=(upstream/mujoco_menagerie)
cleanup() { for f in ${LINKS[@]+"${LINKS[@]}"}; do rm -f "$f"; done; rmdir assets/isaac/G1 upstream 2>/dev/null; true; }
trap cleanup EXIT

for V in .venv .venv-newton152; do
  "$MAIN/$V/bin/python" - <<'EOF' >> "$L/versions.log" 2>&1
import os, subprocess, sys
import warp as wp
wp.config.quiet = True
import mujoco, metalsim
def head(m):
    d = os.path.dirname(os.path.dirname(m.__file__))
    return subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = [sys.prefix, "warp", head(wp), wp.__version__, "mujoco", mujoco.__version__, "metalsim", metalsim.__file__]
try:
    import mujoco_warp; out += ["mujoco_warp", head(mujoco_warp), mujoco_warp.__file__]
except Exception as e:
    out += ["mujoco_warp", repr(e)]
try:
    import newton; out += ["newton", newton.__version__, head(newton)]
except Exception as e:
    out += ["newton", repr(e)]
print(*out)
EOF
done

echo "##### RIGID SUITE (.venv): tests/ minus test_mpm.py (render tests included); $(date)" > "$L/rigid_suite.log"
"$MAIN/.venv/bin/python" -m pytest -q -rs -p no:cacheprovider tests --ignore=tests/test_mpm.py >> "$L/rigid_suite.log" 2>&1
echo "rigid rc=$?" >> "$L/job.log"

echo "##### MPM SUITE (.venv-newton152): tests/test_mpm.py; $(date)" > "$L/test_mpm.log"
"$MAIN/.venv-newton152/bin/python" -m pytest -q -rs -p no:cacheprovider tests/test_mpm.py >> "$L/test_mpm.log" 2>&1
echo "mpm rc=$?" >> "$L/job.log"
echo done >> "$L/job.log"
