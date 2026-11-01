#!/bin/bash
# job.sh NAME LOG -- command args...
# One pipeline job, run by scripts/gpu_run.sh (the GPU queue) from run_all.sh. Writes a provenance header to LOG (date,
# power source, MetalSim / MuJoCo Warp / Warp commits, the loaded mujoco_warp file and its heightfield flags, warp and
# mujoco versions), writes LOG.meta.json (start, power, loaded MuJoCo Warp) for status.jsonl, then EXECs the command with
# stdout/stderr appended to LOG, so gpu_run.sh's pid bookkeeping and kill reach the job itself.
# RR_PYTHONPATH (optional) selects a MuJoCo Warp copy (the pinned 8eb51fe archive for the A/B); RR_EXPECT_MJW is the
# commit prefix that copy must be (default 93e5ce18, the heightfield fix on the fork's metalsim branch).
set -u
NAME=$1; LOG=$2; shift 2; [ "${1:-}" = "--" ] && shift
R=~/MetalSim; cd "$R" || exit 98
PY=$R/.venv/bin/python
[ -n "${RR_PYTHONPATH:-}" ] && export PYTHONPATH=$RR_PYTHONPATH
EXPECT_MJW=${RR_EXPECT_MJW:-93e5ce18}; EXPECT_WARP=52c09cdd
mkdir -p "$(dirname "$LOG")"; rm -f "$LOG.meta.json"
START=$(date '+%Y-%m-%dT%H:%M:%S')
SRC=$(pmset -g batt | head -1 | sed -e "s/^Now drawing from '//" -e "s/'.*$//")
BATT=$(pmset -g batt | sed -n 2p | sed -e 's/^.*)[[:space:]]*//' | cut -c1-60)
PMODE=$(pmset -g | awk '/powermode/ {print $2}')
WHEAD=$(git -C upstream/warp-metalsim rev-parse HEAD 2>/dev/null)
{
  echo "=== job $NAME start $START on $(sysctl -n hw.model) ($(scutil --get ComputerName 2>/dev/null))"
  echo "power: $SRC | battery $BATT | powermode $PMODE"
  [ "$SRC" = "AC Power" ] || echo "WARNING: on battery at job start (these runs must be on AC power); the job runs anyway, recorded here and in status.jsonl"
  echo "metalsim HEAD $(git rev-parse HEAD) $(git log -1 --format='%ad %s' --date=format:'%Y-%m-%d %H:%M' | cut -c1-100)"
  echo "metalsim local changes in the training / evaluation path (git status --short, empty = clean):"
  git status --short -- metalsim/learn metalsim/physics metalsim/interop metalsim/parity/export_policy.py scripts/diagnostics/g1_rough_transfer.py \
    scripts/diagnostics/g1_tp_variants.py tests/test_terrain.py tests/test_g1_task_terms.py tests/test_solver_presets.py | sed 's/^/    /'
  echo "upstream/mujoco_warp HEAD $(git -C upstream/mujoco_warp rev-parse HEAD) branch $(git -C upstream/mujoco_warp rev-parse --abbrev-ref HEAD) (pinned-fix expectation 93e5ce18)"
  git -C upstream/mujoco_warp status --short | head -5 | sed 's/^/    mjw local change: /'
  echo "upstream/warp-metalsim HEAD $WHEAD (expected $EXPECT_WARP)"
  case "$WHEAD" in $EXPECT_WARP*) ;; *) echo "WARNING: warp-metalsim HEAD is not $EXPECT_WARP";; esac
  echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
} >> "$LOG" 2>&1
RR_EXPECT_MJW=$EXPECT_MJW RR_LOG=$LOG RR_START=$START RR_SRC=$SRC RR_BATT=$BATT RR_PMODE=$PMODE $PY - >> "$LOG" 2>&1 <<'EOF'
import json, os, subprocess
import mujoco, torch, warp as wp
import mujoco_warp
from mujoco_warp._src import collision_convex as cc, collision_gjk as cg
import metalsim
root = os.path.dirname(os.path.dirname(os.path.abspath(mujoco_warp.__file__)))
cf = os.path.join(root, "COMMIT")
if os.path.exists(cf):
    commit, how = open(cf).read().strip(), "COMMIT file of a git archive"
else:
    top = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if top == root else "?"
    how = "git HEAD of the checkout"
flags = {k: getattr(m, k) for m in (cc, cg) for k in dir(m) if k.startswith("HFIELD_")}
print(f"mujoco_warp loaded from {mujoco_warp.__file__} commit {commit} ({how})")
print(f"mujoco_warp heightfield flags {flags}")
print(f"warp {wp.__version__} from {wp.__file__}; mujoco {mujoco.__version__}; torch {torch.__version__}; metalsim from {metalsim.__file__}")
exp = os.environ["RR_EXPECT_MJW"]
if not commit.startswith(exp):
    print(f"WARNING: loaded MuJoCo Warp commit {commit} is not the expected {exp}")
meta = {"start": os.environ["RR_START"], "power": os.environ["RR_SRC"], "battery": os.environ["RR_BATT"], "powermode": os.environ["RR_PMODE"],
        "mjw_commit": commit, "mjw_file": mujoco_warp.__file__, "mjw_expected": commit.startswith(exp),
        "hfield_plane_contacts": flags.get("HFIELD_PLANE_CONTACTS"), "warp": wp.__version__}
json.dump(meta, open(os.environ["RR_LOG"] + ".meta.json", "w"))
EOF
echo "cmd: $*" >> "$LOG"
echo "=== output" >> "$LOG"
exec "$@" >> "$LOG" 2>&1
