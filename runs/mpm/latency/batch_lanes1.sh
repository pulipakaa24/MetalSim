#!/bin/bash
# Ordering-lanes prototype (research_notes/gpu_latency/LANES.md): build the pinned Warp fork (31cf79d7) plus the lanes
# patch (runs/mpm/latency/warp_lanes.diff) into a separate tree, check that the default single-lane path is unchanged
# (tests/test_mpm.py), then measure robot / sand overlap on two independent ANYmal instances with a bitwise check
# (probes/lanes_overlap_probe.py). Nothing is pushed to the forks; the shared Warp checkout is not touched.
# No spin loops are added (lane waits are not used by the probe); the probe runs under a watchdog. Each GPU job is an
# exclusive gpu_run.sh `timing` job with a health check after it; only build failures and GPU-health problems stop it.
# Log: runs/mpm/latency/batch_lanes1.log. Resumable.
set -u
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
W=/Users/aditya/robosim/.claude/worktrees/agent-a8f809a11de334dd0
R=$W/runs/mpm/latency
O=$R/lanes1
M=/Users/aditya/robosim
PY=$M/.venv-newton152/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
F=$O/failures.txt
mkdir -p $O
exec > >(tee -a "$R/batch_lanes1.log") 2>&1
echo "== $(date) batch_lanes1 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_lanes1 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
sjob() { local name=$1 min=$2 log=$3; shift 4; $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?; health "$name"
         [ $rc -eq 0 ] || { echo "FAIL: $name exit $rc ($log)"; echo "$name exit $rc" >> $F; }; return 0; }

# ---- 1. build the lanes tree (CPU only)
[ -d $T/warp_fork/warp ] || stop "pinned Warp clone missing (upstream/trees/warp_fork)"
[ "$(git -C $T/warp_fork rev-parse HEAD)" = 31cf79d781e4b10d8ff85c8cbd98fc0e40f75953 ] || stop "warp_fork not at 31cf79d7"
L=$T/warp_lanes
if [ ! -f $L/.built ]; then
  rm -rf $L && cp -R $T/warp_fork $L || stop "copy"
  patch -p1 -d $L < $R/warp_lanes.diff || stop "lanes patch"
  (cd $L && $PY build_lib.py --no-cuda --no-standalone > $O/build.log 2>&1) || { tail -30 $O/build.log; stop "build failed (lanes1/build.log)"; }
  tail -3 $O/build.log
  touch $L/.built
fi
export PYTHONPATH=$L:$T/newton_fork:$W WARP_CACHE_PATH=$R/kcache_lanes
$PY -P -c "import warp as wp; wp.config.quiet=True; wp.init(); from warp._src.context import runtime; print('lanes API', hasattr(runtime.core, 'wp_metal_select_lane'), wp.__file__)" | tee $O/api.txt
grep -q 'lanes API True' $O/api.txt || stop "built library has no wp_metal_select_lane"

# ---- 2. default path unchanged: tests/test_mpm.py on the lanes build (lane 0 only)
if ! grep -qE '[0-9]+ passed' $O/test_mpm.log 2>/dev/null; then
  sjob lanes_test_mpm 30 $O/test_mpm.log -- $PY -P -m pytest -q -rf -p no:cacheprovider $W/tests/test_mpm.py
fi
echo "test_mpm on the lanes build: $(grep -E '[0-9]+ (passed|failed)' $O/test_mpm.log | tail -1)"

# ---- 3. overlap prototype (watchdog: 25 min, 120 s without CPU progress)
if ! grep -qE '^(OK|FAIL)$' $O/overlap_watch.log 2>/dev/null; then
  sjob lanes_overlap 30 $O/overlap_job.log -- python3 $W/runs/rigid17/tools/watch.py --log $O/overlap_watch.log --timeout 1500 \
      --stall 120 -- $PY -P $W/metalsim/physics/mpm/probes/lanes_overlap_probe.py --frames 40 --json $O/overlap.json
fi
grep -E 'ms/frame|bitwise|^OK|^FAIL|Error|error' $O/overlap_watch.log | grep -v '^Module'

# ---- 4. commit and push
cd $W
git add runs/mpm/latency research_notes/gpu_latency metalsim/physics/mpm/probes/lanes_overlap_probe.py
git commit -s -q -m "Ordering-lanes prototype for concurrent robot / sand work on Metal (design, patch, logs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q -u origin mpm-latency 2>&1 | grep -v '^remote:'; git log --oneline -1
echo "failures: $(cat $F 2>/dev/null | tr '\n' ';')"
echo "== $(date) batch_lanes1 done"
