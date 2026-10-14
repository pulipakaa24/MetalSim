#!/bin/bash
# GPU latency in the coupled MPM scenes: measurement only (research_notes/gpu_latency/NOTE.md). No code change kept,
# nothing pushed to the forks. Waits for batch_10.sh to finish first (the overnight GPU queue), then runs each probe as
# an exclusive gpu_run.sh `timing` job with a health check after it; only GPU-health problems stop the batch.
# Trees: the pinned forks (Warp 31cf79d7, Newton addd10cb) as cloned by batch_08 (upstream/trees/{warp_fork,newton_fork}).
#   1  tiled bsr_mv: Warp's kernel vs order-preserving unrolled copies (bitwise + us/launch), under a watchdog
#      (new kernels): IJtm shape (12 rows x 1,170 blocks) and a wide shape (8,192 rows x 96 blocks)
#   2  Gauss-Seidel structure + in-replay pass time + dispatch floor: ANYmal, two-way, granular
#   3  ANYmal robot side by subtraction: stage graphs replayed, us and dispatches per replay (2 runs)
#   4  device utilization sampled at 5 Hz during two-way captured and ANYmal captured timing runs
#   5  commit and push the logs (branch mpm-latency)
# Log: runs/mpm/latency/batch_lat1.log. Resumable. Optional, by the user in another terminal during step 4:
#   sudo powermetrics --samplers gpu_power -i 500 -n 60 > runs/mpm/latency/lat1/powermetrics.txt
set -u
export GIT_AUTHOR_NAME="Aditya Pulipaka" GIT_AUTHOR_EMAIL="adipu@utexas.edu"
export GIT_COMMITTER_NAME="Aditya Pulipaka" GIT_COMMITTER_EMAIL="adipu@utexas.edu"
W=/Users/aditya/robosim/.claude/worktrees/agent-a8f809a11de334dd0
R=$W/runs/mpm/latency
O=$R/lat1
M=/Users/aditya/robosim
PY=$M/.venv-newton152/bin/python
GR=$M/scripts/gpu_run.sh
T=$W/upstream/trees
P=$W/metalsim/physics/mpm/probes
F=$O/failures.txt
mkdir -p $O
exec > >(tee -a "$R/batch_lat1.log") 2>&1
echo "== $(date) batch_lat1 start"
export METALSIM_ROOT=$M
stop() { echo "STOP: $*"; echo "== $(date) batch_lat1 stopped"; exit 1; }
health() {
  [ -e $M/runs/GPU_RUNAWAY_ALERT ] && stop "runs/GPU_RUNAWAY_ALERT exists"
  local u
  for i in $(seq 1 60); do u=$($M/scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && return 0; sleep 2; done
  python3 $M/scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && stop "GPU at ${u}% with no owning process after '$1'"
  echo "note: GPU at ${u}% after '$1', owned by another process"
}
sjob() { local name=$1 min=$2 log=$3; shift 4; $GR "$name" timing "$min" -- "$@" > "$log" 2>&1; local rc=$?; health "$name"
         [ $rc -eq 0 ] || { echo "FAIL: $name exit $rc ($log)"; echo "$name exit $rc" >> $F; }; return 0; }

# ---- 0. wait for the overnight queue (batch_10.sh), then check the trees
while pgrep -f 'batch_10\.sh' > /dev/null; do echo "$(date +%H:%M) waiting for batch_10.sh"; sleep 120; done
[ -d $T/warp_fork/warp ] && [ -d $T/newton_fork/newton ] || stop "pinned fork clones missing (batch_08)"
[ "$(git -C $T/warp_fork rev-parse HEAD)" = 31cf79d781e4b10d8ff85c8cbd98fc0e40f75953 ] || stop "warp_fork not at 31cf79d7"
[ "$(git -C $T/newton_fork rev-parse HEAD)" = addd10cbeb25e946b3a4c2bb1d4097acc7a98ca2 ] || stop "newton_fork not at addd10cb"
export PYTHONPATH=$T/warp_fork:$T/newton_fork:$W WARP_CACHE_PATH=$R/kcache
cd $W

# ---- 1. tiled bsr_mv unroll (new kernels: watchdog)
for shape in ijtm wide; do
  [ -f $O/tiled_$shape.json ] && continue
  case $shape in ijtm) a=(--rows 12 --blocks 1170 --ncol 120000) ;; wide) a=(--rows 8192 --blocks 96 --ncol 60000) ;; esac
  sjob lat_tiled_$shape 10 $O/tiled_$shape.log -- python3 $W/runs/rigid17/tools/watch.py --log $O/tiled_${shape}_watch.log \
      --timeout 400 --stall 60 -- $PY -P $P/tiled_mv_unroll_probe.py "${a[@]}" --json $O/tiled_$shape.json
  grep -v '^Module' $O/tiled_${shape}_watch.log | grep -E 'bitwise|OK|FAIL'
done

# ---- 2. Gauss-Seidel structure and pass time
for s in anymal twoway_coupling granular; do
  [ -f $O/gs_$s.json ] && continue
  sjob lat_gs_$s 15 $O/gs_$s.log -- $PY -P $P/gs_structure_probe.py example:mpm_$s --json $O/gs_$s.json
  grep -v '^Module' $O/gs_$s.log | grep -E '^\{|colour 0:' | cut -c1-400
done

# ---- 3. robot side by subtraction
for rep in 1 2; do
  [ -f $O/robot_r$rep.json ] && continue
  sjob lat_robot_r$rep 15 $O/robot_r$rep.log -- $PY -P $P/robot_graph_probe.py --reps 200 --json $O/robot_r$rep.json
  grep -E 'us/replay|stage sum|error' $O/robot_r$rep.log
done

# ---- 4. device utilization during timing runs
for s in twoway_capture anymal_capture; do
  [ -f $O/busy_$s.json ] && continue
  case $s in
    twoway_capture) a=(--scene example:mpm_twoway_coupling --set capture=true) ;;
    anymal_capture) a=(--scene example:mpm_anymal --set capture=true) ;;
  esac
  sjob lat_busy_$s 15 $O/busy_$s.log -- python3 $P/gpu_busy_sampler.py --out $O/busy_$s.json -- $PY -P \
      $W/metalsim/parity/isaac_side/record_mpm.py --device metal:0 --chunk-graph --gpu-exit --no-log --seconds 2 --stride 1000 \
      "${a[@]}" --out $O/busy_${s}_rec
  grep -E 'device utilization mean|^done' $O/busy_$s.log
done

# ---- 5. commit and push
git add runs/mpm/latency research_notes/gpu_latency metalsim/physics/mpm/probes/gs_structure_probe.py \
    metalsim/physics/mpm/probes/robot_graph_probe.py metalsim/physics/mpm/probes/tiled_mv_unroll_probe.py \
    metalsim/physics/mpm/probes/gpu_busy_sampler.py runs/mpm/coupled_speed/batch_09.log runs/mpm/coupled_speed/b09
git commit -s -q -m "GPU latency in the coupled MPM scenes: research note, probes, measurement logs (batch_lat1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit)"
git push -q -u origin mpm-latency 2>&1 | grep -v '^remote:'; git log --oneline -1
echo "failures: $(cat $F 2>/dev/null | tr '\n' ';')"
echo "== $(date) batch_lat1 done"
