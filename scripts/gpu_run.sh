#!/bin/bash
# scripts/gpu_run.sh NAME KIND MINUTES -- command args...   (KIND: timing | render | train | low)
# Acquires the GPU through the priority queue, runs the command, releases on any exit. The lock records the
# job's own pid (not this wrapper's), and the job is killed if this wrapper is terminated. The body is a
# function so bash parses the whole file before running it (edits cannot break queued instances).
# Timing jobs also check that the device is actually idle before starting (system services and out-of-queue
# processes are invisible to the lock): they wait up to 120 s for device utilisation to drop under 10 %, and
# log the value at start and end so contamination is detectable afterwards.
# While the job runs, runs/gpu_hb/<pid> is touched every 5 s (sandboxed waiters cannot signal the pid; see gpu_lock.alive).
set -u
main() {
  NAME=$1; KIND=$2; MIN=$3; shift 3; [ "$1" = "--" ] && shift
  cd "$(dirname "$0")/.."
  python3 scripts/gpu_lock.py acquire "$NAME" --kind "$KIND" --minutes "$MIN" --pid $$ --cmd "$*"
  # Runaway guard (2026-10-12): GPU work can outlive its killed process (utilisation stays ~100 % with no process owning
  # GPU time) and later wedge the machine. No job of any kind starts on such a GPU: wait up to 10 min for it to drain,
  # then pause the queue, write runs/GPU_RUNAWAY_ALERT and fail this job (a reboot is the only known recovery).
  orphan() { python3 scripts/gpu_top.py 2>/dev/null | grep -q "none >= 1 %" && [ "$(scripts/gpu_util.sh)" -ge 90 ]; }
  if orphan; then
    for i in $(seq 1 120); do sleep 5; orphan || break; done
    if orphan; then
      msg="[gpu_run] $(date +%H:%M:%S) $NAME refused: GPU at $(scripts/gpu_util.sh)% with no owning process for 10 min (runaway GPU work). Queue paused; reboot to recover."
      echo "$msg" | tee -a runs/GPU_RUNAWAY_ALERT; scripts/gpu_pause.sh > /dev/null 2>&1
      python3 scripts/gpu_lock.py release "$NAME" --pid $$ --rc 75; exit 75
    fi
  fi
  if [ "$KIND" = "timing" ]; then
    for i in $(seq 1 60); do u=$(scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && break; sleep 2; done
    echo "[gpu_run] $NAME start: device utilisation ${u:-?}% ($(date +%H:%M:%S))"; python3 scripts/gpu_top.py 2>/dev/null | head -4 | sed "s/^/[gpu_run] gpu_top: /"
  fi
  "$@" &
  CHILD=$!
  python3 scripts/gpu_lock.py setpid "$NAME" --pid $CHILD
  # Heartbeat: a sandboxed waiter cannot see this job's pid and would force-release it; gpu_lock.alive() trusts a fresh beat.
  mkdir -p runs/gpu_hb; HBF=runs/gpu_hb/$CHILD
  ( while kill -0 $CHILD 2>/dev/null; do touch "$HBF"; sleep 5; done ) &
  HBP=$!
  trap 'kill $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; kill $HBP 2>/dev/null; rm -f "$HBF"; python3 scripts/gpu_lock.py release "$NAME" --pid $CHILD --rc ${RC:-130}' EXIT INT TERM HUP
  wait $CHILD
  RC=$?
  [ "$KIND" = "timing" ] && { echo "[gpu_run] $NAME end: device utilisation $(scripts/gpu_util.sh)% ($(date +%H:%M:%S)), exit $RC"; python3 scripts/gpu_top.py 2>/dev/null | head -4 | sed "s/^/[gpu_run] gpu_top: /"; }
  return $RC
}
main "$@"; exit $?
