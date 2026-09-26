#!/bin/bash
# scripts/gpu_run.sh NAME KIND MINUTES -- command args...   (KIND: timing | render | train | low)
# Acquires the GPU through the priority queue, runs the command, releases on any exit. The lock records the
# job's own pid (not this wrapper's), and the job is killed if this wrapper is terminated. The body is a
# function so bash parses the whole file before running it (edits cannot break queued instances).
# Timing jobs also check that the device is actually idle before starting (system services and out-of-queue
# processes are invisible to the lock): they wait up to 120 s for device utilisation to drop under 10 %, and
# log the value at start and end so contamination is detectable afterwards.
set -u
main() {
  NAME=$1; KIND=$2; MIN=$3; shift 3; [ "$1" = "--" ] && shift
  cd "$(dirname "$0")/.."
  python3 scripts/gpu_lock.py acquire "$NAME" --kind "$KIND" --minutes "$MIN" --pid $$
  if [ "$KIND" = "timing" ]; then
    for i in $(seq 1 60); do u=$(scripts/gpu_util.sh); [ "${u:-0}" -lt 10 ] && break; sleep 2; done
    echo "[gpu_run] $NAME start: device utilisation ${u:-?}% ($(date +%H:%M:%S))"; python3 scripts/gpu_top.py 2>/dev/null | head -4 | sed "s/^/[gpu_run] gpu_top: /"
  fi
  "$@" &
  CHILD=$!
  python3 scripts/gpu_lock.py setpid "$NAME" --pid $CHILD
  trap 'kill $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; python3 scripts/gpu_lock.py release "$NAME" --pid $CHILD' EXIT INT TERM HUP
  wait $CHILD
  RC=$?
  [ "$KIND" = "timing" ] && { echo "[gpu_run] $NAME end: device utilisation $(scripts/gpu_util.sh)% ($(date +%H:%M:%S)), exit $RC"; python3 scripts/gpu_top.py 2>/dev/null | head -4 | sed "s/^/[gpu_run] gpu_top: /"; }
  return $RC
}
main "$@"; exit $?
