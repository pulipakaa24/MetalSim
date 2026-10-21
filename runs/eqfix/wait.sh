#!/bin/bash
# Wait up to $1 seconds for the eqfix_tests job to finish; print the stage summaries and queue status.
L=runs/eqfix
end=$(( $(date +%s) + ${1:-540} ))
while [ "$(date +%s)" -lt "$end" ]; do
  grep -q "^done" "$L/job.log" 2>/dev/null && break
  pgrep -f "gpu_run.sh eqfix_tests" > /dev/null || break
  sleep 15
done
for f in versions cap_tests fork_tests rigid_suite render_suite; do [ -f "$L/$f.log" ] && echo "$f: $(tail -1 "$L/$f.log")"; done
tail -2 "$L/job.log"
python3 scripts/gpu_lock.py status 2>&1 | cut -c1-120
