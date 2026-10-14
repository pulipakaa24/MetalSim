#!/bin/bash
# submit_probe.sh SCRIPT NAME: run a small diagnostic python script as the exclusive timing job stall_bisect_NAME,
# under a 60 s watchdog; log runs/stall_bisect/NAME.log, then the AGX utilisation 30 s after it ended
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
B=${B:-$REPO/runs/stall_bisect}
S=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); N=$2
cd "$REPO"
nohup scripts/gpu_run.sh stall_bisect_$N timing 5 -- /bin/bash -c "
  /usr/bin/python3 '$REPO/scripts/particle_demo/watchdog.py' --timeout 120 --stall 60 --log '$B/$N.log' -- '$REPO/.venv/bin/python' '$S'
  sleep 30
  echo \"[after 30 s] \$(ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '\"[A-Za-z]* Utilization %\"=[0-9]*' | tr '\n' ' ')\" >> '$B/$N.log'
" > "$B/$N.queue.log" 2>&1 &
echo "queued stall_bisect_$N (log $B/$N.log)"
