#!/bin/bash
# submit_probe.sh SCRIPT NAME: run a small diagnostic python script as the exclusive timing job stall_bisect_NAME,
# under a 60 s watchdog; log runs/stall_bisect/NAME.log, then the AGX utilisation 30 s after it ended
B=/Users/aditya/robosim/runs/stall_bisect
S=$1; N=$2
cd /Users/aditya/robosim
nohup scripts/gpu_run.sh stall_bisect_$N timing 5 -- /bin/bash -c "
  /usr/bin/python3 $(dirname $S)/watchdog.py --timeout 120 --stall 60 --log $B/$N.log -- /Users/aditya/robosim/.venv/bin/python $S
  sleep 30
  echo \"[after 30 s] \$(ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '\"[A-Za-z]* Utilization %\"=[0-9]*' | tr '\n' ' ')\" >> $B/$N.log
" > $B/$N.queue.log 2>&1 &
echo "queued stall_bisect_$N (log $B/$N.log)"
