#!/bin/bash
# submit_stall_bisect.sh STAGE [OUT_DIR] [CODE_DIR]: queue one stall-bisect stage as the exclusive timing job stall_bisect_STAGE
# (scripts/gpu_run.sh), running the frozen code copy in OUT_DIR/code under an outer watchdog; log OUT_DIR/stageSTAGE.log
STAGE=$1; B=${2:-/Users/aditya/robosim/runs/stall_bisect}
C=${3:-$B/code}
cd /Users/aditya/robosim
CODE=$C nohup scripts/gpu_run.sh stall_bisect_$STAGE timing 15 -- /usr/bin/python3 $C/scripts/particle_demo/watchdog.py \
  --timeout 1500 --stall 150 --log $B/stage$STAGE.log -- /bin/bash $C/scripts/particle_demo/stall_bisect.sh $STAGE $B \
  > $B/stage$STAGE.queue.log 2>&1 &
echo "queued stall_bisect_$STAGE (log $B/stage$STAGE.log)"
