#!/bin/bash
# stop.sh  (SMOKE=1 stop.sh for the smoke pipeline). Stops the rough re-run pipeline safely:
#  1. touch STOP: the driver starts no new job (restart later with: rm STOP; FROM=<job> nohup ... run_all.sh, see hand-off);
#  2. TERM to the running job's scripts/gpu_run.sh: its trap kills the job (training python / job script) and releases
#     the GPU lock (runs/.gpu_lock); a waiting (queued) gpu_run.sh just drops its ticket;
#  3. TERM to the driver (it exits after gpu_run.sh returns), then shows the queue.
R=~/MetalSim; O=$R/runs/rough_rerun_2026-10-31; [ "${SMOKE:-0}" = 1 ] && O=$O/smoke
PFX=rr_; [ "${SMOKE:-0}" = 1 ] && PFX=rr_smoke_
touch $O/STOP; echo "STOP file: $O/STOP"
for p in $(pgrep -f "scripts/gpu_run.sh $PFX"); do echo "TERM gpu_run.sh pid $p: $(ps -o command= -p $p | cut -c1-120)"; kill -TERM $p 2>/dev/null; done
DP=$(cat $O/driver.pid 2>/dev/null); [ -n "$DP" ] && { echo "TERM driver pid $DP"; kill -TERM $DP 2>/dev/null; }
sleep 8; pgrep -fl "metalsim.learn.g1_velocity|g1_rough_transfer|rough_rerun_2026-10-31" | grep -v pgrep | sed 's/^/still running: /'
python3 $R/scripts/gpu_lock.py status
