#!/bin/bash
# L4: per-kernel profile of one eager tick and wall of eager / captured ticks after 70 warm-up ticks (2048 worlds)
source $HOME/parity3_scripts/common.sh
S=$HOME/parity3_scripts/mpm/scale; R=$OUT/rigid_late; mkdir -p $R; rm -f $R/DONE
exec >> $R/run.log 2>&1
set -x
export PANDA_XML=$S/franka_emika_panda/panda.xml
NSYS=$(command -v nsys || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
A="--part rigid --device cuda:0 --worlds 2048 --max-triangle-pairs 8388608 --warmup 70"
for ph in tick captured; do
  wait_for_gpu
  python $S/scale_pour_k.py $A --kprof $ph --kprof-ticks 6 --json $R/wall_$ph.json > $R/wall_$ph.log 2>&1
done
$NSYS profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true \
  -o $R/tick python $S/scale_pour_k.py $A --kprof tick --kprof-ticks 4 --json $R/nsys_tick.json > $R/nsys_tick.log 2>&1
$NSYS stats -r cuda_gpu_kern_sum -f csv --force-export true -o $R/tick $R/tick.nsys-rep
tar czf $OUT/rigid_late.tgz --exclude='*.sqlite' --exclude='*.nsys-rep' -C $OUT rigid_late
touch $R/DONE
