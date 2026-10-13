#!/bin/bash
# L4 side of the per-kernel Metal-vs-CUDA comparison (run on aditya-isaac, after copying l4_kernel_profile.py,
# scenes.py and newton_compat.py to ~/parity3_scripts/mpm/). Output ~/parity3/mpm_kprof/, tarball mpm_kprof.tgz.
#   gcloud compute scp --zone us-west1-a --tunnel-through-iap --project mcmc-193822 \
#       metalsim/physics/mpm/probes/l4_kernel_profile.{py,sh} metalsim/physics/mpm/{scenes,newton_compat}.py \
#       aditya-isaac:parity3_scripts/mpm/
#   gcloud compute ssh ... aditya-isaac --command 'nohup bash ~/parity3_scripts/mpm/l4_kernel_profile.sh > /dev/null 2>&1 &' < /dev/null
source $HOME/parity3_scripts/common.sh
M=$HOME/parity3_scripts/mpm; R=$OUT/mpm_kprof; mkdir -p $R; rm -f $R/DONE
exec >> $R/run.log 2>&1
set -x
NSYS=$(command -v nsys || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv
python $M/l4_kernel_profile.py --chain --json $R/chain.json
for ex in granular twoway_coupling multi_material anymal; do
  wait_for_gpu
  python $M/l4_kernel_profile.py example:mpm_$ex --frames 8 --json $R/timing_$ex.json      # wall without nsys
  $NSYS profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite true -o $R/$ex python $M/l4_kernel_profile.py example:mpm_$ex --frames 4
  $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export true -o $R/$ex $R/$ex.nsys-rep
done
for ex in twoway_coupling anymal; do
  wait_for_gpu
  python $M/l4_kernel_profile.py example:mpm_$ex --frames 8 --set capture=true --json $R/timing_${ex}_capture.json
  $NSYS profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true \
      -o $R/${ex}_capture python $M/l4_kernel_profile.py example:mpm_$ex --frames 4 --set capture=true
  $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export true -o $R/${ex}_capture $R/${ex}_capture.nsys-rep
done
tar czf $OUT/mpm_kprof.tgz --exclude='*.sqlite' -C $OUT mpm_kprof
touch $R/DONE
