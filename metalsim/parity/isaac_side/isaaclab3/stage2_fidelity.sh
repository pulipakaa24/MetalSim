#!/bin/bash
# Stage 2: fidelity protocol (A_hold / B_random / C_drop, RTX frames every 5 steps, grey ground) on both backends.
source $HOME/parity3_scripts/common.sh
exec > $OUT/stage2_fidelity.log 2>&1
set -x
python -c "import imageio" 2>/dev/null || uv pip install imageio
for B in newton_mjwarp isaacsim_physx; do
  wait_for_gpu; t0=$(date +%s)
  timeout 3600 python $HOME/parity3_scripts/record_g1.py --out $OUT/fidelity/$B --frame_every 5 --viz none physics=$B > $OUT/fidelity_$B.log 2>&1
  echo "fidelity $B exit $? in $(( $(date +%s) - t0 )) s"
done
touch $OUT/FIDELITY_DONE
