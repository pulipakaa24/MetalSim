#!/bin/bash
until [ -f $HOME/parity3/CHAIN_DONE ]; do sleep 30; done
source $HOME/parity3_scripts/common.sh
S=$HOME/parity3_scripts/mpm/scale; R=$OUT/scale_l4; export PANDA_XML=$S/franka_emika_panda/panda.xml
for id in mpm_rep_off_2048 mpm_rep_on_2048; do
  det=""; [ $id = mpm_rep_on_2048 ] && det="--det on"
  timeout 1800 python $S/scale_pour.py --device cuda:0 --json $R/${id}_v2.json --part mpm --worlds 2048 --mode eager --steps 4 $det --repeat-check > $R/${id}_v2.log 2>&1
  echo "$id exit $?" >> $R/rerun.log
done
tar czf $OUT/scale_l4_rerun.tgz -C $OUT scale_l4/mpm_rep_off_2048_v2.json scale_l4/mpm_rep_on_2048_v2.json scale_l4/mpm_rep_off_2048_v2.log scale_l4/mpm_rep_on_2048_v2.log scale_l4/rerun.log
touch $OUT/RERUN_DONE
