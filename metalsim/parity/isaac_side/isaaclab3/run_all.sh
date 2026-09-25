#!/bin/bash
# parity3 orchestration on the VM: one Kit / GPU job at a time, in order.
exec >> $HOME/parity3/run_all.log 2>&1
echo "start $(date -u +%FT%TZ)"
$HOME/parity3_scripts/stage2_fidelity.sh; echo "fidelity stage done $(date -u +%FT%TZ)"
$HOME/parity3_scripts/stage1_bench.sh;    echo "bench stage done $(date -u +%FT%TZ)"
$HOME/parity3_scripts/stage3_train.sh "flat:newton_mjwarp flat:isaacsim_physx rough:newton_mjwarp"; echo "train stage done $(date -u +%FT%TZ)"
touch $HOME/parity3/ALL_DONE
