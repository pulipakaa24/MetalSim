#!/bin/bash
# scale_pour.py on the L4 (cuda:0), same argument sets as the Metal runs. Output ~/parity3/scale_l4/, tarball ~/parity3/scale_l4.tgz
source $HOME/parity3_scripts/common.sh
S=$HOME/parity3_scripts/mpm/scale; R=$OUT/scale_l4; mkdir -p $R
exec >> $R/run.log 2>&1
export PANDA_XML=$S/franka_emika_panda/panda.xml
rm -f $R/DONE
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
python -c "import newton, warp; print('newton', newton.__version__, 'warp', warp.__version__)"
gpu_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && ! pgrep -f "[n]sys|[r]ecord_mpm" >/dev/null; }
while read -r id rest; do
  [ -z "$id" ] && continue; [ -f $R/$id.json ] && { echo "skip $id"; continue; }
  wait_for_gpu; until gpu_idle; do echo "busy $(date +%T)"; sleep 30; done
  t0=$(date +%s); echo "== $id $(date +%T)"
  timeout 1800 python $S/scale_pour.py --device cuda:0 --json $R/$id.json $rest > $R/$id.log 2>&1; rc=$?
  echo "exit $rc $(( $(date +%s)-t0 ))s"; [ $rc -ne 0 ] && tail -4 $R/$id.log
done < $S/sets.txt
tar czf $OUT/scale_l4.tgz -C $OUT scale_l4
touch $R/DONE
