#!/bin/bash
# CPU evals of the so101-servo stream (so101Sim's venv), 100 seeds each, 4 in parallel. From this directory:
#   bash run_cpu_evals.sh <so101Sim root>
SO=$1; PY=$SO/urlab_bridge/.venv/bin/python
ORIG=../../../film/so101_lift/policy/so101_lift_ppo.pt
CAND=../../../film/so101_lift_smooth/policy/candidates
run() { n=$1; shift; echo "[cpu] $(date +%T) start $n"; $PY eval_cpu.py $SO "$@" > logs/cpu_$n.log 2>&1; echo "[cpu] $(date +%T) done $n: $(tail -1 logs/cpu_$n.log | cut -c1-120)"; }
for s in 7v4 7v4_real; do
  run expert_$s expert $s cpu/expert_$s 100 &
  run orig_$s $ORIG $s cpu/orig_$s 100 &
  run isaac_scaled_$s $CAND/isaac_scaled.pt $s cpu/isaac_scaled_$s 100 &
  run isaac_unscaled_raw_$s $CAND/isaac_unscaled_raw.pt $s cpu/isaac_unscaled_raw_$s 100 &
  wait
done
run expert_7v4_freshkin expert 7v4 cpu/expert_7v4_freshkin 100 --fresh-kin &
run expert_7v4_real_freshkin expert 7v4_real cpu/expert_7v4_real_freshkin 100 --fresh-kin &
run expert_12v_real expert 12v_real cpu/expert_12v_real 100 &
run expert_12v expert 12v cpu/expert_12v 100 &
wait
echo "[cpu] $(date +%T) all done"
