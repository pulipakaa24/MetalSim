#!/bin/bash
# Final timing re-check (one timing-class GPU job): MPM ms/frame, integrated (I) vs final (Ndev = newton_compat's device
# arguments, Nauto = Warp's "auto"), interleaved; then G1 4,096-world throughput (g1_tp_variants.py) I vs N, flat + rough.
A=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/runs/a2
W=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/upstream
R=/Users/aditya/robosim
O=$A/out/timing; mkdir -p $O
export PYTHONDONTWRITEBYTECODE=1
cd $R
args() { case $1 in
    granular) echo "example:mpm_granular" ;; multi_material) echo "example:mpm_multi_material" ;;
    twoway) echo "example:mpm_twoway_coupling" ;; twoway_capture) echo "example:mpm_twoway_coupling --set capture=true" ;;
    anymal) echo "example:mpm_anymal" ;; anymal_capture) echo "example:mpm_anymal --set capture=true" ;;
    plate_drag) echo "plate --set mode=\"drag\" --set travel=0.3" ;; column_a8) echo "column --set aspect=8" ;;
    column_a2) echo "column --set aspect=2" ;; esac; }
mpm() { # label warp newton da scene rep
  local j=$O/$5_$1_r$6.json
  MPM_DA=$4 WARP_CACHE_PATH=$A/kc/t_$1 PYTHONPATH=$2:$3:$R $R/.venv-newton152/bin/python $A/py/speed_da.py $(args $5) \
    --frames 60 --warmup 5 --json $j > ${j%.json}.log 2>&1
  echo "$5 $1 r$6: $($R/.venv/bin/python -c "import json;print(round(json.load(open('$j'))['ms_median'],2))" 2>&1 | tail -1)"; }
for rep in 1 2; do for sc in granular multi_material twoway twoway_capture anymal anymal_capture plate_drag column_a8 column_a2; do
  mpm I $R/upstream/warp-innate-integrated $R/upstream/newton-integrated 1 $sc $rep
  mpm Ndev $W/warp-i2 $W/newton-i2 1 $sc $rep
  mpm Nauto $W/warp-i2 $W/newton-i2 keep $sc $rep
done; done
tp() { # tag tree terrain [--quick]
  PYTHONPATH=$2:. WARP_CACHE_PATH=$A/kc/tp_$1 MJW_TP_TERRAIN=$3 MJW_TP_VARIANT="{\"label\":\"$1_$3\"}" \
    .venv/bin/python -c "import warp,sys,runpy;sys.argv=[\"g1_tp_variants.py\",\"4096\"]+sys.argv[1:];runpy.run_path(\"scripts/diagnostics/g1_tp_variants.py\",run_name=\"__main__\")" $4 > $O/tp_$1_$3.log 2>&1
  /usr/bin/grep SUMMARY $O/tp_$1_$3.log; }
for t in flat rough; do
  tp I $R/upstream/warp-innate-integrated $t --quick > /dev/null; tp N $W/warp-i2 $t --quick > /dev/null
  tp I $R/upstream/warp-innate-integrated $t; tp N $W/warp-i2 $t
done
echo TIMING_DONE
