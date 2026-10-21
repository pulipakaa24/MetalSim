#!/bin/bash
# Final-state verification (each job through the GPU queue): recordings integrated (I) / final (N) and a second N run of
# the coupled scenes, Warp Metal test groups, tests/test_mpm.py, rigid repeat checks, rigid suite.
A=runs/a2
W=upstream
R=.
G=$R/scripts/gpu_run.sh
O=$A/out; mkdir -p $O $A/kc
export PYTHONDONTWRITEBYTECODE=1
echo "start $(date +%H:%M:%S)"
bash $A/rec.sh I $R/upstream/warp-innate-integrated $R/upstream/newton-integrated
bash $A/rec.sh N $W/warp-i2 $W/newton-i2
$R/.venv/bin/python $A/py/bitcmp.py $A/rec/I $A/rec/N > $O/bitwise_I_vs_N.txt; cat $O/bitwise_I_vs_N.txt
# Warp test groups
$G a2v_warp check 40 -- env PYTHONPATH=$W/warp-i2 WARP_CACHE_PATH=$A/kc/wtests bash -c "cd $W/warp-i2 && $R/.venv/bin/python -m unittest \
  warp.tests.test_metal warp.tests.test_metal_primitives warp.tests.test_metal_sparse warp.tests.test_sparse \
  warp.tests.test_metal_deterministic warp.tests.deterministic.test_deterministic_backward \
  warp.tests.deterministic.test_deterministic_counter warp.tests.deterministic.test_deterministic_graph_capture \
  warp.tests.deterministic.test_deterministic_options warp.tests.deterministic.test_deterministic_scatter \
  warp.tests.test_atomic warp.tests.test_atomic_cas warp.tests.test_atomic_bitwise warp.tests.test_graph \
  warp.tests.test_metal_bounded_launch" > $O/warp_tests.log 2>&1
echo "warp tests: $(/usr/bin/grep -E '^Ran|^OK|^FAILED' $O/warp_tests.log | tr '\n' ' ') $(/usr/bin/grep -E '^(ERROR|FAIL):' $O/warp_tests.log | tr '\n' ' ')"
$G a2v_mpm check 40 -- env PYTHONPATH=$W/warp-i2:$W/newton-i2:$R WARP_CACHE_PATH=$A/kc/testmpm \
  bash -c "cd $R && $R/.venv-newton152/bin/python -m pytest -q -rA -p no:cacheprovider tests/test_mpm.py" > $O/test_mpm.log 2>&1
echo "test_mpm: $(tail -1 $O/test_mpm.log)"
for w in 64 2048; do for mode in eager capture; do
  $G a2v_rep_${w}_$mode check 30 -- env PYTHONPATH=$W/warp-i2:$W/newton-i2:$R WARP_CACHE_PATH=$A/kc/rep \
    $R/.venv-newton152/bin/python $R/metalsim/physics/mpm/probes/scale_pour.py --part rigid --device metal:0 --worlds $w \
    --det on --pipeline-det --max-triangle-pairs 8388608 --mode $mode --steps 3 --warmup 2 --repeat-check \
    --json $O/rep_${w}_$mode.json > $O/rep_${w}_$mode.log 2>&1
  echo "repeat $w $mode exit $?"
done; done
$R/.venv/bin/python - $O <<'EOF'
import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + "/rep_*.json")):
    d = json.load(open(f)); print(f.split("/")[-1], {k: v["bitwise_equal"] for k, v in d.items() if k.startswith("repeat")})
EOF
# coupled scenes a second time (run to run)
bash $A/rec.sh N2 $W/warp-i2 $W/newton-i2
$R/.venv/bin/python $A/py/bitcmp.py $A/rec/N $A/rec/N2 > $O/bitwise_N_vs_N2.txt; cat $O/bitwise_N_vs_N2.txt
# rigid suite with the final Warp and the fixed Newton 1.7
SR=$A/suite; mkdir -p $SR
$G a2v_suite check 60 -- env PYTHONPATH=$W/warp-i2:$W/newton-rigid:$R:$R/runs/fma_rigid_parity/harness WARP_CACHE_PATH=$A/kc/suite \
  CAPTURE_OUT=$O/suite_final.jsonl bash -c "cd $R && $R/.venv/bin/python -m pytest -p capture_locals -q -rA --ignore=tests/test_mpm.py -p no:cacheprovider tests" > $O/suite_final.log 2>&1
echo "suite: $(tail -1 $O/suite_final.log)"; /usr/bin/grep "^FAILED" $O/suite_final.log
echo "end $(date +%H:%M:%S)"
