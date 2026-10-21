#!/bin/bash
# Newton hashtable / contact tests on CPU + Metal: final Warp + fixed Newton 1.7 (.venv) and fork branch (.venv-newtonfork),
# old Warp (.venv default, upstream/warp-innate) + stock / fixed Newton 1.7
A=runs/a2
W=upstream
R=.
NT="newton.tests.test_hashtable newton.tests.test_contact_matching newton.tests.test_contact_reduction_global newton.tests.test_contact_reduction"
run() { local tag=$1 pp=$2 venv=$3 cwd=$4
  (cd $cwd && PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=$A/kc/nt_$tag PYTHONPATH=$pp $R/$venv/bin/python $A/py/newton_tests.py $NT > $A/out/nt_$tag.log 2>&1)
  echo "$tag: $(/usr/bin/grep RESULT $A/out/nt_$tag.log) $(/usr/bin/grep -E '^(FAIL|ERROR):' $A/out/nt_$tag.log | sed 's/ (.*//' | tr '\n' ' ')"; }
run final_fixed $W/warp-i2:$W/newton-rigid .venv $W/newton-rigid
run final_fork $W/warp-i2:$W/newton-rigid-fork .venv-newtonfork $W/newton-rigid-fork
run old_stock "" .venv $A
run old_fixed $W/newton-rigid .venv $W/newton-rigid
(cd $R && PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=$A/kc/g1_stock PYTHONPATH=$W/warp-i2:$R $R/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_g1_newton_engine.py > $A/out/g1_final_stock.log 2>&1); echo "G1 Newton-engine tests, final Warp + stock Newton 1.7: $(tail -1 $A/out/g1_final_stock.log)"
(cd $R && PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=$A/kc/g1_fixed PYTHONPATH=$W/warp-i2:$W/newton-rigid:$R $R/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_g1_newton_engine.py > $A/out/g1_final_fixed.log 2>&1); echo "G1 Newton-engine tests, final Warp + fixed Newton 1.7: $(tail -1 $A/out/g1_final_fixed.log)"
(cd $R && PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=$A/kc/g1_fork PYTHONPATH=$W/warp-i2:$W/newton-rigid-fork:$R $R/.venv-newtonfork/bin/python -m pytest -q -p no:cacheprovider tests/test_g1_newton_engine.py > $A/out/g1_final_fork.log 2>&1); echo "G1 Newton-engine tests, final Warp + fixed fork: $(tail -1 $A/out/g1_final_fork.log)"
