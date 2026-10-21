#!/bin/bash
# Item-4 regression check: interleaved A/B. A = integrated-2 (Warp c1bb061b + Newton c1ab1cdc), B = pinned
# (Warp 152808b7 + Newton 35791a07); own clones (upstream/t4, git-ignored), own libwarp builds, own kernel caches;
# metalsim = this worktree; fast_math relaxed; device args "auto" (da=keep) and all-device (da=1).
#   ab.sh warm | ab.sh run REPS [SCENES]
H=.
D=$H/runs/mpm/regress_check; T=$H/upstream/t4; O=${OUT:-$D/out}; mkdir -p $O
PY=.venv-newton152/bin/python
export PYTHONDONTWRITEBYTECODE=1
SCENES=${3:-"granular column_a2"}
args() { case $1 in granular) echo "example:mpm_granular" ;; column_a2) echo "column --set aspect=2" ;; esac; }
one() { # arm da scene rep [frames]
  local arm=$1 da=$2 sc=$3 rep=$4 fr=${5:-60}
  local j=$O/${sc}_${arm}_da${da}_r$rep.json
  (cd $H && MPM_DA=$da WARP_CACHE_PATH=$D/kc_$arm PYTHONPATH=$T/warp-$arm:$T/newton-$arm:$H $PY $D/speed_da.py $(args $sc) \
     --frames $fr --warmup 5 --metal-fast-math relaxed --json $j > ${j%.json}.log 2>&1)
  echo "$sc $arm da=$da r$rep: $($PY -c "import json;print(round(json.load(open('$j'))['ms_median'],2))" 2>&1 | tail -1) util=$(scripts/gpu_util.sh) $(date +%H:%M:%S)"; }
case $1 in
warm) for sc in $SCENES; do for arm in A B; do for da in keep 1; do one $arm $da $sc 0 2; done; done; done ;;
run) for rep in $(seq 1 $2); do for sc in $SCENES; do for da in keep 1; do
       if [ $((rep % 2)) = 1 ]; then one A $da $sc $rep; one B $da $sc $rep; else one B $da $sc $rep; one A $da $sc $rep; fi
     done; done; done ;;
esac
