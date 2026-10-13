#!/bin/bash
# Throughput A/B: g1_tp_variants.py 4096 (task defaults, 2.5 ms), flat and rough, trees interleaved B C F x2
S=/private/tmp/claude-501/-Users-aditya-robosim/3a553738-d88f-4d8d-ac36-4ecff3ac4bc1/scratchpad/rigid_fma
cd /Users/aditya/robosim
O=runs/fma_rigid_parity/tp; mkdir -p $O
tree() { case $1 in B) echo warp-innate;; C) echo warp-innate-ordered;; F) echo warp-innate-fpcontract;; esac; }
run() { # tag tree terrain [--quick]
  PYTHONPATH=upstream/$2:.:$S WARP_CACHE_PATH=$S/cache_$2 MJW_TP_TERRAIN=$3 MJW_TP_VARIANT="{\"label\":\"$1_$3\"}" \
    .venv/bin/python -c "import warp,sys,runpy;print(\"WARP\",warp.__file__,flush=True);sys.argv=[\"g1_tp_variants.py\",\"4096\"]+sys.argv[1:];runpy.run_path(\"scripts/diagnostics/g1_tp_variants.py\",run_name=\"__main__\")" $4 > $O/$1_$3.log 2>&1; /usr/bin/grep '^WARP\|^\[\|Error\|Traceback' $O/$1_$3.log | /usr/bin/grep -v "^\[gpu" ; }
echo "=== start $(date) power: $(pmset -g batt | head -1)"
for k in B C F; do for t in flat rough; do run warm_$k $(tree $k) $t --quick >/dev/null; done; done
echo "=== warm done $(date +%H:%M:%S)"
for r in 1 2; do for t in flat rough; do for k in B C F; do echo "--- $k$r $t $(date +%H:%M:%S)"; run $k$r $(tree $k) $t; done; done; done
echo "=== end $(date) power: $(pmset -g batt | head -1)"
