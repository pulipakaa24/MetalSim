#!/bin/bash
# hangtest.sh ID TREE N [TEST]   Repeat one Newton test N times (one pytest process each) with the pinned Warp build;
# run under watch.py through the GPU queue (timing kind: exclusive). Default test: the one running at the 2026-10-12
# 02:32 GPU hang. Each repetition's result line goes to out/ID.log.
WT=.
ID=$1; TREE=$2; N=$3
T=${4:-newton/tests/test_collision_pipeline.py::TestDeterministicPipeline::test_cylinder_scale_update_keeps_generic_stage_metal_0}
export PYTHONPATH=$WT/upstream/t4/warp-B:$WT/upstream/$TREE
export WARP_CACHE_PATH=$WT/runs/rigid17/kcache/kcT_$TREE
cd $WT/upstream/$TREE
for i in $(seq 1 $N); do
  r=$(.venv-newtonfork/bin/python $WT/runs/rigid17/tools/pt_all.py -q -p no:cacheprovider "$T" 2>&1 | tail -1)
  echo "rep $i $(date +%H:%M:%S): $r"
done
