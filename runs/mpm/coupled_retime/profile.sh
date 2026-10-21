#!/bin/bash
# Where the coupled scenes' time goes on Metal (2026-10-13; same trees as run.sh). One exclusive timing-class job each.
#   parts_*  profile_example_parts.py: synchronized wall time per solver step / collide / MPM step (chunk graph, eager)
#   kprof_*  kernel_profile.py --mode profile (WP_METAL_PROFILE=1): per-kernel GPU time, each dispatch in its own
#            command buffer (isolated), joined with the L4's nsys per-kernel sums by ../../profile/kernels_l4/kjoin.py
# Two-way's frame cost grows over the 2 s run, so it is profiled early (after 5 frames) and late (after 140 frames).
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd)
R=$W/runs/mpm/coupled_retime/profile; mkdir -p $R
PY=.venv-newton152/bin/python
GR=scripts/gpu_run.sh
P=$W/metalsim/physics/mpm/probes
export PYTHONPATH=$W
job() { local id=$1; shift; [ -f $R/$id.json ] && { echo "skip $id"; return; }
        $GR cr_$id timing 20 -- "$@" --json $R/$id.json > $R/$id.log 2>&1; echo "$id exit $?"; }
job parts_twoway_early $PY $P/profile_example_parts.py example:mpm_twoway_coupling --chunk-graph --warmup 5 --frames 20
job parts_twoway_late  $PY $P/profile_example_parts.py example:mpm_twoway_coupling --chunk-graph --warmup 140 --frames 20
job parts_anymal       $PY $P/profile_example_parts.py example:mpm_anymal --chunk-graph --warmup 5 --frames 20
export WP_METAL_PROFILE=1
job kprof_twoway_early $PY $P/kernel_profile.py example:mpm_twoway_coupling --mode profile --warmup 3 --frames 6
job kprof_twoway_late  $PY $P/kernel_profile.py example:mpm_twoway_coupling --mode profile --warmup 140 --frames 6
job kprof_anymal       $PY $P/kernel_profile.py example:mpm_anymal --mode profile --warmup 3 --frames 6
echo PROF_DONE
