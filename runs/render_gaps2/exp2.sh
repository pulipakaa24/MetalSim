#!/bin/bash
# After specializing the ray kernels (GRID_PLANES, HAZE_ON): the any-hit shadow fast path on / off, and the cost of the
# infinite-plane continuation where the scene has one (cloth_inf: the cloth scene's floor infinite). One process each.
set -u
WT=/Users/aditya/robosim/.claude/worktrees/agent-a5314e27bee02c142
PY=/Users/aditya/robosim/.venv/bin/python
OUT=$WT/runs/render_gaps2/exp2.jsonl
run() {  # scene renderer envs envvar json tag
  echo "== $*"
  env $4="$5" $PY $WT/scripts/diagnostics/render_perf_profile.py --scene $1 --renderer $2 --envs $3 --features none \
      --all-off 0 --profile 0 --tag "$6" --out $OUT 2>&1 | grep '^{'
}
run cloth tier1 1024 METALSIM_TIER0_EXP '{}' "t1 specialized"
run cloth tier2 1024 METALSIM_PT_EXP '{}' "t2 specialized anyhit"
run cloth tier2 1024 METALSIM_PT_EXP '{"TRANSMIT_ANYHIT":0}' "t2 specialized no-anyhit"
run glass tier2 64 METALSIM_PT_EXP '{}' "glass t2 anyhit"
run glass tier2 64 METALSIM_PT_EXP '{"TRANSMIT_ANYHIT":0}' "glass t2 no-anyhit"
run cloth_inf tier1 1024 METALSIM_TIER0_EXP '{}' "inf t1 planes=1"
run cloth_inf tier1 1024 METALSIM_TIER0_EXP '{"GRID_PLANES":0}' "inf t1 planes forced 0"
run cloth_inf tier2 1024 METALSIM_PT_EXP '{}' "inf t2 planes=1"
run cloth_inf tier2 1024 METALSIM_PT_EXP '{"GRID_PLANES":0}' "inf t2 planes forced 0"
