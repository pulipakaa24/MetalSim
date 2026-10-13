#!/bin/bash
# Shader-variant experiments (temporary #ifdef toggles, scratchpad exp_patch*.py): which workstream-C addition costs
# the tier-1 fragment stage / the tier-2 path_trace kernel. One process per variant, cloth scene, defaults.
set -u
WT=/Users/aditya/robosim/.claude/worktrees/agent-a5314e27bee02c142
PY=/Users/aditya/robosim/.venv/bin/python
N=${1:-1024}
OUT=$WT/runs/render_gaps2/exp.jsonl
run() {  # renderer, envvar, json, tag
  echo "== $1 $4"
  env $2="$3" $PY $WT/scripts/diagnostics/render_perf_profile.py --scene ${SCENE:-cloth} --renderer $1 --envs $N --features none \
      --all-off 0 --profile 0 --tag "$4" --out $OUT 2>&1 | grep '^{'
}
for v in '{}' '{"EXP_SHARED_LIGHTS":1}' '{"EXP_NO_GRID_PLANE":1}' '{"EXP_NO_REACH":1}' '{"EXP_NO_REFL_PLANE":1}' \
         '{"EXP_SHARED_LIGHTS":1,"EXP_NO_GRID_PLANE":1,"EXP_NO_REACH":1,"EXP_NO_REFL_PLANE":1}'; do
  run tier1 METALSIM_TIER0_EXP "$v" "t1 $v"
done
for v in '{}' '{"EXP_SHARED_LIGHTS":1}' '{"EXP_NO_GRID_PLANE":1}' '{"EXP_NO_HAZE":1}' '{"EXP_NO_ANYHIT":1}' \
         '{"EXP_SHARED_LIGHTS":1,"EXP_NO_GRID_PLANE":1,"EXP_NO_HAZE":1,"EXP_NO_ANYHIT":1}'; do
  run tier2 METALSIM_PT_EXP "$v" "t2 $v"
done
