#!/bin/bash
# Re-runs, on the watchdog fix (branch concurrency-1-handover), the two runs/il3 early trainings whose first or
# eleventh update a GPU watchdog timeout zeroed (docs/research/isaaclab3_like_for_like_2026-09-25.md §6.3).
# Settings as runs/il3/run_early_ab.sh on 2026-09-25: rough_isaac on exact boxes with Isaac's solver preset, and
# rough_il3 with that date's "recommended" contacts (now recommended_pyramidal). Run from the fix worktree.
O=runs/gpu_watchdog_2026-11-01
PY=<repo>/.venv/bin/python
export TORCH_EXTENSIONS_DIR=<torch extensions dir>
$PY -m metalsim.learn.g1_velocity 4096 rough train 20 $O/rerun_rough232task.log - 0.0025 --reward_cfg rough_isaac --contact_cfg recommended --solver_cfg isaaclab3_every_substep_cap20 --terrain_collision boxes_local --seed 0 --base_velocity com --feet_slide_velocity com > $O/rerun_rough232task.out 2>&1
echo "exit $?" >> $O/rerun_rough232task.out
$PY -m metalsim.learn.g1_velocity 4096 rough train 20 $O/rerun_recommended.log - 0.0025 --reward_cfg rough_il3 --contact_cfg recommended_pyramidal --seed 0 --base_velocity com --feet_slide_velocity com > $O/rerun_recommended.out 2>&1
echo "exit $?" >> $O/rerun_recommended.out
