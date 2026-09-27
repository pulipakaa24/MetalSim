#!/bin/bash
# physics-only cost at 4096 envs, each preset at its own cap (3 interleaved repeats, bench_contact_tuning): the default and
# stock tau 5 ms (gap 0: bitwise equal to the fork head, cpu_traj check) and the final-rule speculative preset, one process
cd /Users/aditya/robosim
MJW=/Users/aditya/robosim/upstream/mujoco_warp-spec-27f1fcd MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live MJW_SPEC_FRICIMP=d0 scripts/diagnostics/penetration/sweep.sh cost recommended ellip10_tau5_specgap10mm ellip10_tau5
