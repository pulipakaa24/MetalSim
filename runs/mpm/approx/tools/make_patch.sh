#!/bin/bash
# Patch of the metalsim side (files untracked in the main checkout) against /Users/aditya/robosim.
W=/Users/aditya/robosim/.claude/worktrees/agent-ae8661f0d09092d4b
M=/Users/aditya/robosim
OUT=$W/patches/metalsim_metal_fast_math.patch
: > $OUT
for f in metalsim/physics/mpm/newton_compat.py metalsim/physics/mpm/MPM.md metalsim/physics/mpm/record_variant_suite.sh \
         metalsim/physics/mpm/probes/speed_bench.py metalsim/parity/isaac_side/record_mpm.py; do
  diff -u --label a/$f --label b/$f $M/$f $W/$f >> $OUT
done
for f in metalsim/physics/mpm/probes/mufu_model.py metalsim/physics/mpm/probes/cuda_approx_dump.py \
         metalsim/physics/mpm/probes/cuda_approx_compose.py metalsim/physics/mpm/probes/metal_approx_probe.py \
         metalsim/physics/mpm/probes/approx_emulation.py metalsim/physics/mpm/probes/metal_compose_check.py \
         metalsim/physics/mpm/probes/metal_approx_timing.py; do
  diff -u --label /dev/null --label b/$f /dev/null $W/$f >> $OUT
done
echo "$(grep -c '^+++ ' $OUT) files, $(wc -l < $OUT) lines -> $OUT"
