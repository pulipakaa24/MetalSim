#!/bin/bash
# speculative contacts, task 1: over-travel probe, second pass with joint positions (gait vs Isaac play recordings) (scripts/diagnostics/speculative/overtravel_probe.py) on frozen fork worktrees:
# stock f824af1 (recommended, ellip10_tau5) and the prototype d99186b (MJW_SPECULATIVE_GAP=1; friction cone / none / live; NULL rows)
cd /Users/aditya/robosim
PY=.venv/bin/python; P=scripts/diagnostics/speculative/overtravel_probe.py; O=runs/speculative/probe2
STOCK=/Users/aditya/robosim/upstream/mujoco_warp-f824af1; SPEC=/Users/aditya/robosim/upstream/mujoco_warp-spec-d99186b
mkdir -p $O
echo "=== start $(date) metalsim $(git rev-parse --short HEAD) stock $(git -C $STOCK rev-parse --short HEAD) spec $(git -C $SPEC rev-parse --short HEAD)"
F='^Warning\|^Module\|^Warp\|^   '
run() {  # tag mjw envstr preset
  for n in 4 64; do
    echo "--- $1 n=$n $(date +%H:%M:%S)"
    env PYTHONPATH=$2 $3 $PY $P $4 $O/$1_n$n.npz --envs $n 2>&1 | grep -v "$F"
  done
}
run rec        $STOCK "MJW_SPECULATIVE_GAP=0" recommended
run tau5       $STOCK "MJW_SPECULATIVE_GAP=0" ellip10_tau5
run spec5_cone $SPEC  "MJW_SPECULATIVE_GAP=1" ellip10_tau5_specgap10mm
run spec5_null $SPEC  "MJW_SPECULATIVE_GAP=1 MJW_SPEC_NULL=1" ellip10_tau5_specgap10mm
run spec5_none $SPEC  "MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=none" ellip10_tau5_specgap10mm
run spec5_live $SPEC  "MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live" ellip10_tau5_specgap10mm
run spec10_cone $SPEC "MJW_SPECULATIVE_GAP=1" ellip10_specgap10mm
run spec10_null $SPEC "MJW_SPECULATIVE_GAP=1 MJW_SPEC_NULL=1" ellip10_specgap10mm
run spec10_live $SPEC "MJW_SPECULATIVE_GAP=1 MJW_SPEC_FRICTION=live" ellip10_specgap10mm
echo "=== end $(date)"
