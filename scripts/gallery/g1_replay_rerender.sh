#!/bin/bash
# Re-render the MetalSim replay frames of the fidelity protocols with the full-extent ground plane, then
# re-run the comparison against the grey-ground Isaac recording. Runs after the stage videos, under the GPU lock.
cd /Users/aditya/robosim
until grep -q VIDEOS_DONE runs/g1_stage_videos.log 2>/dev/null; do sleep 60; done
while [ -e runs/.gpu_lock ]; do sleep 30; done
echo "g1_replay_rerender (main session) $(date)" > runs/.gpu_lock; trap "rm -f runs/.gpu_lock" EXIT
.venv/bin/python -m metalsim.parity.record_g1 --isaac runs/parity/isaac/parity_out2/rt --out runs/parity/metalsim2 --physics_dt 0.0025 --frame_every 5 > runs/parity/replay2.log 2>&1
rm -f runs/.gpu_lock
.venv/bin/python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/rt --metalsim runs/parity/metalsim2 --out runs/parity/report_rt3 > runs/parity/compare_rt3.log 2>&1
.venv/bin/python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/pt --metalsim runs/parity/metalsim2 --out runs/parity/report_pt3 > runs/parity/compare_pt3.log 2>&1
echo RERENDER_DONE >> runs/parity/compare_pt3.log
