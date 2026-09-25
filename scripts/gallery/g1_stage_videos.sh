#!/bin/bash
# Side-by-side stage videos (Isaac RTX | MetalSim tier 2) for Isaac's rsl_rl checkpoints and MetalSim's
# own, each rolled out in both simulators from the same state and command. Holds the GPU lock.
cd /Users/aditya/robosim
while [ -e runs/.gpu_lock ]; do sleep 30; done
echo "g1_stage_videos (main session) $(date)" > runs/.gpu_lock; trap "rm -f runs/.gpu_lock" EXIT
P=runs/parity/isaac/parity_out/play; C=runs/parity/isaac/ckpt_out; M=runs/policies
run() { .venv/bin/python -m metalsim.parity.side_by_side --isaac "$1" --ckpt "$2" --out "$3" --label "$4" 2>&1 | grep "^wrote"; }
for it in 100 500 1000; do run $P/isaac_it$it $C/model_$it.pt docs/gallery/g1_stage_isaac_it$it.mp4 "Isaac-trained policy, iteration $it"; done
[ -d $P/isaac_it1499 ] && run $P/isaac_it1499 $C/model_1499.pt docs/gallery/g1_stage_isaac_it1499.mp4 "Isaac-trained policy, iteration 1499"
for it in 100 500 1000 1500; do run $P/metalsim_export_metalsim_it$it $M/g1_flat_dt25_fixed_it$it.pt docs/gallery/g1_stage_metalsim_it$it.mp4 "MetalSim-trained policy, iteration $it"; done
echo VIDEOS_DONE
