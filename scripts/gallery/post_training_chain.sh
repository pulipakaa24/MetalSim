#!/bin/bash
# After the fixed-PPO training process exits: release its (old-style) lock, then queue the replay re-render
# and the video re-render through the priority queue as render jobs.
cd /Users/aditya/robosim
while pgrep -f "metalsim.learn.g1_velocity 4096 flat train 1000" > /dev/null; do sleep 30; done
rm -f runs/.gpu_lock; rmdir runs/.gpu_lock.d 2>/dev/null
scripts/gpu_run.sh replay_rerender render 15 -- sh -c '.venv/bin/python -m metalsim.parity.record_g1 --isaac runs/parity/isaac/parity_out2/rt --out runs/parity/metalsim2 --physics_dt 0.0025 --frame_every 5 > runs/parity/replay2.log 2>&1'
.venv/bin/python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/rt --metalsim runs/parity/metalsim2 --out runs/parity/report_rt3 > runs/parity/compare_rt3.log 2>&1
.venv/bin/python -m metalsim.parity.compare --isaac runs/parity/isaac/parity_out2/pt --metalsim runs/parity/metalsim2 --out runs/parity/report_pt3 > runs/parity/compare_pt3.log 2>&1
echo RERENDER_DONE >> runs/parity/compare_pt3.log
sed -i '' 's|^while \[ -e runs/.gpu_lock \]; do sleep 30; done$|:|; s|^echo "g1_stage_videos (main session) $(date)" > runs/.gpu_lock; trap "rm -f runs/.gpu_lock" EXIT$|:|' scripts/gallery/g1_stage_videos.sh
scripts/gpu_run.sh stage_videos render 10 -- bash scripts/gallery/g1_stage_videos.sh > runs/g1_stage_videos2.log 2>&1
FF=$(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
for f in docs/gallery/g1_stage_*.mp4; do $FF -y -loglevel error -i $f -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart ${f%.mp4}.tmp.mp4 && mv ${f%.mp4}.tmp.mp4 $f; done
$FF -y -loglevel error -ss 6 -i docs/gallery/g1_stage_isaac_it1499.mp4 -frames:v 1 docs/gallery/g1_stage_isaac_it1499_frame150.png
echo CHAIN_DONE >> runs/g1_stage_videos2.log
