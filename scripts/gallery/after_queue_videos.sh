#!/bin/bash
# Re-render the stage videos without the collider geometry once the main queue (fixed-PPO run, replay re-render) is done.
cd /Users/aditya/robosim
until grep -q QUEUE_DONE runs/g1_replay_rerender.log 2>/dev/null; do sleep 60; done
bash scripts/gallery/g1_stage_videos.sh > runs/g1_stage_videos2.log 2>&1
FF=$(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
for f in docs/gallery/g1_stage_*.mp4; do $FF -y -loglevel error -i $f -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart ${f%.mp4}.tmp.mp4 && mv ${f%.mp4}.tmp.mp4 $f; done
$FF -y -loglevel error -ss 6 -i docs/gallery/g1_stage_isaac_it1499.mp4 -frames:v 1 docs/gallery/g1_stage_isaac_it1499_frame150.png
echo VIDEOS2_DONE >> runs/g1_stage_videos2.log
