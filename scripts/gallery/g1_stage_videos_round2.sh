#!/bin/bash
cd /Users/aditya/robosim
P=runs/parity/isaac/parity_out/play2; M=runs/policies
for it in 100 300 500 1000; do .venv/bin/python -m metalsim.parity.side_by_side --isaac $P/metalsim_export_fixed_it$it --ckpt $M/g1_flat_ppowarp_fixed_it$it.pt --out docs/gallery/g1_stage_metalsim_fixed_it$it.mp4 --label "MetalSim-trained (fixed PPO), iteration $it" 2>&1 | grep "^wrote"; done
FF=$(.venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
for f in docs/gallery/g1_stage_metalsim_fixed_it*.mp4; do $FF -y -loglevel error -i $f -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart ${f%.mp4}.tmp.mp4 && mv ${f%.mp4}.tmp.mp4 $f; done
echo ROUND2_DONE
