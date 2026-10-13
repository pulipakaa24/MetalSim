#!/bin/bash
# encode.sh FRAMES_DIR OUT.mp4 [FFMPEG]: H.264 (yuv420p, CRF 14, 30 fps) from frame_%04d.png. FFMPEG defaults to
# `ffmpeg` on PATH; without one, pip install imageio-ffmpeg in a venv and pass its binary
# (python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").
set -e
IN=$1; OUT=$2; FF=${3:-ffmpeg}
"$FF" -y -loglevel error -framerate 30 -i "$IN/frame_%04d.png" -c:v libx264 -preset slow -crf 14 -pix_fmt yuv420p \
    -movflags +faststart "$OUT"
ls -la "$OUT"
