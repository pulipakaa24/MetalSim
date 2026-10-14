#!/bin/bash
# Wait for the Isaac-side playback of the fixed-PPO checkpoints, fetch it from the L4 machine ($L4_HOST), render round-two videos.
cd "$(dirname "$0")/../.."   # repo root
G="ssh ${L4_HOST:?set L4_HOST to the L4 machine ssh host}"
until $G 'test -f ~/STAGE9_DONE' > /dev/null 2>&1; do sleep 60; done
echo "stage9 done $(date +%H:%M:%S)"
$G 'grep "final root z\|grey ground\|Traceback" ~/stage9.log | cut -c1-110'
scp "$L4_HOST:~/parity_play2.tgz" runs/parity/isaac/
tar xzf runs/parity/isaac/parity_play2.tgz -C runs/parity/isaac/parity_out/ && ls runs/parity/isaac/parity_out/play2
scripts/gpu_run.sh stage_videos_round2 render 10 -- bash scripts/gallery/g1_stage_videos_round2.sh > runs/g1_stage_videos_round2.log 2>&1
tail -5 runs/g1_stage_videos_round2.log
echo ROUND2_PIPELINE_DONE
