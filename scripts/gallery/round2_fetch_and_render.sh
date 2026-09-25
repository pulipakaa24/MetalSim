#!/bin/bash
# Wait for the Isaac-side playback of the fixed-PPO checkpoints, fetch it, stop the VM, render round-two videos.
cd /Users/aditya/robosim
G="gcloud compute ssh --zone us-west1-a aditya-isaac --tunnel-through-iap --project mcmc-193822 --quiet"
until $G --command 'test -f ~/STAGE9_DONE' > /dev/null 2>&1; do sleep 60; done
echo "stage9 done $(date +%H:%M:%S)"
$G --command 'grep "final root z\|grey ground\|Traceback" ~/stage9.log | cut -c1-110'
gcloud compute scp --zone us-west1-a --tunnel-through-iap --project mcmc-193822 --quiet aditya-isaac:~/parity_play2.tgz runs/parity/isaac/
gcloud compute instances stop aditya-isaac --zone us-west1-a --project mcmc-193822 --quiet
tar xzf runs/parity/isaac/parity_play2.tgz -C runs/parity/isaac/parity_out/ && ls runs/parity/isaac/parity_out/play2
scripts/gpu_run.sh stage_videos_round2 render 10 -- bash scripts/gallery/g1_stage_videos_round2.sh > runs/g1_stage_videos_round2.log 2>&1
tail -5 runs/g1_stage_videos_round2.log
echo ROUND2_PIPELINE_DONE
