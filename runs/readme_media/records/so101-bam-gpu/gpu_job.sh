#!/bin/bash
# so101-bam-gpu: the GPU work of this stream, one queue slot. From the MetalSim root:
#   scripts/gpu_run.sh so101-bam-gpu render 25 -- bash runs/readme_media/scratch/so101-bam-gpu/gpu_job.sh
# Each part logs to logs/ and the next runs whatever its exit code.
S=runs/readme_media/scratch/so101-bam-gpu
PY=.venv/bin/python
L=$S/logs
mkdir -p $L
echo "[gpu_job] start $(date +%H:%M:%S); mujoco_warp $(git -C upstream/mujoco_warp rev-parse --short HEAD)" | tee $L/gpu_job.log
(pmset -g batt | head -2; pmset -g | grep -i powermode) >> $L/gpu_job.log 2>&1
$PY -m pytest -q -p no:cacheprovider tests/test_bam_friction.py > $L/pytest_gpu.log 2>&1
echo "[gpu_job] pytest rc $? $(date +%H:%M:%S)" | tee -a $L/gpu_job.log
$PY $S/gpu_bam_check.py > $L/check.log 2>&1
echo "[gpu_job] check rc $? $(date +%H:%M:%S)" | tee -a $L/gpu_job.log
$PY $S/eval_metalsim_bam.py 7v4_bam $S/metalsim/deploy_7v4_bam > $L/eval_metalsim.log 2>&1
echo "[gpu_job] eval rc $? $(date +%H:%M:%S)" | tee -a $L/gpu_job.log
# the CPU reference's first-step order (ctrl 0 at the first forward), for an apples-to-apples comparison
$PY $S/eval_metalsim_bam.py 7v4_bam $S/metalsim/deploy_7v4_bam_legacy0 --legacy-ctrl0 > $L/eval_metalsim_legacy0.log 2>&1
echo "[gpu_job] eval legacy0 rc $? $(date +%H:%M:%S)" | tee -a $L/gpu_job.log
# harness check: 7v4_real through this copy, against so101-expert's metalsim/deploy_7v4_real
$PY $S/eval_metalsim_bam.py 7v4_real $S/metalsim/deploy_7v4_real_rerun > $L/eval_metalsim_7v4_real.log 2>&1
echo "[gpu_job] eval 7v4_real rc $? $(date +%H:%M:%S)" | tee -a $L/gpu_job.log
$PY $S/bench_bam.py > $L/bench.log 2>&1
echo "[gpu_job] bench rc $? $(date +%H:%M:%S)" | tee -a $L/gpu_job.log
(pmset -g batt | head -2; pmset -g | grep -i powermode) >> $L/gpu_job.log 2>&1
