#!/bin/bash
# so101-film-bam: record + render of so101_lift_deploy on 7v4_bam as ONE GPU-queue job. From the repository root:
#   scripts/gpu_run.sh so101-film-bam render 20 -- bash runs/readme_media/scratch/so101-film-bam/film_job.sh
# Refuses to run unless the queued Metal test of the BAM friction (logs/pytest_gpu.log) passed all 8 tests.
# Encode (ffmpeg / Pillow, CPU) runs afterwards outside the queue.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
L=runs/readme_media/scratch/so101-film-bam/logs
mkdir -p "$L"
if ! grep -Eq "^8 passed" "$L/pytest_gpu.log" 2>/dev/null || grep -Eqi "failed|error" "$L/pytest_gpu.log"; then
  echo "[so101-film-bam] Metal test did not pass 8/8 (see $L/pytest_gpu.log): not filming"; exit 3
fi
{ date; pmset -g batt; pmset -g | grep -i powermode; } > "$L/power_job_start.txt" 2>&1 || true
.venv/bin/python -m metalsim.render.film so101_lift_deploy --stage record 2>&1 | tee "$L/record.log"
.venv/bin/python -m metalsim.render.film so101_lift_deploy --stage render 2>&1 | tee "$L/render.log"
{ date; pmset -g batt; pmset -g | grep -i powermode; } > "$L/power_job_end.txt" 2>&1 || true
echo "[so101-film-bam] record + render done"
