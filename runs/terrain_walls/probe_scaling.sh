#!/bin/bash
# boxes-mode memory/throughput scaling probe (4096 envs ran out of Metal memory)
cd "$(dirname "$0")/../.."; source .venv/bin/activate
python3 scripts/gpu_lock.py status
for N in 512 1024 2048; do
  N=$N OUT=runs/terrain_walls/bench_scaling.jsonl python - <<PY 2>&1 | grep -v "^Module\|^   \|^Warp\|warn"
import subprocess, sys
subprocess.run([sys.executable, "scripts/diagnostics/terrain_walls_bench.py", "--one", "boxes", "$N", "0.0025", "runs/terrain_walls/bench_scaling.jsonl"])
PY
done
