#!/bin/bash
# All throughput numbers of the terrain-walls table in one GPU slot, on MuJoCo Warp fork a8e6485 exported clean
# (another agent was editing upstream/mujoco_warp's working tree during this work).
cd "$(dirname "$0")/../.."
export PYTHONPATH=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/mjw_a8e6485
.venv/bin/python -c "import mujoco_warp; print('mujoco_warp from', mujoco_warp.__file__)"
OUT=runs/terrain_walls/bench_pinned.jsonl .venv/bin/python scripts/diagnostics/terrain_walls_bench.py ${MODES:-hfield hfield@nconmax128 boxes_local boxes_local+exact hfield+exact hfield_fine:0.05 hfield_fine:0.025 boxes_fine:0.025}
[ -n "${MODES:-}" ] && exit 0
for N in 1024 2048; do .venv/bin/python scripts/diagnostics/terrain_walls_bench.py --one boxes $N 0.0025 runs/terrain_walls/bench_pinned.jsonl; .venv/bin/python scripts/diagnostics/terrain_walls_bench.py --one hfield $N 0.0025 runs/terrain_walls/bench_pinned.jsonl; done
.venv/bin/python scripts/diagnostics/terrain_walls_bench.py --one meshes 1024 0.0025 runs/terrain_walls/bench_pinned.jsonl
