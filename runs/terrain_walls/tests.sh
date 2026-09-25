#!/bin/bash
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/gpu_lock.py status
export PYTHONPATH=/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/mjw_a8e6485
.venv/bin/python scripts/diagnostics/terrain_step_edge.py --modes boxes_local --out runs/terrain_walls/step_edge.jsonl 2>&1 | grep '^{'
.venv/bin/python -m pytest tests/test_terrain.py tests/test_g1_task_terms.py -q 2>&1 | tail -15
