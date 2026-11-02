#!/bin/bash
# repeat.sh <worktree> <tag> <n> <seeds...>: n repeats of measure_seeds.py gpu on the given MuJoCo Warp copy
W=$1; T=$2; N=$3; shift 3
for i in $(seq 1 $N); do PYTHONPATH=$W .venv/bin/python runs/readme_media/scratch/hfield-contacts/falls/measure_seeds.py gpu ${T}_r$i "$@"; done
