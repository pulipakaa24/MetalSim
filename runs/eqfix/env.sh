#!/bin/bash
# Run a command with the shared .venv, this worktree's metalsim and the MuJoCo Warp fork clone at the new pin
# (scratch/mjw_fork, metalsim f1b35e2) first on sys.path.
WT=.
export PYTHONPATH=$WT/scratch/mjw_fork:$WT
cd "$WT"
exec "$@"
