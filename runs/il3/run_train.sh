#!/bin/bash
# Step 3: like-for-like training on the Isaac Lab 3.0 task: run_train.sh TERRAIN SOLVER_CFG [ITERATIONS] [SEED]
cd "$(dirname "$0")/../.."
T=$1; P=$2; IT=${3:-1500}; S=${4:-0}
RC=${T}_il3
NAME=train_${T}_il3_${P}_s${S}
LOG=runs/il3/$NAME.log
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } > runs/il3/$NAME.stdout
source .venv/bin/activate
SC=""; [ "$P" != "none" ] && SC="--solver_cfg $P"
python -m metalsim.learn.g1_velocity 4096 $T train $IT $LOG runs/il3/ckpt/${NAME}.pt 0.0025 --reward_cfg $RC $SC --seed $S >> runs/il3/$NAME.stdout 2>&1
echo "exit $? $(date)" >> runs/il3/$NAME.stdout
