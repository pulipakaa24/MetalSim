#!/bin/bash
# run_train2.sh TERRAIN REWARD_CFG CONTACT_CFG SOLVER_CFG ITERATIONS SEED TAG [BASE_VELOCITY] [FEET_SLIDE_VELOCITY]  (unit tests first; abort on failure)
cd "$(dirname "$0")/../.."
T=$1; RC=$2; CC=$3; P=$4; IT=$5; S=$6; TAG=$7; BV=${8:-com}; FSV=${9:-com}
NAME=$TAG
OUT=runs/il3/$NAME.stdout
{ echo "== $(date)"; python3 scripts/gpu_lock.py status; } > $OUT
source .venv/bin/activate
python -m pytest -q tests/test_solver_presets.py tests/test_g1_task_terms.py -p no:cacheprovider >> $OUT 2>&1 || { echo "tests failed; not training" >> $OUT; echo "exit 99 $(date)" >> $OUT; exit 99; }
SC=""; [ "$P" != "none" ] && SC="--solver_cfg $P"
python -m metalsim.learn.g1_velocity 4096 $T train $IT runs/il3/$NAME.log runs/il3/ckpt/${NAME}.pt 0.0025 --reward_cfg $RC --contact_cfg $CC $SC --seed $S --base_velocity $BV --feet_slide_velocity $FSV >> $OUT 2>&1
echo "exit $? $(date)" >> $OUT
