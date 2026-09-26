#!/bin/bash
# Headline re-runs under the new G1 default (elliptic impratio 10, cap 20): run.sh TERRAIN ITERATIONS SEED NAME
# Gated on the task/preset tests so a broken default cannot burn GPU hours.
cd "$(dirname "$0")/../.."
T=$1; IT=$2; S=$3; NAME=$4
source .venv/bin/activate
python -m pytest tests/test_g1_task_terms.py tests/test_contact_tuning.py tests/test_solver_presets.py -q -x > runs/ellip_default/${NAME}.tests.log 2>&1 || { echo "TESTS FAILED, not training"; exit 1; }
RC=flat; [ "$T" = rough ] && RC=rough_isaac
python -m metalsim.learn.g1_velocity 4096 $T train $IT runs/ellip_default/${NAME}.log runs/ellip_default/${NAME}.pt 0.0025 --reward_cfg $RC --seed $S
echo "exit $? $(date)"
