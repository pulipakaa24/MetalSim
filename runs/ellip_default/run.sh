#!/bin/bash
# Headline re-runs under the new G1 default (elliptic impratio 10, cap 20): run.sh TERRAIN ITERATIONS SEED NAME
# Gated on the task/preset tests so a broken default cannot burn GPU hours.
cd "$(dirname "$0")/../.."
T=$1; IT=$2; S=$3; NAME=$4
source .venv/bin/activate
if [ "${GATE:-1}" = 1 ]; then python -m pytest tests/test_g1_task_terms.py tests/test_contact_tuning.py tests/test_solver_presets.py -q -x > runs/ellip_default/${NAME}.tests.log 2>&1 || { echo "TESTS FAILED, not training"; exit 1; }; fi
RC=flat; [ "$T" = rough ] && RC=rough_isaac
# a job granted seconds after another releases can hit a transient Metal out-of-memory while the previous process frees
# its buffers (the rough run did on 2026-09-26): retry once after 60 s if the first attempt dies within 2 minutes
for attempt in 1 2 3; do
  t0=$(date +%s)
  python -m metalsim.learn.g1_velocity 4096 $T train $IT runs/ellip_default/${NAME}.log runs/ellip_default/${NAME}.pt 0.0025 --reward_cfg $RC --seed $S
  rc=$?; [ $rc -eq 0 ] && break
  [ $(( $(date +%s) - t0 )) -gt 120 ] && break
  echo "attempt $attempt failed within 2 minutes (rc $rc); retrying after 120 s"; sleep 120
done
echo "exit $rc $(date)"
