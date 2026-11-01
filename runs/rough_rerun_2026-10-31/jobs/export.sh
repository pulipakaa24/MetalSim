#!/bin/bash
# export.sh SRC=DST ...  MetalSim checkpoints -> the format isaac_side/play_policy.py reads (metalsim/parity/export_policy.py:
# actor state dict + our joint order + scan_ordering), as the original protocol did before stage11.sh played ours in Isaac
# (WHO=metalsim, ~/metalsim_ckpts_rough/*.pt on the L4). CPU only (build_g1_model on MuJoCo C).
set -u
trap 'kill -TERM $P 2>/dev/null; wait $P 2>/dev/null; exit 143' TERM INT HUP
PY=.venv/bin/python; RC=0
for pair in "$@"; do
  src=${pair%%=*}; dst=${pair#*=}; mkdir -p "$(dirname "$dst")"
  if [ ! -e "$src" ]; then echo "missing $src"; RC=2; continue; fi
  echo "--- $(date +%T) export $src -> $dst"; $PY metalsim/parity/export_policy.py "$src" "$dst" & P=$!; wait $P; rc=$?
  echo "--- rc $rc"; [ $rc -ne 0 ] && RC=$rc
done
exit $RC
