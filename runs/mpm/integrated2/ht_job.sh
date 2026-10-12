#!/bin/bash
# HashTable race in the rough G1 Newton-engine task (old Warp = .venv's upstream/warp-innate; stock vs fixed Newton 1.7)
A=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/runs/a2
W=/Users/aditya/robosim/.claude/worktrees/agent-acdd938690dabdae8/upstream
R=/Users/aditya/robosim
O=$A/out/ht; mkdir -p $O
cd $R
P=$A/py/ht_probe.py; PY=$R/.venv/bin/python
F="SUMMARY\|^warp\|Error"
export PYTHONDONTWRITEBYTECODE=1
echo "== 64 envs, contact buffer 16384: record (fixed), replay identical states x2"
WARP_CACHE_PATH=$A/kc/ht_old_fixed PYTHONPATH=$W/newton-rigid:$R $PY $P record --n 64 --steps 30 --contact-max 16384 --out $O/rec64 2>&1 | /usr/bin/grep "$F"
echo "-- stock"; WARP_CACHE_PATH=$A/kc/ht_old_stock PYTHONPATH=$R $PY $P replay --n 64 --contact-max 16384 --states $O/rec64.states.npz --repeat 2 --out $O/rep64_stock 2>&1 | /usr/bin/grep "$F"
echo "-- fixed"; WARP_CACHE_PATH=$A/kc/ht_old_fixed PYTHONPATH=$W/newton-rigid:$R $PY $P replay --n 64 --contact-max 16384 --states $O/rec64.states.npz --repeat 2 --out $O/rep64_fixed 2>&1 | /usr/bin/grep "$F"
echo "-- fixed, final Warp"; WARP_CACHE_PATH=$A/kc/ht_i2_fixed PYTHONPATH=$W/warp-i2:$W/newton-rigid:$R $PY $P replay --n 64 --contact-max 16384 --states $O/rec64.states.npz --out $O/rep64_i2_fixed 2>&1 | /usr/bin/grep "$F"
echo "== 64 envs, default contact buffer, free run"
WARP_CACHE_PATH=$A/kc/ht_old_stock PYTHONPATH=$R $PY $P run --n 64 --steps 30 --out $O/run64_stock 2>&1 | /usr/bin/grep "$F"
WARP_CACHE_PATH=$A/kc/ht_old_fixed PYTHONPATH=$W/newton-rigid:$R $PY $P run --n 64 --steps 30 --out $O/run64_fixed 2>&1 | /usr/bin/grep "$F"
echo "== 4096 envs, default contact buffer, 3 control steps (48 collides)"
WARP_CACHE_PATH=$A/kc/ht_old_stock PYTHONPATH=$R $PY $P run --n 4096 --steps 3 --out $O/run4096_stock 2>&1 | /usr/bin/grep "$F"
WARP_CACHE_PATH=$A/kc/ht_old_fixed PYTHONPATH=$W/newton-rigid:$R $PY $P run --n 4096 --steps 3 --out $O/run4096_fixed 2>&1 | /usr/bin/grep "$F"
echo "== final Warp + stock Newton (expected to raise)"
WARP_CACHE_PATH=$A/kc/ht_i2_stock PYTHONPATH=$W/warp-i2:$R $PY $P run --n 64 --steps 1 --out $O/run_i2_stock 2>&1 | /usr/bin/grep "RuntimeError" | cut -c1-220
