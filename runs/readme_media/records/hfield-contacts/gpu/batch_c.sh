#!/bin/bash
# GPU batch C: random actions with a MuJoCo C reference on 32 worlds (fix, and fix with the mesh plane path); step-time
# split: fix with HFIELD_PLANE_CONTACTS=True (x2), fix default (x1).
R=~/MetalSim; S=$R/runs/readme_media/scratch/hfield-contacts; PY=$R/.venv/bin/python; cd $R
pw() { echo "power $(pmset -g batt | sed -n 2p | cut -c1-70) | $(pmset -g | grep powermode | tr -s ' ') | $(date +%H:%M:%S)"; }
echo "== random actions fix + C(32) $(pw)"; C=32 PYTHONPATH=$S/mjw $PY $S/gpu/random_actions.py 2>&1 | grep -E "^mujoco_warp|^MJW|^C |rror"
echo "== random actions fix, mesh plane path $(pw)"; MJW_SET=collision_convex.HFIELD_PLANE_CONTACTS=1 PYTHONPATH=$S/mjw $PY $S/gpu/random_actions.py 2>&1 | grep -E "^set|^MJW|rror"
T='import sys, runpy, importlib, os
for kv in filter(None, os.environ.get("MJW_SET", "").split(",")):
  k, v = kv.split("="); mod, attr = k.rsplit(".", 1); M = importlib.import_module(f"mujoco_warp._src.{mod}"); setattr(M, attr, type(getattr(M, attr))(eval(v))); print("set", k, getattr(M, attr))
sys.argv = ["g1_velocity", "4096", "rough"]; runpy.run_module("metalsim.learn.g1_velocity", run_name="__main__")'
for rep in 1 2; do
  echo "== timing fix+plane rep $rep $(pw)"; MJW_SET=collision_convex.HFIELD_PLANE_CONTACTS=1 PYTHONPATH=$S/mjw $PY -c "$T" 2>&1 | grep -E "^set|^N=|rror"
  echo "== timing fix rep $rep $(pw)"; PYTHONPATH=$S/mjw $PY -c "$T" 2>&1 | grep -E "^N=|rror"
done
echo "== done $(pw)"
