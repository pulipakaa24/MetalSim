#!/bin/bash
cd /Users/aditya/robosim
python3 scripts/gpu_lock.py status
echo "=== tests/test_physx_cloth.py on metal:0"
.venv/bin/python -m pytest -q tests/test_physx_cloth.py 2>&1 | tail -5
echo "=== throughput (recorder cloth scene: 21x21, 1 m, 0.4 m box; dt 5 ms, 16 iterations + 1 velocity iteration), graph replay"
.venv/bin/python -m metalsim.physics.physx_cloth --envs 64 256 1024 4096 --seconds 3 2>&1 | grep "^{"
echo "=== XPBDSim, same topology (box scene 21x21), 16 substeps, for scale"
.venv/bin/python - <<'PY'
import json, mujoco, warp as wp
wp.config.quiet = True
from metalsim.physics import deformable as dfm
xml = open("tests/test_physx_cloth.py").read().split('DROP_XML = """')[1].split('"""')[0]
m = mujoco.MjModel.from_xml_string(xml)
for n in (1024, 4096):
    print(json.dumps(dfm.benchmark_xpbd(m, n, seconds=3.0, cfg=dfm.XPBDCfg(substeps=16))), flush=True)
PY
