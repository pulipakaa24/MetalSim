"""CPU (Warp's CPU device, single thread) cost of MuJoCo Warp's collision() on a placements set (stream hfield-epa).
Not a performance figure: ratios between fork copies on the same machine and set only.
Usage: PYTHONPATH=<mjw copy> python time_cpu.py <dir with placements.npz> <gap per geom> [reps] [repro|train]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402

out, gap = sys.argv[1], float(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
kind = sys.argv[4] if len(sys.argv) > 4 else "train"
Z = np.load(os.path.join(out, "placements.npz")); Q = Z["qpos"]
if kind == "train":
  import epa_train as R
  m = R.build_model(R.terrain(), Z["verts"], gap)
else:
  import epa_repro as R
  m = R.build_model(R.tile(), gap)
d0 = mujoco.MjData(m)
with wp.ScopedDevice("cpu"):
  mm = mjw.put_model(m); mm.opt.warn_overflow = 0
  dd = mjw.put_data(m, d0, nworld=len(Q), nconmax=16, njmax=64)
  dd.qpos.assign(Q.astype(np.float32)); mjw.kinematics(mm, dd)
  mjw.collision(mm, dd); wp.synchronize()
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter(); mjw.collision(mm, dd); wp.synchronize(); ts.append(time.perf_counter() - t0)
  n = int(dd.nacon.numpy()[0])
print(f"{mjw.__file__.split('/')[-3]} {out} gap {gap}: collision median {np.median(ts) * 1e3:.1f} ms (min {min(ts) * 1e3:.1f}) "
      f"over {reps}; contacts {n}")
