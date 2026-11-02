"""Run a subset of placements (worlds) of a placements.npz through MuJoCo Warp's collision on CPU (stream hfield-epa).
Usage: PYTHONPATH=<mjw copy> python sub.py <placements.npz> <gap> <hzn> <w0,w1,...|all>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epa_repro as R
import mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import mujoco_warp._src.collision_convex as CCV
print("mujoco_warp", mjw.__file__)
path, gap, hzn, sel = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
CCV.MJ_MAX_EPAHORIZON = hzn
Q = np.load(path)["qpos"]
idx = np.arange(len(Q)) if sel == "all" else np.array([int(v) for v in sel.split(",")])
hf = R.tile(); m = R.build_model(hf, gap); d0 = mujoco.MjData(m)
with wp.ScopedDevice("cpu"):
  mm = mjw.put_model(m); mm.opt.warn_overflow = 0
  dd = mjw.put_data(m, d0, nworld=len(idx), nconmax=16, njmax=64)
  dd.qpos.assign(Q[idx].astype(np.float32))
  mjw.kinematics(mm, dd); mjw.collision(mm, dd)
  n = int(dd.nacon.numpy()[0]); ov = dd.overflow.numpy()
print("worlds", len(idx), "contacts", n, "overflow worlds", int((ov != 0).sum()), "->", idx[ov != 0][:20].tolist())
