"""Isaac's G1 foot mesh against box geoms (the boxes_local terrain's stair / box colliders): MuJoCo Warp contacts on
Warp's CPU device for N placements, to compare EPA versions on box-mesh pairs (stream hfield-epa).
Usage: PYTHONPATH=<mjw> python boxmesh.py <out dir> <tag> [N] [seed]"""
import os
import sys

import numpy as np
import mujoco
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epa_train as R  # noqa: E402

wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402

out, tag = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(out, exist_ok=True)
verts = R.collider("foot")
spec = mujoco.MjSpec()
for i, (x, z, hz) in enumerate([(0.0, -0.05, 0.05), (0.3, 0.0, 0.1), (0.6, 0.05, 0.15)]):  # three stair boxes, steps 5 cm
  g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.15, 1.0, hz]; g.pos = [x, 0, z]
b = spec.worldbody.add_body(); b.pos = [0, 0, 1]; b.add_freejoint()
mm_ = spec.add_mesh(); mm_.name = "foot"; mm_.uservert = verts.reshape(-1).tolist()
s_ = b.add_geom(); s_.type = mujoco.mjtGeom.mjGEOM_MESH; s_.meshname = "foot"; s_.mass = 0.1
m = spec.compile(); d = mujoco.MjData(m)
rng = np.random.default_rng(seed)
Q = np.zeros((N, 7))
for k in range(N):
  q = R.upright(rng, True) if k % 2 else R.rand_quat(rng)
  if k % 4 == 1:  # exactly flat, grid-aligned yaw (coplanar faces)
    q = R.qmul(R.axang([0, 0, 1], rng.integers(4) * np.pi / 2), R.axang([0, 1, 0], np.pi / 2))
  x = rng.uniform(-0.1, 0.7); y = rng.uniform(-0.5, 0.5)
  d.qpos[:] = [x, y, 0.5, *q]; mujoco.mj_kinematics(m, d)
  A = m.mesh_vert @ d.geom_xmat[3].reshape(3, 3).T + d.geom_xpos[3]
  top = {0: 0.0, 1: 0.1, 2: 0.2}
  under = min(2, max(0, int((x + 0.15) // 0.3)))
  Q[k] = [x, y, 0.5 - (A[:, 2].min() - top[under]) + rng.uniform(-0.03, 0.01), *q]
with wp.ScopedDevice("cpu"):
  M = mjw.put_model(m); M.opt.warn_overflow = 0
  D = mjw.put_data(m, d, nworld=N, nconmax=32, njmax=64)
  D.qpos.assign(Q.astype(np.float32)); mjw.kinematics(M, D); mjw.collision(M, D)
  n = int(D.nacon.numpy()[0]); ov = D.overflow.numpy()
  np.savez(os.path.join(out, f"run_{tag}.npz"), world=D.contact.worldid.numpy()[:n], dist=D.contact.dist.numpy()[:n],
           pos=D.contact.pos.numpy()[:n], normal=D.contact.frame.numpy()[:n, 0], overflow=ov)
np.savez(os.path.join(out, "placements.npz"), qpos=Q)
print(mjw.__file__, f"[{tag}] {N} worlds, contacts {n}, EPA_HORIZON worlds {int(((ov & int(mjw.OverflowType.EPA_HORIZON)) != 0).sum())}")
