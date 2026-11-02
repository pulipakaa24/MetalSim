"""Replay the EPA_HORIZON states captured on Metal (capture_metal.py) through MuJoCo Warp's collision on Warp's CPU
device, with whatever copy is first on PYTHONPATH (stream hfield-epa).
Usage: PYTHONPATH=<mjw copy> python replay.py <capture dir> [pre|post] [nworld_jitter]
Per captured world: EPA_HORIZON bit, the heightfield contacts (geom, dist) and MuJoCo C's (3.14) contacts for reference.
nworld_jitter > 0: also replays that many copies of each state with qpos jittered by up to 1e-6 (float noise)."""
import os
import sys

import numpy as np
import mujoco
import warp as wp

wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402

cap = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "pre"; nj = int(sys.argv[3]) if len(sys.argv) > 3 else 0
print("mujoco_warp", mjw.__file__)
m = mujoco.MjModel.from_binary_path(os.path.join(cap, "model.mjb"))
C = np.load(os.path.join(cap, "captures.npz")); Q = C["qpos_" + which]
print("captured events", int(C["count"]), "stored", len(Q), "geom_gap", np.unique(m.geom_gap).tolist(), "ccd_iterations", m.opt.ccd_iterations)
rng = np.random.default_rng(0)
QQ = [Q] + [Q + rng.uniform(-1e-6, 1e-6, Q.shape) * (np.arange(Q.shape[1]) < 3) for _ in range(nj)]
QA = np.concatenate(QQ)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
d = mujoco.MjData(m)
with wp.ScopedDevice("cpu"):
  mm = mjw.put_model(m); mm.opt.warn_overflow = 0
  dd = mjw.put_data(m, d, nworld=len(QA), nconmax=64, njmax=256)
  dd.qpos.assign(QA.astype(np.float32)); mjw.kinematics(mm, dd); mjw.collision(mm, dd)
  n = int(dd.nacon.numpy()[0]); ov = dd.overflow.numpy()
  W = dd.contact.worldid.numpy()[:n]; G = dd.contact.geom.numpy()[:n]; D = dd.contact.dist.numpy()[:n]
epa = (ov & int(mjw.OverflowType.EPA_HORIZON)) != 0
print(f"replayed {len(QA)} worlds ({len(Q)} states x {1 + nj}): EPA_HORIZON in {int(epa.sum())}; first-copy {int(epa[:len(Q)].sum())}")
for i in range(len(Q)):
  m_ = W == i
  ours = sorted((names[a] if a >= 0 else "?", names[b], round(float(x) * 1e3, 2)) for (a, b), x in zip(G[m_], D[m_]))
  d.qpos[:] = Q[i]; mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
  cc = sorted((names[d.contact.geom1[k]], names[d.contact.geom2[k]], round(float(d.contact.dist[k]) * 1e3, 2)) for k in range(d.ncon))
  jit = int(epa[i::len(Q)].sum())
  print(f"[{i}] world {int(C['world'][i])} substep {int(C['substep'][i])} EPA {bool(epa[i])} (jittered copies flagged {jit}/{1 + nj}) "
        f"pelvis z {Q[i, 2]:.3f}\n    mjw  {ours}\n    C    {cc}")
