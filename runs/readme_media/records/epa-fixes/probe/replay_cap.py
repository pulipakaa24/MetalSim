"""Replay the rough_il3 training states captured by hfield-epa (cap/: model.mjb at the rough_il3 settings, 12 PPO
iterations x 4096 worlds of qpos snapshots, and the 3 EPA_HORIZON states before/after their substep) through MuJoCo
Warp's kinematics + collision on Warp's CPU device, with whatever copy is first on PYTHONPATH, and save every contact
(world, geoms, dist, pos, frame normal) and the overflow bits for a bitwise comparison (stream epa-fixes).

Batches: snapshot iteration 0..11 (4096 worlds each); "events": the 3 captured states pre and post, each with 20 copies
jittered by up to 1e-6 m in the base position (as hfield-epa's replay.py, rng seed 0).
Usage: PYTHONPATH=<mjw tree> python replay_cap.py <capture dir> <out.npz>"""
import os
import sys

import mujoco
import numpy as np
import warp as wp

wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402

cap, out = sys.argv[1], sys.argv[2]
print("mujoco_warp", mjw.__file__, flush=True)
m = mujoco.MjModel.from_binary_path(os.path.join(cap, "model.mjb"))
S = np.load(os.path.join(cap, "qpos_snapshots.npy"))
C = np.load(os.path.join(cap, "captures.npz"))
rng = np.random.default_rng(0)
ev = []
for which in ("pre", "post"):
  Q = C["qpos_" + which]
  ev += [Q] + [Q + rng.uniform(-1e-6, 1e-6, Q.shape) * (np.arange(Q.shape[1]) < 3) for _ in range(20)]
batches = [(f"it{i}", S[i]) for i in range(len(S))] + [("events", np.concatenate(ev))]
print("ccd_iterations", m.opt.ccd_iterations, "ccd_tolerance", m.opt.ccd_tolerance, "geom_gap", np.unique(m.geom_gap).tolist(),
      "geom_margin", np.unique(m.geom_margin).tolist(), "batches", [(n, len(q)) for n, q in batches], flush=True)
res = {}
d = mujoco.MjData(m)
with wp.ScopedDevice("cpu"):
  mm = mjw.put_model(m)
  mm.opt.warn_overflow = 0
  for name, Q in batches:
    dd = mjw.put_data(m, d, nworld=len(Q), nconmax=64, njmax=256)
    dd.qpos.assign(Q.astype(np.float32))
    mjw.kinematics(mm, dd)
    mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0])
    ov = dd.overflow.numpy().copy()
    res[f"{name}_world"] = dd.contact.worldid.numpy()[:n]
    res[f"{name}_geom"] = dd.contact.geom.numpy()[:n]
    res[f"{name}_dist"] = dd.contact.dist.numpy()[:n]
    res[f"{name}_pos"] = dd.contact.pos.numpy()[:n]
    res[f"{name}_normal"] = dd.contact.frame.numpy()[:n, 0]
    res[f"{name}_overflow"] = ov
    epa = int(((ov & int(mjw.OverflowType.EPA_HORIZON)) != 0).sum())
    print(f"{name}: worlds {len(Q)} contacts {n} EPA_HORIZON worlds {epa} other overflow bits "
          f"{sorted(set(int(v) & ~int(mjw.OverflowType.EPA_HORIZON) for v in ov) - {0})}", flush=True)
np.savez(out, names=np.array([n for n, _ in batches]), mjw_file=mjw.__file__, **res)
print("saved", out)
