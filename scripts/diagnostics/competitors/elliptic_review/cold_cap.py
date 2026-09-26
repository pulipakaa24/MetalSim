"""Warp CPU (float32), elliptic imp10: warm vs cold start x Newton cap, on a dumped state set, 8 substeps per state.
Reference: warm start, cap 100. Floor: warm cap 100 with a 1-ulp qvel nudge."""
import sys, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = sys.argv[1]; z = np.load(pre + ".npz"); W = z[f"t{z['steps'][0]}_qpos"].shape[0]
cone = int(sys.argv[2]) if len(sys.argv) > 2 else 1; imp = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
def run(cold, cap, nudge=False):
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = cap; m.opt.ls_iterations = 20
  if cold: m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W); nit, V = [], []
    for t in z["steps"]:
      k = f"t{t}"; qv = z[k+"_qvel"]
      if nudge: qv = np.nextafter(qv.astype(np.float32), np.float32(np.inf))
      dd.qpos.assign(z[k+"_qpos"]); dd.qvel.assign(qv); dd.ctrl.assign(z[k+"_ctrl"]); dd.qacc_warmstart.assign(z[k+"_qacc_warmstart"])
      r = []
      for s in range(8): mjw.step(mm, dd); r.append(dd.solver_niter.numpy().copy())
      nit.append(np.stack(r, 1)); V.append(dd.qvel.numpy().copy())
  return np.concatenate(nit), np.concatenate(V)
_, ref = run(False, 100)
print(f"{pre} cone {cone} imp {impratio if False else imp}: {W} worlds x {len(z['steps'])} states x 8 substeps")
for tag, cold, cap, nudge in [("floor (warm cap100 nudged)", False, 100, True), ("warm cap 100", False, 100, False), ("warm cap 10", False, 10, False), ("warm cap 15", False, 15, False), ("warm cap 20", False, 20, False),
                              ("cold cap 100", True, 100, False), ("cold cap 10", True, 10, False), ("cold cap 12", True, 12, False), ("cold cap 15", True, 15, False)]:
  n, v = run(cold, cap, nudge); dv = np.abs(v - ref).max(1)
  print(f"  {tag:28s} niter mean {n.mean():.2f} p99 {np.percentile(n,99):.0f} max {n.max():3d} % solves at >=11 {100*(n>=11).mean():.3f} >=13 {100*(n>=13).mean():.3f} >=16 {100*(n>=16).mean():.3f} | |dqvel| vs warm cap100: p99 {np.percentile(dv,99):.1e} max {dv.max():.1e} >1e-3 {(dv>1e-3).sum()} of {len(dv)}", flush=True)
