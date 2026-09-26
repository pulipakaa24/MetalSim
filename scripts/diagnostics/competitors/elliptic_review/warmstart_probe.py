"""Warm start vs cold start (qacc_smooth) on the walking states, 8 substeps each, cap 100.
MuJoCo C (float64, best-of(warmstart, smooth) built in) and MuJoCo Warp CPU (float32, warm start taken as is).
State difference after 8 substeps vs the warm-started reference of the same engine; floor = 1-ulp qvel nudge (Warp)."""
import sys, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = 256
def model(cone, imp, cold):
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = 100
  if cold: m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
  return m
def runC(cone, imp, cold):
  m = model(cone, imp, cold); d = mujoco.MjData(m); nit = []; V = []
  for t in z["steps"]:
    k = f"t{t}"
    for w in range(W):
      mujoco.mj_resetData(m, d); d.qpos[:] = z[k+"_qpos"][w]; d.qvel[:] = z[k+"_qvel"][w]; d.ctrl[:] = z[k+"_ctrl"][w]; d.qacc_warmstart[:] = z[k+"_qacc_warmstart"][w]
      r = []
      for s in range(8): mujoco.mj_step(m, d); r.append(d.solver_niter[0])
      nit.append(r); V.append(d.qvel.copy())
  return np.array(nit), np.array(V)
def runW(cone, imp, cold, nudge=False):
  m = model(cone, imp, cold)
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W)
    nit, V = [], []
    for t in z["steps"]:
      k = f"t{t}"; qv = z[k+"_qvel"]
      if nudge: qv = np.nextafter(qv.astype(np.float32), np.float32(np.inf))
      dd.qpos.assign(z[k+"_qpos"]); dd.qvel.assign(qv); dd.ctrl.assign(z[k+"_ctrl"]); dd.qacc_warmstart.assign(z[k+"_qacc_warmstart"])
      r = []
      for s in range(8): mjw.step(mm, dd); r.append(dd.solver_niter.numpy().copy())
      nit.append(np.stack(r, 1)); V.append(dd.qvel.numpy().copy())
  return np.concatenate(nit), np.concatenate(V)
def line(tag, n, v=None, ref=None):
  s = f"{tag:40s} mean {n.mean():.2f} | substep 1 mean {n[:,0].mean():.2f} >10 {100*(n[:,0]>10).mean():.2f} % | substeps 2-8 mean {n[:,1:].mean():.2f} >10 {100*(n[:,1:]>10).mean():.2f} % max {n[:,1:].max()} | all >10 {100*(n>10).mean():.2f} %"
  if ref is not None:
    dv = np.abs(v - ref).max(1); s += f" | |dqvel| vs warm ref: p99 {np.percentile(dv,99):.1e} max {dv.max():.1e} >1e-3 {(dv>1e-3).sum()}"
  print(s, flush=True)
for cone, imp in [(1, 10.0), (0, 1.0), (1, 1.0)]:
  nw, vw = runC(cone, imp, False); nc, vc = runC(cone, imp, True)
  line(f"C cone {cone} imp {imp} warm", nw); line(f"C cone {cone} imp {imp} cold", nc, vc, vw)
  nw, vw = runW(cone, imp, False); nn, vn = runW(cone, imp, False, nudge=True); nc, vc = runW(cone, imp, True)
  line(f"Warp cone {cone} imp {imp} warm", nw); line(f"Warp cone {cone} imp {imp} warm, 1-ulp nudge (floor)", nn, vn, vw); line(f"Warp cone {cone} imp {imp} cold", nc, vc, vw)
