"""MuJoCo C with ls_iterations 20 / 50 / 100 vs MuJoCo Warp (ls 20) on the walking states (8 substeps, cap 100)."""
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = 256
def runC(cone, imp, ls):
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = 100; m.opt.ls_iterations = ls
  d = mujoco.MjData(m); nit = []; V = []; ne = 0
  for t in z["steps"]:
    k = f"t{t}"
    for w in range(W):
      mujoco.mj_resetData(m, d); d.qpos[:] = z[k+"_qpos"][w]; d.qvel[:] = z[k+"_qvel"][w]; d.ctrl[:] = z[k+"_ctrl"][w]; d.qacc_warmstart[:] = z[k+"_qacc_warmstart"][w]
      r = []
      for s in range(8):
        mujoco.mj_step(m, d); r.append(d.solver_niter[0]); ne += sum(int(d.solver[i].neval) for i in range(min(d.solver_niter[0], mujoco.mjNSOLVER)))
      nit.append(r); V.append(d.qvel.copy())
  nit = np.array(nit); return nit, np.array(V), ne / nit.sum()
def runW(cone, imp):
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = 100
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W); V = []
    for t in z["steps"]:
      k = f"t{t}"; dd.qpos.assign(z[k+"_qpos"]); dd.qvel.assign(z[k+"_qvel"]); dd.ctrl.assign(z[k+"_ctrl"]); dd.qacc_warmstart.assign(z[k+"_qacc_warmstart"])
      for s in range(8): mjw.step(mm, dd)
      V.append(dd.qvel.numpy().copy())
  return np.concatenate(V)
for cone, imp in [(1, 10.0), (0, 1.0), (0, 10.0)]:
  ww = runW(cone, imp)
  for ls in (20, 50, 100):
    n, v, ev = runC(cone, imp, ls); dv = np.abs(v - ww).max(1); mx = np.abs(v).max(1)
    print(f"cone {cone} imp {imp} C ls {ls:3d}: niter mean {n.mean():.2f} >10 {100*(n>10).mean():.2f} % max {n.max()} LS evals/iter {ev:.2f} | vs Warp: p99 {np.percentile(dv,99):.1e} max {dv.max():.1e} worlds >1e-3 {(dv>1e-3).sum()} | worlds max|qvel|>50: {(mx>50).sum()}", flush=True)
