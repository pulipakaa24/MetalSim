"""Warp CPU: per substep, worlds whose line search hit ls_iterations (overflow bit reset per substep), vs niter."""
import sys, os, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = 256
for cone, imp, lsit in [(1, 10.0, 20), (0, 1.0, 20), (1, 10.0, 50)]:
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = 100; m.opt.ls_iterations = lsit
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W)
    ls, nit = [], []
    for t in z["steps"]:
      k = f"t{t}"
      dd.qpos.assign(z[k + "_qpos"]); dd.qvel.assign(z[k + "_qvel"]); dd.ctrl.assign(z[k + "_ctrl"]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"])
      for s in range(8):
        dd.overflow.zero_(); mjw.step(mm, dd)
        ls.append((dd.overflow.numpy() & int(mjw.OverflowType.LS_ITERATIONS)) != 0); nit.append(dd.solver_niter.numpy().copy())
    ls = np.concatenate(ls); nit = np.concatenate(nit)
  print(f"cone {cone} imp {imp} ls_it {lsit}: niter mean {nit.mean():.2f} >10 {100*(nit>10).mean():.2f} % max {nit.max()} | solves with >=1 LS exhaustion {100*ls.mean():.1f} %; "
        f"among solves with niter>10: {100*ls[nit>10].mean() if (nit>10).any() else float('nan'):.0f} %; niter mean with / without exhaustion {nit[ls].mean() if ls.any() else float('nan'):.2f} / {nit[~ls].mean():.2f}", flush=True)
