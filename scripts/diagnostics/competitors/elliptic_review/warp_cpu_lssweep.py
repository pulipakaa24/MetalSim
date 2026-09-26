"""Warp CPU: ls_iterations sweep on the walking states (elliptic imp10): niter, LS exhaustion, and the state after
8 substeps vs a reference (ls_iterations 100). Floor: ls 100 vs ls 200 and the impratio-10 run with a 1-ulp qvel nudge."""
import sys, os, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = 256
cone = int(sys.argv[1]) if len(sys.argv) > 1 else 1; imp = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
def run(lsit, nudge=False, tol=1e-8, lstol=0.01, iters=100):
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = iters; m.opt.ls_iterations = lsit
  m.opt.tolerance = tol; m.opt.ls_tolerance = lstol
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W)
    q, v, ls, nit = [], [], [], []
    for t in z["steps"]:
      k = f"t{t}"; qv = z[k + "_qvel"].copy()
      if nudge: qv = np.nextafter(qv.astype(np.float32), np.float32(np.inf))
      dd.qpos.assign(z[k + "_qpos"]); dd.qvel.assign(qv); dd.ctrl.assign(z[k + "_ctrl"]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"])
      e = np.zeros(W, bool)
      for s in range(8):
        dd.overflow.zero_(); mjw.step(mm, dd)
        e |= (dd.overflow.numpy() & int(mjw.OverflowType.LS_ITERATIONS)) != 0; nit.append(dd.solver_niter.numpy().copy())
      q.append(dd.qpos.numpy().copy()); v.append(dd.qvel.numpy().copy()); ls.append(e)
  return np.concatenate(q), np.concatenate(v), np.concatenate(ls), np.concatenate(nit)
ref = run(100)
def cmp(tag, r):
  dv = np.abs(r[1] - ref[1]).max(1)
  print(f"{tag:26s} niter mean {r[3].mean():.3f} >10 {100*(r[3]>10).mean():.2f} % max {r[3].max()} | control steps with an LS at its cap {100*r[2].mean():.1f} % | "
        f"|dqvel| vs ls100 after 8 substeps: median {np.median(dv):.2e} p99 {np.percentile(dv,99):.2e} max {dv.max():.2e}, worlds >1e-3 {(dv>1e-3).sum()} of {len(dv)}", flush=True)
cmp("floor: ls 200", run(200)); cmp("floor: 1-ulp qvel nudge", run(100, nudge=True))
for l in (5, 10, 15, 20, 30, 50): cmp(f"ls_iterations {l}", run(l))
cmp("ls_tolerance 0.1 (ls 20)", run(20, lstol=0.1)); cmp("tolerance 1e-6 (ls 20)", run(20, tol=1e-6)); cmp("tolerance 1e-7 (ls 20)", run(20, tol=1e-7))
cmp("cap 10 (ls 20)", run(20, iters=10)); cmp("cap 12 (ls 20)", run(20, iters=12)); cmp("cap 15 (ls 20)", run(20, iters=15)); cmp("cap 20 (ls 20)", run(20, iters=20))
