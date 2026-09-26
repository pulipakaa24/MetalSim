"""MuJoCo Warp (fork, float32) on Warp's CPU device: the same dumped walking states as c_iters_substeps.py,
8 substeps per state (ctrl held), Newton cap 100; solver_niter per substep. Read-only import of the fork."""
import sys, os, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
cfgs = [a.split(":") for a in (sys.argv[2:] or ["1:10", "0:1"])]
print("mujoco_warp", os.path.dirname(mjw.__file__), "MJW_ELLIPTIC_INCREMENTAL", os.environ.get("MJW_ELLIPTIC_INCREMENTAL", "2(default)"))
for cone, imp in cfgs:
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = int(cone); m.opt.impratio = float(imp); m.opt.iterations = 100
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); d0 = mujoco.MjData(m)
    dd = mjw.put_data(m, d0, nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W)
    nit = []
    for t in z["steps"]:
      k = f"t{t}"
      dd.qpos.assign(z[k + "_qpos"][:W]); dd.qvel.assign(z[k + "_qvel"][:W]); dd.ctrl.assign(z[k + "_ctrl"][:W]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"][:W])
      row = []
      for s in range(8):
        mjw.step(mm, dd); row.append(dd.solver_niter.numpy().copy())
      nit.append(np.stack(row, 1))
    nit = np.concatenate(nit, 0)
  f = nit[:, 0]; a = nit[:, 1:].ravel()
  name = f"cone {cone} imp {imp}"
  print(f"{name:16s} substep 1: mean {f.mean():.2f} max {f.max()} >10 {100*(f>10).mean():.2f} % | substeps 2-8: mean {a.mean():.2f} p99 {np.percentile(a,99):.0f} max {a.max()} >10 {100*(a>10).mean():.2f} % >15 {100*(a>15).mean():.3f} % | all mean {nit.mean():.2f} >10 {100*(nit>10).mean():.2f} %", flush=True)
  print(f"{'':16s} mean by substep " + " ".join(f"{x:.2f}" for x in nit.mean(0)) + " | % iterating at it 5,8,10,11,13,15,17,20: " + " ".join(f"{100*(nit>=i).mean():.3f}" for i in (5,8,10,11,13,15,17,20)), flush=True)
  np.save(f"runs/competitors/elliptic_review/warp_nit_c{cone}_i{imp}.npy", nit)
