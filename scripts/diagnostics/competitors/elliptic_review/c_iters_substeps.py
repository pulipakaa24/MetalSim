"""MuJoCo C (float64): from each dumped walking state (control-step boundary, new ctrl), run the control step's
8 substeps (mj_step, ctrl held, C's own warm start) with a Newton cap of 100; solver_niter per substep index."""
import sys, numpy as np, mujoco
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
configs = [("ellip imp10 (task)", 1, 10.0, 1e-8), ("ellip imp1", 1, 1.0, 1e-8), ("pyr imp1 (task pyr)", 0, 1.0, 1e-8), ("pyr imp10", 0, 10.0, 1e-8),
           ("ellip imp10 tol 1e-6", 1, 10.0, 1e-6), ("ellip imp10 tol 1e-10", 1, 10.0, 1e-10)]
for name, cone, imp, tol in configs:
  m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = 100; m.opt.tolerance = tol
  d = mujoco.MjData(m); nit = np.zeros((len(z["steps"]) * W, 8), int); r = 0; nev = 0
  for t in z["steps"]:
    k = f"t{t}"
    for w in range(W):
      mujoco.mj_resetData(m, d)
      d.qpos[:] = z[k + "_qpos"][w]; d.qvel[:] = z[k + "_qvel"][w]; d.ctrl[:] = z[k + "_ctrl"][w]; d.qacc_warmstart[:] = z[k + "_qacc_warmstart"][w]
      for s in range(8):
        mujoco.mj_step(m, d); nit[r, s] = d.solver_niter[0]; nev += sum(int(d.solver[i].neval) for i in range(min(d.solver_niter[0], mujoco.mjNSOLVER)))
      r += 1
  a = nit[:, 1:].ravel(); f = nit[:, 0]
  print(f"{name:24s} substep 1 (new ctrl): mean {f.mean():.2f} p99 {np.percentile(f,99):.0f} max {f.max()} >10 {100*(f>10).mean():.2f} % | substeps 2-8: mean {a.mean():.2f} p90 {np.percentile(a,90):.0f} p99 {np.percentile(a,99):.0f} max {a.max()} >10 {100*(a>10).mean():.2f} % >15 {100*(a>15).mean():.3f} % | all: mean {nit.mean():.2f} >10 {100*(nit>10).mean():.2f} % | LS evals/iter {nev/nit.sum():.2f}")
  print(f"{'':24s} mean by substep: " + " ".join(f"{x:.2f}" for x in nit.mean(0)) + " | % of solves iterating at it 5,8,10,11,13,15,20: " + " ".join(f"{100*(nit>=i).mean():.2f}" for i in (5,8,10,11,13,15,20)))
