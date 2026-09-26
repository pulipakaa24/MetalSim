"""MuJoCo C (float64) Newton iteration counts on the dumped G1-task walking states (g1_state_dump.py:
ellip10-trained policy, 256 worlds x 9 snapshots). Per configuration: solver_niter distribution with a
cap of 100, share of solves needing > 10 / > 20, line-search evaluations per iteration (solver stats)."""
import sys, numpy as np, mujoco
pre = sys.argv[1] if len(sys.argv) > 1 else "runs/competitors/g1_states/ellip10"
m0 = mujoco.MjModel.from_binary_path(pre + ".mjb"); z = np.load(pre + ".npz")
W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
configs = [
  ("ellip imp10 (task)", dict(cone=1, impratio=10.0)),
  ("ellip imp1", dict(cone=1, impratio=1.0)),
  ("ellip imp100", dict(cone=1, impratio=100.0)),
  ("pyr imp1 (task pyr)", dict(cone=0, impratio=1.0)),
  ("pyr imp10", dict(cone=0, impratio=10.0)),
  ("ellip imp10 no warmstart", dict(cone=1, impratio=10.0, nows=True)),
  ("pyr imp1 no warmstart", dict(cone=0, impratio=1.0, nows=True)),
  ("ellip imp10 ls_it 50", dict(cone=1, impratio=10.0, ls=50)),
  ("ellip imp10 tol 1e-6", dict(cone=1, impratio=10.0, tol=1e-6)),
  ("pyr imp1 tol 1e-6", dict(cone=0, impratio=1.0, tol=1e-6)),
]
print(f"model nv {m0.nv} tol {m0.opt.tolerance} ls_tol {m0.opt.ls_tolerance} ls_it {m0.opt.ls_iterations} mujoco {mujoco.__version__}; {W} worlds x {len(z['steps'])} snapshots")
for name, c in configs:
  m = mujoco.MjModel.from_binary_path(pre + ".mjb")
  m.opt.cone = c["cone"]; m.opt.impratio = c["impratio"]; m.opt.iterations = 100
  m.opt.ls_iterations = c.get("ls", 20); m.opt.tolerance = c.get("tol", 1e-8)
  if c.get("nows"): m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
  d = mujoco.MjData(m)
  nit, nev, ncone, nefc = [], [], [], []
  for t in z["steps"]:
    k = f"t{t}"
    for w in range(W):
      mujoco.mj_resetData(m, d)
      d.qpos[:] = z[k + "_qpos"][w]; d.qvel[:] = z[k + "_qvel"][w]; d.ctrl[:] = z[k + "_ctrl"][w]
      if z[k + "_act"].size: d.act[:] = z[k + "_act"][w]
      d.qacc_warmstart[:] = z[k + "_qacc_warmstart"][w]
      mujoco.mj_forward(m, d)
      it = int(d.solver_niter[0]); nit.append(it)
      nev.append(sum(int(d.solver[i].neval) for i in range(min(it, mujoco.mjNSOLVER))))
      nefc.append(d.nefc)
      ncone.append(int((d.efc_state[:d.nefc] == mujoco.mjtConstraintState.mjCNSTRSTATE_CONE).sum()))
  nit = np.array(nit); nev = np.array(nev)
  alive = [(nit >= i).mean() * 100 for i in (1, 2, 3, 4, 5, 6, 8, 10, 11, 15, 20, 21)]
  print(f"{name:28s} niter mean {nit.mean():.2f} p50 {np.median(nit):.0f} p90 {np.percentile(nit,90):.0f} p99 {np.percentile(nit,99):.0f} max {nit.max():3d} | "
        f">10: {(nit>10).mean()*100:5.2f} % >20: {(nit>20).mean()*100:5.2f} % | LS evals/iter {nev.sum()/max(1,nit.sum()):.2f} | nefc {np.mean(nefc):.1f} cone rows at exit {np.mean(ncone):.1f}")
  print(f"{'':28s} % solves still iterating at iteration 1,2,3,4,5,6,8,10,11,15,20,21: " + " ".join(f"{a:.1f}" for a in alive))
