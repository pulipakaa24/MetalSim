import numpy as np, mujoco
pre="runs/competitors/g1_states/ellip10"; z=np.load(pre+".npz"); W=256; steps=list(z["steps"])
def run(gi, **o):
  t=steps[gi//W]; w=gi%W; k=f"t{t}"
  m=mujoco.MjModel.from_binary_path(pre+".mjb"); m.opt.iterations=o.get("it",100); m.opt.ls_iterations=o.get("ls",20); m.opt.tolerance=o.get("tol",1e-8)
  if o.get("pyr"): m.opt.cone=0
  if o.get("imp"): m.opt.impratio=o["imp"]
  d=mujoco.MjData(m); d.qpos[:]=z[k+"_qpos"][w]; d.qvel[:]=z[k+"_qvel"][w]; d.ctrl[:]=z[k+"_ctrl"][w]; d.qacc_warmstart[:]=z[k+"_qacc_warmstart"][w]
  r=[]
  for s in range(8):
    mujoco.mj_step(m,d); r.append((int(d.solver_niter[0]), round(float(np.abs(d.qvel).max()),2), int(np.argmax(np.abs(d.qvel)))))
  return r
for gi in (2252, 1692, 1293):
  print(gi, "default ", run(gi)); print(gi, "ls 100  ", run(gi, ls=100)); print(gi, "it 1000 tol 1e-12", run(gi, it=1000, tol=1e-12)); print(gi, "imp 1   ", run(gi, imp=1.0))
m=mujoco.MjModel.from_binary_path(pre+".mjb"); print([m.joint(m.dof_jntid[i]).name for i in (6,20,30,40)])
