"""Warp CPU: per Newton iteration index, share of the still-iterating worlds whose line search hit ls_iterations (20),
and whether that iteration was the world's last (hook on solver._solver_iteration, overflow reset per iteration)."""
import sys, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = 256
cone = int(sys.argv[1]); imp = float(sys.argv[2])
LS = int(mjw.OverflowType.LS_ITERATIONS)
rec = []
orig = solver._solver_iteration
def hooked(m, d, ctx, nsolving, compact=False):
  done0 = ctx.done.numpy().copy(); d.overflow.zero_()
  orig(m, d, ctx, nsolving, compact=compact)
  rec.append((~done0, (d.overflow.numpy() & LS) != 0, ctx.done.numpy().copy()))
solver._solver_iteration = hooked
m = mujoco.MjModel.from_binary_path(pre + ".mjb"); m.opt.cone = cone; m.opt.impratio = imp; m.opt.iterations = 30
stats = {}
with wp.ScopedDevice("cpu"):
  mm = mjw.put_model(m); mm.opt.warn_overflow = 0
  dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W)
  for t in z["steps"]:
    k = f"t{t}"; dd.qpos.assign(z[k+"_qpos"]); dd.qvel.assign(z[k+"_qvel"]); dd.ctrl.assign(z[k+"_ctrl"]); dd.qacc_warmstart.assign(z[k+"_qacc_warmstart"])
    for s in range(8):
      rec.clear(); mjw.step(mm, dd)
      for i, (act, ex, done1) in enumerate(rec):
        a = stats.setdefault(i + 1, [0, 0, 0, 0])
        a[0] += act.sum(); a[1] += (act & ex).sum(); last = act & done1
        a[2] += last.sum(); a[3] += (last & ex).sum()
tot = [sum(v[j] for v in stats.values()) for j in range(4)]
print(f"cone {cone} imp {imp}: LS calls {tot[0]}, exhausted {100*tot[1]/tot[0]:.1f} %; on a world's last iteration {100*tot[3]/max(1,tot[2]):.1f} %, on earlier iterations {100*(tot[1]-tot[3])/max(1,tot[0]-tot[2]):.1f} %")
print("  it: active worlds / % exhausted: " + "  ".join(f"{i}:{v[0]}/{100*v[1]/max(1,v[0]):.0f}" for i, v in sorted(stats.items()) if v[0]))
