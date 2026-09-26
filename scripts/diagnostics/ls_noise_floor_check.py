"""Line-search derivative noise floor (MuJoCo Warp fork, MJW_LS_NOISE_FLOOR) on the CPU device, the review's
warp_cpu_lssweep protocol: dumped G1-task walking states (g1_state_dump.py), 8 substeps per snapshot, Newton cap 100.
Reference: ls_iterations 100 with the floor off. Floors: ls_iterations 200 (off) and a 1-ulp qvel nudge. Variants: the
task's ls_iterations 20 with the floor off / on at several eps multiples. Per variant: Newton iterations (mean, > 10,
max), share of control steps with a line search at its budget, the line-search iteration count per call (mean, and the
mean over launches of the maximum among the still-iterating worlds: the launch's latency), and |dqvel| after 8 substeps
vs the reference (median, p99, max, worlds > 1e-3).
    python scripts/diagnostics/ls_noise_floor_check.py [--worlds 256] [--cone elliptic|pyramidal] [--eps 8,2,32]"""
import argparse, os, sys, numpy as np, mujoco, warp as wp
os.environ["MJW_LS_STATS"] = "1"
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import solver

ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="runs/competitors/g1_states/ellip10"); ap.add_argument("--worlds", type=int, default=256)
ap.add_argument("--cone", default="elliptic"); ap.add_argument("--eps", default="8,2,32"); ap.add_argument("--cap", type=int, default=100)
a = ap.parse_args(); z = np.load(a.prefix + ".npz"); W = a.worlds
print("mujoco_warp", os.path.dirname(mjw.__file__), "cone", a.cone, "worlds", W, flush=True)

stats = {"n": 0, "sum": 0, "launch_max": []}
orig_iter = solver._solver_iteration
def hooked(m, d, ctx, nsolving, compact=False, **kw):
  done0 = ctx.done.numpy().copy()
  orig_iter(m, d, ctx, nsolving, compact=compact, **kw)
  it = ctx.ls_iters.numpy()[~done0]
  if it.size:
    stats["n"] += it.size; stats["sum"] += int(it.sum()); stats["launch_max"].append(int(it.max()))
solver._solver_iteration = hooked


def run(lsit, floor_eps, nudge=False, iters=None):
  solver._LS_NOISE_FLOOR_EPS = float(floor_eps)
  m = mujoco.MjModel.from_binary_path(a.prefix + ".mjb")
  if a.cone == "pyramidal":
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL; m.opt.impratio = 1.0
  m.opt.iterations = iters or a.cap; m.opt.ls_iterations = lsit
  stats.update(n=0, sum=0, launch_max=[])
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, mujoco.MjData(m), nworld=W, njmax=int(z["njmax"]), naconmax=int(z["nconmax"]) * W)
    q, v, ls, nit = [], [], [], []
    for t in z["steps"]:
      k = f"t{t}"; qv = z[k + "_qvel"][:W].copy()
      if nudge:
        qv = np.nextafter(qv.astype(np.float32), np.float32(np.inf))
      dd.qpos.assign(z[k + "_qpos"][:W]); dd.qvel.assign(qv); dd.ctrl.assign(z[k + "_ctrl"][:W]); dd.qacc_warmstart.assign(z[k + "_qacc_warmstart"][:W])
      if z[k + "_act"].size:
        dd.act.assign(z[k + "_act"][:W])
      e = np.zeros(W, bool)
      for s in range(8):
        dd.overflow.zero_(); mjw.step(mm, dd)
        e |= (dd.overflow.numpy() & int(mjw.OverflowType.LS_ITERATIONS)) != 0; nit.append(dd.solver_niter.numpy().copy())
      q.append(dd.qpos.numpy().copy()); v.append(dd.qvel.numpy().copy()); ls.append(e)
  return dict(q=np.concatenate(q), v=np.concatenate(v), ls=np.concatenate(ls), nit=np.concatenate(nit),
              ls_mean=stats["sum"] / max(1, stats["n"]), ls_launch_max_mean=float(np.mean(stats["launch_max"])), ls_launch_max_p50=float(np.median(stats["launch_max"])))


ref = run(100, 0.0)


def cmp(tag, r):
  dv = np.abs(r["v"] - ref["v"]).max(1)
  print(f"{tag:34s} niter mean {r['nit'].mean():.3f} >10 {100 * (r['nit'] > 10).mean():.2f} % max {r['nit'].max():2d} | LS at budget {100 * r['ls'].mean():5.1f} % of control steps | "
        f"LS iterations per call mean {r['ls_mean']:.2f}, launch max mean {r['ls_launch_max_mean']:.1f} p50 {r['ls_launch_max_p50']:.0f} | "
        f"|dqvel| vs ref: median {np.median(dv):.2e} p99 {np.percentile(dv, 99):.2e} max {dv.max():.2e}, worlds >1e-3 {(dv > 1e-3).sum()} of {len(dv)}", flush=True)


cmp("reference ls 100, floor off", ref)
cmp("floor: ls 200, floor off", run(200, 0.0))
cmp("floor: 1-ulp qvel nudge", run(100, 0.0, nudge=True))
cmp("ls 20, floor off (the task)", run(20, 0.0))
for e in [float(x) for x in a.eps.split(",")]:
  cmp(f"ls 20, floor {e:g} eps", run(20, e))
  cmp(f"ls 100, floor {e:g} eps", run(100, e))
