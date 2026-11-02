"""PARITY 1.6's stability check on the G1 rough task: 1024 envs, Isaac-scale random actions (unit-std Gaussian x 0.5),
400 control steps from reset, no resets in between (falls stay on the ground). Reports max |qvel|, non-finite states,
and the heightfield contacts (normal z, depth). C=<k>: also MuJoCo C (mj_step on the task's MjModel, as
tests/test_terrain.py does) on worlds 0..k-1 from the same start states with the same actions. MJW_SET as cc2.py.
Usage: PYTHONPATH=<mjw copy> python random_actions.py"""
import os, importlib, numpy as np, torch, warp as wp
import mujoco, mujoco_warp
for kv in filter(None, os.environ.get("MJW_SET", "").split(",")):
  k, v = kv.split("="); mod, attr = k.rsplit(".", 1); M = importlib.import_module(f"mujoco_warp._src.{mod}")
  setattr(M, attr, type(getattr(M, attr))(eval(v))); print("set", k, getattr(M, attr))
from metalsim.learn.g1_velocity import G1VelocityTask, ACTION_SCALE
from metalsim.interop import torch_bridge as tb
print("mujoco_warp", mujoco_warp.__file__)
n, K, T = 1024, int(os.environ.get("C", "0")), 400
task = G1VelocityTask(n, terrain="rough", seed=0); m = task.model
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
task.reset_all(); task.sim.synchronize()
q0, v0 = task.sim.d.qpos.numpy()[:K].copy(), task.sim.d.qvel.numpy()[:K].copy()
dq = task.default_q.numpy()[7:]
action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); ta = tb.mps_tensor(action)
g = torch.Generator(device="cpu").manual_seed(0)
A = [(0.5 * torch.randn(n, task.act_dim, generator=g)) for _ in range(T)]
qmax, nonfinite, nz, dep = 0.0, 0, [], []
for t in range(T):
  ta.copy_(A[t].to(ta.device))
  task.launch_apply_action(action); task.sim.launch_step(); task.sim.synchronize()
  d = task.sim.d
  qv = d.qvel.numpy(); fin = np.isfinite(qv).all(1)
  nonfinite = max(nonfinite, int((~fin).sum())); qmax = max(qmax, float(np.abs(qv[fin]).max()) if fin.any() else 0.0)
  if t % 10 == 0:
    na = int(d.nacon.numpy()[0]); gm = d.contact.geom.numpy()[:na]; hf = (gm == floor).any(1)
    nz.append(d.contact.frame.numpy()[:na, 0, 2][hf]); dep.append(-d.contact.dist.numpy()[:na][hf])
nz, dep = np.concatenate(nz), np.concatenate(dep)
print(f"MJW 400 steps x {n} envs: max |qvel| {qmax:.1f}, worlds with non-finite qvel {nonfinite}; heightfield contacts sampled {len(nz)}: "
      f"normal z min {nz.min():.3f} p1 {np.percentile(nz, 1):.3f} share < 0.3 {np.mean(nz < 0.3):.4f}; depth p99 {np.percentile(dep, 99) * 1e3:.1f} mm max {dep.max() * 1e3:.1f} mm")
if K:
  dc = mujoco.MjData(m); cq, cnf, cnz, cdep = 0.0, 0, [], []
  for e in range(K):
    dc.qpos[:] = q0[e]; dc.qvel[:] = v0[e]; mujoco.mj_forward(m, dc); bad = False
    for t in range(T):
      dc.ctrl[:] = dq + ACTION_SCALE * np.clip(A[t][e].numpy(), -100, 100)
      for _ in range(task.decimation):
        mujoco.mj_step(m, dc)
      if not np.isfinite(dc.qvel).all():
        bad = True; break
      cq = max(cq, float(np.abs(dc.qvel).max()))
      if t % 10 == 0:
        for c in dc.contact[:dc.ncon]:
          if floor in (c.geom1, c.geom2):
            cnz.append(c.frame[2]); cdep.append(-c.dist)
    cnf += int(bad)
  cnz, cdep = np.array(cnz), np.array(cdep)
  print(f"C   400 steps x {K} envs (same starts and actions): max |qvel| {cq:.1f}, worlds with non-finite qvel {cnf}; heightfield contacts sampled {len(cnz)}: "
        f"normal z min {cnz.min():.3f} p1 {np.percentile(cnz, 1):.3f} share < 0.3 {np.mean(cnz < 0.3):.4f}; depth p99 {np.percentile(cdep, 99) * 1e3:.1f} mm max {cdep.max() * 1e3:.1f} mm")
