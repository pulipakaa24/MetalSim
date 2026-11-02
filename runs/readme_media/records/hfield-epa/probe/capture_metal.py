"""Capture the states in which MuJoCo Warp's EPA horizon overflows during rough_il3 training (stream hfield-epa).

Runs the smoke's rough_il3 training (train_wrap.py's settings: 4096 envs, --seed 0 --reward_cfg rough_il3 --contact_cfg
recommended_pyramidal --solver_cfg isaaclab3_every_substep_cap20 --base_velocity com --feet_slide_velocity com
--initial_reset 1 --effort_limit actuator) for ITERS PPO iterations with a substep hook that, the first time a world's
EPA_HORIZON bit is set, stores that world's qpos before the substep (= the collision's input) and after it.
EPA_HORIZON is not fatal here (as in train_wrap.py). Saves <out>/model.mjb, <out>/captures.npz, <out>/opt.json.

Usage (repo root): PYTHONPATH=<mjw copy> .venv/bin/python <this> <out> [iters] [n]
"""
import json
import os
import sys
import time

import numpy as np
import warp as wp

wp.config.quiet = True
import mujoco
import mujoco_warp as mjw

from metalsim.physics import capacity

_orig = capacity.capacity_overflows


def _cap_ov(flags):
  c = _orig(flags); n = c.pop("EPA_HORIZON", 0)
  print(f"[cap] EPA_HORIZON latched worlds {n}", flush=True)
  return c


capacity.capacity_overflows = _cap_ov

from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config  # noqa: E402
from metalsim.learn.ppo_warp import PPOWarp  # noqa: E402

EPA_BIT = wp.constant(int(mjw.OverflowType.EPA_HORIZON))
MAXCAP = 256


@wp.kernel
def _cap_kernel(overflow: wp.array(dtype=int), qpos: wp.array2d(dtype=float), prevq: wp.array2d(dtype=float),
                seen: wp.array(dtype=int), count: wp.array(dtype=int), substep: wp.array(dtype=int),
                capw: wp.array(dtype=int), capstep: wp.array(dtype=int), capq0: wp.array2d(dtype=float),
                capq1: wp.array2d(dtype=float)):
  w = wp.tid()
  if (overflow[w] & EPA_BIT) != 0:
    if seen[w] == 0:
      seen[w] = 1
      s = wp.atomic_add(count, 0, 1)
      if s < capw.shape[0]:
        capw[s] = w
        capstep[s] = substep[0]
        for i in range(qpos.shape[1]):
          capq0[s, i] = prevq[w, i]
          capq1[s, i] = qpos[w, i]
  else:
    seen[w] = 0
  for i in range(qpos.shape[1]):
    prevq[w, i] = qpos[w, i]


@wp.kernel
def _tick(substep: wp.array(dtype=int)):
  substep[0] = substep[0] + 1


def main():
  out = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 12; n = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
  os.makedirs(out, exist_ok=True)
  print("mujoco_warp", mjw.__file__, flush=True)
  task = G1VelocityTask(n, terrain="rough", seed=0, physics_dt=0.0025, reward_cfg="rough_il3",
                        contact_cfg="recommended_pyramidal", solver_cfg="isaaclab3_every_substep_cap20",
                        base_velocity="com", feet_slide_velocity="com", effort_limit="actuator")
  mujoco.mj_saveModel(task.model, os.path.join(out, "model.mjb"))
  sim = task.sim; d = sim.d; mo = sim.m.opt
  opt = {}
  for k in ("ccd_iterations", "ccd_tolerance", "enableflags", "disableflags"):
    try:
      v = getattr(mo, k); opt[k] = np.asarray(v.numpy() if hasattr(v, "numpy") else v).reshape(-1)[:1].tolist()
    except Exception as e:  # noqa: BLE001
      opt[k] = repr(e)
  try:
    opt["geom_gap"] = sim.m.geom_gap.numpy().reshape(-1)[:8].tolist(); opt["geom_margin"] = sim.m.geom_margin.numpy().reshape(-1)[:8].tolist()
  except Exception as e:  # noqa: BLE001
    opt["geom_gap"] = repr(e)
  opt["mjb_geom_gap"] = task.model.geom_gap.tolist(); opt["mjb_geom_margin"] = task.model.geom_margin.tolist()
  opt["naconmax"] = int(d.naconmax); opt["mjw_file"] = mjw.__file__
  json.dump(opt, open(os.path.join(out, "opt.json"), "w"), indent=1, default=str)
  print("opt", opt, flush=True)
  nq = d.qpos.shape[1]
  dev = sim.device
  with wp.ScopedDevice(dev):
    prevq = wp.clone(d.qpos); seen = wp.zeros(n, dtype=int); count = wp.zeros(1, dtype=int); substep = wp.zeros(1, dtype=int)
    capw = wp.full(MAXCAP, -1, dtype=int); capstep = wp.zeros(MAXCAP, dtype=int)
    capq0 = wp.zeros((MAXCAP, nq), dtype=float); capq1 = wp.zeros((MAXCAP, nq), dtype=float)

  def hook():
    wp.launch(_cap_kernel, dim=n, inputs=[d.overflow, d.qpos, prevq, seen, count, substep, capw, capstep, capq0, capq1])
    wp.launch(_tick, dim=1, inputs=[substep])

  sim.add_substep_hook(hook)
  algo = PPOWarp(task, g1_ppo_config("rough", iters, task.seed, initial_reset=True))
  t0 = time.time()
  snaps = []

  def cb(it, a):  # every world's qpos after each iteration: training-like states for the CPU cost profile
    wp.synchronize_device(dev); snaps.append(d.qpos.numpy().copy())

  algo.train(log=lambda s: print(s, flush=True), callback=cb)
  np.save(os.path.join(out, "qpos_snapshots.npy"), np.stack(snaps).astype(np.float32))
  wp.synchronize_device(dev)
  c = int(count.numpy()[0])
  print(f"captured {c} first-flag events in {iters} iterations x {n} worlds ({time.time() - t0:.0f} s); substeps {int(substep.numpy()[0])}",
        flush=True)
  k = min(c, MAXCAP)
  np.savez(os.path.join(out, "captures.npz"), world=capw.numpy()[:k], substep=capstep.numpy()[:k], qpos_pre=capq0.numpy()[:k],
           qpos_post=capq1.numpy()[:k], count=c, iters=iters, n=n)
  print("flags", sim.overflow_flags(), flush=True)


if __name__ == "__main__":
  main()
