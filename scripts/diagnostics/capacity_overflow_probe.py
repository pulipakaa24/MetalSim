"""Does BatchSim.check_overflow() see every capacity overflow on the G1 flat task? (2026-10-13, fix-known-failures)

Runs the G1 velocity task (flat, 64 worlds) under uniform random actions, as tests/test_capacity.py does, and after every
control step's physics (before the task's reward / done / reset kernels) and again after them records: per-world nefc
(last substep), the global contact count, and the per-world overflow bits. Reports how many worlds overflowed at any
point, and how many of those bits survive to the next check (the task's masked mjw.reset_data zeroes d.overflow of the
worlds it resets).

    .venv/bin/python scripts/diagnostics/capacity_overflow_probe.py [--njmax 32 --nconmax 24 --steps 100]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this checkout's metalsim

import numpy as np
import warp as wp

wp.config.quiet = True

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--njmax", default="32")
ap.add_argument("--nconmax", default="24")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cap = lambda s: s if s == "auto" else int(s)
import metalsim
print("metalsim from", metalsim.__file__)

import mujoco_warp as mjw
from metalsim.learn import g1_velocity as g1v
from metalsim.learn.warp_policy import RolloutBuffers, bump
from metalsim.physics.batch import BatchSimOptions

orig = g1v.BatchSimOptions
g1v.BatchSimOptions = lambda **kw: orig(**{**kw, "njmax": cap(a.njmax), "nconmax": cap(a.nconmax)})
task = g1v.G1VelocityTask(a.n, terrain="flat", physics_dt=0.0025, seed=0)
g1v.BatchSimOptions = orig
task.reset_all()
sim, dev, n = task.sim, task.device, task.n
b = sim.capacity_bounds
print("model: cone", "elliptic" if task.sim.mj_model.opt.cone == 1 else "pyramidal", "| njmax", sim.d.njmax, "naconmax", sim.d.naconmax,
      "| bounds:", None if b is None else dict(ncon=b.ncon, nefc=b.nefc, njmax=b.njmax, nlimit=b.nlimit,
                                               rows=sorted({r for *_, r in b.pairs})))
rng = np.random.default_rng(a.seed)


class _Pol:
    step_idx = wp.zeros(1, dtype=int, device=dev)


pol = _Pol()
bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, n), dtype=float, device=dev); bufs.done = wp.zeros((1, n), dtype=float, device=dev)
NEFC = int(mjw.OverflowType.NEFC); NP = int(mjw.OverflowType.NARROWPHASE)
ever = np.zeros(n, bool); nefc_max = 0; nacon_max = 0; lost = 0; steps_flagged = 0; resets = 0
for k in range(a.steps):
    act = wp.array(rng.uniform(-1, 1, (n, task.act_dim)).astype(np.float32), dtype=float, device=dev)
    with wp.ScopedDevice(dev):
        wp.launch(bump, dim=1, inputs=[pol.step_idx], device=dev)
        task.launch_apply_action(act); sim.launch_step()
    sim.synchronize()
    nefc = sim.d.nefc.numpy(); ov = sim.d.overflow.numpy(); nacon = int(sim.d.nacon.numpy()[0])
    nefc_max = max(nefc_max, int(nefc.max())); nacon_max = max(nacon_max, nacon)
    flagged = (ov & (NEFC | NP)) != 0
    ever |= flagged; steps_flagged += bool(flagged.any())
    lat = getattr(sim, "_overflow_latched", None)
    if lat is not None:
        miss = flagged & ((lat.numpy() & (NEFC | NP)) == 0)
        if miss.any():
            print(f"step {k}: {int(miss.sum())} flagged worlds not latched: ov {ov[miss][:4]} latched {lat.numpy()[miss][:4]} nefc {nefc[miss][:4]}")
    with wp.ScopedDevice(dev):
        task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
    sim.synchronize()
    ov2 = sim.d.overflow.numpy(); done = bufs.done.numpy()[0] > 0.5
    resets += int(done.sum())
    lost += int((flagged & ((ov2 & (NEFC | NP)) == 0)).sum())
    if k < 3 or k % 10 == 9:
        print(f"step {k:3d}: nefc max {int(nefc.max())} (> njmax in {int((nefc > sim.d.njmax).sum())} worlds), nacon {nacon}, "
              f"flagged {int(flagged.sum())}, cleared by this step's resets {int((flagged & ((ov2 & (NEFC | NP)) == 0)).sum())}")
final = sim.overflow_flags()
if getattr(sim, "_overflow_latched", None) is not None:
    latw = np.nonzero(sim._overflow_latched.numpy())[0]
    print("worlds seen flagged:", np.nonzero(ever)[0].tolist(), "| latched:", latw.tolist(), "values", sim._overflow_latched.numpy()[latw].tolist())
try:
    sim.check_overflow(); raised = False
except RuntimeError as e:
    raised = True; print("check_overflow raised:", e)
res = dict(njmax=a.njmax, nconmax=a.nconmax, steps=a.steps, nefc_max=nefc_max, nacon_max=nacon_max, worlds_ever_overflowed=int(ever.sum()),
           steps_with_overflow=steps_flagged, flags_cleared_by_reset=lost, resets=resets, final_flags=final, check_overflow_raised=raised)
print(json.dumps(res))
