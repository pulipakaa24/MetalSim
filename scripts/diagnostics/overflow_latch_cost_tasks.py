"""Throughput cost of BatchSim's capacity-overflow latch (the only per-step change of PR #7) on the shipped training tasks,
in one process: the latch hook attached (A) / detached (B), interleaved repeats, the PPO rollout (G1, Tron1: PPOWarp's
captured rollout graph, recaptured after each toggle) or the env step (SO-101 lift: random actions through env.step).

    bash scripts/gpu_run.sh latch_tasks timing 40 -- .venv/bin/python scripts/diagnostics/overflow_latch_cost_tasks.py g1_flat
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this checkout's metalsim
from metalsim.physics.batch import BatchSim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=["g1_flat", "g1_rough", "tron1", "lift"])
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--rollouts", type=int, default=10, help="per measurement (lift: env steps / 10)")
ap.add_argument("--repeats", type=int, default=5)
a = ap.parse_args()
wp.config.quiet = True

if a.task.startswith("g1"):
    from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config
    from metalsim.learn.ppo_warp import PPOWarp
    terrain = a.task.split("_")[1]; n = a.n or 4096
    task = G1VelocityTask(n, terrain=terrain, seed=42 if terrain == "rough" else 0)
    algo = PPOWarp(task, g1_ppo_config(terrain, 1, 0))
elif a.task == "tron1":
    from metalsim.learn.ppo_warp import PPOWarp
    from metalsim.learn.tron1_wf import Tron1WFTask, ppo_config
    n = a.n or 8192
    task = Tron1WFTask(n, seed=0)
    algo = PPOWarp(task, ppo_config(1, 0))
else:
    from metalsim.learn.so101_lift import LiftConfig, SO101LiftEnv
    n = a.n or 64
    task = SO101LiftEnv(LiftConfig(num_envs=n, decimate_faces=2000, seed=0)); algo = None
sim = task.sim
full = list(sim._substep_hooks)
without = [h for h in full if getattr(h, "__func__", None) is not BatchSim._latch_overflow]
assert len(without) == len(full) - 1
if algo is not None:
    algo.prepare()
else:
    task.reset()


def attach(on):
    sim._substep_hooks[:] = full if on else without
    with wp.ScopedDevice(sim.device):
        sim._capture()
    if algo is not None:
        algo.graph = None                     # PPOWarp recaptures its rollout graph with the current hooks


def measure():
    if algo is not None:
        algo.rollout(); torch.mps.synchronize()     # (re)capture + warm-up
        t0 = time.perf_counter()
        for _ in range(a.rollouts):
            algo.rollout()
        torch.mps.synchronize()
        return a.rollouts * algo.cfg.rollout * n / (time.perf_counter() - t0)
    steps = a.rollouts * 10
    act = torch.zeros((n, task.act_dim if hasattr(task, "act_dim") else 6), device="mps")
    task.step(act.uniform_(-1, 1)); torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        task.step(act.uniform_(-1, 1))
    torch.mps.synchronize()
    return steps * n / (time.perf_counter() - t0)


res = {"latch": [], "none": []}
for r in range(a.repeats):
    for on in ((True, False) if r % 2 == 0 else (False, True)):
        attach(on)
        res["latch" if on else "none"].append(measure())
print(f"{a.task} N={n} substeps {sim.opt.substeps} njmax {sim.d.njmax}")
for k, v in res.items():
    print(f"  {k:5s}: median {np.median(v):,.0f} env-steps/s ({', '.join(f'{x:,.0f}' for x in v)})")
d = 100 * (np.median(res["none"]) / np.median(res["latch"]) - 1)
spread = 100 * (max(res["none"]) - min(res["none"])) / np.median(res["none"])
print(f"RESULT {a.task}: latch cost {d:+.2f} % (run-to-run spread without latch {spread:.1f} %)")
