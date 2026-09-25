"""G1 flat (MuJoCo Warp, 2.5 ms, 4096 envs) throughput under one configuration variant: the cost split of
g1_engine_bench.py (physics only / full env step / rollout + inference / full PPO loop) plus the per-world
constraint and contact maxima and overflow flags seen over the measured steps.

The variant is a JSON object in MJW_TP_VARIANT (default {} = the task as committed), keys:
  njmax, nconmax        per-world constraint rows / contact slots handed to BatchSim
  jacobian              "dense" | "sparse" | "auto" (Warp-side only, BatchSimOptions.jacobian)
  block_dim             {BlockDim field: int}
  no_kin                true: drop the all-world mjw.kinematics after reset (flat task has no reader)
  label                 printed tag
Warp Metal knobs (WP_METAL_INFLIGHT, WP_METAL_ICB_BATCH, ...) are plain environment variables.

usage: MJW_TP_VARIANT='{"njmax":128}' python scripts/diagnostics/g1_tp_variants.py [N] [--quick]"""
import json, os, sys, time
import numpy as np, torch, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import metalsim.learn.g1_velocity as g1v
from metalsim.physics.batch import BatchSimOptions

V = json.loads(os.environ.get("MJW_TP_VARIANT", "{}") or "{}")
N = int(next((a for a in sys.argv[1:] if not a.startswith("--")), 4096))
QUICK = "--quick" in sys.argv
tag = V.get("label") or (json.dumps({k: v for k, v in V.items() if k != "label"}, sort_keys=True) or "baseline")
env_knobs = {k: os.environ[k] for k in os.environ if k.startswith("WP_METAL")}
if env_knobs:
    tag += " " + " ".join(f"{k}={v}" for k, v in sorted(env_knobs.items()))


def _opts(**kw):
    for k in ("njmax", "nconmax", "jacobian", "block_dim", "m_dense_max", "metal_register_cholesky_max"):
        if k in V:
            kw[k] = V[k]
    return BatchSimOptions(**kw)


g1v.BatchSimOptions = _opts
from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step, g1_ppo_config
from metalsim.learn.ppo_warp import PPOWarp

task = G1VelocityTask(N, terrain=os.environ.get("MJW_TP_TERRAIN", "flat"), physics_dt=0.0025)
if V.get("gravcomp_launch"):     # A/B for the MuJoCo Warp fork's no-op gravity-compensation skip
    task.sim.m.has_gravcomp = True
    with wp.ScopedDevice(task.sim.device):
        task.sim._capture()
task.reset_all()
d = task.sim.d
print(f"[{tag}] N={N} njmax {d.njmax} naconmax {d.naconmax} is_sparse {task.sim.m.is_sparse}", flush=True)


def timed(fn, k):
    fn(); task.sim.synchronize(); t0 = time.perf_counter()
    for _ in range(k): fn()
    task.sim.synchronize(); return (time.perf_counter() - t0) / k


if V.get("no_kin"):              # every graph captured from here on (step, rollout) omits the post-reset kinematics
    mjw.kinematics = lambda m, d: None
p = timed(task.sim.step, 30)
print(f"[{tag}] physics only {p*1e3:.1f} ms/step -> {N/p:,.0f} env-steps/s", flush=True)
benchmark_step(task, num_frames=60, warmup=10)
t0 = time.perf_counter(); benchmark_step(task, num_frames=60, warmup=0); task.sim.synchronize(); s = (time.perf_counter() - t0) / 60
print(f"[{tag}] full env step {s*1e3:.1f} ms -> {N/s:,.0f} env-steps/s", flush=True)
nefc = d.nefc.numpy(); ov = task.sim.overflow_flags()
print(f"[{tag}] after random-action steps: nefc max {nefc.max()} mean {nefc.mean():.1f}; nacon {int(d.nacon.numpy()[0])} "
      f"({d.nacon.numpy()[0]/N:.2f}/world); overflow {ov}", flush=True)
if QUICK:
    sys.exit(0)
algo = PPOWarp(task, g1_ppo_config(task.terrain_kind, 1))
algo.rollout(); task.sim.synchronize(); torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(5): algo.rollout()
task.sim.synchronize(); torch.mps.synchronize(); r = (time.perf_counter() - t0) / 5
print(f"[{tag}] rollout + inference {r*1e3/24:.1f} ms/step -> {24*N/r:,.0f} env-steps/s", flush=True)
us = []
for _ in range(3):
    t0 = time.perf_counter(); algo.update(); torch.mps.synchronize(); us.append(time.perf_counter() - t0)
u = sorted(us)[1]
print(f"[{tag}] PPO update {u*1e3:.0f} ms -> full loop {24*N/(r+u):,.0f} env-steps/s", flush=True)
print(f"[{tag}] SUMMARY physics {N/p:,.0f} | step {N/s:,.0f} | rollout {24*N/r:,.0f} | loop {24*N/(r+u):,.0f}  "
      f"(ms: physics {p*1e3:.1f}, step {s*1e3:.1f}, rollout/step {r*1e3/24:.1f}, update {u*1e3:.0f})", flush=True)
