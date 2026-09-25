"""Full-step cost split of the G1 flat task on either engine, measured like runs/bench_clean.log
("cost split"): graph replays, synchronized wall clock, 4096 envs.

  physics only      : task.sim.step() (one control step = decimation substeps, graph)
  full env step     : benchmark_step (apply action + physics + reward/termination/reset + observations,
                      one captured graph per step, random actions written by torch)
  rollout           : PPOWarp.rollout(), 24 steps with Warp policy inference (the training rollout)
  full loop         : rollout + one PPO update (5 epochs x 4 minibatches) = training env-steps/s

Nothing is excluded: resets, observations, rewards and PPO are in the last line. The step-only line
of the same script (`python -m metalsim.learn.g1_velocity N flat`) is the "full env step" row here.

usage: python scripts/diagnostics/g1_engine_bench.py N mjwarp DT | N newton ITERS DT"""
import sys, time, warp as wp, torch
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step, g1_ppo_config
from metalsim.learn.ppo_warp import PPOWarp

n = int(sys.argv[1]); engine = sys.argv[2]
if engine == "newton":
    it, dt = int(sys.argv[3]), float(sys.argv[4])
    task = G1VelocityTask(n, terrain="flat", engine="newton", newton_iterations=it, newton_dt=dt); tag = f"newton {it} it {dt*1e3:.3g} ms"
else:
    dt = float(sys.argv[3]); task = G1VelocityTask(n, terrain="flat", physics_dt=dt); tag = f"mjwarp {dt*1e3:.3g} ms"
task.reset_all()


def timed(fn, k):
    fn(); task.sim.synchronize(); t0 = time.perf_counter()
    for _ in range(k): fn()
    task.sim.synchronize(); return (time.perf_counter() - t0) / k


p = timed(task.sim.step, 30)
print(f"N={n} {tag}: physics only ({task.decimation} substeps, graph) {p*1e3:.1f} ms/step -> {n/p:,.0f} env-steps/s", flush=True)
benchmark_step(task, num_frames=60, warmup=10)
t0 = time.perf_counter(); benchmark_step(task, num_frames=60, warmup=0); task.sim.synchronize(); s = (time.perf_counter() - t0) / 60
print(f"N={n} {tag}: full env step (graph, synchronized) {s*1e3:.1f} ms -> {n/s:,.0f} env-steps/s", flush=True)
algo = PPOWarp(task, g1_ppo_config("flat", 1))
algo.rollout(); task.sim.synchronize(); torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(5): algo.rollout()
task.sim.synchronize(); torch.mps.synchronize(); r = (time.perf_counter() - t0) / 5
print(f"N={n} {tag}: rollout with Warp policy inference ({algo.cfg.hidden}), 24 steps: {r*1e3:.0f} ms -> {24*n/r:,.0f} env-steps/s", flush=True)
us = []
for _ in range(3):
    t0 = time.perf_counter(); algo.update(); torch.mps.synchronize(); us.append(time.perf_counter() - t0)
u = sorted(us)[1]
print(f"N={n} {tag}: PPO update (5 epochs x 4 minibatches) {u*1e3:.0f} ms -> full loop {24*n/(r+u):,.0f} env-steps/s", flush=True)
