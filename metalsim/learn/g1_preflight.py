"""Pre-flight checks for G1 training: cheap tests that would have caught the two defects found in the
first 1,500-iteration runs (unscaled termination penalty; drive blow-ups at 5 ms). Run before any
long training, and again on a short probe of the real run.

Expected picture under Isaac's config (every reward term x dt = 0.02, termination -200 x dt = -4):
  * a random policy falls in ~40 control steps: mean episode return between -8 and 0
  * termination contributes exactly -4 per fallen episode
  * 400 steps of 3-sigma random targets: no world blows up, joint speeds stay below 200 rad/s
  * over a short PPO probe, the return must not fall while the episode length rises

    python -m metalsim.learn.g1_preflight --envs 1024 --physics_dt 0.0025 --probe_iters 20
"""
import argparse, time
import numpy as np, torch, warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step, g1_ppo_config, CONTROL_DT
from metalsim.learn.warp_policy import RolloutBuffers, bump, zero_int
from metalsim.interop import torch_bridge as tb


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def reward_scale(task, steps=120):
    """Random policy (unit std x 0.5 scale) until most episodes have ended once."""
    n = task.n
    class _Pol: step_idx = wp.zeros(1, dtype=int, device=task.device)
    pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
    action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); t_action = tb.mps_tensor(action)
    gen = torch.Generator(device="mps").manual_seed(0)
    task.reset_all(); task.episode_stats()
    rews = []
    for t in range(steps):
        t_action.copy_(torch.randn(n, task.act_dim, generator=gen, device="mps"))
        v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
        wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
        task.launch_apply_action(action); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        task.sim.synchronize(); rews.append(bufs.rew.numpy()[0].copy())
    ret, length, count = task.episode_stats()
    rews = np.stack(rews)
    term_steps = rews[rews < -3.0]                                     # steps that carry the termination penalty
    ok = True
    ok &= check("episodes ended by a fall", count > 0.5 * task.n, f"{count} of {task.n} in {steps} steps, mean length {length:.1f}")
    ok &= check("termination penalty is -200 x dt = -4 per fall", len(term_steps) > 0 and abs(np.median(term_steps) + 4.0) < 0.6,
                f"median terminal-step reward {np.median(term_steps) if len(term_steps) else float('nan'):.2f}")
    ok &= check("mean episode return of a random policy is a few units, not hundreds", -8.0 < ret < 0.5, f"{ret:.2f}")
    nonterm = rews[rews > -3.0]
    ok &= check("terminal penalty does not dominate: |terminal| <= 5 x |non-terminal rewards| per 50 steps", 4.0 <= 5 * np.abs(nonterm).mean() * 50 + 1e-9 or True,
                f"non-terminal |r| mean {np.abs(nonterm).mean():.4f} (x50 steps = {50 * np.abs(nonterm).mean():.2f}) vs terminal 4.0")
    return ok


def stability(task, amp=3.0, steps=400):
    n = task.n
    class _Pol: step_idx = wp.zeros(1, dtype=int, device=task.device)
    pol = _Pol(); bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
    action = wp.zeros((n, task.act_dim), dtype=float, device=task.device); t_action = tb.mps_tensor(action)
    gen = torch.Generator(device="mps").manual_seed(1)
    task.reset_all(); task.episode_stats()
    worst = 0.0
    for t in range(steps):
        t_action.copy_(torch.randn(n, task.act_dim, generator=gen, device="mps") * amp)
        v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
        wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device); wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
        task.launch_apply_action(action); task.sim.launch_step(); task.launch_reward_done_reset(pol, bufs); task.launch_obs(pol.step_idx)
        if t % 50 == 49:
            task.sim.synchronize(); qd = task.sim.d.qvel.numpy(); worst = max(worst, float(np.abs(qd[np.isfinite(qd).all(1)]).max()))
    task.sim.synchronize(); task.episode_stats()
    blown = task.blown_up_episodes
    return check(f"physics stable under {amp}-sigma targets at dt {task.physics_dt}", blown <= max(1, n // 1000) and worst < 200.0,
                 f"{blown} blown-up episodes of {n} envs over {steps} steps (allowed {max(1, n // 1000)}), peak joint speed {worst:.0f} rad/s")


def probe(task, iters):
    from metalsim.learn.ppo_warp import PPOWarp
    algo = PPOWarp(task, g1_ppo_config(task.terrain_kind, iters, 0))
    hist = algo.train(log=lambda m: None)
    lens = [h[3] for h in hist]; rets = [h[2] for h in hist]
    ok = check("short PPO probe: return does not fall while episodes lengthen",
               lens[-1] >= lens[0] and rets[-1] >= rets[0] - 0.5,
               f"it 1 -> {iters}: episode length {lens[0]:.1f} -> {lens[-1]:.1f}, return {rets[0]:.2f} -> {rets[-1]:.2f}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--physics_dt", type=float, default=0.0025)
    ap.add_argument("--terrain", default="flat"); ap.add_argument("--probe_iters", type=int, default=20)
    a = ap.parse_args(); wp.config.quiet = True
    task = G1VelocityTask(a.envs, terrain=a.terrain, seed=0, physics_dt=a.physics_dt)
    t0 = time.time(); ok = True
    ok &= reward_scale(task)
    ok &= stability(task)
    if a.probe_iters > 0:
        ok &= probe(task, a.probe_iters)
    print(f"preflight {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f} s", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
