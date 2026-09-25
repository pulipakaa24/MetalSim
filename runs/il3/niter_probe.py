"""Largest per-substep Newton iteration count over 4096 worlds under isaaclab3 numerics (cap 100) on the flat_il3 task with
uniform random actions in [-1, 1] (the benchmark protocol, which falls and resets), 300 control steps: does a cap of 20 bind?"""
import numpy as np, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
for preset in ("isaaclab3_collide_every_substep", "isaaclab3"):
    task = G1VelocityTask(4096, terrain="flat", seed=0, physics_dt=0.0025, reward_cfg="flat_il3", solver_cfg=preset)
    task.reset_all(); hist = []
    rng = np.random.default_rng(0)
    for k in range(300):
        a = wp.array(rng.uniform(-1, 1, (4096, task.act_dim)).astype(np.float32), dtype=float, device="metal:0")
        task.launch_apply_action(a)
        for s in range(task.decimation):          # one substep at a time to read every substep's count
            import mujoco_warp as mjw
            with wp.ScopedDevice("metal:0"):
                mjw.step(task.sim.m, task.sim.d)
            hist.append(task.sim.d.solver_niter.numpy().copy())
    h = np.stack(hist)
    print(f"{preset}: per-substep iterations mean {h.mean():.2f}, p99.99 {np.percentile(h, 99.99):.0f}, max {h.max()}, "
          f"substep-worlds above 20: {(h > 20).sum()} of {h.size}", flush=True)
