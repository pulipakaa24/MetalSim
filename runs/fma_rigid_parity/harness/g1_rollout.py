"""G1 velocity task rollout with fixed seeded random actions; saves qpos/qvel/rew/done per control step.
usage: g1_rollout.py OUT.npz [N] [terrain] [steps] [physics_dt]"""
import sys, time, hashlib
import numpy as np, warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask, PHYSICS_DT
from metalsim.learn.warp_policy import RolloutBuffers, bump

out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
terrain = sys.argv[3] if len(sys.argv) > 3 else "flat"; steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
dt = float(sys.argv[5]) if len(sys.argv) > 5 else PHYSICS_DT
task = G1VelocityTask(n, terrain=terrain, physics_dt=dt)
dev = task.device
class _Pol:
    step_idx = wp.zeros(1, dtype=int, device=dev)
pol = _Pol()
bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
bufs.rew = wp.zeros((1, n), dtype=float, device=dev); bufs.done = wp.zeros((1, n), dtype=float, device=dev)
acts = np.random.default_rng(1).uniform(-1, 1, (steps, n, task.act_dim)).astype(np.float32)
A = wp.array(acts, dtype=float, device=dev)
task.reset_all()
d = task.sim.d
Q, V, R, D = [], [], [], []
t0 = time.perf_counter()
with wp.ScopedDevice(dev):
    for i in range(steps):
        wp.launch(bump, dim=1, inputs=[pol.step_idx], device=dev)
        task.launch_apply_action(A[i])
        task.sim.launch_step()
        task.launch_reward_done_reset(pol, bufs)
        task.launch_obs(pol.step_idx)
        Q.append(d.qpos.numpy().copy()); V.append(d.qvel.numpy().copy()); R.append(bufs.rew.numpy()[0].copy()); D.append(bufs.done.numpy()[0].copy())
Q, V, R, D = map(np.stack, (Q, V, R, D))
np.savez_compressed(out, qpos=Q, qvel=V, rew=R, done=D)
h = hashlib.sha1(Q.tobytes() + V.tobytes()).hexdigest()[:16]
print(f"{out}: n={n} {terrain} steps={steps} dt={dt} wall {time.perf_counter()-t0:.1f}s sha1(qpos,qvel)={h} "
      f"nonfinite {int((~np.isfinite(Q)).sum())} dones {int(D.sum())} pelvis z mean {Q[-1][:, 2].mean():.5f}", flush=True)
print("WARP", wp.__file__)
