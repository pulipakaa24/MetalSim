"""Replay the real teleop commands on the sim follower (so101-sag): history- and pose-controlled sag comparison.

For every episode of a LeRobot so_follower teleop dataset, the sim arm (so101Sim's SO101LiftEnv model, contacts off:
free air) starts at the real follower's first state (deg -> rad, joint map sign +1 offset 0; gripper 0..100 ->
-0.17..1.75 rad), and the recorded actions (the leader's positions, which lerobot-record writes to the follower as
Goal_Position) are streamed as absolute goals at the dataset's fps (zero-order hold; frame k on physics step
round(k * 200 / fps)) through the variant's servo layer (command delay + firmware profile) and, for the BAM-friction
variants, the per-step friction hook. As in lerobot-record's loop (observe, then act), the sim state of frame k is read
at the start of tick k, before action k is sent. Output: <out>/<dataset>.npz with action (deg) and sim state (deg)
per frame, episode index; the same layout as the real parquet, so compare.py runs one analysis on both.
  <so101Sim venv>/python -I replay.py <so101Sim root> <variant> <out_dir> name=<parquet>@fps ...
"""
import json
import os
import sys
import time

root, variant, out_dir = sys.argv[1:4]
specs = sys.argv[4:]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.insert(0, HERE)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402

import sag_servo as SS  # noqa: E402
from metalsim.render.film.so101 import servo_layer  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
env = SO101LiftEnv(randomize=False)
m = env.model
d = mujoco.MjData(m)
applied, hook = SS.apply(m, d, variant)
m.geom_contype[:] = 0
m.geom_conaffinity[:] = 0
layer = servo_layer(applied, 1)
act = np.array(applied["actuators"])
qa, va = env._arm_qadr, env._arm_vadr
DT = m.opt.timestep


def to_sim(x):
    q = np.radians(np.asarray(x, float))
    q[..., 5] = -0.17 + np.asarray(x, float)[..., 5] / 100 * 1.92
    return q


def to_deg(q):
    x = np.degrees(np.asarray(q, float))
    x[..., 5] = (np.asarray(q, float)[..., 5] + 0.17) / 1.92 * 100
    return x


meta = dict(variant=variant, servo=applied, mujoco=mujoco.__version__, datasets={})
for spec in specs:
    name, rest = spec.split("=", 1)
    path, fps = rest.split("@")
    fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    SIM = np.full_like(S, np.nan, dtype=float)
    t0 = time.time()
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        a = to_sim(A[idx])
        q0 = to_sim(S[idx[0]])
        mujoco.mj_resetData(m, d)
        d.qpos[qa] = q0
        mujoco.mj_forward(m, d)
        layer.reset(q0[None])
        if hook is not None:
            hook.reset()
        cmd = q0.copy()
        nsteps = int(round(len(idx) * (1 / fps) / DT))
        tick = {int(round(k / fps / DT)): k for k in range(len(idx))}
        for n in range(nsteps):
            k = tick.get(n)
            if k is not None:
                SIM[idx[k]] = to_deg(d.qpos[qa])
                cmd = a[k]
            d.ctrl[act] = layer.tick(cmd[None])[0]
            if hook is not None:
                hook.update()
            mujoco.mj_step(m, d)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), action=A, state=SIM, episode_index=ep, fps=fps)
    meta["datasets"][name] = dict(path=path, fps=fps, frames=int(len(A)), episodes=int(len(np.unique(ep))),
                                  wall_s=round(time.time() - t0, 1))
    print(variant, name, meta["datasets"][name], flush=True)
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
