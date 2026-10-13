"""Do the latched capacity-overflow flags trip in real training? (2026-10-13, PR #7)

Runs a short training of a shipped task through its own training path and reports, at the end, BatchSim's overflow
flags (every flag, capacity and solver cap) and the training throughput from the log lines. A capacity overflow in
the PPOWarp loop raises (check_overflow at every log point); the run then reports where it stopped.

    .venv/bin/python scripts/diagnostics/overflow_training_check.py g1_flat --n 4096 --iters 50 --log runs/fixfail/train_g1_flat.log
    tasks: g1_flat, g1_rough (default boxes_local), g1_rough_hfield (train_g1, the shipped G1 path incl. the anomaly monitor), tron1 (PPOWarp over
    Tron1WFTask: the task's own loop does not call check_overflow, so this is what the guard would say there),
    lift (SO-101 lift, torch PPO of train_lift: flags read at the end; its loop has no guard),
    g1_rough_policy[_hfield] (a trained rough policy rolled out, no updates, with the envs spread evenly over all 10
    terrain levels from the start and the curriculum running: the contact-richest terrain a trained policy reaches;
    --iters rollouts of 24 control steps, check_overflow after each)
"""
import argparse
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # this checkout's metalsim
import warp as wp  # noqa: E402

wp.config.quiet = True
ap = argparse.ArgumentParser()
ap.add_argument("task", choices=["g1_flat", "g1_rough", "g1_rough_hfield", "g1_rough_policy", "g1_rough_policy_hfield", "tron1", "lift"])
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--log", required=True)
ap.add_argument("--freeze_levels", action="store_true", help="g1_rough_policy*: no curriculum moves (envs stay on their level)")
ap.add_argument("--legacy", action="store_true", help="g1_rough_policy*: the 2026-09-25 settings the checkpoints were trained "
                "with (pyramidal contact preset, origin base / feet-slide velocity, reward_cfg rough_isaac, exact scan)")
ap.add_argument("--ckpt", default=None, help="g1_rough_policy*: a trained rough checkpoint (net state dict under 'net')")
a = ap.parse_args()
import metalsim  # noqa: E402
print("metalsim from", metalsim.__file__, flush=True)
if os.path.exists(a.log):
    os.remove(a.log)
lines = []


def log(msg):
    print(msg, flush=True); lines.append(msg)
    with open(a.log, "a") as f:
        f.write(msg + "\n")


task = None
t0 = time.perf_counter(); err = None
try:
    if a.task.startswith("g1_rough_policy"):
        import numpy as np
        import torch
        from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config
        from metalsim.learn.ppo_warp import PPOWarp
        kw = dict(contact_cfg="recommended_pyramidal", base_velocity="origin", feet_slide_velocity="origin", reward_cfg="rough_isaac",
                  scan_surface="exact") if a.legacy else {}
        task = G1VelocityTask(a.n, terrain="rough", seed=42, terrain_collision="hfield" if a.task.endswith("hfield") else None, **kw)
        if a.freeze_levels:
            task.curriculum = 0                 # read when the reward/done kernel is launched (before any capture)
        algo = PPOWarp(task, g1_ppo_config("rough", a.iters, 42))
        algo.net.load_state_dict(torch.load(a.ckpt, map_location="mps")["net"])
        nr, nc = task.n_levels, task.n_cols
        lv = (np.arange(a.n) % nr).astype(np.int32); col = task.col.numpy()
        task.level.assign(lv)
        task.origins.assign(task.origin_table.numpy()[lv * nc + col].astype(np.float32))
        algo.prepare()
        log(f"G1 rough trained-policy rollout: N={a.n} collision {task.terrain_collision} njmax {task.sim.d.njmax} "
            f"nconmax {task.sim.d.naconmax // a.n} levels {nr} x {nc} start levels spread 0..{nr - 1}, frozen {a.freeze_levels}, "
            f"legacy settings {a.legacy}, contact_cfg {task.contact_cfg}; ckpt {a.ckpt}")
        t1 = time.perf_counter()
        for it in range(1, a.iters + 1):
            algo.rollout()
            if it % 5 == 0 or it == a.iters:
                ov = task.sim.check_overflow()          # raises on any capacity overflow since construction
                ret, length, count = task.episode_stats()
                lvn = task.level.numpy()
                log(f"it {it:4d} sps {it * 24 * a.n / (time.perf_counter() - t1):8,.0f} | ep_ret {ret:7.2f} ep_len {length:6.1f} "
                    f"(n={count}) | level mean {lvn.mean():.2f} per level {np.bincount(lvn, minlength=nr).tolist()} | flags {ov}")
    elif a.task.startswith("g1"):
        from metalsim.learn import g1_velocity as g1v
        orig = g1v.G1VelocityTask

        class _Keep(orig):                      # keep a handle on the task train_g1 builds
            def __init__(self, *args, **kw):
                global task
                super().__init__(*args, **kw); task = self
        g1v.G1VelocityTask = _Keep
        g1v.train_g1(a.n, a.task.split("_")[1], a.iters, log_path=a.log,
                     terrain_collision="hfield" if a.task.endswith("hfield") else None)
    elif a.task == "tron1":
        from metalsim.learn.ppo_warp import PPOWarp
        from metalsim.learn.tron1_wf import Tron1WFTask, ppo_config
        task = Tron1WFTask(a.n, seed=0); task.reset_all()
        PPOWarp(task, ppo_config(a.iters, 0)).train(log=log)
    else:
        from metalsim.learn.ppo import PPO, PPOConfig
        from metalsim.learn.so101_lift import LiftConfig, SO101LiftEnv
        task = SO101LiftEnv(LiftConfig(num_envs=a.n, decimate_faces=2000, seed=0))
        PPO(task, PPOConfig(total_steps=a.iters * 64 * a.n, rollout=64, epochs=8, minibatches=8, gamma=0.9, ent_coef=0.005,
                            lr=3e-4, seed=0, log_every=5)).train(log=log)
except RuntimeError as e:
    err = str(e); traceback.print_exc()
wall = time.perf_counter() - t0
sim = getattr(task, "sim", None)
flags = sim.overflow_flags() if sim is not None and hasattr(sim, "overflow_flags") else None
latched = hasattr(sim, "_overflow_latched")
lines = open(a.log).read().splitlines() if os.path.exists(a.log) else lines
sps = [float(m.group(1).replace(",", "")) for m in (re.search(r"sps\s+([\d,]+)", l) for l in lines) if m]
its = [int(m.group(1)) for m in (re.search(r"^it\s+(\d+)", l) for l in lines) if m]
res = (f"RESULT task={a.task} n={a.n} iters_logged={its[-1] if its else 0}/{a.iters} latch={'yes' if latched else 'no (main)'} "
      f"njmax={getattr(getattr(sim, 'd', None), 'njmax', None)} naconmax/world={getattr(getattr(sim, 'd', None), 'naconmax', 0) // max(a.n, 1)} "
      f"final_sps={sps[-1] if sps else None} wall={wall:.0f}s flags={flags} raised={err!r}")
with open(a.log, "a") as f:                  # the result also goes to the log (stdout may be a closed pipe)
    f.write(res + "\n")
try:
    print(res, flush=True)
except BrokenPipeError:
    pass
