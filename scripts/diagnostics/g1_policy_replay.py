"""Replay a trained G1 checkpoint (rollouts only, no updates) on the task configuration saved in the checkpoint and
report episode return / length, the fraction of ended episodes that were falls, and the mean terrain level.

    .venv/bin/python scripts/diagnostics/g1_policy_replay.py runs/ellip_default/g1_rough_ellip_default_s0.pt --n 1024 --iters 50
    # checkpoints older than the saved settings: pass them, e.g. the 2026-09-25 rough runs
    ... g1_rough_ppowarp_fixed_it1500.pt --override reward_cfg=rough_isaac scan_ordering=ij terrain_collision=hfield \
        scan_surface=grid contact_cfg=None base_velocity=origin feet_slide_velocity=origin

Prints whether the Warp rollout policy holds the checkpoint's weights after loading and at the end (2026-10-13: before
PPOWarp published loaded weights to the Warp queue, a replay ran the random initial weights; see
tests/test_ppo_warp_rollout.py::test_loaded_weights_reach_the_warp_policy). --random replays the untrained net;
--mean sets the action noise to ~0. Episodes of a walking policy end by time-out (1000 control steps), so with
--iters 50 (1,200 steps) most episodes end once, around iteration 42-45."""
import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # this checkout's metalsim
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--random", action="store_true")
ap.add_argument("--mean", action="store_true")
ap.add_argument("--seed", type=int, default=None, help="task seed (default: the task's own, 42 on rough terrain)")
ap.add_argument("--override", nargs="*", default=[], help="task keyword=value pairs (None for None)")
a = ap.parse_args()
import metalsim  # noqa: E402
from metalsim.learn.g1_velocity import G1VelocityTask, g1_ppo_config  # noqa: E402
from metalsim.learn.ppo_warp import PPOWarp  # noqa: E402

print("metalsim from", metalsim.__file__)
ck = torch.load(a.ckpt, map_location="cpu")
params = inspect.signature(G1VelocityTask.__init__).parameters
kw = {k: ck[k] for k in ("reward_cfg", "scan_ordering", "terrain_collision", "scan_surface", "solver_cfg", "il3_events",
                         "contact_cfg", "base_velocity", "feet_slide_velocity") if k in ck and k in params}
for o in a.override:
    k, v = o.split("=", 1)
    kw[k] = None if v == "None" else v
print("task kwargs", kw, flush=True)
task = G1VelocityTask(a.n, terrain=ck["terrain"], seed=a.seed, **kw)
algo = PPOWarp(task, g1_ppo_config(ck["terrain"], a.iters, 0))


def weights_match():
    wp.synchronize_device("metal:0")
    return bool(np.allclose(algo.pol.actor_layers[0][0].numpy(), ck["net"]["actor.0.weight"].numpy()))


if not a.random:
    sd = {k: v.to("mps") for k, v in ck["net"].items()}
    if a.mean:
        sd["log_std"] = torch.full_like(sd["log_std"], -20.0)
    algo.net.load_state_dict(sd)
    print("warp first-layer weight matches checkpoint:", weights_match(), flush=True)
algo.prepare()
for it in range(1, a.iters + 1):
    algo.rollout()
    if it % 5 == 0 or it == a.iters:
        ret, length, count = task.episode_stats()
        lv = task.level.numpy()
        print(f"it {it:3d} ep_ret {ret:7.2f} ep_len {length:6.1f} (n={count}) falls_frac {task.last_termination_fraction:.3f} "
              f"level mean {lv.mean():.2f}", flush=True)
if not a.random:
    print("end: warp first-layer weight matches checkpoint:", weights_match())
