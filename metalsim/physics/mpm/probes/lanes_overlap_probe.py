"""Prototype: can a robot step and an MPM sand step run concurrently on the M4 Max with the Warp fork's ordering lanes
(wp_metal_select_lane, research_notes/gpu_latency/LANES.md)? Two independent example_mpm_anymal instances:
  A runs only its sand step (eager MPM step: grid build, then the GPU-gated solver loop),
  B runs only its robot side (apply_control + the example's captured robot graph, 4 MuJoCo substeps).
They share no arrays, so running B's work on lane 1 while A's runs on lane 0 must not change a single bit of either.

Modes (fresh A, B per mode, same warm-up, --frames frames each, one device synchronize per frame):
  seq   B robot, then A sand, all on lane 0 (what one queue chain does today)
  conc  A sand on lane 0; when its solver loop starts (after the grid build's host reads), B's robot work is
        submitted on lane 1, then A's solve continues on lane 0; synchronize waits for both lanes
Reports ms per frame (median) for each mode and for robot-only / sand-only frames, the overlap gain, and whether the
per-frame state hashes of A (particle_q) and B (body_q, joint_q) are bitwise equal between seq and conc.

    PYTHONPATH=<warp with lanes>:<newton>:<metalsim> python lanes_overlap_probe.py --frames 40 --json out.json
"""
import argparse
import ctypes
import hashlib
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="metal:0")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from warp._src.context import runtime

from metalsim.physics.mpm import newton_compat, scenes

core = runtime.core
core.wp_metal_select_lane.argtypes = [ctypes.c_int, ctypes.c_int]
core.wp_metal_select_lane.restype = ctypes.c_int
ORD = wp.get_device(args.device).metal_ordinal


def select_lane(k):
    if core.wp_metal_select_lane(ORD, k) != 0:
        raise RuntimeError("wp_metal_select_lane failed: " + runtime.get_error_string())


newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True)
from newton._src.solvers.implicit_mpm import solve_rheology as sr  # noqa: E402

base_loop = sr._run_solver_loop
pending = {"robot": None}


def hooked_loop(*a, **kw):
    """At the start of the sand step's solver loop (its host reads are done), submit the robot work on lane 1."""
    work = pending["robot"]
    if work is not None:
        pending["robot"] = None
        select_lane(1)
        work()
        select_lane(0)
    return base_loop(*a, **kw)


sr._run_solver_loop = hooked_loop


def h(*arrs):
    m = hashlib.sha1()
    for a in arrs:
        if a is not None:
            m.update(np.ascontiguousarray(a.numpy()).tobytes())
    return m.hexdigest()


def make_pair():
    a = scenes.make_scene("example:mpm_anymal", capture=True).example
    b = scenes.make_scene("example:mpm_anymal", capture=True).example
    for ex in (a, b):
        for _ in range(args.warmup):
            ex.step()
    wp.synchronize_device()
    return a, b


def robot_work(b):
    def work():
        b.apply_control()
        wp.capture_launch(b.graph)
    return work


def run(mode):
    a, b = make_pair()
    times, ha, hb = [], [], []
    for _ in range(args.frames):
        wp.synchronize_device()
        t0 = time.perf_counter()
        if mode == "seq":
            robot_work(b)()
            a.simulate_sand()
        elif mode == "conc":
            pending["robot"] = robot_work(b)
            a.simulate_sand()
            if pending["robot"] is not None:   # the loop hook did not run (no solve this frame): submit on lane 1 now
                w = pending["robot"]
                pending["robot"] = None
                select_lane(1)
                w()
                select_lane(0)
        elif mode == "robot_only":
            robot_work(b)()
        elif mode == "sand_only":
            a.simulate_sand()
        wp.synchronize_device()
        times.append((time.perf_counter() - t0) * 1e3)
        ha.append(h(a.state_0.particle_q))
        hb.append(h(b.state_0.body_q, b.state_0.joint_q))
    del a, b
    return {"ms_median": float(np.median(times)), "ms_p10": float(np.percentile(times, 10)),
            "ms_p90": float(np.percentile(times, 90)), "hash_a": ha, "hash_b": hb}


res = {"frames": args.frames, "modes": {}}
with wp.ScopedDevice(args.device):
    for mode in ("robot_only", "sand_only", "seq", "conc"):
        r = run(mode)
        res["modes"][mode] = r
        print(f"{mode:10s} {r['ms_median']:7.2f} ms/frame (p10 {r['ms_p10']:.2f}, p90 {r['ms_p90']:.2f})", flush=True)
    select_lane(0)
s, c = res["modes"]["seq"], res["modes"]["conc"]
res["bitwise_A"] = s["hash_a"] == c["hash_a"]
res["bitwise_B"] = s["hash_b"] == c["hash_b"]
res["gain_ms"] = s["ms_median"] - c["ms_median"]
res["ideal_ms"] = max(res["modes"]["robot_only"]["ms_median"], res["modes"]["sand_only"]["ms_median"])
print(f"seq {s['ms_median']:.2f} -> conc {c['ms_median']:.2f} ms/frame (gain {res['gain_ms']:.2f}; ideal "
      f"{res['ideal_ms']:.2f}); bitwise A {res['bitwise_A']}, B {res['bitwise_B']}")
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump({k: v for k, v in res.items()}, open(args.json, "w"), indent=1)
print("OK" if res["bitwise_A"] and res["bitwise_B"] else "FAIL")
