"""Cost of committing every graph replay as its own command buffer (Warp's Metal default) on back-to-back replays.

A graph of D dependent small dispatches (1024 threads) replayed R times, then one synchronize; wall time per replay
and the command buffers committed. Run under different WP_METAL_ICB_BATCH settings to compare.

    PYTHONPATH=<warp> .venv-newton152/bin/python metalsim/physics/mpm/probes/replay_cb.py --replays 50 --dispatches 55
"""
import argparse, ctypes, json, time

ap = argparse.ArgumentParser()
ap.add_argument("--replays", type=int, default=50)
ap.add_argument("--dispatches", type=int, default=55)
ap.add_argument("--threads", type=int, default=1024)
ap.add_argument("--seconds", type=float, default=2.0)
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from warp._src.context import runtime

core = runtime.core
core.wp_metal_counters.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]


def counters():
    out = (ctypes.c_uint64 * 4)()
    core.wp_metal_counters(0, out, 4)
    return list(out)


@wp.kernel
def step(x: wp.array[float]):
    i = wp.tid()
    x[i] = x[i] * 0.5 + 1.0


with wp.ScopedDevice("metal:0"):
    x = wp.zeros(args.threads, dtype=float)
    wp.launch(step, args.threads, inputs=[x])
    wp.synchronize_device()
    with wp.ScopedCapture(force_module_load=False) as cap:
        for _ in range(args.dispatches):
            wp.launch(step, args.threads, inputs=[x], block_dim=64)
    g = cap.graph
    walls, host = [], []
    t_end = time.perf_counter() + args.seconds
    c0 = counters()
    n = 0
    while time.perf_counter() < t_end:
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(args.replays):
            wp.capture_launch(g)
        t1 = time.perf_counter()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e6)
        host.append((t1 - t0) * 1e6)
        n += 1
    c1 = counters()
res = {"replays": args.replays, "dispatches": args.dispatches,
       "us_per_replay": float(np.median(walls[n // 2:])) / args.replays,
       "host_submit_us_per_replay": float(np.median(host[n // 2:])) / args.replays,
       "command_buffers_per_replay": (c1[1] - c0[1]) / (n * args.replays)}
print(json.dumps(res))
