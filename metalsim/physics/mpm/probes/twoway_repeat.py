"""example_mpm_twoway_coupling on Metal: run the example twice from scratch (whole-step capture, as the example does)
and compare rigid bodies and sand bitwise; time the frames and the body-force sum alone.

    PYTHONPATH=<warp>:<newton>:. python metalsim/physics/mpm/probes/twoway_repeat.py --frames 120 --json out.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=120)
ap.add_argument("--capture", type=int, default=1)
ap.add_argument("--json")
args = ap.parse_args()

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes  # noqa: E402

newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True)


def run():
    sc = scenes.make_scene("example:mpm_twoway_coupling", capture=bool(args.capture))
    ex = sc.example
    ts = []
    for _ in range(args.frames):
        wp.synchronize_device()
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device()
        ts.append((time.perf_counter() - t0) * 1e3)
    out = {"body_q": ex.state_0.body_q.numpy().copy(), "particle_q": sc.state.particle_q.numpy().copy()}
    # the body-force sum alone (the substep's compute_body_forces)
    for _ in range(3):
        ex.compute_body_forces()
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(20):
        ex.compute_body_forces()
    wp.synchronize_device()
    force_ms = (time.perf_counter() - t0) * 1e3 / 20
    return out, ts, force_ms, getattr(ex, "ordered_body_forces", None)


a, ta, fa, ordered = run()
b, tb, fb, _ = run()
res = {
    "frames": args.frames,
    "ordered_body_forces": ordered,
    "bitwise_equal_body_q": bool(np.array_equal(a["body_q"].view(np.uint8), b["body_q"].view(np.uint8))),
    "bitwise_equal_particle_q": bool(np.array_equal(a["particle_q"].view(np.uint8), b["particle_q"].view(np.uint8))),
    "frame_ms_median": [float(np.median(ta[5:])), float(np.median(tb[5:]))],
    "body_force_ms": [fa, fb],
    "max_body_q_diff": float(np.abs(a["body_q"] - b["body_q"]).max()),
}
print(json.dumps(res, indent=1))
if args.json:
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=1)
