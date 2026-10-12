"""ms/frame of one MPM scene in its timing-class configuration on Metal, with the runtime's per-frame counters.

Same loop as ``record_mpm.py`` (CUDA stopping rule, chunk graph + GPU exit, no per-solve log, a device
synchronize around every frame) without recording; after ``--warmup`` frames, ``--frames`` timed frames; reports
the median and the Metal runtime counters per frame (host waits, command buffers, host ops, dispatches encoded).

    PYTHONPATH=<warp>:<newton>:<main> .venv-newton152/bin/python metalsim/physics/mpm/probes/speed_bench.py \
        example:mpm_granular --frames 60 --json out.json
"""
import argparse, ctypes, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--log", action="store_true", help="per-solve residual log on (recording configuration)")
ap.add_argument("--metal-fast-math", choices=["off", "cuda", "relaxed"],
                help="Metal form of fast_math modules (newton_compat.install(metal_fast_math=...)); default: install's")
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from warp._src.context import runtime
from metalsim.physics.mpm import newton_compat, scenes

core = runtime.core
core.wp_metal_counters.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
core.wp_metal_counters.restype = ctypes.c_int


def counters():
    out = (ctypes.c_uint64 * 4)()
    core.wp_metal_counters(wp.get_device(args.device).metal_ordinal, out, 4)
    return np.array(list(out), dtype=np.int64)


newton_compat.install(accumulate="float32", loop="cuda", log=args.log, chunk_graph=True, gpu_exit=True,
                      **({"metal_fast_math": args.metal_fast_math} if args.metal_fast_math else {}))
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    n = min(args.frames, max(1, sc.frames - args.warmup))
    for _ in range(args.warmup):
        sc.step()
    wp.synchronize_device()
    c0 = counters()
    walls = []
    for _ in range(n):
        wp.synchronize_device()
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e3)
    c1 = counters()
res = {"scene": args.scene, "set": kw, "frames": n, "warmup": args.warmup, "ms_median": float(np.median(walls)),
       "ms_mean": float(np.mean(walls)), "ms_p10": float(np.percentile(walls, 10)),
       "ms_p90": float(np.percentile(walls, 90)), "wall_ms": walls,
       "per_frame_counters": dict(zip(["host_waits", "command_buffers", "host_ops", "dispatches_encoded"],
                                      ((c1 - c0) / n).tolist())),
       "warp": wp.__file__, "compat": newton_compat.settings()}
print(json.dumps({k: v for k, v in res.items() if k != "wall_ms"}))
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=1)
