"""Where a coupled MPM example's frame goes: wraps every solver ``step`` and collision call reachable from the example
(and Newton's MPM section timers) with synchronized wall-clock timers.

    PYTHONPATH=. .venv-newton152/bin/python \
        metalsim/physics/mpm/probes/profile_example_parts.py example:mpm_twoway_coupling --frames 10
"""
import argparse, collections, json, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--chunk-graph", action="store_true")
ap.add_argument("--json")
args = ap.parse_args()

import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes

dev = wp.get_device(args.device)
newton_compat.install(accumulate="native" if dev.is_cuda else "float32", loop="cuda", chunk_graph=args.chunk_graph,
                      log=False)
T = collections.defaultdict(float)
N = collections.Counter()
active = [False]


def timed(name, fn):
    def wrapper(*a, **kw):
        if not active[0]:
            return fn(*a, **kw)
        wp.synchronize_device()
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        wp.synchronize_device()
        T[name] += time.perf_counter() - t0
        N[name] += 1
        return r

    return wrapper


with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene)
    ex = sc.example
    for attr, obj in list(vars(ex).items()):
        for meth in ("step", "collide", "update_contacts"):
            if hasattr(obj, meth) and callable(getattr(obj, meth)) and not isinstance(obj, type):
                try:
                    setattr(obj, meth, timed(f"{attr}.{meth} ({type(obj).__name__})", getattr(obj, meth)))
                except (AttributeError, TypeError):
                    pass
    for meth in [m for m in dir(ex) if not m.startswith("__") and callable(getattr(ex, m))]:
        if meth in ("step", "render", "capture", "test_final", "create_parser"):
            continue
        try:
            setattr(ex, meth, timed(f"example.{meth}", getattr(ex, meth)))
        except (AttributeError, TypeError):
            pass
    for _ in range(args.warmup):
        sc.step()
    active[0] = True
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(args.frames):
        sc.step()
    wp.synchronize_device()
    total = (time.perf_counter() - t0) / args.frames
out = {"scene": args.scene, "device": args.device, "frame_ms_synced": total * 1e3,
       "parts_ms_per_frame": {k: T[k] * 1e3 / args.frames for k in sorted(T, key=T.get, reverse=True)},
       "calls_per_frame": {k: N[k] / args.frames for k in N}}
print(json.dumps(out, indent=1))
if args.json:
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
