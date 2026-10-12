"""Where the host waits of an MPM frame on Metal come from.

Wraps every ``wp_*`` entry point of the native library; whenever the runtime's host-wait counter
(``wp_metal_counters`` n_sync) or command-buffer counter moves during a call, records the entry point and the
Python call stack (innermost frames outside Warp's own modules first). Runs the timing-class configuration (chunk
graph + GPU exit, no per-solve log) for ``--frames`` frames after ``--warmup``, and prints the waits per frame
grouped by call site.

    PYTHONPATH=<warp>:<newton>:<main> .venv-newton152/bin/python metalsim/physics/mpm/probes/sync_trace.py \
        example:mpm_granular --frames 4
"""
import argparse, collections, ctypes, json, os, sys, traceback

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=4)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--depth", type=int, default=6)
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
_counters_fn = core.wp_metal_counters
ORD = wp.get_device(args.device).metal_ordinal


def counters():
    out = (ctypes.c_uint64 * 4)()
    _counters_fn(ORD, out, 4)
    return list(out)


ACTIVE = {"on": False}
EVENTS = collections.Counter()
EXAMPLES = {}


def _stack():
    frames = traceback.extract_stack()[:-2]
    keep = [f for f in frames if "/warp/_src/context.py" not in f.filename and "/warp/_src/types.py" not in f.filename]
    keep = keep[-args.depth:]
    return " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(keep))


def wrap(name, fn):
    def call(*a, **k):
        if not ACTIVE["on"]:
            return fn(*a, **k)
        c0 = counters()
        r = fn(*a, **k)
        c1 = counters()
        dw, dcb = c1[0] - c0[0], c1[1] - c0[1]
        if dw:
            key = (name, "wait", _stack())
            EVENTS[key] += dw
        elif dcb and name not in ("wp_metal_graph_launch",):
            key = (name, "commit", _stack())
            EVENTS[key] += dcb
        return r
    call.__name__ = name
    for attr in ("argtypes", "restype"):
        pass
    return call


for name in dir(core):
    if name.startswith("wp_") and name != "wp_metal_counters":
        try:
            setattr(core, name, wrap(name, getattr(core, name)))
        except Exception:
            pass

newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True)
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    wp.synchronize_device()
    c0 = counters()
    ACTIVE["on"] = True
    for _ in range(args.frames):
        sc.step()
        wp.synchronize_device()
    ACTIVE["on"] = False
    c1 = counters()
n = args.frames
print(f"per frame: host waits {(c1[0] - c0[0]) / n:g}, command buffers {(c1[1] - c0[1]) / n:g}, "
      f"dispatches {(c1[3] - c0[3]) / n:g}")
rows = sorted(EVENTS.items(), key=lambda kv: -kv[1])
for (name, kind, stack), cnt in rows:
    print(f"{cnt / n:6.2f} {kind:6s} {name}\n        {stack}")
if args.json:
    json.dump([{"per_frame": c / n, "entry": nm, "kind": kd, "stack": st} for (nm, kd, st), c in rows],
              open(args.json, "w"), indent=1)
