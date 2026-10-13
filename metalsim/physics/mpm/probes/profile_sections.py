"""Per-section wall time of Newton's SolverImplicitMPM step (its own ``enable_timers`` sections, synchronized).

    PYTHONPATH=. .venv-newton152/bin/python metalsim/physics/mpm/probes/profile_sections.py example:mpm_granular \
        --device metal:0 --frames 6 --warmup 2
"""
import argparse, collections, contextlib, io, json, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=6)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--accumulate", default=None)
ap.add_argument("--json")
ap.add_argument("--chunk-graph", action="store_true")
ap.add_argument("--gpu-exit", action="store_true")
args = ap.parse_args()

import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes

dev = wp.get_device(args.device)
newton_compat.install(accumulate=args.accumulate or ("native" if dev.is_cuda else "float32"), loop="cuda", chunk_graph=args.chunk_graph, gpu_exit=args.gpu_exit)
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    sc.solver.enable_timers = True
    pat = re.compile(r"^(.*?) took ([0-9.]+) ms")
    totals = collections.defaultdict(float)
    walls = []
    for _ in range(args.frames):
        buf = io.StringIO()
        wp.synchronize_device()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            sc.step()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e3)
        for line in buf.getvalue().splitlines():
            m = pat.match(line.strip())
            if m:
                totals[m.group(1).strip()] += float(m.group(2))
    log = newton_compat.drain_log()
n = args.frames
print(f"{args.scene} on {args.device}: {sc.model.particle_count} particles, frame {sum(walls) / n:.1f} ms "
      f"(timers on), iterations mean {log['iterations'].mean():.1f}")
for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"  {v / n:9.2f} ms  {k}")
if args.json:
    with open(args.json, "w") as fh:
        json.dump({"scene": args.scene, "device": args.device, "frame_ms": sum(walls) / n,
                   "sections_ms": {k: v / n for k, v in totals.items()},
                   "iterations_mean": float(log["iterations"].mean())}, fh, indent=1)
