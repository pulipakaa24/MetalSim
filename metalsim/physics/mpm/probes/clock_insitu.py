"""GPU clock state during an MPM timing run on Metal, sampled with a dependent-FMA chain kernel.

The M4 Max lowers its GPU clock when lightly loaded (latency_clock.swift: ~10 ns per dependent FMA cold vs ~2.1 ns
after ~0.5 s of back-to-back work). This probe times a fixed dependent chain (1024 threads, ``--trips`` x 16 FMAs,
one command buffer, host-timed between two synchronizes) cold, hot (after ``--hot-s`` s of back-to-back chain
launches) and right after every frame of the scene's timing-class configuration (chunk graph + GPU exit), and
reports each in-frame sample as the hot/cold ratio.

    PYTHONPATH=. .venv-newton152/bin/python \
        metalsim/physics/mpm/probes/clock_insitu.py example:mpm_granular --frames 12
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=12)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--trips", type=int, default=1024)
ap.add_argument("--hot-s", type=float, default=1.5)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes


@wp.kernel
def fma_chain(out: wp.array[float], trips: int):
    i = wp.tid()
    x = float(i) * 1.0e-6
    for _k in range(trips):
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
        x = x * 0.999 + 1.0e-3
    out[i] = x


newton_compat.install(accumulate="float32", loop="cuda", log=True, chunk_graph=True, gpu_exit=True)
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
res = {"scene": args.scene, "trips": args.trips}
with wp.ScopedDevice(args.device):
    out = wp.zeros(1024, dtype=float)

    def sample():
        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.launch(fma_chain, dim=1024, inputs=[out, args.trips], block_dim=64)
        wp.synchronize_device()
        return (time.perf_counter() - t0) * 1e6

    # host-side floor: the same launch with zero trips
    def floor():
        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.launch(fma_chain, dim=1024, inputs=[out, 0], block_dim=64)
        wp.synchronize_device()
        return (time.perf_counter() - t0) * 1e6

    sample()
    time.sleep(2.0)
    res["cold_us"] = float(np.median([(time.sleep(1.0), sample())[1] for _ in range(5)]))
    t_end = time.perf_counter() + args.hot_s
    while time.perf_counter() < t_end:
        for _ in range(50):
            wp.launch(fma_chain, dim=1024, inputs=[out, args.trips], block_dim=64)
        wp.synchronize_device()
    res["hot_us"] = float(np.median([sample() for _ in range(9)]))
    res["floor_us"] = float(np.median([floor() for _ in range(9)]))
    # decay: how fast the clock falls once the GPU idles
    decay = {}
    for idle_ms in (1, 2, 5, 10, 20, 50, 100, 200):
        t_end = time.perf_counter() + 0.5
        while time.perf_counter() < t_end:
            for _ in range(50):
                wp.launch(fma_chain, dim=1024, inputs=[out, args.trips], block_dim=64)
            wp.synchronize_device()
        time.sleep(idle_ms / 1e3)
        decay[idle_ms] = sample()
    res["after_idle_us"] = decay

    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    walls, samples = [], []
    for _ in range(args.frames):
        wp.synchronize_device()
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e3)
        samples.append(sample())
    res["frame_ms"] = walls
    res["after_frame_us"] = samples
    res["iterations"] = newton_compat.drain_log()["iterations"].tolist()

c, h, f = res["cold_us"], res["hot_us"], res["floor_us"]
print(f"chain: cold {c:.1f} us, hot {h:.1f} us, host floor {f:.1f} us -> clock ratio hot/cold {(c - f) / (h - f):.2f}")
print("after idle (ms -> us):", {k: round(v, 1) for k, v in res["after_idle_us"].items()})
print("frames (ms):", [round(w, 1) for w in walls])
print("chain right after each frame (us):", [round(s, 1) for s in samples],
      "-> relative clock", [round((h - f) / max(s - f, 1e-9), 2) for s in samples])
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=1)
