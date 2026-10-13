"""CUDA counterpart of kernel_profile.py, run on the L4 under Nsight Systems (per-kernel time inside CUDA graphs).

Warms up, then brackets ``--frames`` frames with cuProfilerStart/Stop so ``nsys --capture-range=cudaProfilerApi``
records only them; Newton's own conditional-graph loop (``--loop native``, as in stage_mpm_timing.sh) and native
(float64) accumulation unless told otherwise. Also prints wall ms/frame of the bracketed frames.

    nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o granular python l4_kernel_profile.py example:mpm_granular
    nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv -o granular granular.nsys-rep

``--chain`` instead measures CUDA's dependent-launch cost: a chain of N tiny kernels (64 / 1024 / 5120 threads)
launched eagerly and from a CUDA graph (the L4 side of dispatch_modes.swift).
"""
import argparse, ctypes, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("scene", nargs="?")
ap.add_argument("--frames", type=int, default=4)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--loop", default="native", choices=["native", "cuda"])
ap.add_argument("--accumulate", default="native", choices=["native", "float32"])
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--chain", action="store_true")
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()

if args.chain:
    @wp.kernel
    def bump(a: wp.array[float], n: int):
        i = wp.tid()
        if i < n:
            a[i] = a[i] * 1.0001 + 1.0

    out = {}
    with wp.ScopedDevice("cuda:0"):
        for threads in (64, 1024, 5120):
            a = wp.zeros(threads, dtype=float)
            n = 2000
            for _ in range(50):
                wp.launch(bump, dim=threads, inputs=[a, threads], block_dim=64)
            wp.synchronize_device()
            t0 = time.perf_counter()
            for _ in range(n):
                wp.launch(bump, dim=threads, inputs=[a, threads], block_dim=64)
            wp.synchronize_device()
            eager = (time.perf_counter() - t0) * 1e6 / n
            with wp.ScopedCapture() as cap:
                for _ in range(n):
                    wp.launch(bump, dim=threads, inputs=[a, threads], block_dim=64)
            wp.capture_launch(cap.graph)
            wp.synchronize_device()
            t0 = time.perf_counter()
            for _ in range(5):
                wp.capture_launch(cap.graph)
            wp.synchronize_device()
            graph = (time.perf_counter() - t0) * 1e6 / (5 * n)
            out[threads] = {"eager_us": eager, "graph_us": graph}
            print(f"threads {threads:5d}: eager {eager:6.2f} us/launch, graph {graph:6.2f} us/node")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    sys.exit(0)

here = os.path.dirname(os.path.abspath(__file__))
for p in (here, os.path.join(here, "..")):
    if os.path.exists(os.path.join(p, "newton_compat.py")):
        sys.path.insert(0, p)
import newton_compat, scenes  # noqa: E401

newton_compat.install(accumulate=args.accumulate, loop=args.loop, log=False)
cuda = ctypes.CDLL("libcuda.so.1")
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice("cuda:0"):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    wp.synchronize_device()
    cuda.cuProfilerStart()
    walls = []
    for _ in range(args.frames):
        wp.synchronize_device()
        t0 = time.perf_counter()
        sc.step()
        wp.synchronize_device()
        walls.append((time.perf_counter() - t0) * 1e3)
    cuda.cuProfilerStop()
res = {"scene": args.scene, "frames": args.frames, "wall_ms": walls, "wall_median_ms": float(np.median(walls))}
print(json.dumps(res))
if args.json:
    json.dump(res, open(args.json, "w"), indent=1)
