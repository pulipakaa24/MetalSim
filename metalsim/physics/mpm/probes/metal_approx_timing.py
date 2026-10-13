"""Cost per operation of Metal float32 division / sqrt forms (throughput: 2^20 threads, dependent chains of 256 ops).

    python metal_approx_timing.py [--json out.json]     (run as a timing-class GPU job)

Reports GPU time per op per thread in picoseconds-per-thread-op and in M4 Max ALU cycles at the measured clock is not
attempted; compare forms against each other and against the baseline loop (a dependent fma), same launch shape.
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--json")
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()

import warp as wp

wp.config.quiet = True
wp.init()

# v -> op(v), kept in [1, 4) so every op sees normal operands
FORMS = {
    "baseline fma": "return metal::fma(v, 0.75f, 0.9f);",
    "div precise": "return metal::precise::divide(3.1f, v + 0.6f);",
    "div fast::divide": "return metal::fast::divide(3.1f, v + 0.6f);",
    "div cuda emulation": "return wp_cuda_div(3.1f, v + 0.6f);",
    "div cuda emulation v2": "return wp_cuda_div2(3.1f, v + 0.6f);",
    "sqrt precise": "return metal::precise::sqrt(v + 3.0f);",
    "sqrt fast::sqrt": "return metal::fast::sqrt(v + 3.0f);",
    "sqrt cuda emulation": "return wp_cuda_sqrt(v + 3.0f);",
    "sqrt cuda emulation v2": "return wp_cuda_sqrt2(v + 3.0f);",
}
N = 1 << 20
CHAIN = 256


def make(snippet):
    @wp.func_native(snippet)
    def op(v: float) -> float: ...

    @wp.kernel(module="unique", enable_backward=False)
    def k(x: wp.array[float], out: wp.array[float]):
        i = wp.tid()
        v = x[i]
        for _j in range(256):
            v = op(v)
        out[i] = v

    return k


res = {}
with wp.ScopedDevice("metal:0"):
    x = wp.array(np.random.default_rng(0).uniform(1, 2, N).astype(np.float32))
    o = wp.zeros(N, dtype=float)
    for name, snip in FORMS.items():
        k = make(snip)
        wp.launch(k, N, inputs=[x, o])
        wp.synchronize_device()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            wp.launch(k, N, inputs=[x, o])
            wp.synchronize_device()
            ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        res[name] = {"ms": t * 1e3, "ps_per_thread_op": t / (N * CHAIN) * 1e12}
        print(f"{name:22s} {t * 1e3:8.3f} ms  {res[name]['ps_per_thread_op']:7.3f} ps/thread-op", flush=True)
base = res["baseline fma"]["ps_per_thread_op"]
for name, r in res.items():
    r["minus_baseline"] = r["ps_per_thread_op"] - base
if args.json:
    json.dump(res, open(args.json, "w"), indent=1)
