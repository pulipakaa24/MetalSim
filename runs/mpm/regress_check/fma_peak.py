"""Achieved float32 FMA throughput of the whole GPU (TFLOP/s), a machine-state probe: if part of the GPU is held by
other work (e.g. a hung command buffer) or clocks are low, this drops below the M4 Max's ~16 TFLOP/s peak."""
import time
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
ITERS = 4096


@wp.kernel
def fma_chain(out: wp.array(dtype=float), seed: float):
    tid = wp.tid()
    a = float(tid) * 1.0e-9 + seed
    b = a + 0.1
    c = a + 0.2
    d = a + 0.3
    e = a + 0.4
    f = a + 0.5
    g = a + 0.6
    h = a + 0.7
    for _ in range(ITERS):
        a = a * 0.9999 + 0.0001
        b = b * 0.9999 + 0.0001
        c = c * 0.9999 + 0.0001
        d = d * 0.9999 + 0.0001
        e = e * 0.9999 + 0.0001
        f = f * 0.9999 + 0.0001
        g = g * 0.9999 + 0.0001
        h = h * 0.9999 + 0.0001
    out[tid] = a + b + c + d + e + f + g + h


n = 40 * 1024 * 16
out = wp.zeros(n, dtype=float, device="metal:0")
for _ in range(3):
    wp.launch(fma_chain, dim=n, inputs=[out, 0.5], device="metal:0")
wp.synchronize_device("metal:0")
res = []
for r in range(10):
    t0 = time.perf_counter()
    for _ in range(5):
        wp.launch(fma_chain, dim=n, inputs=[out, 0.5], device="metal:0")
    wp.synchronize_device("metal:0")
    dt = time.perf_counter() - t0
    res.append(5 * n * ITERS * 8 * 2 / dt / 1e12)
print("FMA_TFLOPS median %.2f min %.2f max %.2f" % (np.median(res), min(res), max(res)))
