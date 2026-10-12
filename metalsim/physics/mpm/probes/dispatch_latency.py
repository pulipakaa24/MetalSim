"""Cost of a chain of tiny dependent Warp launches on Metal: eager vs recorded graph.

    .venv-newton152/bin/python metalsim/physics/mpm/probes/dispatch_latency.py
"""
import time

import warp as wp

wp.config.quiet = True
wp.init()


@wp.kernel
def bump(a: wp.array[float], n: int):
    i = wp.tid()
    if i < n:
        a[i] = a[i] * 1.0001 + 1.0


def chain(a, n_launch, threads):
    for _ in range(n_launch):
        wp.launch(bump, dim=threads, inputs=[a, a.shape[0]])


dev = "metal:0"
with wp.ScopedDevice(dev):
    for threads in (64, 1024, 65536):
        a = wp.zeros(threads, dtype=float)
        n = 400
        chain(a, 10, threads)
        wp.synchronize_device()
        t0 = time.perf_counter()
        chain(a, n, threads)
        wp.synchronize_device()
        eager = (time.perf_counter() - t0) * 1e6 / n
        cmd = wp.launch(bump, dim=threads, inputs=[a, a.shape[0]], record_cmd=True)
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(n):
            cmd.launch()
        wp.synchronize_device()
        recorded = (time.perf_counter() - t0) * 1e6 / n
        with wp.ScopedCapture(force_module_load=False) as cap:
            chain(a, n, threads)
        wp.capture_launch(cap.graph)
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(5):
            wp.capture_launch(cap.graph)
        wp.synchronize_device()
        graph = (time.perf_counter() - t0) * 1e6 / (5 * n)
        print(f"threads {threads:6d}: eager {eager:6.1f} us/launch, record_cmd {recorded:6.1f}, graph {graph:6.1f}")
