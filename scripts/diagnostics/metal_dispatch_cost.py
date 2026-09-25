"""Cost of one dispatch inside a replayed Warp graph on Metal (ICB, barrier after every command): a graph of K
trivial kernels at several launch sizes, replayed and synchronized; per-dispatch time = graph time / K.

usage: python scripts/diagnostics/metal_dispatch_cost.py"""
import time
import warp as wp
wp.config.quiet = True


@wp.kernel
def touch(a: wp.array(dtype=float)):
    i = wp.tid()
    if a[0] > 1.0e30:          # never true: read one value, write nothing
        a[i] = 0.0


dev = "metal:0"
K = 1000
for n in (32, 4096, 4096 * 43, 4096 * 256, 4096 * 946):
    a = wp.zeros(max(n, 1), dtype=float, device=dev)
    with wp.ScopedDevice(dev), wp.ScopedCapture(device=dev) as cap:
        for _ in range(K):
            wp.launch(touch, dim=n, inputs=[a], device=dev)
    g = cap.graph
    wp.capture_launch(g); wp.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(5):
        wp.capture_launch(g)
    wp.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / 5 / K
    print(f"dim {n:>9,}: {dt*1e6:6.2f} us per dispatch (graph of {K})", flush=True)
