"""Tiny GPU liveness probe: one 1-element Warp kernel on Metal, prints how long the round trip took."""
import time

import warp as wp

wp.config.quiet = True
wp.init()


@wp.kernel
def bump(a: wp.array[float]):
    a[0] = a[0] + 1.0


a = wp.zeros(1, dtype=float, device="metal:0")
wp.launch(bump, dim=1, inputs=[a], device="metal:0")  # compile/load
wp.synchronize_device("metal:0")
t0 = time.perf_counter()
wp.launch(bump, dim=1, inputs=[a], device="metal:0")
wp.synchronize_device("metal:0")
print(f"gpu round trip {1e3 * (time.perf_counter() - t0):.2f} ms, value {a.numpy()[0]}", flush=True)
