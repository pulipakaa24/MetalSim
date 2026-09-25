"""Cost of the register tile Cholesky (factor + solve, one SIMD group per matrix) on Metal for 4096 matrices of
size n, as MuJoCo Warp's solver runs it: sizes the Newton Hessian of the G1 (43) and the blocks a partitioned
factorization would leave (core 19, arms 24 / 12).

usage: python scripts/diagnostics/metal_cholesky_cost.py"""
import time
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.metal_register_cholesky_max = 48


def make(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float)):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        rhs = wp.tile_load(b[w], shape=(n,))
        wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))
    return k


dev = "metal:0"; W = 4096; K = 50
rng = np.random.default_rng(0)
for n in (12, 19, 24, 32, 43):
    m = rng.standard_normal((W, n, n)).astype(np.float32)
    A = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
    a = wp.array(A, dtype=float, device=dev); b = wp.array(rng.standard_normal((W, n)).astype(np.float32), dtype=float, device=dev)
    x = wp.zeros((W, n), dtype=float, device=dev)
    kern = make(n)
    with wp.ScopedDevice(dev), wp.ScopedCapture(device=dev) as cap:
        for _ in range(K):
            wp.launch_tiled(kern, dim=W, inputs=[a, b, x], block_dim=32, device=dev)
    wp.capture_launch(cap.graph); wp.synchronize_device(dev)
    t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / K
    err = np.abs(np.einsum("wij,wj->wi", A[:4].astype(np.float64), x.numpy()[:4]) - b.numpy()[:4]).max()
    print(f"n={n:2d}: {dt*1e3:.3f} ms per launch of {W} factor+solve (residual {err:.1e})", flush=True)
