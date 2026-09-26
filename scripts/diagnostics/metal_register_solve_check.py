"""Correctness of the Warp fork's Metal register triangular solve (tile_cholesky_solve, vector RHS) against
the cooperative scalar path and float64 numpy, for n in {8, 31, 32, 43, 48}, 256 matrices each, block_dim 32
and 16. Run with PYTHONPATH pointing at the Warp worktree that carries the change.

usage: python scripts/diagnostics/metal_register_solve_check.py"""
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.metal_register_cholesky_max = 48


def make(n, fill):
    if fill == "upper":
        @wp.kernel(enable_backward=False, module="unique")
        def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float)):
            w = wp.tid()
            t = wp.tile_load(a[w], shape=(n, n))
            wp.tile_cholesky_inplace(t, fill_mode="upper")
            rhs = wp.tile_load(b[w], shape=(n,))
            wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))
    else:
        @wp.kernel(enable_backward=False, module="unique")
        def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float)):
            w = wp.tid()
            t = wp.tile_load(a[w], shape=(n, n))
            wp.tile_cholesky_inplace(t, fill_mode="lower")
            rhs = wp.tile_load(b[w], shape=(n,))
            wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="lower"))
    return k


dev = "metal:0"; W = 256
rng = np.random.default_rng(0)
ok = True
for n in (8, 31, 32, 43, 48):
    m = rng.standard_normal((W, n, n)).astype(np.float32)
    A = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
    B = rng.standard_normal((W, n)).astype(np.float32)
    ref = np.linalg.solve(A.astype(np.float64), B.astype(np.float64)[:, :, None])[:, :, 0]
    for bd in (32, 16):
        for fill in ("upper", "lower"):
            out = {}
            for reg in (False, True):
                wp.config.metal_register_solve = reg
                a = wp.array(A, dtype=float, device=dev); b = wp.array(B, dtype=float, device=dev); x = wp.zeros((W, n), dtype=float, device=dev)
                wp.launch_tiled(make(n, fill), dim=W, inputs=[a, b, x], block_dim=bd, device=dev); wp.synchronize_device(dev)
                out[reg] = x.numpy().astype(np.float64)
            e_scalar = np.abs(out[False] - ref).max() / np.abs(ref).max()
            e_reg = np.abs(out[True] - ref).max() / np.abs(ref).max()
            diff = np.abs(out[True] - out[False]).max() / np.abs(ref).max()
            good = e_reg < 5e-5 and diff < 5e-5
            ok &= good
            print(f"n={n:2d} block_dim {bd:2d} {fill:5s}: scalar vs f64 {e_scalar:.1e} | register vs f64 {e_reg:.1e} | register vs scalar {diff:.1e} {'ok' if good else 'FAIL'}")
print("ALL OK" if ok else "FAILURES")
