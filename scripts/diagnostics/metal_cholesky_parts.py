"""Cost split of the Metal register tile Cholesky at 4096 matrices: factorization only, solve only (from a stored
factor), factor + solve (as MuJoCo Warp's Newton solver runs it), and a plain load/store round trip, for the
Newton Hessian sizes of interest (32: the size the register path was tuned at, 43: the G1, 48: the bound).
Graph-replayed, 50 launches per timing, synchronized; residual check on the first matrices.

usage: python scripts/diagnostics/metal_cholesky_parts.py [sizes...]"""
import os, sys, time
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.metal_register_cholesky_max = 48
if os.environ.get("WP_METAL_REGISTER_SOLVE") is not None and hasattr(wp.config, "metal_register_solve"):
    wp.config.metal_register_solve = os.environ["WP_METAL_REGISTER_SOLVE"] != "0"   # fork worktree knob (A/B)
if os.environ.get("WP_METAL_ROLLED_CHOLESKY") is not None and hasattr(wp.config, "metal_rolled_cholesky"):
    wp.config.metal_rolled_cholesky = int(os.environ["WP_METAL_ROLLED_CHOLESKY"])
if os.environ.get("WP_METAL_CHOL_SPLIT") is not None and hasattr(wp.config, "metal_chol_split"):
    wp.config.metal_chol_split = os.environ["WP_METAL_CHOL_SPLIT"] != "0"
print(f"chol split: {getattr(wp.config, 'metal_chol_split', 'n/a')}", flush=True)
print(f"rolled cholesky above n: {getattr(wp.config, 'metal_rolled_cholesky', 'n/a')}", flush=True)
print(f"register solve: {getattr(wp.config, 'metal_register_solve', 'n/a (not in this Warp)')}", flush=True)


def make(n, mode):
    if mode == "factor":
        @wp.kernel(enable_backward=False, module="unique")
        def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float), f: wp.array3d(dtype=float)):
            w = wp.tid()
            t = wp.tile_load(a[w], shape=(n, n))
            wp.tile_cholesky_inplace(t, fill_mode="upper")
            wp.tile_store(f[w], t)
    elif mode == "solve":
        @wp.kernel(enable_backward=False, module="unique")
        def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float), f: wp.array3d(dtype=float)):
            w = wp.tid()
            t = wp.tile_load(f[w], shape=(n, n))
            rhs = wp.tile_load(b[w], shape=(n,))
            wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))
    elif mode == "both":
        @wp.kernel(enable_backward=False, module="unique")
        def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float), f: wp.array3d(dtype=float)):
            w = wp.tid()
            t = wp.tile_load(a[w], shape=(n, n))
            wp.tile_cholesky_inplace(t, fill_mode="upper")
            rhs = wp.tile_load(b[w], shape=(n,))
            wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))
    else:   # load/store round trip of the matrix tile
        @wp.kernel(enable_backward=False, module="unique")
        def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float), f: wp.array3d(dtype=float)):
            w = wp.tid()
            t = wp.tile_load(a[w], shape=(n, n))
            wp.tile_store(f[w], t)
    return k


dev = "metal:0"; W = 4096; K = 50
BD = int(os.environ.get("CHOL_BLOCK_DIM", "32"))
print(f"block_dim {BD}", flush=True)
sizes = [int(s) for s in sys.argv[1:]] or [32, 43, 48]
rng = np.random.default_rng(0)
for n in sizes:
    m = rng.standard_normal((W, n, n)).astype(np.float32)
    A = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
    a = wp.array(A, dtype=float, device=dev); b = wp.array(rng.standard_normal((W, n)).astype(np.float32), dtype=float, device=dev)
    x = wp.zeros((W, n), dtype=float, device=dev); f = wp.zeros((W, n, n), dtype=float, device=dev)
    res = {}
    for mode in ("copy", "factor", "solve", "both"):
        kern = make(n, mode)
        wp.launch_tiled(kern, dim=W, inputs=[a, b, x, f], block_dim=BD, device=dev); wp.synchronize_device(dev)
        with wp.ScopedDevice(dev), wp.ScopedCapture(device=dev) as cap:
            for _ in range(K):
                wp.launch_tiled(kern, dim=W, inputs=[a, b, x, f], block_dim=BD, device=dev)
        wp.capture_launch(cap.graph); wp.synchronize_device(dev)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev); ts.append((time.perf_counter() - t0) / K)
        res[mode] = sorted(ts)[1]
        if mode in ("solve", "both"):
            err = np.abs(np.einsum("wij,wj->wi", A[:8].astype(np.float64), x.numpy()[:8]) - b.numpy()[:8]).max()
            res[mode + "_res"] = err
    print(f"n={n:2d}: copy {res['copy']*1e3:.3f} ms | factor {res['factor']*1e3:.3f} ms | solve {res['solve']*1e3:.3f} ms "
          f"(residual {res['solve_res']:.1e}) | factor+solve {res['both']*1e3:.3f} ms (residual {res['both_res']:.1e})  "
          f"[per launch of {W}, block_dim {BD}, graph replay]", flush=True)
