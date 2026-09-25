"""Candidate for the 43-dof Newton Hessian on Metal: block (Schur-complement) factor + solve with a trailing block S of
ns dofs (the G1's arms and hands, which no contact row touches) and a core C of nc dofs, all with register-path tiles
(n <= 32), versus the single 43x43 register Cholesky. Same linear system, different arithmetic.

  L_S = chol(S);  W = S^-1 B^T;  C' = C - B W;  x_C = C'^-1 (g_C - B S^-1 g_S);  x_S = S^-1 g_S - W x_C

usage: python scripts/diagnostics/metal_schur_cost.py"""
import time
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.metal_register_cholesky_max = 48
NC, NS = 19, 24
N = NC + NS


@wp.kernel(enable_backward=False, module="unique")
def dense(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float)):
    w = wp.tid()
    t = wp.tile_load(a[w], shape=(N, N))
    wp.tile_cholesky_inplace(t, fill_mode="upper")
    rhs = wp.tile_load(b[w], shape=(N,))
    wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))


@wp.kernel(enable_backward=False, module="unique")
def schur(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float)):
    w = wp.tid()
    S = wp.tile_load(a[w], shape=(NS, NS), offset=(NC, NC))
    wp.tile_cholesky_inplace(S, fill_mode="upper")
    Bm = wp.tile_load(a[w], shape=(NC, NS), offset=(0, NC))           # upper block B (core rows, S columns)
    W = wp.tile_cholesky_solve(S, wp.tile_transpose(Bm), fill_mode="upper")   # NS x NC
    C = wp.tile_load(a[w], shape=(NC, NC))
    wp.tile_matmul(Bm, W, C, alpha=-1.0)                              # C' = C - B W (upper part is what is read)
    gS = wp.tile_load(b[w], shape=(NS,), offset=(NC,))
    yS = wp.tile_cholesky_solve(S, gS, fill_mode="upper")
    gC = wp.tile_load(b[w], shape=(NC,))
    rC = wp.tile_map(wp.sub, gC, wp.tile_squeeze(wp.tile_matmul(Bm, wp.tile_reshape(yS, (NS, 1)))))
    wp.tile_cholesky_inplace(C, fill_mode="upper")
    xC = wp.tile_cholesky_solve(C, rC, fill_mode="upper")
    xS = wp.tile_map(wp.sub, yS, wp.tile_squeeze(wp.tile_matmul(W, wp.tile_reshape(xC, (NC, 1)))))
    wp.tile_store(x[w], xC)
    wp.tile_store(x[w], xS, offset=(NC,))


dev = "metal:0"; Wn = 4096; K = 50
rng = np.random.default_rng(0)
m = rng.standard_normal((Wn, N, N)).astype(np.float32)
A = m @ m.transpose(0, 2, 1) + N * np.eye(N, dtype=np.float32)
A[:, 6:NC, NC:] = 0.0; A[:, NC:, 6:NC] = 0.0                            # legs do not couple to arms (G1 structure)
a = wp.array(A, dtype=float, device=dev); bb = rng.standard_normal((Wn, N)).astype(np.float32)
b = wp.array(bb, dtype=float, device=dev)
for name, kern in (("dense 43", dense), (f"schur {NC}+{NS}", schur)):
    x = wp.zeros((Wn, N), dtype=float, device=dev)
    with wp.ScopedDevice(dev), wp.ScopedCapture(device=dev) as cap:
        for _ in range(K):
            wp.launch_tiled(kern, dim=Wn, inputs=[a, b, x], block_dim=32, device=dev)
    wp.capture_launch(cap.graph); wp.synchronize_device(dev)
    t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / K
    xs = x.numpy()[:8].astype(np.float64)
    res = np.abs(np.einsum("wij,wj->wi", A[:8].astype(np.float64), xs) - bb[:8]).max()
    print(f"{name}: {dt*1e3:.3f} ms per launch of {Wn} (residual {res:.1e})", flush=True)
