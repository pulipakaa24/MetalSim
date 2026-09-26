"""Residency test for the n = 43 register Cholesky: the same factor + solve arithmetic as Warp's tile path
(the fork's metal_register_cholesky_step / forward / backward templates), run as a native snippet on H read from
device memory, in a plain launch with 32 lanes per world and 1, 2, 4 or 8 worlds per threadgroup (block sizes
32 .. 256), against the tile path at block_dim 32. Bitwise check of the search vector against the tile path.

usage: PYTHONPATH=<warp worktree> python scripts/diagnostics/metal_cholesky_residency.py [n=43]"""
import sys, time
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.metal_register_cholesky_max = 48

n = int(sys.argv[1]) if len(sys.argv) > 1 else 43
CPL = (n + 31) // 32
SNIP = f"""
#if defined(__METAL_VERSION__)
    constexpr int N = {n};
    constexpr int BD = 32;
    constexpr int CPL = {CPL};
    thread float col[CPL][N];
#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {{
        const int jc = lane + c * BD;
#pragma clang loop unroll(full)
        for (int i = 0; i < N; ++i)
            col[c][i] = (jc < N && i >= jc) ? h.data[jc * N + i] : 0.0f;   // upper storage: H[jc, i]
    }}
    wp::metal_register_cholesky_step<0, N, CPL, BD, float>(col, lane);
    thread float b[CPL]; thread float y[CPL]; thread float x[CPL];
#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {{
        const int jc = lane + c * BD;
        b[c] = (jc < N) ? g.data[jc] : 0.0f; y[c] = 0.0f; x[c] = 0.0f;
    }}
    wp::metal_register_forward_step<0, N, CPL, BD, float>(col, y, b, lane);
    wp::metal_register_backward_step<N - 1, N, CPL, BD, float>(col, y, x, lane);
#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {{
        const int jc = lane + c * BD;
        if (jc < N) out.data[jc] = -x[c];
    }}
#endif
"""


@wp.func_native(snippet=SNIP)
def chol_solve_native(h: wp.array2d[float], g: wp.array[float], out: wp.array[float], lane: int):
    pass


SNIP_SPLIT = SNIP.replace("wp::metal_register_cholesky_step<0, N, CPL, BD, float>(col, lane);",
                          "wp::metal_register_cholesky_step_split<0, N, CPL, BD, float>(col, lane);")


@wp.kernel(enable_backward=False, module="unique")
def k_native(h: wp.array3d(dtype=float), g: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, lane = wp.tid()
    chol_solve_native(h[w], g[w], out[w], lane)


@wp.func_native(snippet=SNIP_SPLIT)
def chol_solve_split(h: wp.array2d[float], g: wp.array[float], out: wp.array[float], lane: int):
    pass


@wp.kernel(enable_backward=False, module="unique")
def k_split(h: wp.array3d(dtype=float), g: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, lane = wp.tid()
    chol_solve_split(h[w], g[w], out[w], lane)


@wp.kernel(enable_backward=False, module="unique")
def k_tile(h: wp.array3d(dtype=float), g: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w = wp.tid()
    t = wp.tile_load(h[w], shape=(n, n))
    wp.tile_cholesky_inplace(t, fill_mode="upper")
    rhs = wp.tile_load(g[w], shape=(n,))
    wp.tile_store(out[w], wp.tile_map(wp.mul, wp.tile_cholesky_solve(t, rhs, fill_mode="upper"), -1.0))


dev = "metal:0"; W = 4096; K = 50
rng = np.random.default_rng(0)
m = rng.standard_normal((W, n, n)).astype(np.float32)
A = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
h = wp.array(A, dtype=float, device=dev); g = wp.array(rng.standard_normal((W, n)).astype(np.float32), dtype=float, device=dev)
o_tile = wp.zeros((W, n), dtype=float, device=dev); o_nat = wp.zeros((W, n), dtype=float, device=dev)


def timed(fn):
    fn(); wp.synchronize_device(dev)
    with wp.ScopedDevice(dev), wp.ScopedCapture(device=dev) as cap:
        for _ in range(K):
            fn()
    wp.capture_launch(cap.graph); wp.synchronize_device(dev)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize_device(dev); ts.append((time.perf_counter() - t0) / K)
    return sorted(ts)[1] * 1e3


t = timed(lambda: wp.launch_tiled(k_tile, dim=W, inputs=[h, g, o_tile], block_dim=32, device=dev))
print(f"n={n}: tile path (block_dim 32, one world per threadgroup): {t:.3f} ms per 4096 factor+solve", flush=True)
ref = o_tile.numpy().copy()
for bd in (32, 64, 128, 256):
    t = timed(lambda: wp.launch(k_native, dim=(W, 32), inputs=[h, g, o_nat], block_dim=bd, device=dev))
    same = np.array_equal(o_nat.numpy(), ref)
    print(f"n={n}: native snippet, block {bd:3d} ({bd // 32} worlds per threadgroup): {t:.3f} ms  ({'bitwise = tile path' if same else 'DIFF max %.2e' % np.abs(o_nat.numpy() - ref).max()})", flush=True)
for bd in (32, 128):
    try:
        t = timed(lambda: wp.launch(k_split, dim=(W, 32), inputs=[h, g, o_nat], block_dim=bd, device=dev))
        same = np.array_equal(o_nat.numpy(), ref)
        print(f"n={n}: split-loop snippet, block {bd:3d}: {t:.3f} ms  ({'bitwise = tile path' if same else 'DIFF max %.2e' % np.abs(o_nat.numpy() - ref).max()})", flush=True)
    except Exception as e:
        print(f"n={n}: split-loop snippet, block {bd}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
