"""Bitwise A/B of the Warp fork's Metal register Cholesky variants: dumps the factor (upper) and the solve of 512
SPD matrices for n in {33, 40, 43, 48, 32, 16} to an npz, or compares against a reference npz. Run once with the
installed Warp (--out), once with a worktree (--ref), or within one Warp with WP_METAL_COMPACT_REGISTER_CHOLESKY /
WP_METAL_REGISTER_SOLVE knobs.

usage: python scripts/diagnostics/metal_register_cholesky_ab.py --out F.npz | --ref F.npz"""
import sys
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.metal_register_cholesky_max = 48
import os as _os
if _os.environ.get("WP_METAL_CHOL_SPLIT") is not None and hasattr(wp.config, "metal_chol_split"):
    wp.config.metal_chol_split = _os.environ["WP_METAL_CHOL_SPLIT"] != "0"


def make(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float), f: wp.array3d(dtype=float), x: wp.array2d(dtype=float)):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        wp.tile_store(f[w], t)
        rhs = wp.tile_load(b[w], shape=(n,))
        wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))
    return k


dev = "metal:0"; W = 512
import os
BD = int(os.environ.get("CHOL_BLOCK_DIM", "32"))
rng = np.random.default_rng(0)
out = {}
for n in (33, 40, 43, 48, 32, 16):
    m = rng.standard_normal((W, n, n)).astype(np.float32)
    A = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
    B = rng.standard_normal((W, n)).astype(np.float32)
    a = wp.array(A, dtype=float, device=dev); b = wp.array(B, dtype=float, device=dev)
    f = wp.zeros((W, n, n), dtype=float, device=dev); x = wp.zeros((W, n), dtype=float, device=dev)
    wp.launch_tiled(make(n), dim=W, inputs=[a, b, f, x], block_dim=BD, device=dev); wp.synchronize_device(dev)
    out[f"f{n}"] = f.numpy(); out[f"x{n}"] = x.numpy()
    ref = np.linalg.solve(A.astype(np.float64), B.astype(np.float64)[:, :, None])[:, :, 0]
    print(f"n={n:2d}: solve vs float64 {np.abs(out[f'x{n}'] - ref).max() / np.abs(ref).max():.1e}; "
          f"factor residual {np.abs(np.einsum('wji,wjk->wik', out[f'f{n}'].astype(np.float64), out[f'f{n}'].astype(np.float64)) - A).max() / np.abs(A).max():.1e}")
if "--out" in sys.argv:
    np.savez_compressed(sys.argv[sys.argv.index("--out") + 1], **out); print("saved")
if "--ref" in sys.argv:
    z = np.load(sys.argv[sys.argv.index("--ref") + 1]); ok = True
    for k in out:
        same = np.array_equal(out[k], z[k]); ok &= same
        print(f"  {k}: {'bitwise' if same else f'DIFF max {np.abs(out[k] - z[k]).max():.2e}'}")
    print("ALL BITWISE" if ok else "DIFFERENCES")
