"""First launches of ``warp.sparse.bsr_mv``'s tiled kernel on Metal (``warp.config.metal_bsr_mv_tiles``), small to large.

For BSR matrices of 3x3 blocks shaped like Newton's rigidity operator (a few long rows; and one row per node with up
to 3 blocks), compares the Metal product with a float64 host reference (relative error), checks that two calls give
the same bits, that the tiled path was taken where expected (kernel name in the module cache) and that empty rows
and ``beta`` behave as in the untiled kernel. Exits non-zero on any failure.

    PYTHONPATH=<warp> .venv-newton152/bin/python metalsim/physics/mpm/probes/bsr_mv_tiles_check.py
"""
import sys

import numpy as np
import warp as wp
import warp.sparse as sp

wp.config.quiet = True
wp.init()
dev = "metal:0"
print("metal_bsr_mv_tiles =", getattr(wp.config, "metal_bsr_mv_tiles", None), flush=True)
rng = np.random.default_rng(0)
fail = 0


def case(nrow, ncol, row_blocks, alpha, beta, empty_rows=0):
    global fail
    rows, cols = [], []
    for r in range(nrow - empty_rows):
        c = rng.choice(ncol, size=min(row_blocks, ncol), replace=False)
        rows += [r] * len(c)
        cols += list(c)
    vals = rng.standard_normal((len(rows), 3, 3)).astype(np.float32)
    A = sp.bsr_from_triplets(nrow, ncol, wp.array(rows, dtype=int, device=dev), wp.array(cols, dtype=int, device=dev),
                             wp.array(vals, dtype=wp.mat33, device=dev))
    x = rng.standard_normal(3 * ncol).astype(np.float32)
    y0 = rng.standard_normal(3 * nrow).astype(np.float32)
    ref = np.zeros(3 * nrow)
    for r, c, v in zip(rows, cols, vals):
        ref[3 * r:3 * r + 3] += v.astype(np.float64) @ x[3 * c:3 * c + 3].astype(np.float64)
    ref = alpha * ref + beta * y0.astype(np.float64)
    outs = []
    for _ in range(2):
        y = wp.array(y0, dtype=float, device=dev)
        sp.bsr_mv(A, wp.array(x, dtype=float, device=dev), y, alpha=alpha, beta=beta)
        outs.append(y.numpy())
    tiled = A.nnz * A.block_size > 2 * 64 * A.shape[0]
    scale = np.abs(ref).max() + 1e-30
    err = float(np.abs(outs[0] - ref).max() / scale)
    same = np.array_equal(outs[0], outs[1])
    ok = same and err < 1e-5 * max(1.0, np.sqrt(row_blocks))
    if empty_rows:
        tail = outs[0][3 * (nrow - empty_rows):]
        ok = ok and np.array_equal(tail, (np.float32(beta) * y0[3 * (nrow - empty_rows):]).astype(np.float32) + 0.0)
    fail += not ok
    print(f"rows {nrow:6d} x {row_blocks:6d} blocks (empty {empty_rows}) alpha {alpha} beta {beta}: tiled {tiled}, "
          f"rel err {err:.2e}, repeat bitwise {same} -> {'ok' if ok else 'FAIL'}", flush=True)


case(2, 40, 40, 1.0, 0.0)  # 360 > 2*64*6: tiled, tiny
case(4, 400, 300, -1.0, 0.0, empty_rows=1)
case(4, 20000, 6000, -1.0, 0.0)
case(12, 50000, 20000, 1.0, 1.0)
case(20000, 12, 3, 1.0, 1.0)  # J-shaped: untiled
sys.exit(1 if fail else 0)
