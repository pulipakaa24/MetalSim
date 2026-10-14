"""Warp's tiled bsr_mv on Metal vs copies whose lane loop issues its loads 4 or 8 terms ahead (research_notes/gpu_latency).

The tiled kernel (warp/_src/sparse.py, make_bsr_mv_tiled_kernel) sums, per (row, subrow), the scalar columns
lane, lane + T, lane + 2T, ... in one lane (T = 64), then tile_sum over the lanes. Each term is a dependent load pair
(column index, then x). The unrolled copies load the indices and values of 4 / 8 consecutive terms of the lane first,
then accumulate them in the original order (same operations, same order: bitwise expected), so more loads are in
flight per thread. Checks bitwise equality with warp.sparse.bsr_mv's own tiled path and times each variant (graph
replay of 200 launches, sustained clocks) on an IJtm-like matrix: --rows rows of --blocks 3x3 blocks.

    PYTHONPATH=<warp> python tiled_mv_unroll_probe.py [--rows 12 --blocks 1170 --ncol 120000] [--json out.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp
import warp.sparse as sp
from warp._src.sparse import _bsr_row_count

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="metal:0")
ap.add_argument("--rows", type=int, default=12)
ap.add_argument("--blocks", type=int, default=1170)
ap.add_argument("--ncol", type=int, default=120000)
ap.add_argument("--seconds", type=float, default=1.0)
ap.add_argument("--json")
args = ap.parse_args()
wp.config.quiet = True
wp.init()
T = 64


@wp.kernel(enable_backward=False, module="unique")
def tiled_u1(alpha: float, A_offsets: wp.array[int], A_row_counts: wp.array[int], A_columns: wp.array[int],
             A_values: wp.array3d[float], x: wp.array[float], y: wp.array[float]):
    row, subrow, lane = wp.tid()
    yi = row * 3 + subrow
    subrow_sum = wp.tile_zeros(shape=(1,), dtype=float)
    block_beg = A_offsets[row]
    col_count = _bsr_row_count(A_offsets, A_row_counts, row) * 3
    lane_sum = float(0.0)
    for col in range(lane, col_count, T):
        block = col // 3
        block_col = col - block * 3
        block += block_beg
        xi = x[A_columns[block] * 3 + block_col]
        lane_sum += A_values[block, subrow, block_col] * xi
    lane_sum *= alpha
    subrow_sum += wp.tile_sum(wp.tile(lane_sum))
    wp.tile_store(y, subrow_sum, yi)


@wp.func
def _term(A_columns: wp.array[int], A_values: wp.array3d[float], x: wp.array[float], block_beg: int, subrow: int,
          col: int, col_count: int):
    v = float(0.0)
    xi = float(0.0)
    if col < col_count:
        block = col // 3
        block_col = col - block * 3
        block += block_beg
        xi = x[A_columns[block] * 3 + block_col]
        v = A_values[block, subrow, block_col]
    return v, xi


@wp.kernel(enable_backward=False, module="unique")
def tiled_u4(alpha: float, A_offsets: wp.array[int], A_row_counts: wp.array[int], A_columns: wp.array[int],
             A_values: wp.array3d[float], x: wp.array[float], y: wp.array[float]):
    row, subrow, lane = wp.tid()
    yi = row * 3 + subrow
    subrow_sum = wp.tile_zeros(shape=(1,), dtype=float)
    block_beg = A_offsets[row]
    col_count = _bsr_row_count(A_offsets, A_row_counts, row) * 3
    lane_sum = float(0.0)
    for c in range(lane, col_count, 4 * T):
        v0, x0 = _term(A_columns, A_values, x, block_beg, subrow, c, col_count)
        v1, x1 = _term(A_columns, A_values, x, block_beg, subrow, c + T, col_count)
        v2, x2 = _term(A_columns, A_values, x, block_beg, subrow, c + 2 * T, col_count)
        v3, x3 = _term(A_columns, A_values, x, block_beg, subrow, c + 3 * T, col_count)
        lane_sum += v0 * x0
        if c + T < col_count:
            lane_sum += v1 * x1
        if c + 2 * T < col_count:
            lane_sum += v2 * x2
        if c + 3 * T < col_count:
            lane_sum += v3 * x3
    lane_sum *= alpha
    subrow_sum += wp.tile_sum(wp.tile(lane_sum))
    wp.tile_store(y, subrow_sum, yi)


@wp.kernel(enable_backward=False, module="unique")
def tiled_u8(alpha: float, A_offsets: wp.array[int], A_row_counts: wp.array[int], A_columns: wp.array[int],
             A_values: wp.array3d[float], x: wp.array[float], y: wp.array[float]):
    row, subrow, lane = wp.tid()
    yi = row * 3 + subrow
    subrow_sum = wp.tile_zeros(shape=(1,), dtype=float)
    block_beg = A_offsets[row]
    col_count = _bsr_row_count(A_offsets, A_row_counts, row) * 3
    lane_sum = float(0.0)
    for c in range(lane, col_count, 8 * T):
        v0, x0 = _term(A_columns, A_values, x, block_beg, subrow, c, col_count)
        v1, x1 = _term(A_columns, A_values, x, block_beg, subrow, c + T, col_count)
        v2, x2 = _term(A_columns, A_values, x, block_beg, subrow, c + 2 * T, col_count)
        v3, x3 = _term(A_columns, A_values, x, block_beg, subrow, c + 3 * T, col_count)
        v4, x4 = _term(A_columns, A_values, x, block_beg, subrow, c + 4 * T, col_count)
        v5, x5 = _term(A_columns, A_values, x, block_beg, subrow, c + 5 * T, col_count)
        v6, x6 = _term(A_columns, A_values, x, block_beg, subrow, c + 6 * T, col_count)
        v7, x7 = _term(A_columns, A_values, x, block_beg, subrow, c + 7 * T, col_count)
        lane_sum += v0 * x0
        if c + T < col_count:
            lane_sum += v1 * x1
        if c + 2 * T < col_count:
            lane_sum += v2 * x2
        if c + 3 * T < col_count:
            lane_sum += v3 * x3
        if c + 4 * T < col_count:
            lane_sum += v4 * x4
        if c + 5 * T < col_count:
            lane_sum += v5 * x5
        if c + 6 * T < col_count:
            lane_sum += v6 * x6
        if c + 7 * T < col_count:
            lane_sum += v7 * x7
    lane_sum *= alpha
    subrow_sum += wp.tile_sum(wp.tile(lane_sum))
    wp.tile_store(y, subrow_sum, yi)


rng = np.random.default_rng(0)
res = {"rows": args.rows, "blocks_per_row": args.blocks, "ncol": args.ncol, "variants": {}}
ok = True
with wp.ScopedDevice(args.device):
    rows, cols = [], []
    for r in range(args.rows):
        c = np.sort(rng.choice(args.ncol, size=args.blocks, replace=False))
        rows += [r] * c.size
        cols += list(c)
    vals = rng.standard_normal((len(rows), 3, 3)).astype(np.float32)
    A = sp.bsr_from_triplets(args.rows, args.ncol, wp.array(rows, dtype=int), wp.array(cols, dtype=int),
                             wp.array(vals, dtype=wp.mat33))
    x = wp.array(rng.standard_normal(3 * args.ncol).astype(np.float32), dtype=float)
    alpha = -1.0
    y_ref = wp.zeros(3 * args.rows, dtype=float)
    sp.bsr_mv(A, x, y_ref, alpha=alpha, beta=0.0, tile_size=T)   # warp.sparse's own tiled path
    ref = y_ref.numpy()
    empty = wp.empty(0, dtype=int)
    row_counts = A.row_counts if A.row_counts is not None else empty
    variants = {"warp_bsr_mv_tiled": None, "copy_u1": tiled_u1, "unroll_u4": tiled_u4, "unroll_u8": tiled_u8}
    for name, k in variants.items():
        y = wp.zeros(3 * args.rows, dtype=float)

        def launch(k=k, y=y):
            if k is None:
                sp.bsr_mv(A, x, y, alpha=alpha, beta=0.0, tile_size=T)
            else:
                wp.launch(k, dim=(args.rows, 3, T), block_dim=T,
                          inputs=[alpha, A.offsets, row_counts, A.columns, A.scalar_values, x, y])

        launch()
        out = y.numpy()
        same = bool(np.array_equal(out, ref))
        ok &= same
        wp.synchronize_device()
        n = 200
        with wp.ScopedCapture(force_module_load=False) as cap:
            for _ in range(n):
                launch()
        per = []
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            wp.synchronize_device()
            t0 = time.perf_counter()
            wp.capture_launch(cap.graph)
            wp.synchronize_device()
            per.append((time.perf_counter() - t0) * 1e6 / n)
        us = float(np.median(per[len(per) // 2:]))
        res["variants"][name] = {"bitwise_equal_to_warp_tiled": same, "us_per_launch": us}
        print(f"{name:20s} bitwise {same}  {us:7.2f} us per launch", flush=True)
print("OK" if ok else "FAIL (a variant differs)")
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=1)
sys.exit(0 if ok else 1)
