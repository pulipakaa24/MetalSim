"""Compare dump_first_solve.py outputs of two devices, array by array (pre / post first solve, frame)."""
import sys

import numpy as np

A, B = sys.argv[1], sys.argv[2]
for f in ["solve0_pre", "solve0_post", "frame"]:
    a, b = np.load(f"{A}/{f}.npz"), np.load(f"{B}/{f}.npz")
    print(f"== {f}: {len(a.files)} arrays (A) / {len(b.files)} (B); only A: {sorted(set(a.files) - set(b.files))[:5]}"
          f" only B: {sorted(set(b.files) - set(a.files))[:5]}")
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"   {k}: shape {x.shape} vs {y.shape}")
            continue
        if x.dtype.kind not in "fiub":
            continue
        if np.array_equal(x, y) or (x.dtype.kind == "f" and np.array_equal(x, y, equal_nan=True)):
            continue
        if x.dtype.kind == "f":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            rel = d / np.maximum(np.abs(x.astype(np.float64)), 1e-30)
            xr = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(-1, 1)
            yr = y.reshape(xr.shape)
            nrow = int((xr != yr).any(1).sum())
            print(f"   {k}: {x.shape} rows differ {nrow}/{xr.shape[0]}, max abs {d.max():.3e}, max rel {np.nanmax(rel):.3e}, "
                  f"rms {np.sqrt((d ** 2).mean()):.3e} (|A| rms {np.sqrt((x.astype(np.float64) ** 2).mean()):.3e})")
        else:
            print(f"   {k}: {x.shape} int differs on {(x != y).sum()}")
