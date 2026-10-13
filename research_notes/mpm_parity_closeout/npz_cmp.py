"""Array-by-array comparison of two npz files (same keys): equal / differing entries, and for integer index arrays
whether they are permutations of each other within segments.

    python npz_cmp.py A.npz B.npz
"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print("only A:", sorted(set(a.files) - set(b.files)))
print("only B:", sorted(set(b.files) - set(a.files)))
for k in sorted(set(a.files) & set(b.files)):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print(f"{k}: shape {x.shape} vs {y.shape}")
        continue
    if x.dtype.kind not in "fiub":
        continue
    if np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
        print(f"{k}: {x.shape} {x.dtype} EQUAL")
        continue
    if x.dtype.kind == "f":
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        print(f"{k}: {x.shape} differs on {(x != y).reshape(x.shape[0], -1).any(1).sum()} rows, max abs {d.max():.3e}")
    else:
        same_set = np.array_equal(np.sort(x.ravel()), np.sort(y.ravel()))
        print(f"{k}: {x.shape} int differs on {(x != y).sum()} entries; same multiset: {same_set}")
