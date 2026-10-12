"""Bitwise comparison of scale_pour --dump files across trees: python3 cmpdump.py A_eager.npz B_eager.npz ..."""
import sys

import numpy as np

pairs = sys.argv[1:]
for a, b in zip(pairs[::2], pairs[1::2]):
    A, B = np.load(a), np.load(b)
    res = {}
    for k in sorted(A.files):
        if k not in B.files:
            res[k] = "missing"
            continue
        x, y = A[k], B[k]
        if x.shape != y.shape:
            res[k] = f"shape {x.shape} vs {y.shape}"
        elif np.array_equal(x.view(np.uint8), y.view(np.uint8)):
            res[k] = "bitwise"
        else:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            res[k] = f"DIFF max {d.max():.3g} n {int((d > 0).sum())}"
    print(a.split("/")[-1], "vs", b.split("/")[-1], res)
