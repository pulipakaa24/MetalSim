"""Bitwise comparison of two Metal floor-recording directories (every array of traj.npz and solves.npz).

    python bitwise_dirs.py DIR_A DIR_B
"""
import os
import sys

import numpy as np

A, B = sys.argv[1], sys.argv[2]
for s in sorted(os.listdir(A)):
    pa, pb = os.path.join(A, s), os.path.join(B, s)
    if not (os.path.isdir(pa) and os.path.exists(os.path.join(pa, "traj.npz")) and os.path.exists(os.path.join(pb, "traj.npz"))):
        continue
    out = []
    for f in ("traj.npz", "solves.npz"):
        a, b = np.load(os.path.join(pa, f)), np.load(os.path.join(pb, f))
        for k in a.files:
            if k not in b.files or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k], equal_nan=True):
                out.append(f"{f}:{k}")
    print(f"{s}: {'BITWISE EQUAL' if not out else 'differs in ' + ', '.join(out)}")
