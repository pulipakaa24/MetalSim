"""Compare libm_hash.py outputs of two devices: per function, the fraction of 4,096-input chunks with equal
results over all 2^32 inputs (and over chunks covering given ranges), plus agreement / ulp histogram on the random
sample and against a float64 reference.

    python libm_cmp.py DIR_A DIR_B [fns]
"""
import sys

import numpy as np

A, B = sys.argv[1], sys.argv[2]
fns = sys.argv[3].split(",") if len(sys.argv) > 3 else ["log", "exp", "sinh"]
CH = 4096


def chunk_of(x):
    return int(np.array([x], np.float32).view(np.uint32)[0]) // CH


RANGES = {"log": [(0.5, 2.0), (0.99, 1.01), (1e-6, 1e6)], "exp": [(-1e-3, 1e-3), (-1.0, 1.0), (-80.0, 80.0)],
          "sinh": [(-1e-3, 1e-3), (-1.0, 1.0), (-10.0, 10.0)]}
for fn in fns:
    a, b = np.load(f"{A}/{fn}.npz"), np.load(f"{B}/{fn}.npz")
    eq = a["hash"] == b["hash"]
    print(f"== {fn}: equal chunks {eq.mean() * 100:.3f} % of all 2^32 inputs ({(~eq).sum()} chunks differ)")
    for lo, hi in RANGES[fn]:
        idx = []
        for l_, h_ in ([(lo, hi)] if lo >= 0 else [(0.0, -lo), (0.0, hi)]):
            c0, c1 = chunk_of(l_), chunk_of(h_)
            idx.append(np.arange(c0, c1 + 1))
        if lo < 0:  # negative half: sign bit set
            c0, c1 = chunk_of(-0.0), chunk_of(lo)
            idx[0] = np.arange(min(c0, c1), max(c0, c1) + 1)
        ii = np.unique(np.concatenate(idx))
        print(f"   [{lo:g}, {hi:g}]: equal chunks {eq[ii].mean() * 100:.2f} % of {ii.size}")
    x = a["sample_x"].view(np.float32)
    ya, yb = a["sample_y"], b["sample_y"]
    fin = np.isfinite(ya.view(np.float32)) & np.isfinite(yb.view(np.float32))
    d = yb.view(np.int32).astype(np.int64) - ya.view(np.int32).astype(np.int64)
    print(f"   random sample: equal {np.mean(ya == yb) * 100:.3f} %; finite-pair ulp diffs:",
          dict(zip(*[v.tolist() for v in np.unique(np.clip(d[fin], -4, 4), return_counts=True)])))
    f64 = {"log": np.log, "exp": np.exp, "sinh": np.sinh}[fn]
    with np.errstate(all="ignore"):
        ref = f64(x.astype(np.float64)).astype(np.float32)
    ok = np.isfinite(ref) & fin
    print(f"   correctly rounded (finite): A {np.mean(ya[ok] == ref[ok].view(np.uint32)) * 100:.3f} %, "
          f"B {np.mean(yb[ok] == ref[ok].view(np.uint32)) * 100:.3f} %")
