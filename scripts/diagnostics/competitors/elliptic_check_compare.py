"""Offline comparison of two elliptic_check.py snapshot files: python elliptic_check_compare.py A.npz B.npz"""
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print(f"A: {a['label']}\nB: {b['label']}")
for k, qa, va, qb, vb in zip(a["steps"], a["qpos"], a["qvel"], b["qpos"], b["qvel"]):
    d = np.abs(qa - qb)
    print(f"   step {int(k):4d}: qpos max {d.max():.2e} median {np.median(d):.1e} | qvel max {np.abs(va - vb).max():.2e} | worlds with |dq| > 1e-3: {100 * np.mean(d.max(axis=1) > 1e-3):.1f} %")
print("   A oracle: " + "  ".join(f"t{int(k)} {mx:.2e}/{md:.1e}" for k, mx, md in a["proto"]))
print("   B oracle: " + "  ".join(f"t{int(k)} {mx:.2e}/{md:.1e}" for k, mx, md in b["proto"]))
