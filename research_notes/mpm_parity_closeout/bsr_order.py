"""Row-internal column order of a BSR matrix dumped by dump_first_solve.py on two devices."""
import sys

import numpy as np

A, B, name = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "contact.collision.transposed_collider_mat"
a, b = np.load(f"{A}/solve0_pre.npz"), np.load(f"{B}/solve0_pre.npz")
print([k for k in a.files if k.startswith(name)])
off_a, off_b = a[f"{name}.offsets"], b[f"{name}.offsets"]
ca, cb = a[f"{name}.columns"], b[f"{name}.columns"]
va, vb = a[f"{name}.values"], b[f"{name}.values"]
print("offsets equal:", np.array_equal(off_a, off_b), "nrow", off_a.size - 1, "nnz", off_a[-1], "array len", ca.size)
nnz = int(off_a[-1])
asc_a = asc_b = same_set = same_vals = 0
for r in range(off_a.size - 1):
    s = slice(off_a[r], off_a[r + 1])
    x, y = ca[s], cb[s]
    asc_a += bool(np.all(np.diff(x) > 0))
    asc_b += bool(np.all(np.diff(y) > 0))
    same_set += bool(np.array_equal(np.sort(x), np.sort(y)))
    oa, ob = np.argsort(x, kind="stable"), np.argsort(y, kind="stable")
    same_vals += bool(np.array_equal(va[s][oa], vb[s][ob]))
n = off_a.size - 1
print(f"rows ascending: A {asc_a}/{n}, B {asc_b}/{n}; same column set {same_set}/{n}; same values after sorting {same_vals}/{n}")
r = int(np.argmax(np.diff(off_a)))
print("example row", r, "A", ca[off_a[r]:off_a[r + 1]][:12], "B", cb[off_b[r]:off_b[r + 1]][:12])
