"""Bitwise comparison of recordings: this branch's off / relaxed modes against the integrated tree's recordings."""
import numpy as np

W = "runs/mpm/"
M = "runs/mpm/"
for mine, ref in (("fm_off", "metal_floors_integrated_nofastmath"), ("fm_relaxed", "metal_floors_integrated")):
    for s in ("granular", "multi_material", "twoway", "anymal", "plate_drag", "column_a2"):
        a = np.load(f"{W}{mine}/F_{s}/traj.npz")
        b = np.load(f"{M}{ref}/F_{s}/traj.npz")
        eq = all(np.array_equal(a[k], b[k]) for k in a.files if k in b.files)
        print(f"{mine:10s} vs {ref:36s} {s:15s} bitwise {'EQUAL' if eq else 'DIFFERENT'}")
