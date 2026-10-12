"""Stopping iterations of every L4 floor run and the Metal runs, per solve, against the L4 base.

    python stops_all.py SCENE N_SOLVES [metal_name=dir ...]
"""
import glob
import os
import sys

import numpy as np

L4 = ["/Users/aditya/robosim/runs/parity/isaaclab3_mpm_floors/mpm_floors",
      "/Users/aditya/robosim/runs/parity/isaaclab3_mpm_floors/mpm_floors2"]
scene, n = sys.argv[1], int(sys.argv[2])
runs = {}
for d in L4:
    for p in sorted(glob.glob(f"{d}/{scene}_*")):
        v = os.path.basename(p)[len(scene) + 1:]
        if os.path.isdir(p) and v not in runs and os.path.exists(p + "/solves.npz"):
            runs[v] = np.load(p + "/solves.npz")["iterations"]
for a in sys.argv[3:]:
    k, d = a.split("=", 1)
    runs["M:" + k] = np.load(d + "/solves.npz")["iterations"]
base = runs["base"]
print(f"{'run':14s} first-{n} iterations (. = same as base)")
for k, it in runs.items():
    m = min(n, it.size, base.size)
    s = " ".join(f"{x:3d}" if x != b else "  ." for x, b in zip(it[:m], base[:m]))
    nd = int((it[:min(it.size, base.size)] != base[:min(it.size, base.size)]).sum())
    print(f"{k:14s} {s} | differs on {nd}/{min(it.size, base.size)}, mean {it.mean():.2f}")
