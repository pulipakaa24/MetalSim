"""Per-run RMS against the L4 base for the first frames (each L4 perturbation and Metal runs), plus body positions.

    python per_run.py SCENE NFRAMES [metal=dir ...]
"""
import glob
import os
import sys

import numpy as np

L4 = ["/Users/aditya/robosim/runs/parity/isaaclab3_mpm_floors/mpm_floors",
      "/Users/aditya/robosim/runs/parity/isaaclab3_mpm_floors/mpm_floors2"]
scene, n = sys.argv[1], int(sys.argv[2])


def ld(p):
    t = np.load(p + "/traj.npz")
    return t["pos"][:n].astype(np.float64), (t["body_q"][:n, :, :3].astype(np.float64) if "body_q" in t.files else None)


bp, bb = ld(f"{L4[0]}/{scene}_base")
runs = {}
for d in L4:
    for p in sorted(glob.glob(f"{d}/{scene}_*")):
        v = os.path.basename(p)[len(scene) + 1:]
        if os.path.isdir(p) and v != "base" and v not in runs:
            runs[v] = ld(p)
for a in sys.argv[3:]:
    k, d = a.split("=", 1)
    runs["M:" + k] = ld(d)
for k, (p, b) in runs.items():
    r = np.sqrt(((p - bp) ** 2).sum(-1).mean(-1))
    s = " ".join(f"{x:8.2e}" for x in r)
    if b is not None and bb is not None:
        rb = np.sqrt(((b - bb) ** 2).sum(-1).mean(-1))
        s += " | body " + " ".join(f"{x:8.2e}" for x in rb)
    print(f"{k:14s} {s}")
