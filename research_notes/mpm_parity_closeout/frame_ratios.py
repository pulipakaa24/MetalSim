"""Per-frame Metal / max-floor ratio for a floor scene (particle-position RMS against the L4 base), and the same for
the rigid bodies when recorded.

    python frame_ratios.py SCENE NFRAMES METAL_RUN_DIR [METAL_RUN_DIR ...]
"""
import glob
import os
import sys

import numpy as np

L4 = ["/Users/aditya/robosim/runs/parity/isaaclab3_mpm_floors/mpm_floors",
      "/Users/aditya/robosim/runs/parity/isaaclab3_mpm_floors/mpm_floors2"]
scene, nf = sys.argv[1], int(sys.argv[2])


def ld(p):
    t = np.load(p + "/traj.npz")
    return t["frame"], t["pos"].astype(np.float64), (t["body_q"].astype(np.float64) if "body_q" in t.files else None)


f0, base, bb = ld(f"{L4[0]}/{scene}_base")
runs = {}
for d in L4:
    for p in sorted(glob.glob(f"{d}/{scene}_*")):
        v = os.path.basename(p)[len(scene) + 1:]
        if os.path.isdir(p) and not v.startswith("base") and v not in runs:
            runs[v] = ld(p)
n = min(nf, len(f0))


def rms(a, b):
    d = a[:n] - b[:n]
    return np.sqrt((d * d).sum(-1).mean(-1))


D = np.array([rms(r[1], base) for r in runs.values()])
print("frames", f0[:n].tolist())
print("floor max  ", " ".join(f"{x:9.2e}" for x in D.max(0)))
print("floor min  ", " ".join(f"{x:9.2e}" for x in D.min(0)))
for m in sys.argv[3:]:
    _, mp, mb = ld(m)
    M = rms(mp, base)
    print(f"{m.split('/')[-2][:10]:10s} M", " ".join(f"{x:9.2e}" for x in M))
    print(f"{'':10s} r", " ".join(f"{x:9.2f}" for x in M / np.maximum(D.max(0), 1e-30)))
    if mb is not None and bb is not None:
        DB = np.array([rms(r[2][..., :3], bb[..., :3]) for r in runs.values() if r[2] is not None])
        MB = rms(mb[..., :3], bb[..., :3])
        print(f"{'':10s} body pos M", " ".join(f"{x:9.2e}" for x in MB))
        print(f"{'':10s} body floor", " ".join(f"{x:9.2e}" for x in DB.max(0)))
