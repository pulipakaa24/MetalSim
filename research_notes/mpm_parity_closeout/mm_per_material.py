"""Per-material position RMS over frames for multi_material: Metal vs L4 base and each L4 perturbation vs base.

    python mm_per_material.py [METAL_RUN_DIR ...]
"""
import glob
import os
import sys

import numpy as np

L4 = ["runs/parity/isaaclab3_mpm_floors/mpm_floors",
      "runs/parity/isaaclab3_mpm_floors/mpm_floors2"]
METAL = sys.argv[1:] or ["runs/mpm/metal_floors_integrated_nofastmath/F_multi_material"]


def counts():
    out = []
    for lo, hi in [([-.5, -.5, 0], [.5, .5, .25]), ([-.5, .25, .5], [.5, .75, .75]),
                   ([-.5, -.75, .5], [.5, -.25, .75]), ([-.25, -.5, 1.], [.25, .5, 1.5])]:
        res = np.ceil(3 * (np.array(hi) - np.array(lo)) / 0.05).astype(int)
        out.append(int(np.prod(res + 1)))
    return out


NAMES = ["kinematic", "sand", "snow", "mud"]
C = np.cumsum([0] + counts())


def load(p):
    t = np.load(os.path.join(p, "traj.npz"))
    return t["frame"], t["pos"].astype(np.float64)


def rms(a, b, s=slice(None)):
    d = a[:, s] - b[:, s]
    return np.sqrt((d * d).sum(-1).mean(-1))


def main():
    fb, base = load(f"{L4[0]}/multi_material_base")
    assert C[-1] == base.shape[1], (C, base.shape)
    runs = {}
    for d in L4:
        for p in sorted(glob.glob(f"{d}/multi_material_*")):
            v = os.path.basename(p)[len("multi_material_"):]
            if os.path.isdir(p) and v != "base" and v not in runs:
                runs[v] = load(p)[1]
    for m in METAL:
        runs["M:" + "/".join(m.rstrip("/").split("/")[-2:])[:26]] = load(m)[1]
    print("frames", fb.tolist())
    for k in range(4):
        s = slice(C[k], C[k + 1])
        print(f"\n== {NAMES[k]} ({C[k + 1] - C[k]} particles), RMS vs L4 base [mm]")
        for v, r in runs.items():
            print(f"{v:30s}", " ".join(f"{x * 1e3:7.3f}" for x in rms(base, r, s)))
    print("\n== all [mm]")
    for v, r in runs.items():
        print(f"{v:30s}", " ".join(f"{x * 1e3:7.3f}" for x in rms(base, r)))


if __name__ == "__main__":
    main()
