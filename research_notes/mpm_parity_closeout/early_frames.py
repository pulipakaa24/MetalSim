"""Early-frame (pre-chaos) differences: every run against every reference, per material, in nm.

    python early_frames.py SCENE_PREFIX REF_RUN [RUN ...]   (runs as name=path)
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm_per_material import C, NAMES  # noqa: E402

def ld(p):
    t = np.load(p + "/traj.npz")
    return t["frame"], t["pos"].astype(np.float64)

def main():
    refs = [a.split("=", 1) for a in sys.argv[1:]]
    data = {n: ld(p) for n, p in refs}
    names = list(data)
    nf = int(__import__("os").environ.get("NF", "3"))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, pa = data[a]; fb, pb = data[b]
            d = pa[:nf] - pb[:nf]
            r = np.sqrt((d * d).sum(-1))
            row = []
            for k in range(4):
                s = slice(C[k], C[k + 1])
                row.append(" ".join(f"{np.sqrt((r[f, s] ** 2).mean()) * 1e9:8.1f}" for f in range(nf)))
            print(f"{a:>10s} vs {b:10s} | " + " | ".join(row))
    print("columns: kinematic | sand | snow | mud, frames", data[names[0]][0][:nf].tolist(), "(nm)")

if __name__ == "__main__":
    main()
