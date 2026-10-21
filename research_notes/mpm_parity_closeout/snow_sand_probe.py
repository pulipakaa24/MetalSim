"""Structure of the early Metal-vs-L4 difference in multi_material: snow at frame 6, sand at frame 18."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm_per_material import C  # noqa: E402

F = "runs/parity/isaaclab3_mpm_floors/mpm_floors"
FM = "runs/mpm"
MET = sys.argv[1] if len(sys.argv) > 1 else FM + "/fm_off/F_multi_material"


def ld(p):
    t = np.load(p + "/traj.npz")
    return t["pos"].astype(np.float64), t["vel"].astype(np.float64)


b, bv = ld(F + "/multi_material_base")
n, nv = ld(F + "/multi_material_nofma")
m, mv = ld(MET)
i0 = np.load(F + "/multi_material_base/init.npz")["q"]
s = slice(C[2], C[3])
for name, (x, xv) in {"nofma": (n, nv), "metal": (m, mv)}.items():
    d = (x - b)[0, s]
    dv = (xv - bv)[0, s]
    print(name, "snow f6 mean disp (nm)", (d.mean(0) * 1e9).round(2), "rms", np.sqrt((d ** 2).sum(-1).mean()) * 1e9,
          "vel mean (um/s)", (dv.mean(0) * 1e6).round(3), "nz", int((np.abs(d).max(-1) > 0).sum()))
    z = i0[s, 2]
    for zz in np.linspace(z.min(), z.max(), 6):
        k = np.abs(z - zz) < 0.03
        print("  z0~%.2f mean dz nm %.1f rms %.1f" % (zz, d[k, 2].mean() * 1e9, np.sqrt((d[k] ** 2).sum(-1).mean()) * 1e9))
    y = i0[s, 1]
    for yy in np.linspace(y.min(), y.max(), 6):
        k = np.abs(y - yy) < 0.05
        print("  y0~%.2f mean dz nm %.1f rms %.1f" % (yy, d[k, 2].mean() * 1e9, np.sqrt((d[k] ** 2).sum(-1).mean()) * 1e9))
s = slice(C[1], C[2])
for name, x in {"metal": m, "nofma": n}.items():
    d = np.sqrt(((x - b)[2, s] ** 2).sum(-1))
    o = np.argsort(d)[::-1][:8]
    print(name, "sand f18 top (um)", (d[o] * 1e6).round(1), "idx", (o + C[1]).tolist())
    print("   init pos", i0[o[:4] + C[1]].round(4).tolist(), "\n   base pos f18", b[2, o[:4] + C[1]].round(4).tolist())
    print("   n > 10um", int((d > 1e-5).sum()))
