"""Exact box-vs-prism penetration (separating axis test, float64) for the prisms under one box placement of cc2's box
set; compares with MuJoCo C's and MuJoCo Warp's contacts. Usage: sat_box.py <cc dir> <tag> <world> [<world> ...]"""
import sys, itertools, numpy as np
sys.path.insert(0, "runs/readme_media/scratch/hfield-contacts"); sys.path.insert(0, "runs/film/g1_bumps/measure")
import cc2, contact_check as CC
d, tag, worlds = sys.argv[1], sys.argv[2], [int(w) for w in sys.argv[3:]]
hf = cc2.tile(); D, res, x0, y0 = CC.surface(hf); base = -hf["size"][3] + hf["zmin"]
Q = np.load(f"{d}/placements.npz")["qpos"]
hs = np.array(cc2.SHAPES["box"][1])
def sat(A, B, axes):
  best = (np.inf, None)
  for u in axes:
    n = np.linalg.norm(u)
    if n < 1e-9: continue
    u = u / n; a, b = A @ u, B @ u
    o1, o2 = a.max() - b.min(), b.max() - a.min()   # o1: push A along -u ... overlap either way
    if o1 <= 0 or o2 <= 0: return 0.0, None
    if o1 < best[0]: best = (o1, -u)   # move A by -u*o1 (B's normal toward A is -u)
    if o2 < best[0]: best = (o2, u)
  return best
for w in worlds:
  q = Q[w]; R = cc2.quat2mat(q[3:]); c = q[:3]
  A = np.array([c + R @ (hs * np.array(s)) for s in itertools.product((-1, 1), repeat=3)])
  ea = [R[:, i] for i in range(3)]
  lo, hi = A.min(0), A.max(0)
  out = []
  for r in range(int(np.floor((lo[1] - y0) / res)), int(np.floor((hi[1] - y0) / res)) + 1):
    for cc in range(int(np.floor((lo[0] - x0) / res)), int(np.floor((hi[0] - x0) / res)) + 1):
      P = lambda a_, k: np.array([x0 + k * res, y0 + a_ * res, D[a_, k]])
      for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
        top = np.array(t); bot = top.copy(); bot[:, 2] = base; B = np.vstack([bot, top])
        te = [top[1] - top[0], top[2] - top[1], top[0] - top[2]]
        be = [e * np.array([1, 1, 0]) for e in te]
        axes = ea + [np.cross(te[0], te[1]), np.array([0, 0, 1.0])] + [np.cross(e, [0, 0, 1.0]) for e in be]
        axes += [np.cross(x, y) for x in ea for y in te + be + [np.array([0, 0, 1.0])]]
        dep, n = sat(A, B, axes)
        if dep > 0: out.append((round(dep * 1e3, 3), np.round(n, 2).tolist()))
  out.sort(key=lambda t: -t[0])
  print(f"world {w}: exact per-prism penetrations (mm, normal from terrain to box):", out)
  for e in ("c", f"mjw_{tag}"):
    z = np.load(f"{d}/contacts_{e}.npz"); m = z["world"] == w
    print(f"  {e:10s}", sorted([(round(-x * 1e3, 3), np.round(nn, 2).tolist()) for x, nn in zip(z["dist"][m], z["normal"][m])], key=lambda t: -t[0]))
