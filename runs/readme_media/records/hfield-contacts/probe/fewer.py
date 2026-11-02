"""List spheres where MJW has fewer contacts than the exact set (contact_check.py outputs in <dir>)."""
import sys, numpy as np
sys.path.insert(0, "runs/film/g1_bumps/measure")
import contact_check as CC
from metalsim.render.film.unitree import random_rough_tile
out = sys.argv[1]; eng = sys.argv[2] if len(sys.argv) > 2 else "plane"
D, res, x0, y0 = CC.surface(random_rough_tile(None, 0))
C = np.load(f"{out}/contacts_exact.npz")["centre"]
z = np.load(f"{out}/contacts_{eng}.npz")
cnt = np.bincount(z["world"], minlength=len(C))
bad_n = []
for w, n in zip(z["world"], z["normal"]):
  E = CC.exact_set(D, res, x0, y0, C[w])
  e = min(np.degrees(np.arccos(np.clip(n @ en, -1, 1))) for _, en in E)
  if e > 10: bad_n.append((w, e))
print("normal > 10 deg:", bad_n)
for w in bad_n:
  E = CC.exact_set(D, res, x0, y0, C[w[0]]); m = z["world"] == w[0]
  print(" exact", [(round(d * 1e3, 3), n.round(3)) for d, n in E], "engine", list(zip((-z["dist"][m] * 1e3).round(3), z["normal"][m].round(3))))
k = 0
for i, c in enumerate(C):
  E = CC.exact_set(D, res, x0, y0, c)
  if cnt[i] < min(len(E), 4):
    k += 1
    # exact contact points and pairwise separations / angles
    ci, ri = int(np.floor((c[0] - x0) / res)), int(np.floor((c[1] - y0) / res))
    m = z["world"] == i
    print(i, "true", len(E), "engine", cnt[i], "depths", [round(d * 1e3, 3) for d, _ in E], "angles to first",
          [round(float(np.degrees(np.arccos(np.clip(E[0][1] @ n, -1, 1)))), 2) for _, n in E],
          "engine depths", (-z["dist"][m] * 1e3).round(3))
print("fewer (true <= 4):", k)
