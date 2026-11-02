"""Worst box/mesh placements: C's contacts vs MuJoCo Warp's (depth mm, normal), cc2 outputs. Usage: box_cases.py <dir> <tag> [k]"""
import sys, numpy as np
d, tag = sys.argv[1], sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 6
c = np.load(f"{d}/contacts_c.npz"); e = np.load(f"{d}/contacts_mjw_{tag}.npz")
n = len(np.load(f"{d}/placements.npz")["qpos"])
def per(z):
  out = [[] for _ in range(n)]
  for w, dd, p, nn in zip(z["world"], z["dist"], z["pos"], z["normal"]): out[w].append((-dd * 1e3, p, nn))
  return out
C, E = per(c), per(e)
bad = []
for w in range(n):
  errs = [min(np.degrees(np.arccos(np.clip(x[2] @ y[2], -1, 1))) for y in C[w]) for x in E[w]]
  if errs and max(errs) > 10: bad.append((max(errs), w))
bad.sort(reverse=True)
print("worlds with an MJW normal > 10 deg from every C normal:", len(bad))
for a, w in bad[:k]:
  print(f"world {w}: worst {a:.1f} deg")
  print("  C  ", [(round(x[0], 2), x[2].round(2).tolist()) for x in sorted(C[w], key=lambda t: -t[0])])
  print("  MJW", [(round(x[0], 2), x[2].round(2).tolist()) for x in sorted(E[w], key=lambda t: -t[0])])
