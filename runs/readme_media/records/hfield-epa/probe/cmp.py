"""Compare two epa_train.py / epa_repro.py runs world by world (stream hfield-epa).
Usage: python cmp.py <dir> <tagA> <tagB> [nonloop_log]
Prints: worlds whose contact sets are bitwise identical, differing worlds (and, if a dbg log with HZN lines is given,
how many of them had a non-loop horizon in the instrumented base run), contact count changes."""
import re
import sys

import numpy as np


def per_world(z, nw):
  W = z["world"]; o = np.lexsort((z["pos"][:, 2], z["pos"][:, 1], z["pos"][:, 0], z["dist"], W))
  rec = np.concatenate([z["dist"][o, None], z["pos"][o], z["normal"][o]], 1).astype(np.float32)
  out = [rec[W[o] == w] for w in range(nw)]
  return out


d, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
import os
ld = lambda t: np.load(f"{d}/run_{t}.npz") if os.path.exists(f"{d}/run_{t}.npz") else np.load(f"{d}/contacts_{t}.npz")
A, B = ld(a), ld(b)
nw = len(A["overflow"]) if "overflow" in A else len(np.load(f"{d}/placements.npz")["qpos"])
pa, pb = per_world(A, nw), per_world(B, nw)
same = np.array([x.shape == y.shape and np.array_equal(x.view(np.uint32), y.view(np.uint32)) for x, y in zip(pa, pb)])
diff = np.flatnonzero(~same)
print(f"{d}: {a} vs {b}: worlds {nw}, bitwise identical {same.sum()}, differ {len(diff)}; contacts {len(A['dist'])} vs {len(B['dist'])}")
if "overflow" in A:
  print("  overflow worlds:", int((A["overflow"] != 0).sum()), "vs", int((B["overflow"] != 0).sum()))
if len(sys.argv) > 4:
  bad = set(int(m.group(1)) for m in re.finditer(r"HZN world (\d+)", open(sys.argv[4]).read()))
  print(f"  differing worlds with a non-loop horizon in {sys.argv[4]}: {len(set(diff) & bad)} of {len(diff)} (non-loop worlds {len(bad)})")
for w in diff[:8]:
  x, y = pa[w], pb[w]
  print(f"  world {w}: n {len(x)} -> {len(y)}; min dist {x[:, 0].min() if len(x) else None} -> {y[:, 0].min() if len(y) else None}")
