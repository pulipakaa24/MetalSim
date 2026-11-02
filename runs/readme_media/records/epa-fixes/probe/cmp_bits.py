"""Bitwise comparison of two contact recordings (stream epa-fixes).

Usage: python cmp_bits.py <a.npz> <b.npz> [nworld]
Handles epa_train.py / epa_repro.py / boxmesh.py run_<tag>.npz, cc2.py contacts_mjw_<tag>.npz (keys world, dist, pos,
normal[, overflow]) and replay_cap.py output (the same keys prefixed by batch name, plus geom). Per world, the contacts
are sorted (world, dist, pos, normal as float32 bit patterns) and compared bit for bit, together with the overflow bits.
Prints one line per batch: worlds, contacts a/b, worlds whose contact sets differ, worlds whose overflow differs; then
details for up to 8 differing worlds. Exit 1 if anything differs."""
import sys

import numpy as np


def recs(z, p, nw):
  w = z[p + "world"].astype(np.int64)
  cols = [z[p + "dist"].astype(np.float32)[:, None], z[p + "pos"].astype(np.float32), z[p + "normal"].astype(np.float32)]
  r = np.concatenate(cols, 1).view(np.uint32).astype(np.int64)
  if p + "geom" in z.files:
    r = np.concatenate([z[p + "geom"].astype(np.int64).reshape(len(w), -1), r], 1)
  o = np.lexsort(tuple(r[:, i] for i in range(r.shape[1] - 1, -1, -1)) + (w,))
  w, r = w[o], r[o]
  cut = np.searchsorted(w, np.arange(nw + 1))
  return [r[cut[i]:cut[i + 1]] for i in range(nw)]


a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
prefixes = [f"{n}_" for n in a["names"]] if "names" in a.files else [""]
bad = False
for p in prefixes:
  if p + "overflow" in a.files:
    nw = len(a[p + "overflow"])
  else:
    nw = int(sys.argv[3])
  ra, rb = recs(a, p, nw), recs(b, p, nw)
  diff = [i for i in range(nw) if ra[i].shape != rb[i].shape or not np.array_equal(ra[i], rb[i])]
  ovd = []
  if p + "overflow" in a.files:
    ovd = np.flatnonzero(a[p + "overflow"] != b[p + "overflow"]).tolist()
  print(f"{p or 'all'}: worlds {nw} contacts {len(a[p + 'world'])} / {len(b[p + 'world'])} "
        f"contact sets differ {len(diff)} overflow differs {len(ovd)}")
  for i in diff[:8]:
    da = np.sort(ra[i][:, -7].astype(np.uint32).view(np.float32)) if len(ra[i]) else []
    db = np.sort(rb[i][:, -7].astype(np.uint32).view(np.float32)) if len(rb[i]) else []
    print(f"  world {i}: n {len(ra[i])} -> {len(rb[i])}; dists {list(da)} -> {list(db)}")
  bad |= bool(diff) or bool(ovd)
sys.exit(1 if bad else 0)
