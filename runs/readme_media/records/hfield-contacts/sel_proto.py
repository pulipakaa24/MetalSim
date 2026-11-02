"""Prototype: how MuJoCo Warp's 4-contact heightfield selection treats the exact sphere contact sets of the g1_bumps
contact check (3000 x 5 mm spheres). Candidates = the exact set (per-triangle closest points within the sphere,
deduplicated by position). Rules: upstream (MIN_DIST 1 mm, contact 2/3 tested on m^2 quantities) and variants.

Usage: .venv/bin/python runs/readme_media/scratch/hfield-contacts/sel_proto.py
"""
import sys

import numpy as np

sys.path.insert(0, "runs/film/g1_bumps/measure")
import contact_check as CC  # noqa: E402
from metalsim.render.film.unitree import random_rough_tile  # noqa: E402

R = CC.R


def exact_set_pts(D, res, x0, y0, c, tol=1e-6):
  ci, ri = int(np.floor((c[0] - x0) / res)), int(np.floor((c[1] - y0) / res))
  pts = []
  for r in range(ri - 1, ri + 3):
    for cc in range(ci - 1, ci + 3):
      P = lambda rr, k: np.array([x0 + k * res, y0 + rr * res, D[rr, k]])
      for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
        q = CC.closest_on_triangle(c, *t)
        d = np.linalg.norm(c - q)
        if d < R and not any(np.linalg.norm(q - q2) < tol for q2, _ in pts):
          pts.append((q, d))
  out = []
  for q, d in pts:
    n = (c - q) / d
    dist = d - R
    out.append((dist, q + 0.5 * dist * n, n))
  return out


def select(cands, min_dist=1e-3, mode="upstream"):
  """Return the indices written by the selection (contact 0..3)."""
  if not cands:
    return []
  dist = np.array([c[0] for c in cands]); pos = np.array([c[1] for c in cands]); nrm = np.array([c[2] for c in cands])
  i0 = int(np.argmin(dist)); out = [i0]
  p0, n0 = pos[i0], nrm[i0]
  rest = [i for i in range(len(cands)) if i != i0]
  if not rest:
    return out
  d1 = [np.linalg.norm(pos[i] - p0) for i in rest]
  k = int(np.argmax(d1)); i1, dist1 = rest[k], d1[k]
  if mode == "upstream":
    if 0.0 < dist1 < min_dist:
      return out
  else:
    if dist1 < min_dist:
      return out
  out.append(i1); p1 = pos[i1]
  rest = [i for i in rest if i != i1]
  if not rest:
    return out
  v = np.cross(n0, p0 - p1)
  d2 = [abs((pos[i] - p0) @ v) for i in rest]
  k = int(np.argmax(d2)); i2, dist2 = rest[k], d2[k]
  if mode == "upstream":
    if 0.0 < dist2 < min_dist:
      return out
  else:
    if dist2 / max(np.linalg.norm(p0 - p1), 1e-12) < min_dist:
      return out
  out.append(i2); p2 = pos[i2]
  rest = [i for i in rest if i != i2]
  if not rest:
    return out
  vm2 = np.cross(n0, p0 - p2); v12 = np.cross(n0, p1 - p2)
  d3 = [abs((pos[i] - p0) @ vm2) + abs((p1 - pos[i]) @ v12) for i in rest]
  k = int(np.argmax(d3)); i3, dist3 = rest[k], d3[k]
  if mode == "upstream":
    if 0.0 < dist3 < min_dist:
      return out
  else:
    dmin = min(np.linalg.norm(pos[i3] - pos[j]) for j in out)
    if dmin < min_dist:
      return out
  out.append(i3)
  return out


def main():
  hf = random_rough_tile(None, 0)
  D, res, x0, y0 = CC.surface(hf)
  C = np.load("runs/readme_media/scratch/g1-bumps/measure_npz/contact_check/contacts_exact.npz")["centre"]
  E = [exact_set_pts(D, res, x0, y0, c) for c in C]
  ne = np.array([len(e) for e in E])
  print("true counts", {k: int((ne == k).sum()) for k in sorted(set(ne.tolist()))})
  # pairwise separations of contact positions in 2-contact sets
  sep = np.array([np.linalg.norm(e[0][1] - e[1][1]) for e in E if len(e) == 2])
  ang = np.array([np.degrees(np.arccos(np.clip(e[0][2] @ e[1][2], -1, 1))) for e in E if len(e) == 2])
  print("2-contact sets: separation mm p5/p50/p95", np.percentile(sep * 1e3, [5, 50, 95]).round(3),
        "share < 1 mm", round(float((sep < 1e-3).mean()), 3), "angle deg p5/p50", np.percentile(ang, [5, 50]).round(2))
  nz = np.array([n[2] for e in E for (_, _, n) in e])
  print("exact normals with n_z <= 0:", int((nz <= 0).sum()), "min n_z", nz.min().round(3))
  for mode, md in (("upstream", 1e-3), ("fixed", 1e-3), ("fixed", 1e-4), ("fixed", 1e-5), ("fixed", 1e-6)):
    k = np.array([len(select(e, md, mode)) for e in E])
    eq = (k == ne)
    print(f"{mode:9s} min_dist {md:g}: count equal {eq.mean():.4f} ({eq.sum()}), fewer {(k < ne).sum()}")


if __name__ == "__main__":
  main()


def select_dup(cands, min_dist=1e-3, cos_dup=np.cos(np.radians(2.0))):
  """Selection with a duplicate test on position and normal: candidate j duplicates a selected i if
  |p_j - p_i| < min_dist and n_j . n_i > cos_dup."""
  if not cands:
    return []
  dist = np.array([c[0] for c in cands]); pos = np.array([c[1] for c in cands]); nrm = np.array([c[2] for c in cands])
  out = [int(np.argmin(dist))]
  dup = lambda j: any(np.linalg.norm(pos[j] - pos[i]) < min_dist and nrm[j] @ nrm[i] > cos_dup for i in out)
  for stage in (1, 2, 3):
    rest = [j for j in range(len(cands)) if j not in out and not dup(j)]
    if not rest:
      break
    p0, n0 = pos[out[0]], nrm[out[0]]
    if stage == 1:
      sc = [np.linalg.norm(pos[j] - p0) for j in rest]
    elif stage == 2:
      v = np.cross(n0, p0 - pos[out[1]]); sc = [abs((pos[j] - p0) @ v) for j in rest]
    else:
      p1, p2 = pos[out[1]], pos[out[2]]
      vm2 = np.cross(n0, p0 - p2); v12 = np.cross(n0, p1 - p2)
      sc = [abs((pos[j] - p0) @ vm2) + abs((p1 - pos[j]) @ v12) for j in rest]
    out.append(rest[int(np.argmax(sc))])
  return out


def main2():
  hf = random_rough_tile(None, 0)
  D, res, x0, y0 = CC.surface(hf)
  C = np.load("runs/readme_media/scratch/g1-bumps/measure_npz/contact_check/contacts_exact.npz")["centre"]
  E = [exact_set_pts(D, res, x0, y0, c) for c in C]
  ne = np.array([len(e) for e in E])
  for th in (0.5, 1.0, 2.0, 5.0):
    k = np.array([len(select_dup(e, 1e-3, np.cos(np.radians(th)))) for e in E])
    print(f"dup = within 1 mm and normals within {th} deg: count equal {(k == ne).mean():.4f} ({(k == ne).sum()}), "
          f"vs min(true,4) {(k == np.minimum(ne, 4)).mean():.4f}")
  # angles between true contacts closer than 1 mm
  a = []
  for e in E:
    for i in range(len(e)):
      for j in range(i + 1, len(e)):
        if np.linalg.norm(e[i][1] - e[j][1]) < 1e-3:
          a.append(np.degrees(np.arccos(np.clip(e[i][2] @ e[j][2], -1, 1))))
  a = np.array(a)
  print("true contact pairs < 1 mm apart:", len(a), "angle deg min/p5/p50", a.min().round(3), np.percentile(a, [5, 50]).round(2))


if __name__ == "__main__":
  main2()
