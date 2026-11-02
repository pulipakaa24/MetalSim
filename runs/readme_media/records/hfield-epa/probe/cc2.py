"""Heightfield contacts of primitive and mesh geoms on the g1_bumps tile (Isaac's random_rough, seed 0): MuJoCo C vs
MuJoCo Warp (whatever copy is first on PYTHONPATH, Warp's CPU device), and for spheres and capsules the exact contact
set of the triangulated surface. Extends runs/film/g1_bumps/measure/contact_check.py (spheres only) to more shapes.

Shapes (size in MuJoCo's convention):
  sphere   r 5 mm (Unitree G1 sole sphere); the 3000 placements of contact_check.py (contacts_exact.npz)
  capsule  r 0.02, half-length 0.05            box   half-sizes (0.05, 0.03, 0.02)
  cylinder r 0.03, half-height 0.02            ellipsoid (0.04, 0.03, 0.02)
  mesh     Isaac's G1 USD foot collider ("Cube", 8 vertices, 0.0185 x 0.0655 x 0.203 m; probe/g1_foot_mesh.npz)
Placements (non-sphere, N = 1000, seed 11): x, y uniform in [-3.6, 3.6] m, uniform random orientation, z bisected
(40 steps) so that MuJoCo C's deepest contact is pen = U(0.5, 3) mm deep.

Exact set (sphere, capsule): per triangle of the cells under the geom's AABB, the closest point pair between the
triangle and the sphere centre / capsule segment; a contact if closer than r; deduplicated by triangle point (1e-6 m).

Usage (repo root): .venv/bin/python cc2.py place|c|mjw|compare <shape> <out_dir> [tag]
  mjw writes contacts_mjw_<tag>.npz; compare reads every contacts_*.npz and writes compare.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hfield-epa: probe/ lives one level down
FOOT = os.path.join(HERE, "..", "hfield-contacts", "probe", "g1_foot_mesh.npz")
sys.path.insert(0, "runs/film/g1_bumps/measure")
import contact_check as CC  # noqa: E402

N, SEED = 1000, 11
SHAPES = dict(sphere=("sphere", [0.005, 0, 0]), capsule=("capsule", [0.02, 0.05, 0]), box=("box", [0.05, 0.03, 0.02]),
              cylinder=("cylinder", [0.03, 0.02, 0]), ellipsoid=("ellipsoid", [0.04, 0.03, 0.02]), mesh=("mesh", None))


def tile():
  from metalsim.render.film.unitree import random_rough_tile
  return random_rough_tile(None, 0)


def build_model(hf, shape):
  import mujoco
  spec = mujoco.MjSpec()
  t = spec.add_hfield(); t.name = "t"; t.nrow, t.ncol = int(hf["nrow"]), int(hf["ncol"])
  t.size = [float(v) for v in hf["size"]]; t.userdata = np.asarray(hf["data"], np.float32).reshape(-1).tolist()
  g = spec.worldbody.add_geom(); g.name = "floor"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "t"
  g.pos = [0, 0, float(hf["zmin"])]
  b = spec.worldbody.add_body(); b.name = "obj"; b.pos = [0, 0, 1]; b.add_freejoint()
  kind, size = SHAPES[shape]
  s_ = b.add_geom(); s_.name = "obj"; s_.mass = 0.1
  if kind == "mesh":
    z = np.load(FOOT)
    mm = spec.add_mesh(); mm.name = "foot"; mm.uservert = z["vert"].reshape(-1).tolist()
    s_.type = mujoco.mjtGeom.mjGEOM_MESH; s_.meshname = "foot"
  else:
    s_.type = getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}"); s_.size = size
  return spec.compile()


def rand_quat(rng):
  u1, u2, u3 = rng.uniform(0, 1, 3)
  return np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                   np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])


def c_contacts(m, d, q):
  import mujoco
  d.qpos[:] = q
  mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
  n = d.ncon
  return d.contact.dist[:n].copy(), d.contact.pos[:n].copy(), d.contact.frame[:n, :3].copy()


def place(shape, out):
  import mujoco
  hf = tile(); m = build_model(hf, shape); d = mujoco.MjData(m)
  if shape == "sphere":
    C = np.load(os.path.join(HERE, "cc", "base", "contacts_exact.npz"))["centre"]
    Q = np.tile(np.array([0, 0, 0, 1, 0, 0, 0.0]), (len(C), 1)); Q[:, :3] = C
    np.savez(os.path.join(out, "placements.npz"), qpos=Q, pen=np.full(len(C), np.nan))
    return
  rng = np.random.default_rng(SEED)
  Q, P = [], []
  while len(Q) < N:
    x, y = rng.uniform(-3.6, 3.6, 2); quat = rand_quat(rng); pen = rng.uniform(0.0005, 0.003)
    q = np.concatenate([[x, y, 0.0], quat])
    lo, hi = -0.02, 0.6  # the prisms reach down to the base at -0.05 m
    depth = lambda z: (lambda dd: -dd.min() if len(dd) else -np.inf)(c_contacts(m, d, np.concatenate([[x, y, z], quat]))[0])
    if depth(hi) > -np.inf or depth(lo) < pen:
      continue
    for _ in range(40):
      mid = 0.5 * (lo + hi)
      if depth(mid) > pen:
        lo = mid
      else:
        hi = mid
    q[2] = 0.5 * (lo + hi)
    Q.append(q); P.append(pen)
  np.savez(os.path.join(out, "placements.npz"), qpos=np.array(Q), pen=np.array(P))


def run_c(shape, out):
  import mujoco
  hf = tile(); m = build_model(hf, shape); d = mujoco.MjData(m)
  Q = np.load(os.path.join(out, "placements.npz"))["qpos"]
  W, D, Pp, Nn = [], [], [], []
  for i, q in enumerate(Q):
    dist, pos, nrm = c_contacts(m, d, q)
    W += [i] * len(dist); D += list(dist); Pp += list(pos); Nn += list(nrm)
  np.savez(os.path.join(out, "contacts_c.npz"), world=np.array(W, int), dist=np.array(D), pos=np.array(Pp).reshape(-1, 3),
           normal=np.array(Nn).reshape(-1, 3))


def run_mjw(shape, out, tag):
  import mujoco
  import warp as wp
  wp.config.quiet = True
  import mujoco_warp as mjw
  import importlib
  for kv in filter(None, os.environ.get("MJW_SET", "").split(",")):  # e.g. collision_convex.HFIELD_PLANE_CONTACTS=0
    k, v = kv.split("="); mod, attr = k.rsplit(".", 1)
    setattr(importlib.import_module(f"mujoco_warp._src.{mod}"), attr, type(getattr(importlib.import_module(f"mujoco_warp._src.{mod}"), attr))(eval(v)))
    print("set", k, "=", getattr(importlib.import_module(f"mujoco_warp._src.{mod}"), attr))
  hf = tile(); m = build_model(hf, shape); d0 = mujoco.MjData(m)
  Q = np.load(os.path.join(out, "placements.npz"))["qpos"]
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m)
    dd = mjw.put_data(m, d0, nworld=len(Q), nconmax=64, njmax=256)
    dd.qpos.assign(Q.astype(np.float32))
    mjw.kinematics(mm, dd); mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0])
    W = dd.contact.worldid.numpy()[:n]; D = dd.contact.dist.numpy()[:n]; Pp = dd.contact.pos.numpy()[:n]
    Nn = dd.contact.frame.numpy()[:n, 0, :]
  np.savez(os.path.join(out, f"contacts_mjw_{tag}.npz"), world=W.astype(int), dist=D.astype(np.float64),
           pos=Pp.astype(np.float64), normal=Nn.astype(np.float64), mjw_file=mjw.__file__)
  print("mjw", os.environ.get("MJW_SET", ""), mjw.__file__, "contacts", n, "worlds with contact", len(set(W.tolist())), "of", len(Q))


# ---------------------------------------------------------------- exact sets (sphere, capsule)

def seg_closest(p0, p1, q0, q1):
  """Closest points between segments p0p1 and q0q1 (Ericson 5.1.9)."""
  d1, d2, r = p1 - p0, q1 - q0, p0 - q0
  a, e, f = d1 @ d1, d2 @ d2, d2 @ r
  c = d1 @ r; b = d1 @ d2; den = a * e - b * b
  s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-18 else 0.0
  t = (b * s + f) / e
  if t < 0:
    t, s = 0.0, np.clip(-c / a, 0, 1)
  elif t > 1:
    t, s = 1.0, np.clip((b - c) / a, 0, 1)
  return p0 + d1 * s, q0 + d2 * t


def seg_tri_closest(p0, p1, a, b, c):
  """Closest pair (point on segment, point on triangle) for a segment that does not cross the triangle."""
  if np.array_equal(p0, p1):  # sphere
    q = CC.closest_on_triangle(p0, a, b, c)
    return np.linalg.norm(p0 - q), p0, q
  best = (np.inf, None, None)
  for p in (p0, p1):
    q = CC.closest_on_triangle(p, a, b, c); dd = np.linalg.norm(p - q)
    if dd < best[0]:
      best = (dd, p, q)
  for u, v in ((a, b), (b, c), (c, a)):
    p, q = seg_closest(p0, p1, u, v); dd = np.linalg.norm(p - q)
    if dd < best[0]:
      best = (dd, p, q)
  n = np.cross(b - a, c - a); n /= np.linalg.norm(n)
  s0, s1 = (p0 - a) @ n, (p1 - a) @ n
  if s0 * s1 < 0:  # crosses the plane: check the crossing point
    x = p0 + (p1 - p0) * (s0 / (s0 - s1))
    q = CC.closest_on_triangle(x, a, b, c)
    if np.linalg.norm(q - x) < 1e-12:
      return 0.0, x, q
  return best


def quat2mat(q):
  w, x, y, z = q
  return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def exact_set(shape, D, res, x0, y0, q, tol=1e-6):
  r, hl = SHAPES[shape][1][0], (SHAPES[shape][1][1] if shape == "capsule" else 0.0)
  c = q[:3]; ax = quat2mat(q[3:])[:, 2]
  p0, p1 = c - hl * ax, c + hl * ax
  lo = np.minimum(p0, p1) - r; hi = np.maximum(p0, p1) + r
  c0, c1 = int(np.floor((lo[0] - x0) / res)), int(np.floor((hi[0] - x0) / res))
  r0, r1 = int(np.floor((lo[1] - y0) / res)), int(np.floor((hi[1] - y0) / res))
  pts = []
  for rr in range(r0, r1 + 1):
    for cc in range(c0, c1 + 1):
      P = lambda a_, k: np.array([x0 + k * res, y0 + a_ * res, D[a_, k]])
      for t in ((P(rr, cc), P(rr, cc + 1), P(rr + 1, cc + 1)), (P(rr, cc), P(rr + 1, cc + 1), P(rr + 1, cc))):
        dd, ps, qt = seg_tri_closest(p0, p1, *t)
        if dd < r and not any(np.linalg.norm(qt - q2) < tol for q2, _, _ in pts):
          pts.append((qt, ps, dd))
  return [(r - dd, (ps - qt) / dd, qt) for qt, ps, dd in pts]


def run_exact(shape, out):
  hf = tile(); D, res, x0, y0 = CC.surface(hf)
  Q = np.load(os.path.join(out, "placements.npz"))["qpos"]
  W, Dd, Pp, Nn = [], [], [], []
  for i, q in enumerate(Q):
    for dep, n, p in exact_set(shape, D, res, x0, y0, q):
      W.append(i); Dd.append(-dep); Nn.append(n); Pp.append(p)
  np.savez(os.path.join(out, "contacts_exact.npz"), world=np.array(W, int), dist=np.array(Dd), pos=np.array(Pp).reshape(-1, 3),
           normal=np.array(Nn).reshape(-1, 3))


def sat(A, B, axes):
  """Minimum translation between convex point sets A (geom) and B (prism) over the candidate axes: (depth, normal from
  B to A), or (0, None) if separated along some axis."""
  best = (np.inf, None)
  for u in axes:
    n = np.linalg.norm(u)
    if n < 1e-12:
      continue
    u = u / n; a, b = A @ u, B @ u
    o1, o2 = b.max() - a.min(), a.max() - b.min()   # move A along +u by o1, or along -u by o2
    if o1 <= 0 or o2 <= 0:
      return 0.0, None
    if o1 < best[0]:
      best = (o1, u)
    if o2 < best[0]:
      best = (o2, -u)
  return best


def run_exact_prism(shape, out):
  """Box / mesh: per heightfield prism (triangle column down to the base), the exact penetration depth and direction
  by the separating axis test (face normals of both, cross products of edge directions); the semantics of MuJoCo C's
  mjc_ConvexHField (one contact per prism) computed exactly. Contact point: the prism's top centroid (no dedup)."""
  import mujoco
  hf = tile(); D, res, x0, y0 = CC.surface(hf); base = hf["zmin"] - hf["size"][3]
  m = build_model(hf, shape); d = mujoco.MjData(m)
  Q = np.load(os.path.join(out, "placements.npz"))["qpos"]
  g = 1
  if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
    va, vn = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
    V0 = m.mesh_vert[va:va + vn].astype(np.float64)
    if vn != 8:
      raise SystemExit("exactprism: box-like meshes only (edge directions = local axes)")
  else:
    import itertools
    V0 = np.array([m.geom_size[g] * np.array(s_) for s_ in itertools.product((-1, 1), repeat=3)])
  W, Dd, Pp, Nn = [], [], [], []
  z = np.array([0, 0, 1.0])
  for i, q in enumerate(Q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    R = d.geom_xmat[g].reshape(3, 3); A = V0 @ R.T + d.geom_xpos[g]
    ea = [R[:, k] for k in range(3)]
    lo, hi = A.min(0), A.max(0)
    for r in range(int(np.floor((lo[1] - y0) / res)), int(np.floor((hi[1] - y0) / res)) + 1):
      for cc in range(int(np.floor((lo[0] - x0) / res)), int(np.floor((hi[0] - x0) / res)) + 1):
        P = lambda a_, k: np.array([x0 + k * res, y0 + a_ * res, D[a_, k]])
        for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
          top = np.array(t); bot = top.copy(); bot[:, 2] = base; B = np.vstack([bot, top])
          te = [top[1] - top[0], top[2] - top[1], top[0] - top[2]]
          be = [e * np.array([1, 1, 0]) for e in te]
          axes = ea + [np.cross(te[0], te[1]), z] + [np.cross(e, z) for e in be]
          axes += [np.cross(x, y) for x in ea for y in te + be + [z]]
          dep, n = sat(A, B, axes)
          if dep > 0:
            W.append(i); Dd.append(-dep); Nn.append(n); Pp.append(top.mean(0))
  np.savez(os.path.join(out, "contacts_exact.npz"), world=np.array(W, int), dist=np.array(Dd), pos=np.array(Pp).reshape(-1, 3),
           normal=np.array(Nn).reshape(-1, 3))


# ---------------------------------------------------------------- compare

def per_world(z, n):
  out = [[] for _ in range(n)]
  for w, d, p, nn in zip(z["world"], z["dist"], z["pos"], z["normal"]):
    out[w].append((float(d), np.asarray(p), np.asarray(nn)))
  return out


def distinct(cs, ptol=float(os.environ.get("CC2_PTOL", 1e-4)), ntol_deg=float(os.environ.get("CC2_NTOL", 0.05)),
             by_depth=False):
  """Contacts with duplicates removed (same point within ptol and normal within ntol_deg; by_depth: same depth within
  1e-6 m and normal, for references without witness points), deepest first."""
  out = []
  cn = np.cos(np.radians(ntol_deg))
  for c in sorted(cs, key=lambda x: x[0]):
    if by_depth:
      dup = any(abs(c[0] - o[0]) < 1e-6 and c[2] @ o[2] > cn for o in out)
    else:
      dup = any(np.linalg.norm(c[1] - o[1]) < ptol and c[2] @ o[2] > cn for o in out)
    if not dup:
      out.append(c)
  return out


def compare_sets(E, R, cap=4, by_depth=False):
  """E, R: per-world contact lists. Returns metrics of E against reference R."""
  n = len(R)
  cnt_eq, fewer, more, no_c, miss_ref = 0, 0, 0, 0, 0
  nerr, derr, werr, cover = [], [], [], []
  for e, r in zip(E, R):
    rd = distinct(r, by_depth=by_depth) if r else []
    k = min(len(rd), cap)
    if len(e) == k:
      cnt_eq += 1
    elif len(e) < k:
      fewer += 1
    else:
      more += 1
    if r and not e:
      no_c += 1
    if not r or not e:
      continue
    for c in e:
      nerr.append(min(np.degrees(np.arccos(np.clip(c[2] @ o[2], -1, 1))) for o in r))
    derr.append(abs(min(c[0] for c in e) - min(c[0] for c in r)) * 1e3)
    ws = lambda cs: (lambda v: v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v)(sum(max(-c[0], 0) * c[2] for c in cs))
    a, b = ws(e), ws(rd)
    if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
      werr.append(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
    # coverage: reference contacts (deepest `cap`, distinct) matched by an engine contact within 10 deg
    top = rd[:cap]
    cover.append(np.mean([any(np.degrees(np.arccos(np.clip(c[2] @ o[2], -1, 1))) < 10 for c in e) for o in top]))
  nerr, derr, werr = np.array(nerr), np.array(derr), np.array(werr)
  pc = lambda a, qs: [round(float(np.percentile(a, q)), 2) for q in qs] if len(a) else None
  return dict(worlds=n, count_equal_min_ref_cap=round(cnt_eq / n, 4), fewer=fewer, more=more, no_contact=no_c,
              contacts=int(sum(len(e) for e in E)), normal_err_deg_p50_p95_p99=pc(nerr, (50, 95, 99)),
              share_normal_over_10deg=round(float((nerr > 10).mean()), 4) if len(nerr) else None,
              deepest_depth_err_mm_p50_p95_max=pc(derr, (50, 95, 100)), weighted_normal_err_deg_p95=pc(werr, (95,)),
              coverage_of_ref_mean=round(float(np.mean(cover)), 4) if cover else None)


def compare(shape, out):
  Q = np.load(os.path.join(out, "placements.npz"))["qpos"]; n = len(Q)
  sets = {f[len("contacts_"):-4]: per_world(np.load(os.path.join(out, f)), n)
          for f in sorted(os.listdir(out)) if f.startswith("contacts_") and f.endswith(".npz")}
  res = dict(shape=shape, size=SHAPES[shape][1], placements=n, engines={})
  refs = [r for r in ("exact", "c") if r in sets]
  for ref in refs:
    bd = ref == "exact" and shape in ("box", "mesh")   # SAT reference: no witness points
    rd = [distinct(r, by_depth=bd) for r in sets[ref]]
    k = np.array([len(x) for x in rd])
    res[f"distinct_contacts_per_world_{ref}"] = {str(v): int((k == v).sum()) for v in sorted(set(k.tolist()))}
    for name, E in sets.items():
      if name == ref:
        continue
      res["engines"].setdefault(name, {})[f"vs_{ref}"] = compare_sets(E, sets[ref], by_depth=bd)
  json.dump(res, open(os.path.join(out, "compare.json"), "w"), indent=1)
  print(json.dumps(res, indent=1))


def main():
  mode, shape, out = sys.argv[1], sys.argv[2], sys.argv[3]
  os.makedirs(out, exist_ok=True)
  if mode == "place":
    place(shape, out)
  elif mode == "c":
    run_c(shape, out)
  elif mode == "exact":
    run_exact(shape, out)
  elif mode == "exactprism":
    run_exact_prism(shape, out)
  elif mode == "mjw":
    run_mjw(shape, out, sys.argv[4])
  elif mode == "compare":
    compare(shape, out)


if __name__ == "__main__":
  main()
