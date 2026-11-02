"""Minimal reproduction of MuJoCo Warp's EPA_HORIZON overflow on heightfields (stream hfield-epa, 2026-10-31).

Isaac's G1 foot collider (8-vertex box mesh, hfield-contacts/probe/g1_foot_mesh.npz) on Isaac's random rough tile
(seed 0, the g1_bumps tile), with rough_il3's per-geom gap 0.01 (solver preset _IL3: pair gap 0.02, margin 0), on
Warp's CPU device, one collision call over N worlds (one placement per world).

Placements (seed S): x, y uniform in [-3.6, 3.6]; orientation: half uniform random, half "sole down" (local +x to world
-z, random yaw, tilt N(0, 10 deg)); height: MuJoCo C (gap 0) bisected to touching (deepest contact 0.01 mm), then
shifted by a clearance uniform in [-CLO, CHI] (negative = penetration).

Usage (repo root): PYTHONPATH=<mjw copy> .venv/bin/python probe/epa_repro.py place <out> [N] [seed] [clo] [chi]
                   PYTHONPATH=<mjw copy> .venv/bin/python probe/epa_repro.py run <out> <tag> [gap]
  run writes <out>/run_<tag>.npz (per world overflow bits; contacts) and prints the overflow count.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOOT = os.path.join(HERE, "..", "..", "hfield-contacts", "probe", "g1_foot_mesh.npz")


def tile():
  from metalsim.render.film.unitree import random_rough_tile
  return random_rough_tile(None, 0)


def build_model(hf, gap=0.0):
  import mujoco
  spec = mujoco.MjSpec()
  t = spec.add_hfield(); t.name = "t"; t.nrow, t.ncol = int(hf["nrow"]), int(hf["ncol"])
  t.size = [float(v) for v in hf["size"]]; t.userdata = np.asarray(hf["data"], np.float32).reshape(-1).tolist()
  g = spec.worldbody.add_geom(); g.name = "floor"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "t"
  g.pos = [0, 0, float(hf["zmin"])]; g.gap = gap
  b = spec.worldbody.add_body(); b.name = "obj"; b.pos = [0, 0, 1]; b.add_freejoint()
  s_ = b.add_geom(); s_.name = "obj"; s_.mass = 0.1; s_.gap = gap
  z = np.load(FOOT)
  mm = spec.add_mesh(); mm.name = "foot"; mm.uservert = z["vert"].reshape(-1).tolist()
  s_.type = mujoco.mjtGeom.mjGEOM_MESH; s_.meshname = "foot"
  return spec.compile()


def rand_quat(rng):
  u1, u2, u3 = rng.uniform(0, 1, 3)
  return np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                   np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])


def qmul(a, b):
  w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def axang(ax, a):
  ax = np.asarray(ax, float); ax = ax / np.linalg.norm(ax)
  return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * ax])


def sole_down(rng):
  q = axang([0, 1, 0], np.pi / 2)  # local +x -> world -z
  q = qmul(axang([0, 0, 1], rng.uniform(0, 2 * np.pi)), q)
  tilt = axang([*rng.normal(size=2), 0], np.deg2rad(rng.normal(0, 10)))
  return qmul(tilt, q)


def c_depth(m, d, q):
  import mujoco
  d.qpos[:] = q
  mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
  return -d.contact.dist[:d.ncon].min() if d.ncon else -np.inf


def place(out, n=2000, seed=0, clo=0.02, chi=0.025):
  import mujoco
  os.makedirs(out, exist_ok=True)
  hf = tile(); m = build_model(hf, 0.0); d = mujoco.MjData(m)
  rng = np.random.default_rng(seed)
  Q, C, K = [], [], []
  while len(Q) < n:
    x, y = rng.uniform(-3.6, 3.6, 2)
    kind = len(Q) % 2
    quat = sole_down(rng) if kind else rand_quat(rng)
    lo, hi = -0.02, 0.6
    dep = lambda z: c_depth(m, d, np.concatenate([[x, y, z], quat]))
    if dep(hi) > -np.inf or dep(lo) < 1e-5:
      continue
    for _ in range(40):
      mid = 0.5 * (lo + hi)
      if dep(mid) > 1e-5:
        lo = mid
      else:
        hi = mid
    c = rng.uniform(-clo, chi)
    Q.append(np.concatenate([[x, y, 0.5 * (lo + hi) + c], quat])); C.append(c); K.append(kind)
  np.savez(os.path.join(out, "placements.npz"), qpos=np.array(Q), clearance=np.array(C), sole_down=np.array(K))
  print("placed", n, "->", out)


def run(out, tag, gap=0.01):
  import mujoco
  import warp as wp
  wp.config.quiet = True
  import mujoco_warp as mjw
  print("mujoco_warp", mjw.__file__)
  if os.environ.get("HZN"):  # runtime horizon size (the bound is pt.horizon.shape[0]); probes the size distribution
    import mujoco_warp._src.collision_convex as CCV
    CCV.MJ_MAX_EPAHORIZON = int(os.environ["HZN"]); print("horizon buffer", CCV.MJ_MAX_EPAHORIZON)
  hf = tile(); m = build_model(hf, gap); d0 = mujoco.MjData(m)
  Z = np.load(os.path.join(out, "placements.npz")); Q = Z["qpos"]
  print("ccd_iterations", m.opt.ccd_iterations, "ccd_tolerance", m.opt.ccd_tolerance, "gap per geom", gap)
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m)
    mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, d0, nworld=len(Q), nconmax=16, njmax=64)
    dd.qpos.assign(Q.astype(np.float32))
    mjw.kinematics(mm, dd); mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0])
    ov = dd.overflow.numpy().copy()
    W = dd.contact.worldid.numpy()[:n]; D = dd.contact.dist.numpy()[:n]; P = dd.contact.pos.numpy()[:n]
    N = dd.contact.frame.numpy()[:n, 0, :]
  epa = (ov & int(mjw.OverflowType.EPA_HORIZON)) != 0
  np.savez(os.path.join(out, f"run_{tag}.npz"), overflow=ov, world=W, dist=D, pos=P, normal=N, mjw_file=mjw.__file__)
  cl = Z["clearance"]
  print(f"[{tag}] worlds {len(Q)} contacts {n} EPA_HORIZON worlds {int(epa.sum())} "
        f"(sole-down {int(epa[Z['sole_down'] == 1].sum())}, random {int(epa[Z['sole_down'] == 0].sum())}); "
        f"other overflow bits {sorted(set(int(v) & ~int(mjw.OverflowType.EPA_HORIZON) for v in ov) - {0})}")
  if epa.any():
    print("  overflow worlds:", np.flatnonzero(epa)[:20].tolist(), "clearance mm:", np.round(cl[epa][:20] * 1e3, 2).tolist())


if __name__ == "__main__":
  cmd, out = sys.argv[1], sys.argv[2]
  if cmd == "place":
    a = sys.argv[3:]
    place(out, *(int(a[0]) if i == 0 else int(a[1]) if i == 1 else float(a[i]) for i in range(len(a))))
  elif cmd == "run":
    run(out, sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.01)
