"""EPA_HORIZON reproduction on the G1 rough task's own terrain (stream hfield-epa, 2026-10-31).

Isaac's rough terrain for env seed 0 as the training's 0.1 m heightfield (isaac_rough_terrain(seed=0, collision="hfield")),
one of the G1's terrain colliders (foot = left_ankle_roll_link "Cube", 8-vertex box mesh; torso = torso_link_Cube) as a
free mesh body, per-geom gap GAP (rough_il3: 0.01 per geom, pair 0.02; margin 0), Warp's CPU device, one collision call
over N worlds (one placement per world).

Placements (seed S): x, y uniform over the sub-terrain area (border excluded); orientation: half uniform random, half
"sole down" (local +x to world -z for the foot / upright for the torso, random yaw, tilt N(0, 10 deg)); height: MuJoCo C
(gap 0) bisected to touching (deepest contact 0.01 mm), then shifted by a clearance uniform in [-CLO, CHI].

Usage: python epa_train.py place <out> <foot|torso> [N] [seed] [clo] [chi]
       PYTHONPATH=<mjw> python epa_train.py run <out> <tag> [gap] [hzn] [worlds]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epa_repro import rand_quat, qmul, axang  # noqa: E402

_HF = None


def terrain():
  global _HF
  if _HF is None:
    from metalsim.learn.terrain import isaac_rough_terrain
    _HF = isaac_rough_terrain(seed=0, collision="hfield")
  return _HF


def collider(name):
  import mujoco
  from metalsim.learn.g1_velocity import build_g1_model
  m, _ = build_g1_model("flat")
  g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, {"foot": "Cube", "torso": "torso_link_Cube"}[name])
  mid = m.geom_dataid[g]
  return m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()


def build_model(hf, verts, gap=0.0):
  import mujoco
  spec = mujoco.MjSpec()
  t = spec.add_hfield(); t.name = "t"; t.nrow, t.ncol = int(hf["nrow"]), int(hf["ncol"])
  t.size = [float(v) for v in hf["size"]]; t.userdata = np.asarray(hf["data"], np.float32).reshape(-1).tolist()
  g = spec.worldbody.add_geom(); g.name = "floor"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "t"
  g.pos = [0, 0, float(hf["zmin"])]; g.gap = gap
  b = spec.worldbody.add_body(); b.name = "obj"; b.pos = [0, 0, 1]; b.add_freejoint()
  s_ = b.add_geom(); s_.name = "obj"; s_.mass = 0.1; s_.gap = gap
  mm = spec.add_mesh(); mm.name = "col"; mm.uservert = np.asarray(verts).reshape(-1).tolist()
  s_.type = mujoco.mjtGeom.mjGEOM_MESH; s_.meshname = "col"
  return spec.compile()


TILT = float(os.environ.get("TILT_DEG", "10"))  # sd of the tilt from upright; 0 = exactly upright (flat treads)


def upright(rng, foot):
  q = axang([0, 1, 0], np.pi / 2) if foot else np.array([1.0, 0, 0, 0])
  q = qmul(axang([0, 0, 1], rng.uniform(0, 2 * np.pi)), q)
  return qmul(axang([*rng.normal(size=2), 0], np.deg2rad(rng.normal(0, TILT))), q) if TILT > 0 else q


def place(out, name, n=20000, seed=0, clo=0.06, chi=0.02):
  import mujoco
  os.makedirs(out, exist_ok=True)
  hf = terrain(); verts = collider(name); m = build_model(hf, verts, 0.0); d = mujoco.MjData(m)
  sx, sy = float(hf["size"][0]) - 20.0, float(hf["size"][1]) - 20.0
  rng = np.random.default_rng(seed)
  Q, C, K = [], [], []

  def dep(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
    return -d.contact.dist[:d.ncon].min() if d.ncon else -np.inf

  while len(Q) < n:
    x, y = rng.uniform(-sx, sx), rng.uniform(-sy, sy)
    kind = 1 if os.environ.get("ALL_UPRIGHT") else len(Q) % 2
    quat = upright(rng, name == "foot") if kind else rand_quat(rng)
    zt = float(hf["zmax"]) + 0.5; lo, hi = float(hf["zmin"]) - 0.3, zt
    if dep(np.concatenate([[x, y, hi], quat])) > -np.inf or dep(np.concatenate([[x, y, lo], quat])) < 1e-5:
      continue
    for _ in range(40):
      mid = 0.5 * (lo + hi)
      if dep(np.concatenate([[x, y, mid], quat])) > 1e-5:
        lo = mid
      else:
        hi = mid
    c = rng.uniform(-clo, chi)
    Q.append(np.concatenate([[x, y, 0.5 * (lo + hi) + c], quat])); C.append(c); K.append(kind)
  np.savez(os.path.join(out, "placements.npz"), qpos=np.array(Q), clearance=np.array(C), upright=np.array(K),
           verts=verts, name=name)
  print("placed", n, name, "->", out)


def run(out, tag, gap=0.01, hzn=0, worlds="all"):
  import mujoco
  import warp as wp
  wp.config.quiet = True
  import mujoco_warp as mjw
  import mujoco_warp._src.collision_convex as CCV
  print("mujoco_warp", mjw.__file__)
  if hasattr(CCV, "epa_horizon_size"):  # hfield-epa fix: buffer derived from ccd_iterations
    if hzn:
      CCV.epa_horizon_size = lambda it, h=hzn: h
    print("horizon buffer", CCV.epa_horizon_size(35), "(at ccd_iterations 35)")
  else:
    if hzn:
      CCV.MJ_MAX_EPAHORIZON = hzn
    print("horizon buffer", CCV.MJ_MAX_EPAHORIZON)
  Z = np.load(os.path.join(out, "placements.npz")); Q = Z["qpos"]
  idx = np.arange(len(Q)) if worlds == "all" else np.array([int(v) for v in worlds.split(",")])
  hf = terrain(); m = build_model(hf, Z["verts"], gap); d0 = mujoco.MjData(m)
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); mm.opt.warn_overflow = 0
    dd = mjw.put_data(m, d0, nworld=len(idx), nconmax=16, njmax=64)
    dd.qpos.assign(Q[idx].astype(np.float32))
    mjw.kinematics(mm, dd); mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0]); ov = dd.overflow.numpy().copy()
    W = dd.contact.worldid.numpy()[:n]; D = dd.contact.dist.numpy()[:n]; P = dd.contact.pos.numpy()[:n]
    N = dd.contact.frame.numpy()[:n, 0, :]
  epa = (ov & int(mjw.OverflowType.EPA_HORIZON)) != 0
  np.savez(os.path.join(out, f"run_{tag}.npz"), overflow=ov, world=W, dist=D, pos=P, normal=N, idx=idx, mjw_file=mjw.__file__)
  print(f"[{tag}] worlds {len(idx)} contacts {n} EPA_HORIZON worlds {int(epa.sum())}; other bits "
        f"{sorted(set(int(v) & ~int(mjw.OverflowType.EPA_HORIZON) for v in ov) - {0})}")
  if epa.any():
    print("  overflow worlds:", idx[epa][:30].tolist(), "clearance mm:", np.round(Z["clearance"][idx[epa]][:30] * 1e3, 1).tolist())


if __name__ == "__main__":
  cmd, out = sys.argv[1], sys.argv[2]
  a = sys.argv[3:]
  if cmd == "place":
    place(out, a[0], *([int(a[1])] if len(a) > 1 else []), *([int(a[2])] if len(a) > 2 else []),
          *[float(v) for v in a[3:5]])
  elif cmd == "run":
    run(out, a[0], *([float(a[1])] if len(a) > 1 else []), *([int(a[2])] if len(a) > 2 else []), *a[3:4])
