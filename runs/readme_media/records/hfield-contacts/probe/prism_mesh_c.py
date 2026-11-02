"""Cross-check: one heightfield prism of a cc2 box placement as a standalone convex MESH in MuJoCo C (mjc_Convex, plain
GJK/EPA, multiccd off) against the same box; prints C's mesh-box penetration per prism next to the SAT value.
Usage: prism_mesh_c.py <cc dir> <world>"""
import sys, itertools, numpy as np, mujoco
sys.path.insert(0, "runs/readme_media/scratch/hfield-contacts"); sys.path.insert(0, "runs/film/g1_bumps/measure")
import cc2, contact_check as CC
d, w = sys.argv[1], int(sys.argv[2])
hf = cc2.tile(); D, res, x0, y0 = CC.surface(hf); base = -hf["size"][3] + hf["zmin"]
q = np.load(f"{d}/placements.npz")["qpos"][w]; hs = cc2.SHAPES["box"][1]
R = cc2.quat2mat(q[3:]); A = np.array([q[:3] + R @ (np.array(hs) * np.array(s)) for s in itertools.product((-1, 1), repeat=3)])
lo, hi = A.min(0), A.max(0)
for r in range(int(np.floor((lo[1] - y0) / res)), int(np.floor((hi[1] - y0) / res)) + 1):
  for cc in range(int(np.floor((lo[0] - x0) / res)), int(np.floor((hi[0] - x0) / res)) + 1):
    P = lambda a_, k: np.array([x0 + k * res, y0 + a_ * res, D[a_, k]])
    for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
      top = np.array(t); bot = top.copy(); bot[:, 2] = base; V = np.vstack([bot, top]); ctr = V.mean(0)
      spec = mujoco.MjSpec(); spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
      me = spec.add_mesh(); me.name = "p"; me.uservert = (V - ctr).reshape(-1).tolist()
      g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = "p"; g.pos = ctr.tolist()
      b = spec.worldbody.add_body(); b.pos = q[:3].tolist(); b.quat = q[3:].tolist()
      b.add_freejoint(); gb = b.add_geom(); gb.type = mujoco.mjtGeom.mjGEOM_BOX; gb.size = hs
      m = spec.compile(); dd = mujoco.MjData(m); dd.qpos[:] = q; mujoco.mj_forward(m, dd)
      if dd.ncon:
        print(f"prism r{r} c{cc}: C mesh-box",
              [(round(-c.dist * 1e3, 3), np.round(c.frame[:3], 2).tolist()) for c in dd.contact[:dd.ncon]])
