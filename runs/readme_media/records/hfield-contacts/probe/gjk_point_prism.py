"""Why the pinned fork's GJK/EPA prism path returns depths of 5-30 mm with normal (0, 0, +-1) for 5 mm spheres:
per prism of one bad placement, run MJW's gjk() on the sphere shrunk to a point (as gjk_phase does) and the full
ccd(), and compare with the float64 point-to-prism distance (scipy-free: sample the prism's closest point by
projecting on its 5 faces). Usage: PYTHONPATH=<worktree> .venv/bin/python probe/gjk_point_prism.py <placement index>"""
import sys
import numpy as np, warp as wp
sys.path.insert(0, "runs/film/g1_bumps/measure")
import contact_check as CC
from metalsim.render.film.unitree import random_rough_tile
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src.collision_gjk import gjk, ccd
from mujoco_warp._src.collision_primitive import Geom
from mujoco_warp._src.types import GeomType, mat63, MJ_MAX_EPAFACES, MJ_MAX_EPAHORIZON
print(mjw.__file__)
i = int(sys.argv[1])
hf = random_rough_tile(None, 0)
D, res, x0, y0 = CC.surface(hf)
c = np.load("runs/readme_media/scratch/g1-bumps/measure_npz/contact_check/contacts_exact.npz")["centre"][i]
ci, ri = int(np.floor((c[0] - x0) / res)), int(np.floor((c[1] - y0) / res))
prisms = []
for r in range(ri - 1, ri + 2):
  for cc in range(ci - 1, ci + 2):
    P = lambda rr, k: np.array([x0 + k * res, y0 + rr * res, D[rr, k]])
    for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
      top = np.stack(t); bot = top.copy(); bot[:, 2] = -0.05
      prisms.append(np.concatenate([bot, top]))
prisms = np.array(prisms, np.float32)

@wp.kernel
def k(pr: wp.array(dtype=mat63), cen: wp.vec3, out_g: wp.array2d(dtype=float), out_c: wp.array2d(dtype=float),
      ev: wp.array2d(dtype=wp.vec3), evi: wp.array2d(dtype=int), ef: wp.array2d(dtype=int), epr: wp.array2d(dtype=wp.vec3),
      en: wp.array2d(dtype=float), eh: wp.array2d(dtype=int), ov: wp.array(dtype=int)):
  t = wp.tid()
  g1 = Geom(); g1.pos = wp.vec3(0.0); g1.rot = wp.identity(3, float); g1.polyvert = pr[t]; g1.margin = 0.0; g1.index = -1
  g2 = Geom(); g2.pos = cen; g2.rot = wp.identity(3, float); g2.size = wp.vec3(0.005, 0.0, 0.0); g2.margin = 0.0; g2.index = -1
  x1 = (pr[t][0] + pr[t][1] + pr[t][2] + pr[t][3] + pr[t][4] + pr[t][5]) / 6.0
  g2p = g2; g2p.size = wp.vec3(0.0)
  r = gjk(1.0e-6, 35, g1, g2p, x1, cen, int(GeomType.HFIELD), int(GeomType.SPHERE), 0.005, False)
  out_g[t, 0] = r.dist; out_g[t, 1] = float(r.dim); out_g[t, 2] = wp.where(r.separated, 1.0, 0.0)
  dist, nc, w1, w2, idx = ccd(1.0e-6, 0.0, 35, 35, g1, g2, int(GeomType.HFIELD), int(GeomType.SPHERE), x1, cen,
                              ev[t], evi[t], ef[t], epr[t], en[t], eh[t], False, 0, ov)
  out_c[t, 0] = dist; out_c[t, 1] = float(nc)
  n = wp.normalize(w1 - w2)
  out_c[t, 2] = n[0]; out_c[t, 3] = n[1]; out_c[t, 4] = n[2]

with wp.ScopedDevice("cpu"):
  n = len(prisms); E = MJ_MAX_EPAFACES
  pr = wp.array(prisms, dtype=mat63); og = wp.zeros((n, 3), float); oc = wp.zeros((n, 5), float)
  ev = wp.zeros((n, 5 + 35), wp.vec3); evi = wp.zeros((n, 5 + 35), int); ef = wp.zeros((n, 6 * E), int)
  epr = wp.zeros((n, 6 * E), wp.vec3); en = wp.zeros((n, 6 * E), float); eh = wp.zeros((n, 2 * MJ_MAX_EPAHORIZON), int)
  wp.launch(k, n, inputs=[pr, wp.vec3(*c.astype(np.float32)), og, oc, ev, evi, ef, epr, en, eh, wp.zeros(1, int)])
  og, oc = og.numpy(), oc.numpy()

def point_tri(p, a, b, cc): return CC.closest_on_triangle(p, a, b, cc)
for j, P6 in enumerate(prisms.astype(np.float64)):
  top = P6[3:]
  q = point_tri(c, *top); d = np.linalg.norm(c - q)
  print(f"prism {j:2d}: f64 dist to top tri {d*1e3:7.3f} mm | gjk(point) dist {og[j,0]*1e3:9.3f} mm dim {int(og[j,1])} sep {int(og[j,2])}"
        f" | ccd dist {oc[j,0]*1e3:8.3f} mm n {oc[j,2:].round(3)}")
