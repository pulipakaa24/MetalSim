"""Exact (float64) signed distance between a G1 collider mesh and each heightfield prism it can touch (stream hfield-epa).

Per world of an epa_train.py placements set: prisms (triangle columns down to the base, MuJoCo's split along
(c, r)-(c+1, r+1)) under the mesh's AABB grown by HFM (the pair's margin + gap); per prism the signed distance of the
two convex polytopes from the Minkowski difference hull (scipy ConvexHull): inside -> minus the distance to the
nearest facet plane (penetration depth), outside -> the distance to the hull. The heightfield kernel's contact dist
(HFIELD_TRUE_DISTANCE) is this signed distance for prisms closer than margin + gap.
Usage: python exact_sd.py <dir> <hfm> [nworlds] -> <dir>/exact_sd.npz (world, sd per prism, prism top centroid)
"""
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "runs/film/g1_bumps/measure")
import contact_check as CC  # noqa: E402
import epa_train as R  # noqa: E402


def signed_distance(A, B):
  P = (A[:, None, :] - B[None, :, :]).reshape(-1, 3)
  h = ConvexHull(P)
  off = h.equations[:, 3]  # n . x + off <= 0 inside; at the origin: off
  if off.max() <= 0:
    return off.max()
  best = np.inf
  for s in h.simplices:
    a, b, c = P[s]
    q = CC.closest_on_triangle(np.zeros(3), a, b, c)
    best = min(best, np.linalg.norm(q))
  return best


def main():
  out, hfm = sys.argv[1], float(sys.argv[2]); nw = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
  import mujoco
  hf = R.terrain(); D, res, x0, y0 = CC.surface(hf); base = hf["zmin"] - hf["size"][3]
  Z = np.load(os.path.join(out, "placements.npz")); Q = Z["qpos"][:nw]
  m = R.build_model(hf, Z["verts"], 0.0); d = mujoco.MjData(m); g = 1
  va, vn = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
  V0 = m.mesh_vert[va:va + vn].astype(np.float64)
  W, S, C = [], [], []
  for i, q in enumerate(Q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    Rm = d.geom_xmat[g].reshape(3, 3); A = V0 @ Rm.T + d.geom_xpos[g]
    lo, hi = A.min(0) - hfm, A.max(0) + hfm
    for r in range(int(np.floor((lo[1] - y0) / res)), int(np.floor((hi[1] - y0) / res)) + 1):
      for cc in range(int(np.floor((lo[0] - x0) / res)), int(np.floor((hi[0] - x0) / res)) + 1):
        P = lambda a_, k: np.array([x0 + k * res, y0 + a_ * res, D[a_, k]])
        for t in ((P(r, cc), P(r, cc + 1), P(r + 1, cc + 1)), (P(r, cc), P(r + 1, cc + 1), P(r + 1, cc))):
          top = np.array(t)
          if top[:, 2].max() < lo[2]:
            continue
          bot = top.copy(); bot[:, 2] = base
          W.append(i); S.append(signed_distance(A, np.vstack([bot, top]))); C.append(top.mean(0))
  np.savez(os.path.join(out, "exact_sd.npz"), world=np.array(W), sd=np.array(S), centroid=np.array(C), nworld=len(Q), hfm=hfm)
  print("exact sd:", len(Q), "worlds", len(S), "prisms")


if __name__ == "__main__":
  main()
