"""Per-collider heightfield prism counts in training states (stream hfield-epa).

For a captured MjModel (model.mjb) and a qpos snapshot (qpos_snapshots.npy, [iteration, world, nq]): per terrain
collider of each world, the prisms the fork's heightfield kernel processes (tight AABB of the mesh in the heightfield
frame, the kernel's cells, the height test against zmin - (margin + gap)) and, from the exact float64 signed distance
(exact_sd.signed_distance), how many penetrate (EPA in every version) and how many are apart but within margin + gap
(EPA on the inflated shapes on 93e5ce1; GJK only with HFIELD_MARGIN_SHRINK).
Usage: python profile_counts.py <capture dir> <iteration index> [nworlds]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_sd import signed_distance  # noqa: E402
import mujoco  # noqa: E402

cap, it = sys.argv[1], int(sys.argv[2]); nw = int(sys.argv[3]) if len(sys.argv) > 3 else 512
m = mujoco.MjModel.from_binary_path(os.path.join(cap, "model.mjb")); d = mujoco.MjData(m)
Q = np.load(os.path.join(cap, "qpos_snapshots.npy"))[it]
idx = np.random.default_rng(0).choice(len(Q), size=min(nw, len(Q)), replace=False)
g0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground"); hid = m.geom_dataid[g0]
nrow, ncol = m.hfield_nrow[hid], m.hfield_ncol[hid]; sx, sy, sz, sb = m.hfield_size[hid]
H = m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid] + nrow * ncol].reshape(nrow, ncol) * sz + m.geom_pos[g0][2]
base = m.geom_pos[g0][2] - sb
dx, dy = 2 * sx / (ncol - 1), 2 * sy / (nrow - 1)
cols = [g for g in range(m.ngeom) if g != g0 and (m.geom_contype[g] & m.geom_conaffinity[g0] or m.geom_contype[g0] & m.geom_conaffinity[g])]
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in cols]
hfm = float(sys.argv[4]) if len(sys.argv) > 4 else float(m.geom_margin[g0] + m.geom_gap[g0] + m.geom_margin[cols[0]] + m.geom_gap[cols[0]])
stats = {n: [] for n in names}
for w in idx:
  d.qpos[:] = Q[w]; mujoco.mj_kinematics(m, d)
  for g, n in zip(cols, names):
    va, vn = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
    A = m.mesh_vert[va:va + vn].astype(np.float64) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    lo, hi = A.min(0), A.max(0)
    if lo[2] - hfm > H.max() or lo[0] < -sx or hi[0] > sx:
      stats[n].append((0, 0, 0)); continue
    cmin = max(0, int(np.floor((lo[0] + sx) / dx))); cmax = min(ncol - 1, int(np.ceil((hi[0] + sx) / dx)))
    rmin = max(0, int(np.floor((lo[1] + sy) / dy))); rmax = min(nrow - 1, int(np.ceil((hi[1] + sy) / dy)))
    ncand = npen = ngap = 0
    for r in range(rmin, rmax):
      for c in range(cmin + 1, cmax + 1):
        P = lambda rr, cc: np.array([cc * dx - sx, rr * dy - sy, H[rr, cc]])
        for t in ((P(r + 1, c - 1), P(r, c - 1), P(r + 1, c)), (P(r, c - 1), P(r + 1, c), P(r, c))):
          top = np.array(t)
          if top[:, 2].max() < lo[2] - hfm:
            continue
          ncand += 1
          bot = top.copy(); bot[:, 2] = base
          sd = signed_distance(A, np.vstack([bot, top]))
          npen += sd < 0; ngap += 0 <= sd < hfm
    stats[n].append((ncand, npen, ngap))
print(f"{cap} snapshot {it}, {len(idx)} worlds, pair margin + gap {hfm}")
for n in names:
  a = np.array(stats[n]); act = a[:, 0] > 0
  print(f"  {n:28s} worlds with candidates {act.sum():4d}: per pair candidates {a[act, 0].mean():.1f}, penetrating "
        f"{a[act, 1].mean():.2f}, apart within gap {a[act, 2].mean():.2f}; EPA calls / GJK-only calls 93e5ce1 "
        f"{(a[act, 1] + a[act, 2]).mean():.2f} / {(a[act, 0] - a[act, 1] - a[act, 2]).mean():.2f}, fix {a[act, 1].mean():.2f} / "
        f"{(a[act, 0] - a[act, 1]).mean():.2f}")
