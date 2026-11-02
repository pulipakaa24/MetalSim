"""Why upstream MuJoCo Warp's per-prism GJK/EPA gives wrong sphere normals/depths on the g1_bumps tile: run the
pinned fork (primitives on GJK/EPA, b630530) on the 3000 placements, list the worst cases, then re-run those with the
heightfield base (prism bottom) lowered/raised to test the tall-prism hypothesis.
Usage: PYTHONPATH=<worktree> .venv/bin/python epa_probe.py"""
import sys, numpy as np, mujoco, warp as wp
sys.path.insert(0, "runs/film/g1_bumps/measure")
import contact_check as CC
from metalsim.render.film.unitree import random_rough_tile
wp.config.quiet = True
import mujoco_warp as mjw
print(mjw.__file__)
hf = random_rough_tile(None, 0)
z = np.load("runs/readme_media/scratch/g1-bumps/measure_npz/contact_check/contacts_exact.npz")
C, dep, nrm = z["centre"], z["depth"], z["normal"]

def run(hf, C):
  m = CC.build_model(hf); d0 = mujoco.MjData(m)
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); dd = mjw.put_data(m, d0, nworld=len(C), nconmax=64, njmax=256)
    q = np.tile(np.array([0, 0, 1, 1, 0, 0, 0], np.float32), (len(C), 1)); q[:, :3] = C
    dd.qpos.assign(q); mjw.forward(mm, dd)
    n = int(dd.nacon.numpy()[0])
    return dd.contact.worldid.numpy()[:n], dd.contact.dist.numpy()[:n], dd.contact.frame.numpy()[:n, 0, :]

W, Dd, Nn = run(hf, C)
deep = np.full(len(C), np.nan); nd = np.zeros((len(C), 3))
for w, d, n in zip(W, Dd, Nn):
  if np.isnan(deep[w]) or -d > deep[w]: deep[w] = -d; nd[w] = n
err = (deep - dep) * 1e3
ang = np.degrees(np.arccos(np.clip((nd * nrm).sum(1), -1, 1)))
print("deepest depth err mm p50/p95/max", np.nanpercentile(np.abs(err), [50, 95, 100]).round(3), "normal err deg p95", np.nanpercentile(ang, 95).round(2))
bad = np.flatnonzero((np.abs(err) > 1.0) | (ang > 10))
print("bad", len(bad), "examples:")
for i in bad[:6]:
  print(i, "exact depth mm", round(dep[i]*1e3, 2), "mjw", round(deep[i]*1e3, 2), "normal", nd[i].round(3), "exact", nrm[i].round(3))
print("hfield size (x, y, elev, base)", hf["size"])
for base in (0.05, 0.5, 1.0):
  h2 = dict(hf); h2["size"] = list(hf["size"][:3]) + [base]
  W2, D2, N2 = run(h2, C[bad])
  deep2 = np.full(len(bad), np.nan); nd2 = np.zeros((len(bad), 3))
  for w, d, n in zip(W2, D2, N2):
    if np.isnan(deep2[w]) or -d > deep2[w]: deep2[w] = -d; nd2[w] = n
  e2 = np.abs(deep2 - dep[bad]) * 1e3; a2 = np.degrees(np.arccos(np.clip((nd2 * nrm[bad]).sum(1), -1, 1)))
  print(f"base {base}: bad cases still bad: {int(((e2 > 1.0) | (a2 > 10)).sum())} of {len(bad)}; depth err p95 {np.nanpercentile(e2, 95):.2f} mm")
