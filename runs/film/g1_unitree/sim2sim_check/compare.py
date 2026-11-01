"""The MuJoCo Warp side of the g1_unitree sim2sim check: the film adapter (metalsim.render.film.unitree.UnitreeG1, MuJoCo Warp BatchSim) for
8 s with n envs, measured as the CPU reference (reference.py) and compared with it. A GPU job:
scripts/gpu_run.sh <name> render 15 -- .venv/bin/python runs/film/g1_unitree/sim2sim_check/compare.py <ref_dir> <out_dir> 4 8
(this take: ref_dir runs/readme_media/scratch/g1-unitree/ref, out_dir .../cmp; compare.json here is its output)"""
import json
import os
import sys
import time

import numpy as np

from metalsim.render.film import cfg as C
from metalsim.render.film import record as REC
from metalsim.render.film.unitree import UnitreeG1, unitree_feet

ref_dir, out_dir = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
os.makedirs(out_dir, exist_ok=True)
task = C.FilmTask(name="g1_unitree_compare", adapter=UnitreeG1, title="", summary="", film=C.FilmCfg(num_envs=n),
                  viewer=C.ViewerCfg(), look=C.LookCfg(), assets=[], doc="")
ad = UnitreeG1(task, task.film, None)
ad.reset()
frames = int(round(seconds / ad.frame_dt))
snaps, ankles, actions, t0 = [], [], [], time.perf_counter()
for k in range(frames):
    s, done, info, tm = ad.step()
    snaps.append(s)
    ankles.append(unitree_feet(ad.model, ad._fk, s["qpos"], ad.foot_geoms)[2])
    actions.append(ad.action.copy())
wall = time.perf_counter() - t0
print(ad.summary(np.full(n, -1)), "| capacity overflow:", ad.overflow)
g = REC.gait_metrics(snaps, np.full(n, -1), np.zeros(n, bool), ad.frame_dt)
print("\n".join(REC.gait_lines(g, list(range(n)))))
Q = np.stack([s["qpos"] for s in snaps])            # (F, n, 19): frame k = after physics step 10 (k + 1)
A = np.stack(ankles)
Cc = np.stack([s["feet_contact"] for s in snaps])
k0 = int(round(1.0 / ad.frame_dt))
swing = [[float(np.percentile(A[k0:, e, f, 2][~Cc[k0:, e, f]], 95)) for f in range(2)] for e in range(n)]
R = np.load(os.path.join(ref_dir, "reference.npz"))
ref = json.load(open(os.path.join(ref_dir, "reference.json")))
Rq = R["qpos"]                                       # (F, 19), same frame times
F = min(len(Rq), len(Q))
dpos = np.linalg.norm(Q[:F, :, :3] - Rq[:F, None, :3], axis=2)      # (F, n) base-position difference [m]
dxy_w = np.linalg.norm(Q[:F, :, :2] - Q[:F, :1, :2], axis=2)         # spread between the GPU envs
rows = {}
for t in (0.02, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
    k = int(round(t / ad.frame_dt)) - 1
    if 0 <= k < F:
        rows[f"{t:g}s"] = dict(ref=Rq[k, :3].round(4).tolist(), metalsim=Q[k, :, :3].round(4).tolist(),
                               dpos_m=dpos[k].round(4).tolist(), env_spread_m=float(dxy_w[k].max().round(4)))
        print(f"t={t:4.2f}s  ref {Rq[k, :3].round(3)}  metalsim env0 {Q[k, 0, :3].round(3)}  |d| per env "
              f"{dpos[k].round(3)}  env spread {dxy_w[k].max():.3f}")
res = dict(n_envs=n, seconds=seconds, frames=frames, wall_s=wall, capacity_overflow=ad.overflow, at=rows,
           ankle_origin_swing_p95_m=swing, gait=g, reference_gait=ref["gait"]["per_env"]["0"],
           reference_ankle_origin_swing_p95_m=ref["ankle_origin_swing_p95_m"],
           mean_base_z=Q[k0:, :, 2].mean(0).round(4).tolist(), reference_mean_base_z=ref["mean_base_z"])
json.dump(res, open(os.path.join(out_dir, "compare.json"), "w"), indent=1, default=float)
np.savez(os.path.join(out_dir, "metalsim.npz"), qpos=Q, contact=Cc, ankles=A, actions=np.stack(actions))
print("ankle-origin swing p95 per env", np.round(swing, 4).tolist(), "| ref", ref["ankle_origin_swing_p95_m"])
print("wrote", os.path.join(out_dir, "compare.json"))
