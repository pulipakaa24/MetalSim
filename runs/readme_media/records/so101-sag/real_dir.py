"""Real SO-101 follower steady error, split by approach direction and by TCP height (so101-sag, hypotheses a, b).

Same steady test as so101-expert's real_sag_bins.py (action and state each moved < 0.2 deg per frame over 0.3 s;
rest pose shoulder_lift < -85 deg dropped). For each steady sample of joint j:
  * e = action - state (rad). In lerobot-record, action is written to the follower as Goal_Position with the
    follower's own calibration (so_follower.py send_action -> bus.sync_write, normalize=True), and state is the
    follower's Present_Position with the same calibration, so e = (Goal - Present) of the follower servo; the
    leader / follower calibration difference does not enter it (checked here: state sits on the 360/4095 deg lattice).
  * g = qfrc_bias_j of the pose in so101Sim's model (the torque the servo must supply), sag s = e * sign(g) > 0 when
    the joint sits beyond its goal in the direction gravity pulls.
  * approach: the sign of the joint's last motion before the steady run (state moved > MOVE deg over the last
    moving frames), relative to gravity: "with" (moving the way gravity pulls: lowering) or "against" (raising).
  * TCP height: z of so101Sim's gripperframe site (base on the table at z = 0, joint map sign +1 offset 0).
Prints, per joint and |g| quintile: median sag for all, approach with / against, TCP z < 3 cm / > 6 cm; and the
7v4_real band (|g| -/+ c) / kp.
  <so101Sim venv>/python -I real_dir.py <so101Sim root> <out.json> name=<parquet>@fps ...
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

root, dst = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "rl", "native"))
import mujoco  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402

KP, C = 5.262, 0.0515
MOVE = 0.3                     # deg per frame: "moving"
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
m = SO101LiftEnv().model
d = mujoco.MjData(m)
qa = np.array([m.joint(j).qposadr[0] for j in J])
va = np.array([m.joint(j).dofadr[0] for j in J])
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
out = {}
for spec in sys.argv[3:]:
    name, rest = spec.split("=", 1)
    path, fps = rest.split("@")
    fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    lat = 360 / 4095
    fr = (S[:, :5] / lat) % 1.0
    # spacing check: distance of each state value to the nearest lattice point after removing a per-joint phase
    ph = np.array([np.median(fr[:, j]) for j in range(5)])
    res = np.abs(((S[:, :5] / lat) - ph + 0.5) % 1.0 - 0.5)
    print(f"== {name}: state on the 360/4095 deg lattice: p99 residual {np.percentile(res, 99):.3f} step (0 = on it)")
    hold = int(round(0.3 * fps))
    rows = []
    for e in np.unique(ep):
        a, s = A[ep == e].astype(float), S[ep == e].astype(float)
        ds = np.diff(s, axis=0)
        still = (np.abs(np.diff(a, axis=0)) < 0.2) & (np.abs(ds) < 0.2)
        run = np.zeros_like(still, int)
        for k in range(len(still)):
            run[k] = np.where(still[k], (run[k - 1] if k else 0) + 1, 0)
        ok = np.vstack([np.zeros((1, 6), bool), run >= hold])
        # last motion direction before each frame, per joint: sign of ds at the last frame with |ds| > MOVE
        last = np.zeros((len(s), 6))
        cur = np.zeros(6)
        for k in range(1, len(s)):
            mv = np.abs(ds[k - 1]) > MOVE
            cur = np.where(mv, np.sign(ds[k - 1]), cur)
            last[k] = cur
        for i in np.nonzero(ok[:, 1:4].any(1) & (s[:, 1] > -85))[0]:
            rows.append((s[i], a[i] - s[i], ok[i], last[i]))
    G, Z = [], []
    for s, _, _, _ in rows:
        d.qpos[:] = m.qpos0
        d.qpos[qa[:5]] = np.radians(s[:5])
        d.qpos[qa[5]] = -0.17 + s[5] / 100 * 1.92
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        G.append(d.qfrc_bias[va].copy())
        Z.append(d.site_xpos[site][2])
    G, Z = np.array(G), np.array(Z)
    E = np.radians(np.array([r[1] for r in rows]))
    OK = np.array([r[2] for r in rows])
    L = np.array([r[3] for r in rows])
    print(f"   {len(rows)} steady non-rest samples; TCP z p10/p50/p90 {np.percentile(Z, [10, 50, 90]).round(3).tolist()} m")
    res_ds = {}
    for j in (1, 2, 3):
        k = OK[:, j] & (L[:, j] != 0)
        g, e, z, dirn = G[k, j], E[k, j], Z[k], L[k, j]
        sg = np.sign(g)
        sag = e * sg
        with_g = dirn == -sg                  # moved the way gravity pulls (gravity force = -g)
        edges = np.percentile(np.abs(g), [0, 20, 40, 60, 80, 100])
        print(f"  {J[j]} n {k.sum()}  approach with gravity {with_g.mean() * 100:.0f} %")
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (np.abs(g) >= lo) & (np.abs(g) <= hi)
            gm = float(np.median(np.abs(g[b])))
            med = lambda x: (round(float(np.median(x)) * 1e3, 1), int(x.size)) if x.size >= 10 else (None, int(x.size))
            c = dict(g=round(gm, 3), all=med(sag[b]), with_=med(sag[b & with_g]), against=med(sag[b & ~with_g]),
                     z_low=med(sag[b & (z < 0.03)]), z_high=med(sag[b & (z > 0.06)]),
                     model_lo=round((gm - C) / KP * 1e3, 1), model=round(gm / KP * 1e3, 1),
                     model_hi=round((gm + C) / KP * 1e3, 1))
            cells.append(c)
            print(f"    |g| {c['g']:.2f}: sag all {c['all']}  with {c['with_']}  against {c['against']}  "
                  f"TCP<3cm {c['z_low']}  TCP>6cm {c['z_high']}  | 7v4_real {c['model_lo']}..{c['model_hi']} mrad")
        res_ds[J[j]] = dict(n=int(k.sum()), with_frac=float(with_g.mean()), bins=cells)
    out[name] = res_ds
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)
