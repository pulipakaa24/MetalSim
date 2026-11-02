"""Binned view of real_sag_model.py: per dataset, steady samples of shoulder_lift / elbow_flex / wrist_flex binned by
the model's gravity torque g; median real error e (action - state) per bin against the 7v4_real prediction g / kp.
Rest-pose samples (shoulder_lift < -85 deg: the arm folded on its stop, held by contact, not by the servo) are dropped.
  <so101Sim venv>/python -I real_sag_bins.py <so101Sim root> name=<parquet>[@fps] ..."""
import os, sys
import numpy as np
import pyarrow.parquet as pq
root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "rl", "native"))
import mujoco  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402
KP = 5.262
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
m = SO101LiftEnv().model; d = mujoco.MjData(m)
qa = np.array([m.joint(j).qposadr[0] for j in J]); va = np.array([m.joint(j).dofadr[0] for j in J])
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1); path, fps = rest.split("@"); fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    hold = int(round(0.3 * fps)); rows = []
    for e in np.unique(ep):
        a, s = A[ep == e].astype(float), S[ep == e].astype(float)
        still = (np.abs(np.diff(a, axis=0)) < 0.2) & (np.abs(np.diff(s, axis=0)) < 0.2)
        run = np.zeros_like(still, int)
        for k in range(len(still)):
            run[k] = np.where(still[k], (run[k - 1] if k else 0) + 1, 0)
        ok = np.vstack([np.zeros((1, 6), bool), run >= hold])
        for i in np.nonzero(ok[:, 1:4].any(1) & (s[:, 1] > -85))[0]:
            rows.append((s[i], a[i] - s[i], ok[i]))
    G = []
    for s, _, _ in rows:
        d.qpos[:] = m.qpos0; d.qpos[qa[:5]] = np.radians(s[:5]); d.qpos[qa[5]] = -0.17 + s[5] / 100 * 1.92
        d.qvel[:] = 0; mujoco.mj_forward(m, d); G.append(d.qfrc_bias[va].copy())
    G = np.array(G); E = np.radians(np.array([r[1] for r in rows])); OK = np.array([r[2] for r in rows])
    print(f"== {name}: {len(rows)} steady non-rest samples")
    for j in (1, 2, 3):
        k = OK[:, j]; g, e = G[k, j], E[k, j]
        edges = np.percentile(g, [0, 20, 40, 60, 80, 100])
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (g >= lo) & (g <= hi)
            cells.append(f"g {np.median(g[b]):+.2f}: e {np.median(e[b])*1e3:+6.1f} (model {np.median(g[b])/KP*1e3:+6.1f})")
        print(f"  {J[j]:13s} n {k.sum():5d} | " + " | ".join(cells))
