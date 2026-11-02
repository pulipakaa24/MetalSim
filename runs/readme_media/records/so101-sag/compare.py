"""One sag analysis on the real followers and on the sim replays of the same commands (so101-sag).

Per dataset and joint (shoulder_lift, elbow_flex, wrist_flex): steady samples as so101-expert's real_sag_bins.py
(action and state each moved < 0.2 deg per frame over the last 0.3 s; shoulder_lift < -85 deg rest pose dropped);
e = action - state; g = qfrc_bias of the state's pose in so101Sim's model; sag s = e * sign(g) (> 0: beyond the goal
the way gravity pulls); approach = sign of the joint's last motion > 0.3 deg/frame, "with" gravity or "against".
Bins: |g| quintiles of the REAL samples (the same edges for every source). Cell: median sag (mrad) and count.
Summary per source: median |sim - real| over bins with >= 30 samples on both sides, per joint and approach, and the
"all" median (both approaches pooled), against the real data.
  <so101Sim venv>/python -I compare.py <so101Sim root> <out.json> real:name=<parquet>@fps ... sim:<variant>=<dir> ...
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

J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MOVE, MINN = float(os.environ.get("SAG_MOVE", 0.3)), 30   # deg/frame "moving"; SAG_MOVE for the sensitivity check
m = SO101LiftEnv(randomize=False).model
d = mujoco.MjData(m)
qa = np.array([m.joint(j).qposadr[0] for j in J])
va = np.array([m.joint(j).dofadr[0] for j in J])
real, sims = {}, {}
for a in sys.argv[3:]:
    kind, rest = a.split(":", 1)
    name, val = rest.split("=", 1)
    (real if kind == "real" else sims)[name] = val


def samples(A, S, ep, fps):
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
        last = np.zeros((len(s), 6))
        cur = np.zeros(6)
        for k in range(1, len(s)):
            cur = np.where(np.abs(ds[k - 1]) > MOVE, np.sign(ds[k - 1]), cur)
            last[k] = cur
        for i in np.nonzero(ok[:, 1:4].any(1) & (s[:, 1] > -85))[0]:
            rows.append((s[i], a[i] - s[i], ok[i], last[i]))
    G = []
    for s, _, _, _ in rows:
        d.qpos[:] = m.qpos0
        d.qpos[qa[:5]] = np.radians(s[:5])
        d.qpos[qa[5]] = -0.17 + s[5] / 100 * 1.92
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        G.append(d.qfrc_bias[va].copy())
    return (np.array(G), np.radians(np.array([r[1] for r in rows])), np.array([r[2] for r in rows]),
            np.array([r[3] for r in rows]))


def cells(G, E, OK, L, j, edges):
    k = OK[:, j] & (L[:, j] != 0)
    g, sag, dirn = G[k, j], E[k, j] * np.sign(G[k, j]), L[k, j]
    w = dirn == -np.sign(g)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (np.abs(g) >= lo) & (np.abs(g) <= hi)
        f = lambda x: [round(float(np.median(x)) * 1e3, 1) if x.size else None, int(x.size)]
        out.append(dict(g=round(float(np.median(np.abs(g[b]))), 3) if b.any() else None, all=f(sag[b]),
                        with_=f(sag[b & w]), against=f(sag[b & ~w])))
    return out, float(w.mean()) if w.size else None


res = {"bins": {}, "with_frac": {}, "summary": {}}
for name, spec in real.items():
    path, fps = spec.split("@")
    fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    src = {"real": samples(A, S, ep, fps)}
    for v, dr in sims.items():
        z = np.load(os.path.join(dr, f"{name}.npz"))
        src[v] = samples(z["action"], z["state"], z["episode_index"], float(z["fps"]))
    Gr, Er, OKr, Lr = src["real"]
    res["bins"][name], res["with_frac"][name] = {}, {}
    for j in (1, 2, 3):
        k = OKr[:, j] & (Lr[:, j] != 0)
        edges = np.percentile(np.abs(Gr[k, j]), [0, 20, 40, 60, 80, 100])
        edges[0], edges[-1] = 0.0, 10.0
        res["bins"][name][J[j]], res["with_frac"][name][J[j]] = {}, {}
        for v, (G, E, OK, L) in src.items():
            c, wf = cells(G, E, OK, L, j, edges)
            res["bins"][name][J[j]][v], res["with_frac"][name][J[j]][v] = c, wf
        print(f"== {name} {J[j]}  (cells: median sag mrad [n], all / with gravity / against)")
        for v in src:
            wf = res["with_frac"][name][J[j]][v]
            print(f"  {v:14s} with {wf * 100:4.0f} % | " + " | ".join(
                f"|g| {c['g']}: {c['all'][0]} / {c['with_'][0]} / {c['against'][0]}" for c in res["bins"][name][J[j]][v]))
# summary: median |sim - real| per joint and approach, pooled over datasets and bins
for v in sims:
    S = {}
    for j in J[1:4]:
        for key in ("all", "with_", "against"):
            diffs = []
            for name in real:
                for cr, cs in zip(res["bins"][name][j]["real"], res["bins"][name][j][v]):
                    if cr[key][0] is not None and cs[key][0] is not None and cr[key][1] >= MINN and cs[key][1] >= MINN:
                        diffs.append(cs[key][0] - cr[key][0])
            S[f"{j} {key}"] = dict(median_abs=round(float(np.median(np.abs(diffs))), 1) if diffs else None,
                                   median_signed=round(float(np.median(diffs)), 1) if diffs else None, n=len(diffs))
    allabs = [abs(x["median_abs"]) for k, x in S.items() if x["median_abs"] is not None and not k.endswith(" all")]
    res["summary"][v] = S
    print(f"\n{v}: median |sim - real| mrad (signed) [bins]  " + "  ".join(
        f"{k} {x['median_abs']} ({x['median_signed']}) [{x['n']}]" for k, x in S.items()))
json.dump(res, open(dst, "w"), indent=1)
print("wrote", dst)
