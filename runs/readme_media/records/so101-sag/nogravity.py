"""Steady |e| (median, p90 mrad) of the gravity-free joints (shoulder_pan, wrist_roll), real vs sim replays: the
no-load friction band (so101-sag). Same steady test as compare.py.
  python3 nogravity.py <hfdata dir>"""
import os, sys
import numpy as np
import pyarrow.parquet as pq
H = sys.argv[1]
D = {"hungho77": (f"{H}/hungho77/file-000.parquet", 30.0), "IlyaGusev": (f"{H}/ilya/file-000.parquet", 20.0)}
def steady(A, S, ep, fps):
    hold = int(round(0.3 * fps)); out = []
    for e in np.unique(ep):
        a, s = A[ep == e].astype(float), S[ep == e].astype(float)
        still = (np.abs(np.diff(a, axis=0)) < 0.2) & (np.abs(np.diff(s, axis=0)) < 0.2)
        run = np.zeros_like(still, int)
        for k in range(len(still)):
            run[k] = np.where(still[k], (run[k - 1] if k else 0) + 1, 0)
        ok = np.vstack([np.zeros((1, 6), bool), run >= hold])
        out.append(np.where(ok, np.radians(a - s), np.nan))
    return np.vstack(out)
for name, (p, fps) in D.items():
    t = pq.read_table(p).to_pandas()
    src = {"real": steady(np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy(), fps)}
    for v in ("7v4_real", "bam_m1", "bam_m6"):
        z = np.load(f"replay/{v}/{name}.npz"); src[v] = steady(z["action"], z["state"], z["episode_index"], fps)
    for j, jn in ((0, "shoulder_pan"), (4, "wrist_roll")):
        print(f"{name:9s} {jn:12s} " + "  ".join(
            f"{k} {np.nanmedian(np.abs(E[:, j]))*1e3:5.1f}, {np.nanpercentile(np.abs(E[:, j]), 90)*1e3:5.1f}" for k, E in src.items()))
