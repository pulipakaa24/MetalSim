"""Steady tracking error and joint speeds of real SO-101 followers in LeRobot teleop datasets (v3.0 parquet).

action = the leader's position written as Goal_Position, observation.state = the follower's Present_Position read at
the start of the same tick (lerobot-record's loop: observe, then teleop action, then send), in degrees (use_degrees
default True since lerobot 6600b60e, 2026-02-13; checked below: |values| exceed 100 in degree mode is not required,
so units are inferred from the recording dates). Steady sample of joint j: action_j and state_j each changed by less
than 0.2 deg per frame over the last HOLD = 0.3 s; error = action - state. This includes the leader / follower
calibration mismatch (each arm's zero is set by hand at mid-range), so it bounds sag + dead band + calibration offset.
Also: Fontaine's lag metric mean |action[t] - state[t+1]| and the follower's joint speed percentiles.
  python -I real_sag.py <name>=<file.parquet>[@fps] ...
"""
import sys
import numpy as np
import pyarrow.parquet as pq

J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
for spec in sys.argv[1:]:
    name, rest = spec.split("=", 1)
    path, fps = rest.split("@") if "@" in rest else (rest, "30")
    fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    hold = int(round(0.3 * fps))
    errs, lags, speeds = [], [], []
    for e in np.unique(ep):
        a, s = A[ep == e].astype(float), S[ep == e].astype(float)
        da, ds = np.abs(np.diff(a, axis=0)), np.abs(np.diff(s, axis=0))
        still = (da < 0.2) & (ds < 0.2)
        run = np.zeros_like(still, int)
        for k in range(len(still)):
            run[k] = np.where(still[k], (run[k - 1] if k else 0) + 1, 0)
        ok = np.vstack([np.zeros((1, 6), bool), run >= hold])
        errs.append(np.where(ok, a - s, np.nan))
        lags.append(np.abs(a[:-1] - s[1:]))
        speeds.append(ds * fps)
    E, L, V = np.vstack(errs), np.vstack(lags), np.vstack(speeds)
    print(f"== {name}: {len(np.unique(ep))} episodes, {len(A)} frames at {fps:g} fps; value range per joint (deg) "
          f"{np.round(S.min(0)).tolist()} .. {np.round(S.max(0)).tolist()}")
    for j in range(5):
        x = E[:, j][~np.isnan(E[:, j])]
        r = np.radians
        print(f"  {J[j]:13s} steady n {len(x):6d}  |e| median {r(np.median(np.abs(x)))*1e3:6.1f} mrad  p90 "
              f"{r(np.percentile(np.abs(x), 90))*1e3:6.1f}  mean signed {r(np.mean(x))*1e3:7.1f}  | lag mean {np.mean(L[:, j]):5.2f} deg"
              f"  | speed p99 {r(np.percentile(V[:, j], 99)):4.2f} max {r(V[:, j].max()):4.2f} rad/s")
    print(f"  lag mean over arm joints {np.mean(L[:, :5]):.2f} deg (Fontaine: 2.19 deg)")
