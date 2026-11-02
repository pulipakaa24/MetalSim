"""Pose-controlled comparison of steady-state sag: real SO-101 followers (LeRobot teleop datasets) against 7v4_real.

The raw medians (real_sag.py, sag.py) compare different poses: the sim's are mostly the lift pose with the box in hand.
Here each steady real sample (real_sag.py's test: action and state each moved < 0.2 deg per frame over the last 0.3 s)
is put into the MuJoCo model (so101Sim's SO101LiftEnv scene, the arm alone, qvel 0) and its gravity torque g_j =
qfrc_bias_j is computed. A P-only servo (7v4_real: kp 5.262 N.m/rad, kv 0) holds still with kp * e = g +- c (Coulomb
c 0.0515 N.m), so e = action - state should be g / kp within +-c / kp (9.8 mrad). Per joint, the real error is
regressed on g (OLS with intercept: the intercept takes the leader / follower calibration mismatch, which is constant
per pair of arms); the slope is the arm's measured compliance 1 / kp_eff (model: 1 / 5.262 = 0.190 rad/N.m).
The same regression on the sim's own steady samples (free phase, no box) checks the method.

Joint map real deg -> sim rad: sign +1 and offset 0 on every arm joint (so101Sim rl/real/so101_real.py DEFAULT_MAP;
the owner's calibrated rl/real/joint_map.json also has signs +1, offsets -0.7..8.0 deg: run again with --offsets to
use them). Gripper 0..100 % -> -0.17..1.75 rad (so101_real.py docstring).
  <so101Sim venv>/python -I real_sag_model.py <so101Sim root> <sim run dir> name=<parquet>[@fps] ... [--offsets]
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

args = sys.argv[1:]
OFFS = "--offsets" in args
args = [a for a in args if a != "--offsets"]
root, sim_dir, specs = args[0], args[1], args[2:]
sys.path.insert(0, os.path.join(root, "rl", "native"))
import mujoco  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402

KP, C = 5.262, 0.0515
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
off = np.zeros(5)
if OFFS:
    off = np.array(json.load(open(os.path.join(root, "rl", "real", "joint_map.json")))["offsets_deg"])
env = SO101LiftEnv()
m = env.model
d = mujoco.MjData(m)
qa = np.array([m.joint(j).qposadr[0] for j in J])
va = np.array([m.joint(j).dofadr[0] for j in J])


def gravity(q6):
    """(N, 6) sim joint angles -> (N, 6) gravity torque the servo must supply (qfrc_bias at rest)."""
    out = np.empty_like(q6)
    for i, q in enumerate(q6):
        d.qpos[:] = m.qpos0
        d.qpos[qa] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        out[i] = d.qfrc_bias[va]
    return out


def fit(g, e):
    A = np.vstack([g, np.ones_like(g)]).T
    (b, a), *_ = np.linalg.lstsq(A, e, rcond=None)
    r = np.corrcoef(g, e)[0, 1] if g.std() > 0 else float("nan")
    return b, a, r


def report(name, Q, E_, mask):
    """Q (N, 6) sim rad, E_ (N, 5) error rad, mask (N, 5) steady."""
    G = gravity(Q)
    rows = {}
    print(f"== {name}")
    for j in range(5):
        k = mask[:, j]
        g, e = G[k, j], E_[k, j]
        if len(g) < 20:
            continue
        b, a, r = fit(g, e)
        pred = g / KP
        res = e - a - pred                       # after the calibration intercept, what the model leaves
        rows[J[j]] = dict(n=int(k.sum()), g_p10_p90=[round(float(x), 3) for x in np.percentile(g, [10, 90])],
                          e_median_abs_mrad=round(float(np.median(np.abs(e))) * 1e3, 1),
                          pred_median_abs_mrad=round(float(np.median(np.abs(pred))) * 1e3, 1),
                          slope_rad_per_Nm=round(float(b), 3), kp_eff=round(float(1 / b), 2) if b > 0 else None,
                          intercept_mrad=round(float(a) * 1e3, 1), r=round(float(r), 3),
                          resid_model_median_abs_mrad=round(float(np.median(np.abs(res))) * 1e3, 1))
        R = rows[J[j]]
        print(f"  {J[j]:13s} n {R['n']:6d}  g p10..p90 {R['g_p10_p90']} N.m  |e| med {R['e_median_abs_mrad']:6.1f} mrad"
              f"  |g/kp| med {R['pred_median_abs_mrad']:6.1f}  slope {R['slope_rad_per_Nm']:6.3f} rad/N.m (model {1 / KP:.3f})"
              f"  kp_eff {R['kp_eff']}  intercept {R['intercept_mrad']:6.1f} mrad  r {R['r']:.2f}")
    return rows


out = {"kp": KP, "c": C, "offsets_deg": off.tolist(), "datasets": {}}
for spec in specs:
    name, rest = spec.split("=", 1)
    path, fps = rest.split("@") if "@" in rest else (rest, "30")
    fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    hold = int(round(0.3 * fps))
    Qs, Es, Ms = [], [], []
    for e in np.unique(ep):
        a, s = A[ep == e].astype(float), S[ep == e].astype(float)
        da, ds = np.abs(np.diff(a, axis=0)), np.abs(np.diff(s, axis=0))
        still = (da < 0.2) & (ds < 0.2)
        run = np.zeros_like(still, int)
        for k in range(len(still)):
            run[k] = np.where(still[k], (run[k - 1] if k else 0) + 1, 0)
        ok = np.vstack([np.zeros((1, 6), bool), run >= hold])[:, :5]
        keep = ok.any(1)
        q = np.empty((len(s), 6))
        q[:, :5] = np.radians(s[:, :5] - off)
        q[:, 5] = -0.17 + s[:, 5] / 100 * (1.75 + 0.17)
        Qs.append(q[keep]); Es.append(np.radians(a[keep, :5] - s[keep, :5])); Ms.append(ok[keep])
    out["datasets"][name] = report(f"{name} ({os.path.basename(path)}, {fps:g} fps)", np.vstack(Qs), np.vstack(Es), np.vstack(Ms))

# method check on the sim's own steady samples (free phase: approach / descend, no box in hand)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sag import HOLD, V_STEADY  # noqa: E402
R = np.load(os.path.join(sim_dir, "traj.npz"))
q, qd, tgt, cmd, ph, dt = R["q"], R["qd"], R["tgt"], R["cmd"], R["phase"], float(R["dt"])
same = np.zeros(cmd.shape, bool)
same[:, 1:] = np.abs(np.diff(cmd, axis=1)) < 1e-12
run = np.zeros(cmd.shape, np.int32)
for k in range(1, cmd.shape[1]):
    run[:, k] = np.where(same[:, k], run[:, k - 1] + 1, 0)
ok = (run >= int(round(HOLD / dt))) & (np.abs(tgt - cmd) < 1e-9) & (np.abs(qd) < V_STEADY) & ~np.isnan(q)
ok &= ((ph == 0) | (ph == 1))[..., None]
keep = ok[..., :5].any(-1)
out["sim_free_phase"] = report(f"sim {os.path.basename(sim_dir)} free phase (method check)",
                               q[keep], (tgt - q)[keep][:, :5], ok[keep][:, :5])
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_sag_model" + ("_offsets" if OFFS else "") + ".json")
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)
