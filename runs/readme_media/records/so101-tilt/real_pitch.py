"""so101-tilt: how much real SO-101 operators pitch the hand while lifting a grasped object (LeRobot teleop datasets),
against the scripted expert's lift.

Per episode (observation.state, degrees; joint map sign +1, offset 0 = so101Sim rl/real/so101_real.py DEFAULT_MAP):
  grasp   the first squeeze: the leader's gripper command is >= 3 points below the follower's gripper position for
          >= 0.5 s while the follower's gripper is > 3 % open (it is blocked by the object); the grasp frame is the
          first frame of that run at which the follower's gripper has stopped (|change| < 0.5 point / frame).
  lift    in the same squeeze run, within 2.0 s after the grasp frame, the frame where the gripperframe site is highest;
          kept only if it rose >= 3 cm.
  pitch   hand_tilt = angle between the gripper body's z axis (wrist-roll axis, along the jaws) and world z, by MuJoCo
          FK of the SO-101 model (assets/so101/so101.xml = Menagerie's); dpitch = change of shoulder_lift + elbow_flex
          + wrist_flex (their axes are parallel: the hand's pitch in the arm plane), grasp -> lift, deg. A calibration
          offset per joint cancels in dpitch; it does not cancel in hand_tilt.

    <so101Sim>/urlab_bridge/.venv/bin/python -I real_pitch.py <MetalSim>/assets/so101/scene_box_rl.xml \
        name=<file.parquet>@fps ... [--offsets=<so101Sim>/rl/real/joint_map.json]
  --offsets: subtract the owner's calibrated offsets (another arm's; a sensitivity check of hand_tilt only).
"""
import json
import os
import sys

import mujoco
import numpy as np
import pyarrow.parquet as pq

OFFS = [a for a in sys.argv[1:] if a.startswith("--offsets=")]
OFF = np.array(json.load(open(OFFS[0].split("=", 1)[1]))["offsets_deg"]) if OFFS else np.zeros(5)
args = [a for a in sys.argv[1:] if not a.startswith("--offsets=")]
xml, specs = args[0], args[1:]
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
J = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
qa = np.array([m.joint(j).qposadr[0] for j in J])
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")


def fk(s_deg):
    q = np.radians(s_deg[:5] - OFF)
    d.qpos[:] = m.qpos0
    d.qpos[qa[:5]] = q
    mujoco.mj_kinematics(m, d)
    Rg = d.xmat[gb].reshape(3, 3)
    return float(d.site_xpos[site, 2]), float(np.degrees(np.arccos(np.clip(Rg[2, 2], -1, 1))))


def runs(mask):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j < len(mask) and mask[j]:
                j += 1
            out.append((k, j))
            k = j
        else:
            k += 1
    return out


res = {}
for spec in specs:
    name, rest = spec.split("=", 1)
    path, fps = rest.split("@")
    fps = float(fps)
    t = pq.read_table(path).to_pandas()
    A, S, ep = np.stack(t["action"].to_numpy()), np.stack(t["observation.state"].to_numpy()), t["episode_index"].to_numpy()
    rows = []
    for e in np.unique(ep):
        a, s = A[ep == e].astype(float), S[ep == e].astype(float)
        squeeze = (a[:, 5] <= s[:, 5] - 3.0) & (s[:, 5] > 3.0)
        cand = [(i, j) for i, j in runs(squeeze) if j - i >= int(0.5 * fps)]
        if not cand:
            rows.append(dict(ep=int(e), skip="no squeeze"))
            continue
        i, j = cand[0]
        k = i
        while k + 1 < j and abs(s[k + 1, 5] - s[k, 5]) >= 0.5:
            k += 1
        hi = min(j, k + int(2.0 * fps) + 1)
        z = np.array([fk(s[n])[0] for n in range(k, hi)])
        n_l = k + int(np.argmax(z))
        z0, tilt0 = fk(s[k])
        z1, tilt1 = fk(s[n_l])
        if z1 - z0 < 0.03:
            rows.append(dict(ep=int(e), skip=f"rose {100 * (z1 - z0):.1f} cm"))
            continue
        dp = float((s[n_l, 1] + s[n_l, 2] + s[n_l, 3]) - (s[k, 1] + s[k, 2] + s[k, 3]))
        rows.append(dict(ep=int(e), t_grasp=round(k / fps, 2), t_lift=round(n_l / fps, 2), rise_cm=round(100 * (z1 - z0), 1),
                         hand_tilt_grasp=round(tilt0, 1), hand_tilt_lift=round(tilt1, 1), dpitch=round(dp, 1),
                         grip_pct=round(float(s[k, 5]), 1)))
    ok = [r for r in rows if "skip" not in r]
    arr = lambda key: np.array([r[key] for r in ok])
    q = lambda x: [round(float(np.percentile(x, p)), 1) for p in (10, 50, 90)]
    summ = dict(episodes=len(rows), used=len(ok), skipped={r["ep"]: r["skip"] for r in rows if "skip" in r},
                rise_cm_p10_50_90=q(arr("rise_cm")), hand_tilt_grasp_p10_50_90=q(arr("hand_tilt_grasp")),
                hand_tilt_lift_p10_50_90=q(arr("hand_tilt_lift")), dpitch_p10_50_90=q(arr("dpitch")),
                abs_dpitch_p10_50_90=q(np.abs(arr("dpitch"))),
                frac_abs_dpitch_ge_30=round(float(np.mean(np.abs(arr("dpitch")) >= 30)), 3))
    print(name, json.dumps(summ))
    res[name] = dict(path=os.path.basename(path), fps=fps, summary=summ, episodes=rows)
json.dump(dict(offsets_deg=OFF.tolist(), datasets=res), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
          "real_pitch_offsets.json" if OFFS else "real_pitch.json"), "w"), indent=1)
