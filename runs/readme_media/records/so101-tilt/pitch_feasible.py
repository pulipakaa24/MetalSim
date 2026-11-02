"""so101-tilt: can the SO-101 reach the expert's lift target (TCP at the grasp xy, z 0.20 m) with the hand pitch it
had when the lift started, within its joint limits, with the wrist roll held at its lift-start value (a free roll spins the
hand, and the box with it, about the jaw axis)? Bounded least squares (scipy) on the kinematic model, 5 arm
joints, residual = [TCP - target (m), w * (pitch - pitch0) (rad)], w = 1e-3 m/rad (position dominates: 1 rad of pitch error weighs as 1 mm), 24 starts per seed (the lift-start
pose, the baseline's 6 s pose, 22 random within the limits). Reports the best feasible pitch error per seed.
    .venv/bin/python runs/readme_media/scratch/so101-tilt/pitch_feasible.py   (reads out/base.json)"""
import json
import os
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../..")))
import metalsim.render.film.so101_expert as X  # noqa: E402
from metalsim.render.film.so101 import ARM_JOINTS, ROOT, SCENE  # noqa: E402

m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, SCENE))
d = mujoco.MjData(m)
qa = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS[:5]])
lo, hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS[:5]]].T
site = m.site("gripperframe").id
ROW = np.array([0.0, 1.0, 1.0, 1.0, 0.0])


def tcp(q):
    d.qpos[qa] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[site] + d.site_xmat[site].reshape(3, 3) @ X.TCP_LOCAL


J = json.load(open(os.path.join(HERE, "out", "base.json")))
rng = np.random.default_rng(0)
rows = []
for e in J["episodes"]:
    if not (10000 <= e["seed"] < 10100):
        continue
    q0, q6 = np.array(e["q_lift_start"]), np.array(e["q_6s"])
    tgt = tcp(q0).copy()
    tgt[2] = X.LIFT_Z
    p0 = ROW @ q0
    best = None
    starts = [q0, q6] + [rng.uniform(lo, hi) for _ in range(22)]
    lo_s, hi_s = lo.copy(), hi.copy()
    lo_s[4], hi_s[4] = q0[4] - 1e-6, q0[4] + 1e-6           # wrist roll held at its lift-start value
    for s in starts:
        s = s.copy()
        s[4] = q0[4]
        r = least_squares(lambda q: np.concatenate([tcp(q) - tgt, [1e-3 * (ROW @ q - p0)]]), np.clip(s, lo_s, hi_s),
                          bounds=(lo_s, hi_s))
        pos_err = float(np.linalg.norm(tcp(r.x) - tgt))
        if pos_err < 1e-3 and (best is None or abs(ROW @ r.x - p0) < abs(best[1])):
            best = (r.x, float(ROW @ r.x - p0), pos_err)
    rows.append(dict(seed=e["seed"], pitch_change_needed_deg=None if best is None else round(float(np.degrees(best[1])), 1),
                     baseline_pitch_change_deg=round(float(np.degrees(ROW @ q6 - p0)), 1),
                     q=None if best is None else best[0].round(3).tolist()))
pc = np.array([r["pitch_change_needed_deg"] for r in rows if r["pitch_change_needed_deg"] is not None])
print("seeds", len(rows), "with a solution", len(pc), "| smallest reachable |pitch change| at the lift target, deg: p10/50/90",
      np.percentile(np.abs(pc), [10, 50, 90]).round(1), "| seeds that can hold the pitch within 2 deg:", int((np.abs(pc) < 2).sum()))
print("baseline pitch change p10/50/90", np.percentile([r["baseline_pitch_change_deg"] for r in rows], [10, 50, 90]).round(1))
print("seed 10051", [r for r in rows if r["seed"] == 10051][0])
json.dump(rows, open(os.path.join(HERE, "out", "pitch_feasible.json"), "w"), indent=1)
