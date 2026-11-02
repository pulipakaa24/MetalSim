"""so101-tilt: torque budget about the jaw-closing axis for the detail seed's hold (from out/base.json's detail).
Gravity torque of the box about the axis through the jaw contacts' centre of pressure (along the fixed pads' normal,
gripper x) against (a) MuJoCo's torsional friction limit sum(mu_t * fn) (elliptic cone, no tangential load), (b) the
sliding-friction moment the contact spread can supply, sum(mu * fn * r_i), r_i = distance of contact i from that axis,
and what the solver actually used: sum |torsion| and the moment of the tangential forces about the axis.
    python3 runs/readme_media/scratch/so101-tilt/torque_budget.py [out/base.json] [seed]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "base.json")
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10051
e = [e for e in json.load(open(f))["episodes"] if e["seed"] == seed][0]
print("seed", seed, "box mass", e["mass_kg"], "kg")
for t, D in e["detail"].items():
    jaw = [c for c in D["contacts"] if c["geom"] != "floor"]
    P = np.array([c["pos_grip_mm"] for c in jaw]) * 1e-3          # relative to the box COM, gripper frame
    fn = np.array([c["force"][0] for c in jaw])
    mu = np.array([c["friction"][0] for c in jaw])
    mut = np.array([c["friction"][2] for c in jaw])
    tors = np.array([c["force"][3] for c in jaw])
    ft = np.array([np.hypot(c["force"][1], c["force"][2]) for c in jaw])
    cop = (fn[:, None] * P).sum(0) / fn.sum()
    r = np.linalg.norm((P - cop)[:, 1:], axis=1)                     # distance from the gripper-x axis through the CoP
    util = ft / (mu * fn)
    tg = abs(D["gravity_torque_about_x_Nm"])
    print(f"t {t} s: {len(jaw)} jaw contacts, fn sum {fn.sum():.1f} N | gravity torque about the axis {tg:.4f} N.m "
          f"(max at 90 deg: {D['weight_N'] * np.linalg.norm(np.array(D['cop_box_mm'])) * 1e-3:.4f}) | "
          f"torsional limit sum(mu_t fn) {np.sum(mut * fn):.3f} N.m, used sum|torsion| {np.abs(tors).sum():.4f} | "
          f"sliding moment limit sum(mu fn r) {np.sum(mu * fn * r):.3f} N.m (r {r.min() * 1e3:.1f}-{r.max() * 1e3:.1f} mm) | "
          f"tangential cone use |ft|/(mu fn) max {util.max():.3f}, median {np.median(util):.3f}")
