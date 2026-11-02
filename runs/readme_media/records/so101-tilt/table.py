"""so101-tilt: the ablation table from out/*.json (medians over seeds 10000-10099; seed 10051 alone in the last cols).
    .venv/bin/python runs/readme_media/scratch/so101-tilt/table.py [name ...]"""
import json
import os
import sys

import numpy as np

P = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[1:] or ["base", "servo_7v4_real", "condim3", "condim4", "tors1e-5", "tors0p03", "mu0p5", "solref0p02",
                         "impratio1", "impratio100", "pyramidal", "noslip10", "dz0p016", "dz0", "dz-0p008"]
print("| run | success | tilt lift start | tilt 3 s | tilt 6 s (p90) | hand pitch change lift->6 s | box in-plane change "
      "| box out-of-plane lift / 6 s | in-hand rotation (p90) | 10051 tilt lift / 6 s | 10051 in-hand |")
print("|" + "---|" * 11)
for n in names:
    f = os.path.join(P, "out", n + ".json")
    if not os.path.exists(f):
        continue
    J = json.load(open(f))
    E = [e for e in J["episodes"] if 10000 <= e["seed"] < 10100]
    g = lambda k: np.array([e.get(k, np.nan) for e in E], float)
    med = lambda a: float(np.nanmedian(a))
    p90 = lambda a: float(np.nanpercentile(a, 90))
    dh = g("grip_inplane_6s_deg") - g("grip_inplane_lift_start_deg")
    db = g("box_inplane_6s_deg") - g("box_inplane_lift_start_deg")
    d51 = [e for e in J["episodes"] if e["seed"] == 10051][0]
    ok = sum(e["success"] for e in E)
    print(f"| {n} | {ok} | {med(g('tilt_lift_start_deg')):.1f} | {med(g('tilt_3s_deg')):.1f} | "
          f"{med(g('tilt_6s_deg')):.1f} ({p90(g('tilt_6s_deg')):.1f}) | {med(dh):.1f} | {med(db):.1f} | "
          f"{med(g('box_outplane_lift_start_deg')):.1f} / {med(g('box_outplane_6s_deg')):.1f} | "
          f"{med(g('inhand_rot_deg')):.1f} ({p90(g('inhand_rot_deg')):.1f}) | "
          f"{d51.get('tilt_lift_start_deg')} / {d51['tilt_6s_deg']} | {d51.get('inhand_rot_deg')} |")
