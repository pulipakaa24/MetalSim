"""ms/frame per scene x fast-math mode from runs/mpm/speed_fm/<mode>/<scene>_r<rep>.json (median over reps)."""
import glob
import json
import os

import numpy as np

B = "runs/mpm/speed_fm/"
MODES = ["off", "cuda", "relaxed"]
SCENES = ["granular", "multi_material", "twoway_capture", "anymal_capture", "plate_drag", "column_a2", "column_a8"]
print("| scene | " + " | ".join(MODES) + " | cuda / off | relaxed / off |")
print("|---" * (len(MODES) + 3) + "|")
for s in SCENES:
    v = {}
    for m in MODES:
        ms = [json.load(open(f))["ms_median"] for f in glob.glob(os.path.join(B, m, f"{s}_r*.json"))]
        v[m] = float(np.median(ms)) if ms else float("nan")
    print(f"| {s} | " + " | ".join(f"{v[m]:.1f}" for m in MODES)
          + f" | {v['cuda'] / v['off']:.2f} | {v['relaxed'] / v['off']:.2f} |")
