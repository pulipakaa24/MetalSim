"""Collect the so101-smooth evaluations (cpu/<name>/metrics.json, metalsim/<name>/metrics.json) into one table.
Usage (any python with numpy): python table.py [names...] -> prints markdown, writes results.json next to this file."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = sys.argv[1:] or ["expert_7v4", "orig_7v4", "isaac_scaled_7v4", "isaac_unscaled_raw_7v4", "orig_12v",
                         "expert_12v"]
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
rows, out = [], {}
for n in NAMES:
    for where in ("cpu", "metalsim"):
        p = os.path.join(HERE, where, n, "metrics.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        M = d["metrics"]
        sw = d.get("software", {}).get("mujoco_warp", {})
        out[f"{where}/{n}"] = dict(M, mujoco_warp=sw.get("git") if isinstance(sw, dict) else sw,
                                   cpu_agreement=d.get("cpu_agreement"), capacity_overflow=d.get("capacity_overflow"))
        s = M["torque_sat_frac"]
        ttl = M["time_to_lift_mean_s"]
        rows.append(f"| {n} | {where} | {M['success']} | {M['up_at_end']} | {'%.2f s' % ttl if ttl is not None else '-'} "
                    f"| {M['action_rate']:.4f} | {M['ctrl_rate_rad_per_step']:.4f} | {100 * M['action_at_limit_frac']:.1f} % "
                    f"| {100 * M['torque_sat_frac_arm5_mean']:.0f} % ({' '.join('%.0f' % (100 * s[j]) for j in J[:5])}) "
                    f"| {100 * s['gripper']:.0f} % | {M['peak_speed_max_rad_s']:.2f} | {M['peak_speed_episode_median_rad_s']:.2f} |")
hdr = ("| controller | sim | success | up at 6 s | time to lift | action change / step | target change / step (rad) "
       "| actions at limit | torque sat, arm mean (pan lift elbow flex roll) | torque sat, gripper | peak speed, max (rad/s) "
       "| peak speed, episode median |\n|" + "---|" * 12)
print(hdr)
print("\n".join(rows))
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
