"""12 V against 7.4 V servos for so101Sim's lift teacher (docs/renders/SO101_LIFT.md, "7.4 V servo model"): outcome
counts, eval figures, torque saturation and the failure modes, from the CPU references (reference_servo.py ->
reference.npz) and the MetalSim runs (compare_servo.py -> compare.npz / compare.json). CPU, numpy only.

Usage: .venv/bin/python runs/film/so101_lift_74v/port/analysis.py <scratch dir> [out.json]
  <scratch dir> holds ref_7v4/, ref_7v4_keepdamp/, ref_7v4_gripfull/, compare_7v4/, compare_12v/; the 12 V CPU
  reference is read from runs/readme_media/scratch/so101/ref (the so101_lift stream's).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
scr = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "analysis.json")
REF12 = os.path.join(ROOT, "runs", "readme_media", "scratch", "so101", "ref")
LIFT_Z, DT = 0.10, 0.02


def tilt_deg(q):                                         # box z axis against world z, from (w, x, y, z)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1)))


def outcomes(height, ends=None):
    """Per episode: success step (10 consecutive steps above 0.10 m) or -1, max height, height at the end."""
    E, T = height.shape
    st, hmax, hend = np.full(E, -1), np.zeros(E), np.zeros(E)
    for e in range(E):
        n = (int(ends[e]) + 1) if ends is not None else T
        h, run = height[e, :n], 0
        for t, x in enumerate(h > LIFT_Z):
            run = run + 1 if x else 0
            if run >= 10 and st[e] < 0:
                st[e] = t
        hmax[e], hend[e] = h.max(), h[-1]
    return st, hmax, hend


def categories(st, hmax, hend, oob):
    ok = st >= 0
    return dict(success_held_at_6s=int((ok & (hend > LIFT_Z)).sum()),
                success_then_dropped=int((ok & (hend <= LIFT_Z)).sum()),
                no_success_box_never_above_6cm=int((~ok & (hmax < 0.06)).sum()),
                no_success_box_6_to_10cm=int((~ok & (hmax >= 0.06) & (hmax <= LIFT_Z)).sum()),
                no_success_box_above_10cm_briefly=int((~ok & (hmax > LIFT_Z)).sum()),
                box_left_workspace=int(oob.sum()))


def cpu(d):
    R = np.load(os.path.join(d, "reference.npz"))
    J = json.load(open(os.path.join(d, "reference.json")))
    st, hmax, hend = outcomes(R["height"], R["ends"])
    oob = np.array([e["terminated"] for e in J["episodes"]])
    E = len(st)
    c = R["box_contacts"]
    gq_contact, tilt_before_lift, fn_hold = [], [], []
    for e in range(E):
        n = int(R["ends"][e]) + 1
        jaw = c[e, :n, 0] > 0
        t0 = int(np.argmax(jaw)) if jaw.any() else None
        gq_contact.append(float(R["qpos"][e, t0 + 1, 5]) if t0 is not None else np.nan)   # gripper opening at first jaw contact
        # largest box tilt while the box is still low (centre below 4 cm: before any lift): knocked over at the grasp
        q = R["qpos"][e, 1:n + 1]
        low = R["box"][e, 1:n + 1, 2] < 0.04
        tilt_before_lift.append(float(tilt_deg(q[low, 9:13]).max()) if low.any() else 0.0)
        if st[e] >= 0:
            fn_hold.append(float(np.median(R["box_normal_force"][e, st[e]:n])))
    gq, tb = np.array(gq_contact), np.array(tilt_before_lift)
    ok = st >= 0
    grasp = dict(gripper_opening_at_first_jaw_contact_rad=dict(
                     median_all=float(np.nanmedian(gq)), median_success=float(np.nanmedian(gq[ok])),
                     median_no_success=float(np.nanmedian(gq[~ok])) if (~ok).any() else None),
                 box_tipped_over_60deg_before_lift=dict(all=int((tb > 60).sum()), success=int((tb[ok] > 60).sum()),
                                                        no_success=int((tb[~ok] > 60).sum())),
                 hold_box_normal_force_median_N=float(np.median(fn_hold)) if fn_hold else None)
    return dict(summary=J["summary"], servo=J.get("servo"), wrapper=J.get("wrapper", J.get("script")),
                categories=categories(st, hmax, hend, oob), grasp=grasp,
                no_success_seeds=[10_000 + int(e) for e in np.flatnonzero(~ok)],
                dropped_seeds=[10_000 + int(e) for e in np.flatnonzero(ok & (hend <= LIFT_Z))])


def metalsim(d):
    C = np.load(os.path.join(d, "compare.npz"))
    J = json.load(open(os.path.join(d, "compare.json")))
    st, hmax, hend = outcomes(C["height"])
    oob = (np.linalg.norm(C["box"][:, 1:, :2], axis=2) > 0.45).any(1)
    ok = st >= 0
    return dict(summary=J["metalsim"], servo=J.get("servo"), per_seed=J["per_seed"],
                categories=categories(st, hmax, hend, oob), torque_saturation=J.get("torque_saturation"),
                capacity_overflow=J.get("capacity_overflow"),
                no_success_seeds=[10_000 + int(e) for e in np.flatnonzero(~ok)],
                dropped_seeds=[10_000 + int(e) for e in np.flatnonzero(ok & (hend <= LIFT_Z))],
                seed_10013=dict(success_t_s=round(float(st[13]) * DT, 2) if st[13] >= 0 else None,
                                max_height_m=round(float(hmax[13]), 3), end_height_m=round(float(hend[13]), 3)))


def sat_cpu(path):
    J = json.load(open(path))
    return {j: round(r["torque_saturated_frac"], 4) for j, r in J["joints"].items()}


out = dict(script="runs/film/so101_lift_74v/port/analysis.py",
           cpu={"12v": cpu(REF12), "7v4": cpu(os.path.join(scr, "ref_7v4")),
                "7v4_keep_damping_0.60 (sensitivity)": cpu(os.path.join(scr, "ref_7v4_keepdamp")),
                "7v4_gripper_full_torque (diagnostic)": cpu(os.path.join(scr, "ref_7v4_gripfull"))},
           metalsim={"12v (re-run for saturation)": metalsim(os.path.join(scr, "compare_12v")),
                     "7v4": metalsim(os.path.join(scr, "compare_7v4"))},
           metalsim_12v_film_stream=json.load(open(os.path.join(ROOT, "runs/film/so101_lift/port/compare.json")))["metalsim"],
           cpu_torque_saturation={"12v": sat_cpu(os.path.join(ROOT, "runs/film/so101_lift/port/realism.json")),
                                  "7v4": sat_cpu(os.path.join(HERE, "realism_7v4.json"))})
json.dump(out, open(out_path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for side in ("cpu", "metalsim"):
    for k, v in out[side].items():
        s = v["summary"]
        print(f"{side:8s} {k:38s} succ {s['successes']:3d}  held@6s {s['lifted_at_end']:3d}  hmax {s['max_height_mean']:.3f}  "
              f"t_lift {s['time_to_lift_mean_s']:.2f}  {v['categories']}")
        if "grasp" in v:
            print(" " * 9, v["grasp"])
print("cpu saturation", out["cpu_torque_saturation"])
print("metalsim 10013", {k: v["seed_10013"] for k, v in out["metalsim"].items()})
