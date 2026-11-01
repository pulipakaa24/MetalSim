"""Realism check of so101Sim's lift teacher on CPU MuJoCo (the so101_lift film, docs/renders/SO101_LIFT.md), from
reference.py's rollouts (reference.npz: 100 episodes, every 5 ms physics step).

Servo limits (Feetech STS3215; the SO-101 follower's six servos are the 7.4 V, 1/345-gear C001 in The Robot Studio's
bill of materials, github.com/TheRobotStudio/SO-ARM100 README):
* C001 at 7.4 V: stall torque 19.5 kg.cm = 1.91 N.m, no-load speed 52 rpm = 5.45 rad/s (Feetech datasheet
  "STS3215 200620a": "Stall Torque 19.5kg.cm@7.4V", "Stall Speed 52RPM@7.4V", i.e. 0.192 s/60 deg);
  at 6 V: 0.238 s/60 deg = 4.40 rad/s (vendor listings); the SO-ARM100 README quotes 16.5 kg.cm at 6 V (1.62 N.m);
* the 12 V version (C018): 30 kg.cm = 2.94 N.m stall, 0.222 s/60 deg = 4.72 rad/s; 2.94 N.m is the forcerange of
  every actuator in the Menagerie model the scene uses.
A DC servo at full supply voltage is bounded by |tau / tau_stall + omega / omega_0| <= 1 (its linear torque-speed line,
in all four quadrants); the check counts the physics steps outside that envelope (by more than 2 %), with tau the
drive torque at the joint (the servo force minus the joint's viscous damping).

Usage: .venv/bin/python runs/film/so101_lift/port/realism.py <reference_dir> [out.json]   (CPU, numpy only)
"""
import json
import os
import sys

import numpy as np

ref_dir = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "realism.json")
R = np.load(os.path.join(ref_dir, "reference.npz"))
RJ = json.load(open(os.path.join(ref_dir, "reference.json")))
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RPM = 2 * np.pi / 60
SERVO = {"C001_7.4V": dict(stall_Nm=19.5 * 0.0980665, noload_rad_s=52 * RPM),
         "C001_6V": dict(stall_Nm=16.5 * 0.0980665, noload_rad_s=(np.pi / 3) / 0.238),
         "C018_12V": dict(stall_Nm=30 * 0.0980665, noload_rad_s=(np.pi / 3) / 0.222)}
S = R["sub"]                                   # (E, steps, 18): q, qd, ctrl at the start of each physics step
E = S.shape[0]
q, qd, u = S[..., 0:6], S[..., 6:12], S[..., 12:18]
A = R["actuator"]                              # gain, bias1, bias2, force lo, force hi per actuator
tau = np.clip(A[:, 0] * u + A[:, 1] * q + A[:, 2] * qd, A[:, 3], A[:, 4])
ok = ~np.isnan(qd[..., 0])
dt = 0.005

joints = {}
for j, name in enumerate(J):
    v, t = np.abs(qd[..., j][ok]), np.abs(tau[..., j][ok])
    per_ep = np.nanmax(np.abs(qd[..., j]), axis=1)
    row = dict(peak_speed_rad_s=float(v.max()), p99_speed_rad_s=float(np.percentile(v, 99)),
               median_episode_peak_rad_s=float(np.median(per_ep)),
               peak_torque_Nm=float(t.max()), torque_saturated_frac=float((t >= A[j, 4] - 1e-9).mean()))
    # the drive torque at the joint: the position servo's force minus the joint's viscous damping. A DC servo at full
    # supply voltage (either polarity) is bounded, in all four quadrants, by |tau / tau_stall + omega / omega_0| <= 1
    # (motoring: tau <= tau_stall (1 - omega / omega_0); braking may exceed tau_stall). The model's clamped servo
    # force minus damping (forcerange / damping = omega_0) is exactly such a motor.
    w = qd[..., j][ok]
    tn = (tau[..., j] - R["joint_damping"][j] * qd[..., j])[ok]
    row["model_terminal_speed_rad_s"] = float(A[j, 4] / R["joint_damping"][j])
    for k, sv in SERVO.items():
        x = np.abs(tn / sv["stall_Nm"] + w / sv["noload_rad_s"])
        row[f"frac_over_noload_{k}"] = float((v > sv["noload_rad_s"]).mean())
        row[f"episodes_over_noload_{k}"] = int((per_ep > sv["noload_rad_s"]).sum())
        row[f"frac_outside_dc_envelope_{k}"] = float((x > 1.02).mean())
        row[f"dc_envelope_p99_{k}"] = float(np.percentile(x, 99))
        row[f"frac_torque_over_stall_{k}"] = float((t > sv["stall_Nm"]).mean())
    joints[name] = row

# grasp and hold: from the first success step to the episode end
eps = RJ["episodes"]
hold = dict(grip_q=[], jaw_contacts=[], floor_contacts=[], other_contacts=[], normal_force=[], pen=[], slip=[],
            grip_tau=[], tilt=[], sat=[], flips=[], dq_amp=[])
lift_v, lift_a, approach_s = [], [], []
for e, ep in enumerate(eps):
    T = int(R["ends"][e]) + 1
    box, grip = R["box"][e, 1:T + 1], R["grip"][e, 1:T + 1]
    vz = np.diff(R["box"][e, :T + 1, 2]) / 0.02
    lift_v.append(float(vz.max()))
    lift_a.append(float(np.abs(np.diff(vz)).max() / 0.02))
    c = R["box_contacts"][e, :T]
    first_jaw = int(np.argmax(c[:, 0] > 0)) if (c[:, 0] > 0).any() else None
    approach_s.append(first_jaw * 0.02 if first_jaw is not None else None)
    if ep["success_t"] is None:
        continue
    s = slice(ep["success_t"], T)
    hold["grip_q"].append(np.median(R["qpos"][e, 1:T + 1][s, 5]))
    hold["jaw_contacts"].append(np.median(c[s, 0]))
    hold["floor_contacts"].append(float((c[s, 1] > 0).mean()))
    hold["other_contacts"].append(float((c[s, 2] > 0).mean()))
    hold["normal_force"].append(np.median(R["box_normal_force"][e, s]))
    hold["pen"].append(float(-R["max_pen"][e, s].min()))
    rel = box[s] - grip[s]
    hold["slip"].append(float(np.linalg.norm(rel - rel[0], axis=1).max()))
    hold["grip_tau"].append(float(np.median(tau[e, 4 * ep["success_t"]:4 * T, 5])))
    a = R["action"][e, :T]
    hold["sat"].append(float((np.abs(a[s]) > 0.99).mean()))
    sgn = np.sign(a[s][:, :5])
    hold["flips"].append(float((sgn[1:] * sgn[:-1] < 0).mean()))       # arm joints: sign flips per control step
    hold["dq_amp"].append(float(np.median(np.abs(R["qvel"][e, 1:T + 1][s, :5]))))
    qb = R["qpos"][e, 1:T + 1][s, 9:13]                                   # box quaternion (w, x, y, z)
    upz = 1 - 2 * (qb[:, 1] ** 2 + qb[:, 2] ** 2)                         # box z axis . world z
    hold["tilt"].append(float(np.degrees(np.arccos(np.clip(upz, -1, 1))).max()))

med = lambda x: float(np.median(x))
grasp = dict(episodes_held=len(hold["grip_q"]),
             gripper_joint_median_rad=med(hold["grip_q"]),
             jaw_contacts_median=med(hold["jaw_contacts"]),
             floor_contact_frac_median=med(hold["floor_contacts"]),
             non_jaw_arm_contact_frac_median=med(hold["other_contacts"]),
             non_jaw_arm_contact_frac_max=float(np.max(hold["other_contacts"])),
             box_normal_force_median_N=med(hold["normal_force"]),
             max_penetration_mm_median=1e3 * med(hold["pen"]), max_penetration_mm_max=1e3 * float(np.max(hold["pen"])),
             box_slip_in_gripper_mm_median=1e3 * med(hold["slip"]),
             box_slip_in_gripper_mm_max=1e3 * float(np.max(hold["slip"])),
             gripper_torque_median_Nm=med(hold["grip_tau"]),
             box_tilt_deg_median=med(hold["tilt"]), box_tilt_deg_max=float(np.max(hold["tilt"])),
             hold_action_saturated_frac_median=med(hold["sat"]),
             hold_arm_action_sign_flip_per_step_median=med(hold["flips"]),
             hold_arm_joint_speed_median_rad_s=med(hold["dq_amp"]))
act = R["action"]
okA = np.zeros(act.shape[:2], bool)
for e in range(E):
    okA[e, :int(R["ends"][e]) + 1] = True
A_ = act[okA]
policy = dict(action_saturated_frac_all=float((np.abs(A_) > 0.99).mean()),
              action_saturated_frac_per_joint=dict(zip(J, (np.abs(A_) > 0.99).mean(0).round(3).tolist())),
              sign_flip_per_step_per_joint=dict(zip(J, np.mean([(np.sign(act[e, 1:int(R["ends"][e]) + 1])
                                                                 * np.sign(act[e, :int(R["ends"][e])]) < 0).mean(0)
                                                                for e in range(E)], 0).round(3).tolist())),
              ctrl_jump_median_rad=float(np.median(np.abs(np.diff(R["ctrl"], axis=1))[okA[:, 1:]])))
motion = dict(box_peak_vertical_speed_m_s_median=med(lift_v), box_peak_vertical_speed_m_s_max=float(np.max(lift_v)),
              box_peak_vertical_accel_m_s2_median=med(lift_a),
              first_jaw_contact_s_median=med([a for a in approach_s if a is not None]),
              time_to_lift_s_mean=RJ["summary"]["time_to_lift_mean_s"])
out = dict(script="runs/film/so101_lift/port/realism.py", reference=RJ["summary"], episodes=E,
           servo_specs={k: dict(stall_Nm=round(v["stall_Nm"], 3), noload_rad_s=round(v["noload_rad_s"], 3))
                        for k, v in SERVO.items()},
           model_actuator=dict(kp=A[:, 0].tolist(), kv=(-A[:, 2]).tolist(), forcerange=A[:, 4].tolist()),
           joints=joints, grasp_hold=grasp, motion=motion, policy=policy)
json.dump(out, open(out_path, "w"), indent=1)
print(f"{'joint':14s} {'peak':>6s} {'p99':>6s} {'ep-med':>6s} {'>5.45':>7s} {'>4.72':>7s} {'>4.40':>7s} dc7.4 p99   dc12 p99   {'tau pk':>6s} sat  >1.91Nm term")
for name, r in joints.items():
    print(f"{name:14s} {r['peak_speed_rad_s']:6.2f} {r['p99_speed_rad_s']:6.2f} {r['median_episode_peak_rad_s']:6.2f} "
          f"{r['frac_over_noload_C001_7.4V']:7.3f} {r['frac_over_noload_C018_12V']:7.3f} {r['frac_over_noload_C001_6V']:7.3f} "
          f"{r['frac_outside_dc_envelope_C001_7.4V']:5.3f} {r['dc_envelope_p99_C001_7.4V']:4.2f} {r['frac_outside_dc_envelope_C018_12V']:5.3f} {r['dc_envelope_p99_C018_12V']:4.2f} "
          f"{r['peak_torque_Nm']:6.2f} {r['torque_saturated_frac']:.3f} {r['frac_torque_over_stall_C001_7.4V']:.3f} {r['model_terminal_speed_rad_s']:.2f}"
          f"  eps>5.45: {r['episodes_over_noload_C001_7.4V']}")
print(json.dumps(dict(grasp_hold=grasp, motion=motion, policy=policy), indent=1))
