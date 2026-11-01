"""Control-style metrics for the so101-smooth stream (numpy only; used by eval_cpu.py and eval_metalsim.py).

Every metric covers each episode from its first to its last control step (300 steps = 6 s, or until the box leaves the
workspace), for every controller alike, so the scripted expert, the unpenalised policy and the smooth policy are
measured on the same window.

* torque-saturation fraction, per joint: share of samples at which the servo force is at its force limit. CPU: every
  5 ms physics step, the force MuJoCo applies (kp (ctrl - q) - kv qdot clamped to the forcerange, as realism.py);
  MetalSim: the last physics step of every 20 ms control step (MuJoCo Warp ``actuator_force``, as compare_servo.py).
* action rate: mean |a_t - a_{t-1}| over the six joints and the control steps after the first, in normalised action
  units (so101Sim scripted_expert.py's "mean |dctrl|/tick (normalized)"), and the same in radians of joint target
  per 20 ms step.
* actions at the limit: share of action components with |a| > 0.99 (scripted_expert.py's "saturation").
* peak joint speed: the largest |qdot| per joint over all episodes, and the median over episodes of each episode's
  largest arm-joint speed. CPU: every physics step; MetalSim: every control step.
* success (box centre above 0.10 m for 10 control steps), box still above 0.10 m at the episode's end ("up at 6 s"),
  time to lift (the control step at which success first holds, x 0.02 s, as eval_lift_native.py).
"""
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CTRL_DT = 0.02


def control_style(action, ctrl, ends, sat, sat_ok, speed, speed_ok, success_t, lifted_end, hmax):
    """action, ctrl: (E, T, 6) per control step; ends: (E,) last control step index; sat: (E, K, 6) bool at the force
    limit, sat_ok: (E, K) samples inside the episode; speed: (E, K2, 6) joint speeds, speed_ok: (E, K2);
    success_t: (E,) first success step or -1; lifted_end: (E,) bool; hmax: (E,) highest box centre."""
    E = action.shape[0]
    da, dc, lim = [], [], []
    for e in range(E):
        n = int(ends[e]) + 1
        da.append(np.abs(np.diff(action[e, :n], axis=0)))
        dc.append(np.abs(np.diff(ctrl[e, :n], axis=0)))
        lim.append(np.abs(action[e, :n]) > 0.99)
    da, dc, lim = np.concatenate(da), np.concatenate(dc), np.concatenate(lim)
    v = np.abs(speed)
    v_ok = v[speed_ok]                                              # (samples, 6)
    ep_peak = np.array([np.nanmax(v[e][speed_ok[e]]) for e in range(E)])
    ok = success_t >= 0
    return dict(
        episodes=int(E), success=int(ok.sum()), up_at_end=int(np.sum(lifted_end)),
        time_to_lift_mean_s=float(success_t[ok].mean() * CTRL_DT) if ok.any() else None,
        time_to_lift_median_s=float(np.median(success_t[ok]) * CTRL_DT) if ok.any() else None,
        max_height_mean_m=float(np.mean(hmax)),
        torque_sat_frac=dict(zip(JOINTS, [float(sat[..., j][sat_ok].mean()) for j in range(6)])),
        torque_sat_frac_arm5_mean=float(np.mean([sat[..., j][sat_ok].mean() for j in range(5)])),
        action_rate=float(da.mean()), action_rate_per_joint=dict(zip(JOINTS, da.mean(0).round(4).tolist())),
        ctrl_rate_rad_per_step=float(dc.mean()), ctrl_rate_rad_per_step_median=float(np.median(dc)),
        action_at_limit_frac=float(lim.mean()),
        action_at_limit_frac_per_joint=dict(zip(JOINTS, lim.mean(0).round(4).tolist())),
        peak_speed_rad_s=dict(zip(JOINTS, v_ok.max(0).round(3).tolist())),
        peak_speed_max_rad_s=float(v_ok.max()), peak_speed_episode_median_rad_s=float(np.median(ep_peak)),
        speed_p99_rad_s=float(np.percentile(v_ok, 99)))


def row(name, M):
    s = M["torque_sat_frac"]
    return (f"{name:28s} succ {M['success']:3d} up {M['up_at_end']:3d} ttl {M['time_to_lift_mean_s'] or float('nan'):.2f}s "
            f"| sat% " + " ".join(f"{100 * s[j]:5.1f}" for j in JOINTS)
            + f" | da {M['action_rate']:.4f} dctrl {M['ctrl_rate_rad_per_step']:.4f} rad | a@lim {100 * M['action_at_limit_frac']:.1f}%"
            + f" | vpk {M['peak_speed_max_rad_s']:.2f} ep-med {M['peak_speed_episode_median_rad_s']:.2f}")
