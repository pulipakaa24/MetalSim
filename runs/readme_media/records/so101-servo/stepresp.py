"""Step responses of the SO-101 servo options, the orchestrator's protocol (reproduced for the old model within 4
points of overshoot, step_probe.py): so101Sim's scene (MuJoCo 3.10, so101Sim's venv), all joints at qpos0, targets
= positions, 1 s settling, then one joint's target +STEP rad; 1 s recorded at every 5 ms physics step.

Metrics per joint: overshoot past the target (orchestrator's: (max - target) / step) and past the final value
((max - final) / |final - start|), steady-state error (target - final, final = mean of the last 100 ms), target and
final-value crossings, settle time to 2 % of the step around the final value, peak speed, peak acceleration
(finite difference of qvel at 5 ms). Variants: the old options, the new ones, and the ablation of the new 7.4 V one.

  <so101Sim>/urlab_bridge/.venv/bin/python stepresp.py <out.json>
"""
import json
import sys

import mujoco
import numpy as np

sys.path.append("~/MetalSim")
from metalsim.render.film.so101 import ARM_JOINTS, apply_servo, servo_layer  # noqa: E402

XML = "~/so101Sim/mujoco_menagerie/robotstudio_so101/scene_box_rl.xml"
INF = float("inf")
NODYN = dict(servo_delay=0.0, host_delay=0.0, speed_limit=INF, accel_limit=INF)
VARIANTS = {
    "12v (old)": ("12v", {}),
    "7v4 (old)": ("7v4", {}),
    "12v_real": ("12v_real", {}),
    "7v4_real": ("7v4_real", {}),
    # ablation of 7v4_real, cumulative
    "A0 gains+line only": ("7v4_real", NODYN),
    "A1 +(a) servo delay 1.05 ms": ("7v4_real", dict(NODYN, servo_delay=0.0010469000118575051)),
    "A2 +(b) speed+accel profile": ("7v4_real", dict(host_delay=0.0)),
    "A3 +(c) host latency 3 ms (= 7v4_real)": ("7v4_real", {}),
    # one at a time off the full model
    "full -(d) no back-EMF line": ("7v4_real", dict(line=False)),
    "full -(b) accel only off": ("7v4_real", dict(accel_limit=INF)),
    "full, D=32 @ 10 ms loop": ("7v4_real", dict(d_cycle_s=0.010)),
    "full, BAM kt/R stall 3.41 N.m": ("7v4_real", dict(stall_Nm=3.4103, noload_rad_s=5.4225)),
    "full, host latency 10 ms": ("7v4_real", dict(host_delay=0.010)),
}


def run(servo, over, j, step, settle=200, rec=200):
    m = mujoco.MjModel.from_xml_path(XML)
    ap = apply_servo(m, servo, **over)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    act = np.arange(m.nu)
    cmd = d.qpos[:6].copy()
    lay = servo_layer(ap, 1)
    if lay is not None:
        lay.reset(cmd)

    def go(n):
        qs, vs = [], []
        for _ in range(n):
            d.ctrl[:] = lay.tick(cmd[None])[0] if lay is not None else cmd
            mujoco.mj_step(m, d)
            qs.append(d.qpos[j]), vs.append(d.qvel[j])
        return np.array(qs), np.array(vs)

    go(settle)
    q0, tgt = d.qpos[j], cmd[j] + step
    sag = cmd[j] - q0
    cmd[j] = tgt
    nc = []
    q, v = go(rec)
    dt = m.opt.timestep
    y = q - q0
    yf = y[-20:].mean()
    acc = np.diff(np.concatenate([[0.0], v])) / dt
    out = np.where(np.abs(y - yf) > 0.02 * abs(step))[0]
    sgn = np.sign(step)
    e = y - yf                                      # crossings of the final value, 0.5 % of the step hysteresis
    s_ = np.sign(e)[np.abs(e) > 0.005 * abs(step)]
    et = y - step                                   # crossings of the commanded change (orchestrator's "target")
    st_ = np.sign(et)[np.abs(et) > 0.005 * abs(step)]
    return dict(overshoot_pct=float(100 * max(0.0, (sgn * e).max()) / max(abs(yf), 1e-9)),
                overshoot_cmd_pct=float(100 * max(0.0, (sgn * et).max()) / abs(step)),
                sag_before_mrad=float(1e3 * sag), ss_error_mrad=float(1e3 * (step - yf)),
                crossings=int(np.sum(np.diff(s_) != 0)), crossings_cmd=int(np.sum(np.diff(st_) != 0)),
                settle_ms=float((out[-1] + 1) * dt * 1e3) if len(out) else 0.0,
                t90_ms=float((np.argmax(sgn * y >= 0.9 * abs(yf)) + 1) * dt * 1e3),
                peak_speed=float(np.abs(v).max()), peak_acc=float(np.abs(acc).max()), max_ncon=int(d.ncon),
                ), ap


res = {}
for name, (servo, over) in VARIANTS.items():
    res[name] = {}
    for step in (0.06, -0.5):
        for j in (1, 2, 3):
            r, ap = run(servo, over, j, step)
            res[name][f"{ARM_JOINTS[j]}@{step}"] = r
    res[name]["servo"] = {k: ap[k] for k in ap if k in ("kp", "kv", "forcerange_Nm", "damping_Nms", "setpoint",
                                                         "overrides", "armature", "frictionloss")}
    r = res[name]
    k6 = lambda j: r[f"{ARM_JOINTS[j]}@0.06"]
    print(f"{name:40s}", " | ".join(f"{ARM_JOINTS[j][:5]} os {k6(j)['overshoot_pct']:4.1f}% x{k6(j)['crossings']}"
                                     f" st {k6(j)['settle_ms']:3.0f} sag {k6(j)['sag_before_mrad']:4.0f} v {k6(j)['peak_speed']:.2f}"
                                     f" a {k6(j)['peak_acc']:4.0f} V {r[f'{ARM_JOINTS[j]}@-0.5']['peak_speed']:.2f}"
                                     f" A {r[f'{ARM_JOINTS[j]}@-0.5']['peak_acc']:4.0f}" for j in (1, 2, 3)))
json.dump(dict(script="runs/readme_media/scratch/so101-servo/stepresp.py", mujoco=mujoco.__version__,
               protocol="qpos0, targets = positions, 1 s settle, +step on one joint, 1 s at 5 ms", results=res),
          open(sys.argv[1], "w"), indent=1)
