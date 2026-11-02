"""Hold test: approach a pose from above and from below, hold 3 s, report the sag and its creep (so101-sag).

so101Sim's scene (SO101LiftEnv model; box moved out of reach, floor contact off so the arm is in free air). The arm
starts 0.3 rad from the goal on the tested joint (the others at the goal), the goal is a step through the servo layer
(command delay + firmware profile, 7v4_real's / the variant's), the setpoint ramps at the firmware limits. Sag
s = (goal - q) * sign(qfrc_bias) (> 0 = beyond the goal the way gravity pulls), at 0.5, 1 and 3 s after the step.
  <so101Sim venv>/python -I hold_test.py <so101Sim root> [variants]
"""
import json
import os
import sys

root = sys.argv[1]
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["7v4_real", "bam_m1", "bam_m6", "bam_m6_nostiff",
                                                             "7v4_real+m6f"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.insert(0, HERE)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402

import sag_servo as SS  # noqa: E402
from metalsim.render.film.so101 import servo_layer  # noqa: E402

POSES = {  # (name, joint index, goal q of the 5 arm joints) chosen to load the tested joint
    "shoulder |g|~0.6": (1, [0.0, 0.9, -0.6, 0.6, 0.0]),
    "shoulder |g|~0.4": (1, [0.0, 0.3, 0.0, 0.8, 0.0]),
    "elbow |g|~0.4": (2, [0.0, 0.0, -0.3, 0.9, 0.0]),
    "wrist |g|~0.1": (3, [0.0, 0.0, 0.6, -0.6, 0.0]),
}
out = {}
for variant in VARIANTS:
    env = SO101LiftEnv(randomize=False)
    m = env.model
    d = mujoco.MjData(m)
    applied, hook = SS.apply(m, d, variant)
    m.geom_contype[:] = 0                         # free air: no contacts at all (the box is irrelevant here)
    m.geom_conaffinity[:] = 0
    layer = servo_layer(applied, 1)
    act = np.array(applied["actuators"])
    qa, va = env._arm_qadr, env._arm_vadr
    res = {}
    for pname, (j, goal5) in POSES.items():
        goal = np.r_[goal5, 0.5]
        # gravity direction at the goal
        d.qpos[:] = m.qpos0; d.qpos[qa] = goal; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        g = float(d.qfrc_bias[va[j]])
        for side, sgn in (("with", +1), ("against", -1)):
            # start 0.3 rad "up-gravity" of the goal to move with gravity, "down-gravity" to move against it
            q0 = goal.copy()
            q0[j] += sgn * np.sign(g) * 0.3          # gravity force is -g: moving with gravity = moving along -sign(g)
            mujoco.mj_resetData(m, d)
            d.qpos[qa] = q0
            mujoco.mj_forward(m, d)
            layer.reset(q0[None])
            if hook is not None:
                hook.reset()
            # settle 1 s holding q0, then step the goal
            for _ in range(200):
                d.ctrl[act] = layer.tick(q0[None])[0]
                if hook is not None:
                    hook.update()
                mujoco.mj_step(m, d)
            rec = {}
            for n in range(1, 801):
                d.ctrl[act] = layer.tick(goal[None])[0]
                if hook is not None:
                    hook.update()
                mujoco.mj_step(m, d)
                if n in (100, 200, 600):
                    rec[f"{n * 0.005:.1f}s"] = round(float((goal[j] - d.qpos[qa[j]]) * np.sign(g)) * 1e3, 1)
            rec["qd_end"] = round(float(d.qvel[va[j]]), 5)
            res[f"{pname} {side}"] = rec
        res[f"{pname} |g|"] = round(abs(g), 3)
    out[variant] = res
    print(f"== {variant}: kp {applied['kp'][1]} forcerange {applied['forcerange_Nm'][1]} damping {applied['damping_Nms'][1]}"
          f" armature {applied['armature'][1]} setpoint {applied['setpoint']}")
    for pname in POSES:
        w, a = res[f"{pname} with"], res[f"{pname} against"]
        print(f"   {pname:18s} |g| {res[pname + ' |g|']:.3f}  with gravity: {w}   against: {a}")
json.dump(out, open(os.path.join(HERE, "hold_test.json"), "w"), indent=1)
