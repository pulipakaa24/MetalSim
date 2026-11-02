"""Check the GPU path of the "_real" servos: SO101Lift (MuJoCo Warp, CtrlPlanFeeder) against CPU MuJoCo (this venv's,
same scene assets/so101/scene_box_rl.xml, same reset_draw, same SetpointLayer fed per physics step), for a fixed
action schedule without contact (the box is moved out of reach). Prints the largest per-physics-step joint difference.

  scripts/gpu_run.sh so101-servo-check render 5 -- .venv/bin/python runs/readme_media/scratch/so101-servo/gpu_feeder_check.py
"""
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

import metalsim.render.film.so101 as S
from metalsim.render.film.so101 import ARM_JOINTS, SO101Lift, apply_servo, reset_draw, servo_layer

E, T = 4, 40
ROOT = S.ROOT


def schedule(t):
    a = np.zeros((E, 6), np.float32)
    a[:, 1] = -0.2 if t < 10 else 0.3            # shoulder_lift: a step at t = 10
    a[:, 3] = 0.4 if t < 20 else -0.4            # wrist_flex: a step at t = 20
    a[:, 5] = 0.5
    a[1] *= -1                                    # world 1 the other way
    return a


_t = [0]


def _act(P, obs):
    a = schedule(_t[0])
    _t[0] += 1
    return a


S.policy_act = _act
ad = SO101Lift(SimpleNamespace(adapter_kw={"servo": "7v4_real"}), SimpleNamespace(num_envs=E, seed=10_000), None)
ad.reset()
qg = np.zeros((E, T, 6))
for t in range(T):
    ad.step()
    qg[:, t] = ad.state[0][:, ad.arm_qadr]
# CPU
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, S.SCENE))
ap = apply_servo(m, "7v4_real")
qc = np.zeros((E, T, 6))
qh = np.zeros((E, T, 6))                          # negative control: the plan's first target held for 4 steps
for e, hold in [(e, h) for h in (False, True) for e in range(E)]:
    d = mujoco.MjData(m)
    q, ms, fs = reset_draw(10_000 + e, m.qpos0.copy(), ad.box_qadr, ad.arm_qadr, True)
    m.body_mass[ad.box_body] = ad.mass0 * ms
    m.geom_friction[ad.box_geom] = ad.fric0 * fs
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    lay = servo_layer(ap, 1)
    q0 = np.zeros(m.nu); q0[ap["actuators"]] = d.qpos[ad.arm_qadr]
    lay.reset(q0)
    for t in range(T):
        a = schedule(t)[e].astype(np.float64)
        cmd = ad.lo + 0.5 * (a + 1.0) * (ad.hi - ad.lo)
        P = lay.plan(cmd[None], ad.n_sub)[0]
        for k, u in enumerate(P):
            d.ctrl[:] = P[0] if hold else u
            mujoco.mj_step(m, d)
        (qh if hold else qc)[e, t] = d.qpos[ad.arm_qadr]
diff = np.abs(qg - qc)
print("max |q_gpu - q_cpu| per joint (rad):", ["%.2e" % x for x in diff.max((0, 1))])
print("negative control, max |q_gpu - q_cpu(first target held)| (rad):", ["%.2e" % x for x in np.abs(qg - qh).max((0, 1))])
print("motion range per joint (rad):", (qc.max(1) - qc.min(1)).max(0).round(3).tolist())
# the same with the layer bypassed on CPU (targets = commands): the GPU must be far from this if the feeder works
print("box z (GPU, CPU):", ad.state[2][:, 2].round(3).tolist())
