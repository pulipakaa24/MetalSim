"""Is the 7v4_real speed limit right? Bang-bang commands (full range, flipped every 0.4 s) on all six joints:
(1) gravity and contacts off: the joints' own line must cap the speed at the datasheet no-load 5.45 rad/s (inertial
coupling between joints aside); (2) gravity on, contacts off; (3) as the env (both on). Peak |qdot| per joint."""
import sys, numpy as np, mujoco
sys.path.append("~/MetalSim")
from metalsim.render.film.so101 import apply_servo, servo_layer
XML = "~/so101Sim/mujoco_menagerie/robotstudio_so101/scene_box_rl.xml"
for name, grav, cont in (("no gravity, no contacts", 0, 0), ("gravity, no contacts", 1, 0), ("gravity + contacts", 1, 1)):
    m = mujoco.MjModel.from_xml_path(XML); ap = apply_servo(m, "7v4_real")
    if not grav: m.opt.gravity[:] = 0
    if not cont: m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); lay = servo_layer(ap, 1); lay.reset(d.qpos[:6].copy())
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    pk = np.zeros(6); over = 0; n = 0
    for k in range(2000):
        cmd = hi if (k // 80) % 2 == 0 else lo
        d.ctrl[:] = lay.tick(cmd[None])[0]; mujoco.mj_step(m, d)
        pk = np.maximum(pk, np.abs(d.qvel[:6])); over += (np.abs(d.qvel[:6]) > 5.445).any(); n += 1
    print(f"{name:26s} peak rad/s {np.round(pk, 2).tolist()}  steps > 5.45: {100 * over / n:.1f} %")
