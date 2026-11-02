"""Probe: reproduce the orchestrator's old-model step table (protocol check). so101Sim venv."""
import sys, numpy as np, mujoco
sys.path.append("~/MetalSim")
from metalsim.render.film.so101 import apply_servo, ARM_JOINTS
XML = "~/so101Sim/mujoco_menagerie/robotstudio_so101/scene_box_rl.xml"
for servo in ("12v", "7v4"):
    for j in (1, 2, 3):
        m = mujoco.MjModel.from_xml_path(XML); apply_servo(m, servo); d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d); d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
        d.ctrl[:] = d.qpos[:6]
        for _ in range(200): mujoco.mj_step(m, d)
        q0 = d.qpos[j]; tgt = d.ctrl[j] + 0.06; d.ctrl[j] = tgt
        ys = []
        for _ in range(200): mujoco.mj_step(m, d); ys.append(d.qpos[j])
        y = np.array(ys) - q0; A = tgt - q0
        os_ = (y.max() - A) / 0.06
        cr = int(np.sum(np.diff(np.sign(y - A)) != 0))
        out = np.where(np.abs(y - A) > 0.02 * 0.06)[0]
        st = (out[-1] + 1) * 5 if len(out) else 0
        print(servo, ARM_JOINTS[j], f"start err {q0 - (tgt-0.06):+.4f} overshoot {os_*100:.0f}% crossings {cr} settle {st} ms")
