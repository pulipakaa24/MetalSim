"""Box contacts during the deployment expert's episodes (so101Sim venv): eval_cpu.py's loop on the given seeds; every
20 ms, the bodies touching the box and the deepest penetration; reported per seed for the lift phase.
  <so101Sim>/urlab_bridge/.venv/bin/python contacts.py <so101Sim root> seed [seed ...]"""
import collections, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); MS = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(sys.argv[1], "rl", "native")); sys.path.append(MS)
import mujoco, numpy as np
from so101_lift import SO101LiftEnv
import metalsim.render.film.so101_expert as X
from metalsim.render.film.so101 import apply_servo, servo_layer
for seed in map(int, sys.argv[2:]):
    env = SO101LiftEnv(); m, d = env.model, env.data
    ap = apply_servo(m, "7v4_real"); layer = servo_layer(ap, 1); act = np.array(ap["actuators"])
    ids = dict(grip_site=env._grip_site, box_body=env._box_body, box_qadr=env._box_qadr, arm_qadr=env._arm_qadr, arm_vadr=env._arm_vadr)
    ctl = X.DeployExpert(m, ids); kin = mujoco.MjData(m)
    ticks = {int(round(k / X.CTRL_HZ / m.opt.timestep)) for k in range(181)}
    env.reset(seed=seed); qa = d.qpos[env._arm_qadr].copy(); layer.reset(qa[None]); ctl.reset(qa); cmd = qa.copy()
    bodies, pen, pen_link, frc = collections.Counter(), 0.0, 0.0, []
    for n in range(1200):
        if n in ticks:
            kin.qpos[:], kin.qvel[:] = d.qpos, d.qvel; mujoco.mj_forward(m, kin); cmd = ctl.act(kin, d.time)
        d.ctrl[act] = layer.tick(cmd[None])[0]; mujoco.mj_step(m, d)
        if (n + 1) % 4 == 0 and ctl.phase == "lift" and n * 0.005 > 3.0:
            for c in d.contact[:d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if env._box_body in (b1, b2):
                    other = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == env._box_body else b1)
                    bodies[other] += 1; pen = min(pen, c.dist)
                    if "jaw" not in other and other != "gripper": pen_link = min(pen_link, c.dist)
            frc.append(float(abs(d.actuator_force[5])))
    print(seed, "lift hold (t > 3 s) box contacts by body (count over 20 ms samples):", dict(bodies),
          f"deepest {pen*1e3:.1f} mm, deepest with a non-jaw link {pen_link*1e3:.1f} mm, gripper force median {np.median(frc):.3f} N.m")
