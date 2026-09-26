"""Unit-level proof of the clamp/derivative mechanism (MuJoCo C, CPU): G1 at its init pose, gravity off, one finger joint
(left_four_joint, range [-1.84, 0]) driven towards its lower limit at V rad/s with a far target (the saturating policy
actions: target default + 0.5 * 100 = -50 rad), implicitfast at 2.5 ms, Isaac's finger limit solref. Two layouts:
(a) MetalSim's (one affine actuator, forcerange +-300: MuJoCo skips its velocity derivative while clamped,
engine_derivative.c:2416 / mujoco_warp derivative.py:133) and (b) Newton's (effort limit on the joint's actfrcrange,
applied after the actuator sum in engine_forward.c:998, invisible to the derivative).
usage: python runs/il3/finger_unit.py"""
import copy, numpy as np, mujoco
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import solver_presets
base, _ = build_g1_model("flat", physics_dt=0.0025)
base.opt.gravity[:] = 0
import sys
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else -50.0
out = {}
for layout, preset in (("ours (actuator forcerange)", "isaaclab3_every_substep_cap20"), ("Newton (joint actfrcrange)", "isaaclab3_every_substep_cap20_jointeffort")):
    for V in (10.0, 30.0, 60.0):
        m = copy.deepcopy(base); solver_presets.apply(m, preset)
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
        m.body_pos[1] = m.body_pos[1]   # free-floating robot, gravity off: only the finger moves
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_four_joint"); dof = m.jnt_dofadr[j]; qa = m.jnt_qposadr[j]
        a = [i for i in range(m.nu) if m.actuator_trnid[i][0] == j][0]
        d.ctrl[:] = m.key_qpos[0][7:]; d.ctrl[a] = TARGET
        d.qpos[qa] = -1.70; d.qvel[dof] = -V
        speeds = []; bad = None
        for s in range(40):
            mujoco.mj_step(m, d)
            v = float(d.qvel[dof]); speeds.append(round(v, 1))
            if not np.isfinite(v) or abs(v) > 1000:
                bad = s; break
        out[(layout, V)] = (bad, max(abs(x) for x in speeds if np.isfinite(x)), round(float(d.qpos[qa]), 3) if np.isfinite(d.qpos[qa]) else None, speeds[:10])
for k, v in out.items():
    print(f"{k[0]:28s} V {k[1]:5.0f} rad/s: blown at substep {v[0]}, max |qd| {v[1]:.0f}, final q {v[2]}, first speeds {v[3]}")
