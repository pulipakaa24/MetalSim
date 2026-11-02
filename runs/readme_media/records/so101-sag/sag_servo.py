"""Servo variants for the so101-sag experiments, on top of metalsim.render.film.so101.apply_servo (not edited).

Variants (all at LeRobot's registers: P 16, Acceleration 254, gripper 50 %; 3 ms host latency, as 7v4_real):
  7v4_real       as in so101.py (datasheet stall 1.912 N.m -> kp 5.262; BAM m1 Coulomb 0.0515, armature, delay)
  bam_m1         BAM m1 motor throughout (kt / R -> kp 9.38, stall 3.41; Coulomb 0.0515 constant), = so101-expert's
                 "BAM stall 3.41" ablation
  bam_m6         BAM m6 motor throughout (kp 10.58, stall 3.32, back-EMF + viscous 0.619, armature 0.0216,
                 max_velocity 5.096, command_delay 4.98 ms) + m6's load-dependent Stribeck friction budget written to
                 dof_frictionloss every physics step (FrictionHook), joint friction constraint stiffened as BAM's mjlab
                 backend does (solref -5e4 -2e2, solimp 0.99 0.9999 0.001 0.5 2)
  7v4_real+m6f   7v4_real's motor (kp 5.262) with the m6 friction hook (which piece does what)
  bam_m3/4/5     as bam_m6 with BAM's m3 / m4 / m5 file (robustness: the other load-dependent fits)
  bam_m6_nostiff bam_m6 without the stiffened friction constraint (MuJoCo's default soft friction: creep check)
apply(m, d, variant, **over) -> (applied, hook or None). Call hook.update() before every mj_step.
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
import bam_m6 as B  # noqa: E402
from metalsim.render.film.so101 import ARM_JOINTS, BAM_M1, apply_servo  # noqa: E402

STIFF_SOLREF = (-5.0e4, -2.0e2)                         # bam/mjlab.py:256-257 (_STIFF_SOLREF_FRICTION)
STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)          # bam/mjlab.py:257 (_STIFF_SOLIMP_FRICTION)


def bam_over(model: str) -> dict:
    """apply_servo overrides that make the "7v4_real" path produce BAM `model`'s motor at P 16: stall = kt V / R x
    max_pwm, duty per rad with that file's error_gain_ratio (apply_servo multiplies P by 0.166 x m1's ratio, so P is
    scaled), damping = kt^2 / R + viscous (through noload = (stall - coulomb) / damping), armature, max_velocity,
    command delay."""
    p = B.load(model)
    mo = B.motor(p)
    return dict(stall_Nm=mo["stall"], noload_rad_s=(mo["stall"] - mo["coulomb"]) / (mo["emf"] + mo["viscous"]),
                P=16.0 * p["error_gain_ratio"] / BAM_M1["error_gain_ratio"], coulomb=mo["coulomb"],
                armature=mo["armature"], speed_limit=mo["max_velocity"], servo_delay=mo["command_delay"])


class FrictionHook:
    """bam/mujoco.py MujocoController.update (lines 170-203), friction part only: before each mj_step, from the data
    of the previous step, tau_m = qfrc_actuator - kt^2/R * qvel (BAM's motor torque includes the back-EMF, which
    MetalSim's voltage_servo puts in dof_damping), tau_e = -qfrc_bias + qfrc_constraint - (own dof-friction force),
    and dof_frictionloss = BAM budget(tau_m, tau_e, qvel). dof_damping is left as voltage_servo set it."""

    def __init__(self, m, d, dofs, params: dict, emf: float):
        self.m, self.d, self.dofs, self.p, self.emf = m, d, np.asarray(dofs), params, emf
        self.reset()

    def reset(self):
        self.m.dof_frictionloss[self.dofs] = self.p["friction_base"] + self.p.get("friction_stribeck", 0.0)

    def update(self):
        m, d, dofs = self.m, self.d, self.dofs
        fr = np.zeros(m.nv)
        n = d.nefc
        if n:
            t, i, f = d.efc_type[:n], d.efc_id[:n], d.efc_force[:n]
            k = t == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
            np.add.at(fr, i[k], f[k])
        tau_e = -d.qfrc_bias[dofs] + d.qfrc_constraint[dofs] - fr[dofs]
        tau_m = d.qfrc_actuator[dofs] - self.emf * d.qvel[dofs]
        m.dof_frictionloss[dofs] = B.budget(self.p, tau_m, tau_e, d.qvel[dofs])


def apply(m, d, variant: str, **over):
    oid = lambda t, nm: mujoco.mj_name2id(m, t, nm)
    dofs = np.array([int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    hook = None
    if variant == "7v4_real":
        applied = apply_servo(m, "7v4_real", **over)
    elif variant == "bam_m1":
        applied = apply_servo(m, "7v4_real", **dict(bam_over("m1"), **over))
    elif variant in ("bam_m6", "bam_m6_nostiff", "bam_m3", "bam_m4", "bam_m5"):
        mod = variant[4:6]
        applied = apply_servo(m, "7v4_real", **dict(bam_over(mod), **over))
        p = B.load(mod)
        hook = FrictionHook(m, d, dofs, p, B.motor(p)["emf"])
        if variant != "bam_m6_nostiff":
            m.dof_solref[dofs] = STIFF_SOLREF
            m.dof_solimp[dofs] = STIFF_SOLIMP
    elif variant == "7v4_real+m6f":
        applied = apply_servo(m, "7v4_real", **over)
        p = B.load("m6")
        hook = FrictionHook(m, d, dofs, p, 0.0)          # 7v4_real's back-EMF is not BAM's; use qfrc_actuator alone
        m.dof_solref[dofs] = STIFF_SOLREF
        m.dof_solimp[dofs] = STIFF_SOLIMP
    else:
        raise ValueError(variant)
    applied["variant"] = variant
    applied["friction_hook"] = None if hook is None else "bam m6 budget per physics step"
    applied["dof_solref"] = m.dof_solref[dofs].tolist()
    return applied, hook
