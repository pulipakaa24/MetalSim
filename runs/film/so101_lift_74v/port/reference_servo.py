"""so101Sim's CPU reference (runs/film/so101_lift/port/reference.py, run unchanged) with the SO-101's servos switched to
a servo model of metalsim/render/film/so101.py (``SERVOS``, ``apply_servo``): so101Sim's ``SO101LiftEnv`` is wrapped
so that, right after it builds its MjModel, ``apply_servo(env.model, servo)`` sets the arm's force limits and joint
damping in place. Nothing else changes: the same env, policy, seeds, loop and measurements as the 12 V reference.

``--keep-damping`` (a sensitivity check, not the 7.4 V model): after ``apply_servo``, put the joint damping back to the
scene's 0.60 N.m.s/rad (the 7.4 V force limits with the original damping). ``--gripper-full`` (a diagnostic, not the
7.4 V model): the gripper at the full 7.4 V stall torque instead of LeRobot's 50 % limit.

Usage (so101Sim's environment, from its urlab_bridge/ directory; CPU only):
  uv run --no-sync python <MetalSim>/runs/film/so101_lift_74v/port/reference_servo.py <so101Sim root> <policy.pt> \
      <out_dir> <servo> [episodes] [--keep-damping] [--gripper-full]
Writes <out_dir>/reference.json (with ``servo``: what was applied) and reference.npz, as reference.py does.
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METALSIM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REF = os.path.join(METALSIM, "runs", "film", "so101_lift", "port", "reference.py")
keep = "--keep-damping" in sys.argv
full = "--gripper-full" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
root, policy, out_dir, servo = args[:4]
episodes = args[4] if len(args) > 4 else "100"
sys.path.append(METALSIM)                                  # after so101Sim's own paths: no shadowing
sys.path.insert(0, os.path.join(root, "rl", "native"))
import so101_lift  # noqa: E402

from metalsim.render.film.so101 import ARM_JOINTS, SERVO_BASE, SERVOS, apply_servo  # noqa: E402

applied = {}
_init = so101_lift.SO101LiftEnv.__init__


def _init_servo(self, *a, **k):
    _init(self, *a, **k)
    applied.update(apply_servo(self.model, servo))
    if keep:
        import mujoco
        dof = [int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS]
        self.model.dof_damping[dof] = SERVO_BASE["damping"]
        applied.update(damping_Nms=[SERVO_BASE["damping"]] * len(dof), keep_damping=True,
                       noload_line_rad_s=[round(f / SERVO_BASE["damping"], 3) for f in applied["forcerange_Nm"]])
    if full:
        import mujoco
        a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        f = float(self.model.actuator_forcerange[a, 1] / SERVOS[servo]["gripper_torque_limit"])
        self.model.actuator_forcerange[a] = (-f, f)
        fr = list(applied["forcerange_Nm"]); fr[-1] = round(f, 4)
        applied.update(forcerange_Nm=fr, gripper_full=True)


so101_lift.SO101LiftEnv.__init__ = _init_servo
sys.argv = [REF, root, policy, out_dir, episodes]
runpy.run_path(REF, run_name="__main__")

p = os.path.join(out_dir, "reference.json")
J = json.load(open(p))
J["wrapper"] = ("runs/film/so101_lift_74v/port/reference_servo.py" + (" --keep-damping" if keep else "")
                + (" --gripper-full" if full else ""))
J["servo"] = applied
json.dump(J, open(p, "w"), indent=1)
print("servo:", json.dumps(applied))
