"""Diagnostic variant of so101Sim's scripted expert (rl/native/scripted_expert.py, unchanged on disk): NOT so101Sim's
controller. ``CmdBaseExpert`` builds each tick's joint target on its previous *command* instead of the measured joint
positions: q_cmd <- clip(q_cmd + dq, q - E_SAT, q + E_SAT), q_cmd = q at reset. On a P-only servo (LeRobot's I = 0)
the target must lead the joint by load / kp to hold it against gravity; so101Sim's expert re-bases on the measured q
every tick, so its lead never exceeds DQ_MAX (0.06 rad, 0.32 N.m at P = 16 on the 7.4 V model) and the arm sinks
wherever gravity needs more. E_SAT = 0.97 / (16 x 0.166 x 1.0049) = 0.363 rad is the error at which P = 16 saturates
the duty (more lead adds no torque): anti-windup taken from the servo model, not tuned. Everything else (IK, phases,
gains, gripper, DQ_MAX) is the expert's."""
import numpy as np
import scripted_expert as SE

E_SAT = 0.97 / (16 * 0.166 * 1.0049367060648848)


class CmdBaseExpert(SE.Expert):
    def reset(self):
        super().reset()
        self.q_cmd = None

    def _ik_step(self, target):
        d = self.env.data
        q_meas = d.qpos[self.env._arm_qadr[:5]].copy()
        if getattr(self, "q_cmd", None) is None:
            self.q_cmd = q_meas.copy()
        q_new, err = super()._ik_step(target)                 # = clip(q_meas + dq, ctrlrange)
        dq = q_new - q_meas
        lo = self.m.actuator_ctrlrange[:5, 0]; hi = self.m.actuator_ctrlrange[:5, 1]
        self.q_cmd = np.clip(np.clip(self.q_cmd + dq, q_meas - E_SAT, q_meas + E_SAT), lo, hi)
        return self.q_cmd.copy(), err
