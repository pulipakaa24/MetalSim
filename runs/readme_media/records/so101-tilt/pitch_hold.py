"""so101-tilt experiment (not a proposed change): DeployExpert whose lift keeps the hand's pitch.

The SO-101's shoulder_lift, elbow_flex and wrist_flex axes are parallel, so the hand's pitch in the arm plane is
their sum plus a constant (tilt_probe.py checks it: hand in-plane change == the sum's change on every seed). In the
lift phase the IK gets one extra row, d(q_lift + q_elbow + q_wflex) = pitch0 - (q_lift + q_elbow + q_wflex), with
pitch0 the goal's sum when the lift starts; the 3 position rows and this row are solved by the same damped least
squares as so101Sim's IK (5 joints, 4 rows). Everything else is DeployExpert."""
import numpy as np

import metalsim.render.film.so101_expert as X

ROW = np.array([0.0, 1.0, 1.0, 1.0, 0.0])


class PitchHoldExpert(X.DeployExpert):
    def reset(self, q_meas6):
        super().reset(q_meas6)
        self.pitch0 = None

    def _enter(self, phase, t, why):
        super()._enter(phase, t, why)
        self.pitch0 = float(ROW @ self.goal) if phase == "lift" else None

    def _ik_iter(self, target, align):
        if self.phase != "lift" or self.pitch0 is None:
            return super()._ik_iter(target, align)
        m, d = self.m, self.pd
        tcp = self.tcp(d)
        err = target - tcp
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.mj.mj_jac(m, d, jacp, jacr, tcp, self.site_body)
        q = d.qpos[self.qadr[:5]]
        J = np.vstack([jacp[:, self.vadr[:5]], ROW])
        e = np.concatenate([err, [self.pitch0 - ROW @ q]])
        dq = J.T @ np.linalg.solve(J @ J.T + X.IK_LAMBDA * np.eye(4), e)
        dq = np.clip(dq, -X.DQ_MAX, X.DQ_MAX)
        q = np.clip(q + dq, self.lo, self.hi)
        return q, float(np.linalg.norm(err))
