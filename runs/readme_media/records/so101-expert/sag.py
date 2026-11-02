"""Steady-state tracking error (sag) of position servos from per-physics-step traces.

``steady_error(q, qd, tgt, cmd, phase, dt)``: q, qd, tgt (the servo's target after the command delay and firmware
profile), cmd (the streamed command) all (episodes, steps, 6), phase (episodes, steps) int (0 approach, 1 descend,
2 grasp, 3 lift; -1 none). A sample of joint j is steady when the streamed command of joint j has not changed for
``HOLD`` = 0.31 s (the 7v4_real settle time), the servo target equals the command (the profile has landed) and
|qd_j| < 0.05 rad/s. Error e = tgt - q (rad). Per joint: count, median and p90 of |e| in mrad and in encoder steps
(4096 per turn: 1.534 mrad), mean signed e; overall, and split into "free" (approach / descend: no load but the arm)
and "lift" (lift phase: the box in hand). The gripper (index 5) squeezes on purpose in grasp / lift (target -0.15 rad
inside the cube), so it is reported for approach / descend only.
"""
import numpy as np

HOLD, V_STEADY, STEP = 0.31, 0.05, 2 * np.pi / 4096
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _stats(e):
    a = np.abs(e)
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), median_mrad=round(float(np.median(a)) * 1e3, 1), p90_mrad=round(float(np.percentile(a, 90)) * 1e3, 1),
                median_steps=round(float(np.median(a)) / STEP, 1), mean_signed_mrad=round(float(np.mean(e)) * 1e3, 1))


def steady_error(q, qd, tgt, cmd, phase, dt):
    q, qd, tgt, cmd = (np.asarray(x, float) for x in (q, qd, tgt, cmd))
    E, T, J = q.shape
    nhold = int(round(HOLD / dt))
    same = np.zeros_like(cmd, bool)
    same[:, 1:] = np.abs(np.diff(cmd, axis=1)) < 1e-12
    run = np.zeros((E, T, J), np.int32)                 # consecutive steps with an unchanged command
    for t in range(1, T):
        run[:, t] = np.where(same[:, t], run[:, t - 1] + 1, 0)
    ok = (run >= nhold) & (np.abs(tgt - cmd) < 1e-9) & (np.abs(qd) < V_STEADY) & ~np.isnan(q)
    e = tgt - q
    free = (phase == 0) | (phase == 1)
    lift = phase == 3
    out = {}
    for j in range(J):
        oj = ok[..., j]
        row = dict(free=_stats(e[..., j][oj & free]))
        if j < 5:
            row["all"] = _stats(e[..., j][oj])
            row["lift"] = _stats(e[..., j][oj & lift])
        out[NAMES[j]] = row
    out["definition"] = __doc__.split("\n\n")[1].strip() if "\n\n" in __doc__ else ""
    return out
