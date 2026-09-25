"""Contact impulse per control step from recorded STATES (momentum balance), the method of
scripts/diagnostics/contact_research/momentum_impulse.py (contact-research agent, docs/research/
contact_discrepancies_2026-09-25.md) as a library for metalsim.parity.compare.

J_z(t) = P_z(t) - P_z(t-1) + M g dt_ctrl, with P the robot's total linear momentum computed in MuJoCo on the same
asset from each engine's recorded joint and root state. Isaac Lab 2.3.2's recorded root_lin_vel_b is the root COM
velocity (converted to the root-frame origin); MetalSim's is the free joint's (frame origin). J / dt_ctrl is the
contact force averaged over the control step, independent of either sensor's reporting window (Isaac's reports
the last 5 ms PhysX step, ours the last 2.5 ms substep), so protocol comparisons rank on it, not on sampled peaks.
"""
from __future__ import annotations

import mujoco
import numpy as np

CONTROL_DT = 0.02
# event windows (control steps), as the contact-research summary_table.py: the drop's landing and torso impact,
# the hold's torso impact
EVENTS = {"C_drop": {"landing": (5, 30), "torso": (55, 90)}, "A_hold": {"torso": (55, 90)}}

_cache = {}


def _model():
    if "m" not in _cache:
        from metalsim.learn.g1_velocity import build_g1_model
        m, _ = build_g1_model("flat", physics_dt=0.0025)
        _cache["m"] = m; _cache["d"] = mujoco.MjData(m)
    return _cache["m"], _cache["d"]


def momentum_z(R, joint_names, root_vel_is_com: bool, env: int = 0) -> np.ndarray:
    """Total vertical linear momentum (kg m/s) per recorded control step of a protocol recording ``R``
    (npz with joint_pos, joint_vel, root_pos, root_quat, root_lin_vel_b, root_ang_vel_b in ``joint_names`` order)."""
    m, d = _model()
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    idx = np.array([ours.index(n) for n in joint_names])
    M = float(m.body_subtreemass[0])
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); ipos = m.body_ipos[pelvis].copy()
    T = R["joint_pos"].shape[0]; P = np.zeros(T)
    for t in range(T):
        q = np.zeros(m.nq); v = np.zeros(m.nv)
        q[:3] = R["root_pos"][t, env]; q[3:7] = R["root_quat"][t, env]
        q[7:][idx] = R["joint_pos"][t, env]; v[6:][idx] = R["joint_vel"][t, env]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q[3:7]); Rm = Rm.reshape(3, 3)
        w_b = R["root_ang_vel_b"][t, env]; v_b = R["root_lin_vel_b"][t, env]
        if root_vel_is_com:
            v_b = v_b - np.cross(w_b, ipos)
        v[:3] = Rm @ v_b; v[3:6] = w_b
        d.qpos[:] = q; d.qvel[:] = v
        mujoco.mj_forward(m, d); mujoco.mj_subtreeVel(m, d)
        P[t] = M * d.subtree_linvel[0][2]
    return P


def contact_impulse(R, joint_names, root_vel_is_com: bool, env: int = 0) -> np.ndarray:
    """Vertical contact impulse (N s) over each control step t = 1..T-1."""
    m, _ = _model()
    M = float(m.body_subtreemass[0]); g = -m.opt.gravity[2]
    return np.diff(momentum_z(R, joint_names, root_vel_is_com, env)) + M * g * CONTROL_DT


def event_metrics(J_isaac: np.ndarray, J_ours: np.ndarray, tag: str) -> dict:
    """Per event window: impulse (N s) and largest control-step mean force (N) for both engines; whole run too."""
    out = {"whole_run": {"max_20ms_mean_force_N": {"isaac": float(J_isaac.max() / CONTROL_DT), "metalsim": float(J_ours.max() / CONTROL_DT)}}}
    for ev, (lo, hi) in EVENTS.get(tag, {}).items():
        a, b = J_isaac[lo:hi], J_ours[lo:hi]
        out[ev] = {"steps": [lo + 1, hi], "impulse_Ns": {"isaac": float(a.sum()), "metalsim": float(b.sum()),
                                                         "ratio": float(b.sum() / a.sum()) if a.sum() else None},
                   "max_20ms_mean_force_N": {"isaac": float(a.max() / CONTROL_DT), "metalsim": float(b.max() / CONTROL_DT),
                                             "ratio": float(b.max() / a.max()) if a.max() else None}}
    return out
