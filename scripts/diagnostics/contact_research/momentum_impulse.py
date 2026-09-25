"""Contact impulse per control step from the recorded STATES (not from the contact sensor), for Isaac's PhysX
recording and ours: J_z(t) = P_z(t) - P_z(t-1) + M g * 20 ms, with P the robot's total linear momentum
computed on the same asset in MuJoCo from each engine's recorded joint and root state (Isaac's root velocity
is the root COM's (Isaac Lab 2.3.2 ArticulationData.root_lin_vel_b = root_com_lin_vel_b), converted to the
root-frame origin). J_z / 20 ms is the contact force averaged over the whole control step: a quantity both
engines' sensors can be checked against, independent of their reporting windows.

    python scripts/diagnostics/contact_research/momentum_impulse.py [ours_dir]
"""
import sys, os, json
import numpy as np, mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import build_g1_model

ISAAC = os.path.join(ROOT, "runs/parity/isaac/parity_out2/rt")
ours_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs/parity/tuning/default")
m, _ = build_g1_model("flat", physics_dt=0.0025); d = mujoco.MjData(m)
isaac_joints = [str(s) for s in np.load(os.path.join(ISAAC, "action_sequence_B.npz"))["joint_names"]]
our_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
to_isaac = np.array([our_joints.index(n) for n in isaac_joints])
M = float(m.body_subtreemass[0]); g = -m.opt.gravity[2]
pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); ipos = m.body_ipos[pelvis].copy()


def momentum(R, com_vel):
    T = R["joint_pos"].shape[0]; P = np.zeros((T, 3))
    for t in range(T):
        q = np.zeros(m.nq); v = np.zeros(m.nv)
        q[:3] = R["root_pos"][t, 0]; q[3:7] = R["root_quat"][t, 0]
        q[7:][to_isaac] = R["joint_pos"][t, 0]; v[6:][to_isaac] = R["joint_vel"][t, 0]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q[3:7]); Rm = Rm.reshape(3, 3)
        w_b = R["root_ang_vel_b"][t, 0]; v_b = R["root_lin_vel_b"][t, 0]
        if com_vel: v_b = v_b - np.cross(w_b, ipos)          # COM velocity -> root-frame origin velocity (body frame)
        v[:3] = Rm @ v_b; v[3:6] = w_b
        d.qpos[:] = q; d.qvel[:] = v
        mujoco.mj_forward(m, d); mujoco.mj_subtreeVel(m, d)
        P[t] = M * d.subtree_linvel[0]
    return P


out = {}
for tag in ("A_hold", "C_drop"):
    res = {}
    for who, path, comv in (("isaac", ISAAC, True), ("ours", ours_dir, False)):
        R = np.load(os.path.join(path, f"{tag}.npz"))
        P = momentum(R, comv)
        J = np.diff(P[:, 2]) + M * g * 0.02                   # contact impulse over control step t (t = 1..T-1)
        Fs = R["contact"][1:, 0, :, 2].sum(1)                  # sensor sample at the end of step t (sum of bodies, z)
        res[who] = {"J": J, "Fs": Fs}
    Ji, Jo = res["isaac"]["J"], res["ours"]["J"]
    print(f"== {tag} (M = {M:.2f} kg); per control step: contact impulse from momentum / 20 ms (N) vs sensor sample (N)")
    lo, hi = (8, 20) if tag == "C_drop" else (60, 75)
    ranges = [(8, 20), (62, 76)] if tag == "C_drop" else [(60, 76)]
    for lo, hi in ranges:
        for t in range(lo, hi):
            print(f"  step {t+1:3d}  isaac  J/dt {Ji[t]/0.02:7.0f}  sample {res['isaac']['Fs'][t]:7.0f}   |  ours  J/dt {Jo[t]/0.02:7.0f}  sample {res['ours']['Fs'][t]:7.0f}")
        print(f"  steps {lo+1}-{hi}: impulse isaac {Ji[lo:hi].sum():.1f} N s (sensor samples x 20 ms {res['isaac']['Fs'][lo:hi].sum()*0.02:.1f}); "
              f"ours {Jo[lo:hi].sum():.1f} N s (samples x 20 ms {res['ours']['Fs'][lo:hi].sum()*0.02:.1f}); max J/dt isaac {Ji[lo:hi].max()/0.02:.0f} ours {Jo[lo:hi].max()/0.02:.0f}")
    out[tag] = {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in res.items()}
json.dump(out, open(os.path.join(ROOT, "runs/contact_research/momentum_impulse.json"), "w"))
