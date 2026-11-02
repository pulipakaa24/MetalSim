"""so101-film-bam: what a take's filmed world did, frame by frame, from its recording alone (rec/rec.npz geom poses and
rec/twin.xml, the model the renderer used): box tilt (angle between the box's z axis and world z, eval_cpu.py's
box_tilt_end_deg), box height, and the arm's joint angles. A joint angle is recovered from the two bodies' world
rotations, each from one of its geoms' recorded xmat and its local geom_quat: for a hinge MuJoCo sets
R_child = R_parent * body_quat * rot(axis, q - qpos0). Checked against mj_kinematics on random poses first.

    .venv/bin/python runs/readme_media/scratch/so101-film-bam/take_measure.py <rec dir> [<rec dir> ...] [--out f.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import mujoco  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def quat_mat(q):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    return r.reshape(3, 3)


def axis_angle(R, axis):
    """Angle of the rotation R about the unit axis (R assumed a rotation about it)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    v = q[1:]
    a = 2 * np.arctan2(float(v @ axis), q[0])
    return (a + np.pi) % (2 * np.pi) - np.pi          # q and -q are the same rotation


class Extract:
    def __init__(self, m):
        self.m = m
        self.body_geom = {}
        for g in range(m.ngeom):
            self.body_geom.setdefault(int(m.geom_bodyid[g]), g)
        self.j = []
        for name in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            b = int(m.jnt_bodyid[jid])
            self.j.append((name, b, int(m.body_parentid[b]), m.jnt_axis[jid].copy(), float(m.qpos0[m.jnt_qposadr[jid]])))
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")

    def body_R(self, xmat, b):
        if b == 0:
            return np.eye(3)
        g = self.body_geom[b]
        return xmat[g].reshape(3, 3) @ quat_mat(self.m.geom_quat[g]).T

    def joints(self, xmat):
        out = []
        for name, b, p, axis, q0 in self.j:
            Rrel = quat_mat(self.m.body_quat[b]).T @ self.body_R(xmat, p).T @ self.body_R(xmat, b)
            out.append(axis_angle(Rrel, axis) + q0)
        return np.array(out)

    def box(self, xmat, xpos):
        Rb = self.body_R(xmat, self.box_body)
        return float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1)))), float(xpos[self.box_geom][2])


def self_check(m, ex, n=50):
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(n):
        for name, *_ in ex.j:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = m.jnt_range[jid]
            d.qpos[m.jnt_qposadr[jid]] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        q = np.array([d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in JOINTS])
        err = max(err, float(np.abs(ex.joints(d.geom_xmat.copy()) - q).max()))
    return err


args = sys.argv[1:]
out_path = None
if "--out" in args:
    i = args.index("--out"); out_path = args[i + 1]; del args[i:i + 2]
res = {}
for rec in args:
    m = mujoco.MjModel.from_xml_path(os.path.join(rec, "twin.xml"))
    ex = Extract(m)
    chk = self_check(m, ex)
    z = np.load(os.path.join(rec, "rec.npz"))
    xmat, xpos = z["xmat"][:, 0].astype(np.float64), z["xpos"][:, 0].astype(np.float64)
    assert xmat.shape[1] == m.ngeom, (xmat.shape, m.ngeom)
    q = np.stack([ex.joints(xmat[f]) for f in range(len(xmat))])
    tb = np.array([ex.box(xmat[f], xpos[f]) for f in range(len(xmat))])
    meta = json.load(open(os.path.join(rec, "rec.json")))
    res[rec] = dict(servo=meta["adapter"]["servo"]["servo"], envs=meta["envs"], frames=len(xmat), fk_self_check_rad=chk,
                    q=q.round(5).tolist(), tilt_deg=tb[:, 0].round(2).tolist(), box_z=tb[:, 1].round(4).tolist())
    fr = [60, 110, 149, 199, 249, 299]
    print(rec, res[rec]["servo"], f"fk self-check max err {chk:.1e} rad")
    for f in fr:
        print(f"  frame {f:3d} t {f * 0.02:.2f} s  tilt {tb[f, 0]:5.1f} deg  box z {tb[f, 1]:.3f} m  "
              f"q(lift, elbow, wflex) {np.round(q[f, 1:4], 4).tolist()}  grip {q[f, 5]:.3f}")
if out_path:
    json.dump(res, open(out_path, "w"))
