"""so101-tilt: why the box hangs tilted in the SO-101's jaws (CPU MuJoCo, main venv, film scene and reset).

The loop is so101-film-bam's grasp_cpu.py (one CPU world per seed, the deploy expert at 30 Hz, the servo layer and
the BAM friction update before every physics step), plus:
  * one-at-a-time ablations (``ABL``, a JSON dict):
      opt        {"impratio": 1, "cone": 0, "noslip_iterations": 10, ...}   MjOption fields
      grip       {"friction": [s, t, r], "condim": 3, "solref": [..], "solimp": [..]}  on every gripper collision
                 geom (geom_priority 1: these geoms set every jaw-box contact parameter)
      grasp_dz   so101_expert.GRASP_DZ (m; the expert's TCP target above the cube centre, 0.008)
      tcp        so101_expert.TCP_LOCAL replacement (site frame, m), e.g. the measured jaw-pad centre
      over       apply_servo overrides
      pitch_hold true: the lift keeps the hand's pitch (pitch_hold.py; an experiment)
      box_mu     box-table sliding friction (box geom after the reset's scaling, and the floor)
  * tilt (box z vs world z), grip_tilt (gripper body z, the jaws' length axis, vs world z) and box_in_hand (box z vs
    gripper z) at the lift start, 2.5 s, 3 s, 6 s, with the five arm joints; the in-hand rotation (box relative to the
    gripperframe site) from the lift start to 6 s, its component about the jaw-closing direction (gripper body x,
    the fixed pads' normal) and about the hinge axis (gripper body y);
  * the hold detail (``DETAIL`` seed only, at the given times): every box contact with its geom, the position in the
    box frame and in the gripper frame relative to the box centre, normal, dist, dim, friction, solref, solimp,
    mj_contactForce (normal, t1, t2, torsion, roll1, roll2), efc_R of the normal and torsional rows; the gravity
    torque of the box about the jaw-closing axis through the jaw contacts' centre of pressure.

    .venv/bin/python runs/readme_media/scratch/so101-tilt/tilt_probe.py <n seeds> '<ABL json>' <out.json> [detail seed]
"""
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_MS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT_MS)
import mujoco  # noqa: E402

import metalsim.render.film.so101_expert as X  # noqa: E402
from metalsim.render.film.so101 import (ARM_JOINTS, LIFT_Z, ROOT, SCENE, SUCCESS_HOLD, apply_servo,  # noqa: E402
                                        friction_hook, reset_draw, servo_layer)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ABL = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
OUT = sys.argv[3] if len(sys.argv) > 3 else "probe.json"
DETAIL = int(sys.argv[4]) if len(sys.argv) > 4 else 10051
SERVO = ABL.get("servo", "7v4_bam")
SEED0 = 10_000
PATH = os.path.join(ROOT, SCENE)
DETAIL_T = (2.5, 3.0, 4.0, 6.0)
if "grasp_dz" in ABL:
    X.GRASP_DZ = float(ABL["grasp_dz"])
if "tcp" in ABL:
    X.TCP_LOCAL = np.array(ABL["tcp"], float)
if ABL.get("pitch_hold"):                     # experiment: the lift keeps the hand's pitch (pitch_hold.py)
    sys.path.insert(0, HERE)
    from pitch_hold import PitchHoldExpert
    X.DeployExpert = PitchHoldExpert


def gname(m, g):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    return n or f"{b}#{g}"


def rotvec(R):
    """Rotation vector (axis * angle, rad) of the rotation matrix R."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return w * a


def build():
    m = mujoco.MjModel.from_xml_path(PATH)
    for k, v in ABL.get("opt", {}).items():
        setattr(m.opt, k, v)
    grip_geoms = np.where(m.geom_priority > 0)[0]
    g = ABL.get("grip", {})
    if "friction" in g:
        m.geom_friction[grip_geoms] = np.array(g["friction"], float)
    if "condim" in g:
        m.geom_condim[grip_geoms] = int(g["condim"])
    if "solref" in g:
        m.geom_solref[grip_geoms] = np.array(g["solref"], float)
    if "solimp" in g:
        m.geom_solimp[grip_geoms] = np.array(g["solimp"], float)
    return m, grip_geoms


def run(seed, detail=False):
    m, grip_geoms = build()
    ap = apply_servo(m, SERVO, **ABL.get("over", {}))
    fh = friction_hook(ap, m)
    layer = servo_layer(ap, 1)
    act = np.array(ap["actuators"])
    oid = lambda t, s: mujoco.mj_name2id(m, t, s)
    site, box_b, box_g = oid(mujoco.mjtObj.mjOBJ_SITE, "gripperframe"), oid(mujoco.mjtObj.mjOBJ_BODY, "box"), \
        oid(mujoco.mjtObj.mjOBJ_GEOM, "box")
    grip_b = oid(mujoco.mjtObj.mjOBJ_BODY, "gripper")
    box_q = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
    aq = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    av = np.array([int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    ids = dict(grip_site=site, box_body=box_b, box_qadr=box_q, arm_qadr=aq, arm_vadr=av)
    q0, ms, fs = reset_draw(seed, m.qpos0.copy(), box_q, aq, True)
    m.body_mass[box_b] *= ms
    m.geom_friction[box_g] = m.geom_friction[box_g] * fs
    if "box_mu" in ABL:
        m.geom_friction[box_g, 0] = ABL["box_mu"]
        m.geom_friction[oid(mujoco.mjtObj.mjOBJ_GEOM, "floor"), 0] = ABL["box_mu"]
    d = mujoco.MjData(m)
    if fh is not None:
        fh.reset()
    d.qpos[:] = q0
    if fh is not None:
        d.ctrl[act] = q0[aq]
    mujoco.mj_forward(m, d)
    ctl = X.DeployExpert(m, ids, hz=X.CTRL_HZ)
    kin = mujoco.MjData(m)
    layer.reset(d.qpos[aq][None])
    ctl.reset(d.qpos[aq].copy())
    cmd = d.qpos[aq].copy()
    dt = float(m.opt.timestep)
    nph, ns = int(round(6.0 / dt)), int(round(0.02 / dt))
    ticks = {int(round(k / X.CTRL_HZ / dt)) for k in range(int(6.0 * X.CTRL_HZ) + 1)}
    held, success_t, hmax = 0, None, 0.0
    tilt = lambda: float(np.degrees(np.arccos(np.clip(d.xmat[box_b].reshape(3, 3)[2, 2], -1, 1))))
    Rg = lambda: d.xmat[grip_b].reshape(3, 3)
    rel = lambda: Rg().T @ d.xmat[box_b].reshape(3, 3)             # box orientation in the gripper body frame
    gtilt = lambda: float(np.degrees(np.arccos(np.clip(Rg()[2, 2], -1, 1))))          # jaw axis (gripper z) vs vertical
    btilt = lambda: float(np.degrees(np.arccos(np.clip(abs(rel()[2, 2]), 0, 1))))     # box z vs gripper z (in hand)
    jl = oid(mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift")

    def plane(v):
        """(in-plane, out-of-plane) tilt of the unit vector v from world z, deg: in-plane = rotation about the arm's
        pitch axis (the shoulder_lift joint axis, horizontal), signed; out-of-plane = toward / away from that axis."""
        ep = d.xaxis[jl] / np.linalg.norm(d.xaxis[jl])
        er = np.cross(ep, [0.0, 0.0, 1.0])
        return (float(np.degrees(np.arctan2(v @ er, v[2]))), float(np.degrees(np.arcsin(np.clip(v @ ep, -1, 1)))))

    def snap(key):
        bp, gp = plane(d.xmat[box_b].reshape(3, 3)[:, 2]), plane(Rg()[:, 2])
        out.update({f"tilt_{key}_deg": round(tilt(), 2), f"grip_tilt_{key}_deg": round(gtilt(), 2),
                    f"box_in_hand_{key}_deg": round(btilt(), 2), f"q_{key}": d.qpos[aq[:5]].round(4).tolist(),
                    f"box_inplane_{key}_deg": round(bp[0], 2), f"box_outplane_{key}_deg": round(bp[1], 2),
                    f"grip_inplane_{key}_deg": round(gp[0], 2), f"grip_outplane_{key}_deg": round(gp[1], 2),
                    f"box_xy_{key}_mm": (d.xpos[box_b, :2] * 1e3).round(2).tolist(),
                    f"box_in_grip_{key}_mm": (Rg().T @ (d.xpos[box_b] - d.xpos[grip_b]) * 1e3).round(2).tolist()})
    out = dict(seed=seed, mass_kg=round(float(m.body_mass[box_b]), 4))
    rel_lift, t_lift, series, det = None, None, [], {}
    f6 = np.zeros(6)
    for n in range(nph):
        if n in ticks:
            kin.qpos[:], kin.qvel[:] = d.qpos, d.qvel
            mujoco.mj_forward(m, kin)
            cmd = ctl.act(kin, n * dt)
        d.ctrl[act] = layer.tick(cmd[None])[0]
        if fh is not None:
            fh.update(d)
        mujoco.mj_step(m, d)
        t = (n + 1) * dt
        if ctl.phase == "grasp" and "tilt_grasp_start_deg" not in out:
            snap("grasp_start")
        if ctl.phase == "lift" and rel_lift is None:
            rel_lift, t_lift = rel(), t
            snap("lift_start")
            out["t_lift_start"] = round(t, 3)
        if (n + 1) % ns:
            continue
        h = float(d.xpos[box_b, 2])
        hmax = max(hmax, h)
        held = held + 1 if h > LIFT_Z else 0
        if held >= SUCCESS_HOLD and success_t is None:
            success_t = round(((n + 1) // ns - 1) * 0.02, 3)
        for tt in (2.5, 3.0):
            if abs(t - tt) < 1e-9:
                snap(f"{tt:g}s")
        if detail:
            w = rotvec(rel() @ rel_lift.T) if rel_lift is not None else np.zeros(3)   # in-hand, gripper frame
            series.append([round(t, 3), ctl.phase, round(tilt(), 2), round(gtilt(), 2), round(btilt(), 2), round(h, 4)]
                          + np.degrees(w).round(2).tolist() + np.round(plane(d.xmat[box_b].reshape(3, 3)[:, 2]), 2).tolist()
                          + np.round(plane(Rg()[:, 2]), 2).tolist())
            if any(abs(t - tt) < 1e-9 for tt in DETAIL_T) and ctl.phase == "lift":
                det[f"{t:g}"] = hold_detail(m, d, box_b, box_g, grip_b, grip_geoms, f6)
    snap("6s")
    out.update(success=success_t is not None, time_to_lift_s=success_t, retries=ctl.retries, hmax=round(hmax, 4),
               events=ctl.events)
    if rel_lift is not None:
        w = rotvec(rel() @ rel_lift.T)
        out.update(inhand_rot_deg=round(float(np.degrees(np.linalg.norm(w))), 2),
                   inhand_rot_about_x_deg=round(float(np.degrees(w[0])), 2),      # jaw-closing direction
                   inhand_rot_about_y_deg=round(float(np.degrees(w[1])), 2),      # hinge axis
                   inhand_rot_about_z_deg=round(float(np.degrees(w[2])), 2))      # along the jaws
    if detail:
        out["series"] = series
        out["detail"] = det
    return out


def hold_detail(m, d, box_b, box_g, grip_b, grip_geoms, f6):
    Rb, Rg = d.xmat[box_b].reshape(3, 3), d.xmat[grip_b].reshape(3, 3)
    com = d.xipos[box_b]
    cons, P, F = [], np.zeros(3), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if box_g not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == box_g else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        r = c.pos - com
        a = c.efc_address
        row = dict(geom=gname(m, other), pos_box_mm=(Rb.T @ r * 1e3).round(2).tolist(),
                   pos_grip_mm=(Rg.T @ r * 1e3).round(2).tolist(), normal_grip=(Rg.T @ c.frame[:3]).round(3).tolist(),
                   dist_mm=round(c.dist * 1e3, 3), dim=int(c.dim), friction=np.round(c.friction, 6).tolist(),
                   solref=np.round(c.solref, 4).tolist(), solimp=np.round(c.solimp, 4).tolist(),
                   solreffriction=np.round(c.solreffriction, 4).tolist(),
                   force=np.round(f6[:c.dim], 5).tolist(),
                   efc_R=[float(f"{d.efc_R[a + k]:.4g}") for k in range(c.dim)] if a >= 0 else None)
        cons.append(row)
        if other in grip_geoms:
            P += f6[0] * c.pos
            F += f6[0]
    out = dict(n_contacts=len(cons), contacts=cons)
    if F > 0:
        cop = P / F
        ax = Rg[:, 0]                                   # jaw-closing direction (fixed pads' normal)
        wgt = np.array([0, 0, -9.81 * m.body_mass[box_b]])
        tau = np.cross(com - cop, wgt)
        out.update(jaw_fn_sum_N=round(F, 3), cop_box_mm=(Rb.T @ (cop - com) * 1e3).round(2).tolist(),
                   cop_grip_mm=(Rg.T @ (cop - com) * 1e3).round(2).tolist(),
                   gravity_torque_about_x_Nm=round(float(tau @ ax), 5),
                   weight_N=round(float(9.81 * m.body_mass[box_b]), 4))
    return out


def main():
    t0 = time.time()
    eps = [run(SEED0 + e, detail=(SEED0 + e == DETAIL)) for e in range(N)]
    if DETAIL >= SEED0 + N:
        eps.append(run(DETAIL, detail=True))
    base = [e for e in eps if SEED0 <= e["seed"] < SEED0 + N]
    ok = [e for e in base if e["success"]]
    arr = lambda k: np.array([e[k] for e in base if e.get(k) is not None], float)
    pct = lambda a: [round(float(np.median(a)), 1), round(float(np.percentile(a, 90)), 1)] if len(a) else None
    summ = dict(n=N, success=len(ok), retries=int(sum(e["retries"] for e in base)),
                time_to_lift_median=round(float(np.median([e["time_to_lift_s"] for e in ok])), 2) if ok else None)
    for k in ("tilt_lift_start_deg", "tilt_3s_deg", "tilt_6s_deg", "grip_tilt_lift_start_deg", "grip_tilt_6s_deg",
              "box_in_hand_lift_start_deg", "box_in_hand_6s_deg", "box_inplane_lift_start_deg", "box_inplane_6s_deg",
              "box_outplane_lift_start_deg", "box_outplane_6s_deg", "grip_inplane_lift_start_deg", "grip_inplane_6s_deg",
              "grip_outplane_lift_start_deg", "grip_outplane_6s_deg", "inhand_rot_deg", "inhand_rot_about_x_deg",
              "inhand_rot_about_y_deg", "inhand_rot_about_z_deg"):
        summ[k + " (median, p90)"] = pct(arr(k)) if "about" not in k else pct(np.abs(arr(k)))
    det = [e for e in eps if e["seed"] == DETAIL][0]
    summ["detail_seed"] = {k: det.get(k) for k in ("seed", "success", "time_to_lift_s", "tilt_lift_start_deg",
                                                    "tilt_2.5s_deg", "tilt_3s_deg", "tilt_6s_deg",
                                                    "grip_tilt_lift_start_deg", "grip_tilt_6s_deg",
                                                    "box_in_hand_lift_start_deg", "box_in_hand_6s_deg",
                                                    "box_inplane_lift_start_deg", "box_inplane_6s_deg",
                                                    "box_outplane_lift_start_deg", "box_outplane_6s_deg",
                                                    "grip_inplane_lift_start_deg", "grip_inplane_6s_deg",
                                                    "grip_outplane_lift_start_deg", "grip_outplane_6s_deg",
                                                    "q_lift_start", "q_6s", "inhand_rot_deg",
                                                    "inhand_rot_about_x_deg", "inhand_rot_about_y_deg",
                                                    "inhand_rot_about_z_deg")}
    print(json.dumps(dict(abl=ABL, servo=SERVO, **summ)))
    json.dump(dict(script="runs/readme_media/scratch/so101-tilt/tilt_probe.py", mujoco=mujoco.__version__, scene=SCENE,
                   servo=SERVO, abl=ABL, summary=summ, wall_s=round(time.time() - t0, 1), episodes=eps),
              open(os.path.join(HERE, OUT), "w"), indent=1)


if __name__ == "__main__":
    main()
