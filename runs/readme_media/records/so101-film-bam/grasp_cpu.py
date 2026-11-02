"""so101-film-bam: the grasp, the box tilt and the arm's droop of the deploy expert, 7v4_real against 7v4_bam, on CPU
MuJoCo (main venv, MuJoCo 3.14) in the film's scene and reset. A stand-in for the take, which records only the filmed
world's poses: this gives every seed's numbers. The loop is so101-bam-gpu's eval_metalsim_bam.py --cpu-dry (one CPU
world per seed; the friction update before every mj_step; the servos power on holding the start pose), using the live
metalsim/render/film/so101.py, plus measurements every 20 ms:

  tilt_deg        angle between the box's z axis and world z (eval_cpu.py's box_tilt_end_deg), at the lift phase's
                  start, at 3 s and at 6 s
  slip_mm / rot   box position / orientation in the gripper frame ("gripperframe" site), change from 3 s to 6 s
  hold contacts   lift phase, t > 3 s (so101-expert contacts.py's window): geoms touching the box (by name; count of
                  20 ms samples), deepest penetration, jaw-only normal force sum (mj_contactForce), gripper actuator
                  force, joint position and its dof_frictionloss (7v4_bam: the load-dependent budget)
  sag_mrad        held command - joint position in the hold, shoulder_lift and elbow_flex (median over t > 3 s)

    .venv/bin/python runs/readme_media/scratch/so101-film-bam/grasp_cpu.py 100        # -> grasp_cpu.json
    ... grasp_cpu.py 100 7v4_bam '{"gripper_torque_limit": 0.2876}' grasp_cpu_bam_grip0986.json   # ablation
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
SERVOS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["7v4_real", "7v4_bam"]
OVER = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}          # apply_servo overrides (ablations)
OUT = sys.argv[4] if len(sys.argv) > 4 else "grasp_cpu.json"
SEED0 = 10_000
PATH = os.path.join(ROOT, SCENE)


def gname(m, g):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    return n or f"{b}#{g}"


def run(servo, seed):
    m = mujoco.MjModel.from_xml_path(PATH)
    ap = apply_servo(m, servo, **OVER)
    fh = friction_hook(ap, m)
    layer = servo_layer(ap, 1)
    act = np.array(ap["actuators"])
    oid = lambda t, s: mujoco.mj_name2id(m, t, s)
    site, box_b, box_g = oid(mujoco.mjtObj.mjOBJ_SITE, "gripperframe"), oid(mujoco.mjtObj.mjOBJ_BODY, "box"), \
        oid(mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_q = int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, "box")])
    aq = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    av = np.array([int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    jaw_bodies = {oid(mujoco.mjtObj.mjOBJ_BODY, "gripper"), oid(mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")}
    ids = dict(grip_site=site, box_body=box_b, box_qadr=box_q, arm_qadr=aq, arm_vadr=av)
    q0, ms, fs = reset_draw(seed, m.qpos0.copy(), box_q, aq, True)
    m.body_mass[box_b] *= ms
    m.geom_friction[box_g] = m.geom_friction[box_g] * fs
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
    in_grip = lambda: (d.site_xmat[site].reshape(3, 3).T @ (d.xpos[box_b] - d.site_xpos[site]),
                       d.site_xmat[site].reshape(3, 3).T @ d.xmat[box_b].reshape(3, 3))
    out = dict(seed=seed, servo=servo)
    bodies, pen_all, pen_jaw, fn_jaw, fgrip, qgrip, sag, gfric = collections.Counter(), 0.0, 0.0, [], [], [], [], []
    tilt_lift = None
    g3 = None
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
        if ctl.phase == "lift" and tilt_lift is None:
            tilt_lift = tilt()
        if (n + 1) % ns:
            continue
        k = (n + 1) // ns
        h = float(d.xpos[box_b, 2])
        hmax = max(hmax, h)
        held = held + 1 if h > LIFT_Z else 0
        if held >= SUCCESS_HOLD and success_t is None:
            success_t = round((k - 1) * 0.02, 3)
        if abs(t - 3.0) < 1e-9:
            out["tilt_3s_deg"] = round(tilt(), 2)
            g3 = in_grip()
        if ctl.phase == "lift" and t > 3.0 + 1e-9:
            fj = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                if box_g not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == box_g else c.geom1
                bodies[gname(m, other)] += 1
                pen_all = min(pen_all, c.dist)
                if m.geom_bodyid[other] in jaw_bodies:
                    pen_jaw = min(pen_jaw, c.dist)
                    mujoco.mj_contactForce(m, d, i, f6)
                    fj += f6[0]
            fn_jaw.append(fj)
            fgrip.append(abs(float(d.actuator_force[act[5]])))
            qgrip.append(float(d.qpos[aq[5]]))
            gfric.append(float(m.dof_frictionloss[av[5]]))
            sag.append(cmd[1:3] - d.qpos[aq[1:3]])
    g6 = in_grip()
    out.update(success=success_t is not None, time_to_lift_s=success_t, retries=ctl.retries, hmax=round(hmax, 4),
               final_height=round(float(d.xpos[box_b, 2]), 4), tilt_lift_start_deg=None if tilt_lift is None else
               round(tilt_lift, 2), tilt_6s_deg=round(tilt(), 2))
    if g3 is not None and fgrip:
        rel = g3[1].T @ g6[1]
        out.update(slip_3to6_mm=round(float(np.linalg.norm(g6[0] - g3[0])) * 1e3, 3),
                   rot_3to6_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))), 3),
                   hold_contacts=dict(bodies), hold_samples=len(fgrip),
                   pen_all_mm=round(-pen_all * 1e3, 3), pen_jaw_mm=round(-pen_jaw * 1e3, 3),
                   jaw_normal_N_median=round(float(np.median(fn_jaw)), 3),
                   grip_force_Nm_median=round(float(np.median(fgrip)), 4),
                   grip_q_rad_median=round(float(np.median(qgrip)), 4),
                   grip_frictionloss_Nm_median=round(float(np.median(gfric)), 4),
                   sag_mrad_median=(np.median(np.array(sag), 0) * 1e3).round(1).tolist())
    return out


res = {}
t0 = time.time()
for s in SERVOS:
    res[s] = [run(s, SEED0 + e) for e in range(N)]
    eps = res[s]
    ok = [e for e in eps if e["success"]]
    tl = lambda key: np.array([e[key] for e in eps if e.get(key) is not None], float)
    pct = lambda a: [round(float(np.median(a)), 2), round(float(np.percentile(a, 90)), 2), round(float(a.max()), 2)]
    print(s, f"success {len(ok)}/{N}, time to lift median {np.median([e['time_to_lift_s'] for e in ok]):.2f}, "
             f"retries {sum(e['retries'] for e in eps)}")
    for key in ("tilt_lift_start_deg", "tilt_3s_deg", "tilt_6s_deg", "slip_3to6_mm", "rot_3to6_deg", "pen_all_mm",
                "pen_jaw_mm", "jaw_normal_N_median", "grip_force_Nm_median", "grip_q_rad_median",
                "grip_frictionloss_Nm_median"):
        print(f"  {key}: median / p90 / max {pct(tl(key))}")
    sg = np.array([e["sag_mrad_median"] for e in eps if "sag_mrad_median" in e])
    print("  hold sag shoulder_lift / elbow_flex mrad, median over seeds:", np.median(sg, 0).round(1).tolist())
    cnt = collections.Counter()
    for e in eps:
        cnt.update({k: 1 for k in e.get("hold_contacts", {})})
    print("  geoms touching the box in the hold (seeds):", dict(cnt.most_common()))
    print("  seed 10051:", {k: v for k, v in eps[51].items() if k != "hold_contacts"} if N > 51 else "-")
json.dump(dict(script="runs/readme_media/scratch/so101-film-bam/grasp_cpu.py", mujoco=mujoco.__version__, scene=SCENE, over=OVER,
               seeds=[SEED0, SEED0 + N - 1], wall_s=round(time.time() - t0, 1), episodes=res),
          open(os.path.join(HERE, OUT), "w"), indent=1)
