"""Isaac Lab's Unitree G1 velocity task on this stack, from Isaac's own asset.

Robot: `assets/isaac/G1/g1_minimal.usd` (Isaac Lab's asset, converted with `metalsim.scene.usd_to_mjcf`),
with Isaac's actuator overrides (`G1_CFG.actuators`: implicit PD per joint group with stiffness,
damping, effort limit and armature) and Isaac's initial state. Task terms follow
`isaaclab_tasks/manager_based/locomotion/velocity/{velocity_env_cfg.py, config/g1/rough_env_cfg.py}`:

* control 50 Hz (physics 200 Hz, decimation 4), episode 20 s
* actions: joint position targets = default + 0.5 * action (37 joints)
* observations (policy group): base lin vel (body), base ang vel (body), projected gravity,
  velocity commands (3), joint pos - default, joint vel, last action, height scan (rough only;
  187 rays on a 1.6 x 1.0 m grid at 0.1 m, yaw-aligned, clipped to [-1, 1]); noise as configured
* commands: uniform lin_vel_x in [0, 1], lin_vel_y in [-1, 1], heading command with stiffness 0.5
  (ang_vel_z = 0.5 * wrapped heading error, clipped to [-1, 1]), resampled every 10 s,
  2 % of envs standing
* rewards (G1 rough weights): track_lin_vel_xy_yaw_frame_exp 1.0 (std 0.5), track_ang_vel_z_world_exp
  2.0 (std 0.5), feet_air_time_positive_biped 0.25 (threshold 0.4), feet_slide -0.1,
  dof_pos_limits (ankles) -1.0, joint_deviation_l1 hips(yaw/roll) -0.1, arms -0.1, fingers -0.05,
  torso -0.1, termination -200, flat_orientation_l2 -1.0, action_rate_l2 -0.005,
  dof_acc_l2 (hips, knees) -1.25e-7, dof_torques_l2 (hips, knees, ankles) -1.5e-7, ang_vel_xy_l2 -0.05
* terminations: time out; torso_link contact force > 1 N
* resets: base xy in +-0.5 m, yaw in +-pi, default joint pose, zero velocities
* events not reproduced: startup friction randomization (0.8/0.6 applied deterministically),
  base mass / COM randomization (disabled in the G1 config anyway), pushes (disabled in G1 config)

Contact quantities come from MuJoCo touch sensors added to the feet and torso (sites enclosing
the colliders), evaluated by MuJoCo Warp each step.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import warp as wp

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.scene.usd_to_mjcf import load_usd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G1_USD = os.path.join(ROOT, "assets", "isaac", "G1", "g1_minimal.usd")

# Isaac G1_CFG actuators: (joint regex, stiffness, damping, effort limit, armature)
ACTUATORS = [
    (r".*_hip_yaw_joint", 150.0, 5.0, 300.0, 0.01), (r".*_hip_roll_joint", 150.0, 5.0, 300.0, 0.01),
    (r".*_hip_pitch_joint", 200.0, 5.0, 300.0, 0.01), (r".*_knee_joint", 200.0, 5.0, 300.0, 0.01),
    (r"torso_joint", 200.0, 5.0, 300.0, 0.01),
    (r".*_ankle_pitch_joint", 20.0, 2.0, 20.0, 0.01), (r".*_ankle_roll_joint", 20.0, 2.0, 20.0, 0.01),
    (r".*_shoulder_.*", 40.0, 10.0, 300.0, 0.01), (r".*_elbow_.*", 40.0, 10.0, 300.0, 0.01),
    (r".*_(five|three|six|four|zero|one|two)_joint", 40.0, 10.0, 300.0, 0.001),
]
INIT_POS = (0.0, 0.0, 0.74)
INIT_JOINTS = [(r".*_hip_pitch_joint", -0.20), (r".*_knee_joint", 0.42), (r".*_ankle_pitch_joint", -0.23),
               (r".*_elbow_pitch_joint", 0.87), (r"left_shoulder_roll_joint", 0.16), (r"left_shoulder_pitch_joint", 0.35),
               (r"right_shoulder_roll_joint", -0.16), (r"right_shoulder_pitch_joint", 0.35), (r"left_one_joint", 1.0),
               (r"right_one_joint", -1.0), (r"left_two_joint", 0.52), (r"right_two_joint", -0.52)]
ACTION_SCALE = 0.5
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
EPISODE_S = 20.0


def build_g1_model(terrain: str = "flat", hfield=None, visuals: bool = False, physics_dt: float = PHYSICS_DT):
    """MjModel of Isaac's G1 (from its USD) on a plane or a heightfield, with Isaac's actuators,
    initial pose, and touch sensors for contact terms. Returns (model, info)."""
    spec = load_usd(G1_USD, lossless=False, drives=False, visuals=visuals)   # visuals: 43 meshes for rendering only
    spec.option.timestep = physics_dt      # 0.005 = Isaac; 0.0025 keeps Isaac's kp 200 drives stable in MuJoCo (explicit stiffness)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL   # mjlab's G1 training setting; elliptic is costlier
    # solver budget as in MuJoCo Warp's own G1 benchmark (benchmarks/unitree_g1/unitree_g1_mjlab.xml):
    # iterations 10, ls_iterations 20, implicitfast, eulerdamp disabled
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    # ground
    if terrain == "flat":
        g = spec.worldbody.add_geom(); g.name = "ground"; g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [0, 0, 0.05]
        g.friction = [0.8, 0.005, 0.0001]
    else:
        hf = spec.add_hfield(); hf.name = "terrain"; hf.nrow, hf.ncol = hfield["nrow"], hfield["ncol"]
        hf.size = hfield["size"]; hf.userdata = hfield["data"].reshape(-1).tolist()
        g = spec.worldbody.add_geom(); g.name = "ground"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "terrain"
        g.pos = [0.0, 0.0, hfield["zmin"]]       # MuJoCo's surface is data*size_z above the geom: put it at the true heights
        g.friction = [0.8, 0.005, 0.0001]
    # actuators: Isaac's implicit PD groups (USD drives were not converted)
    joints = [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    for j in joints:
        for pat, kp, kv, eff, arm in ACTUATORS:
            if re.fullmatch(pat, j.name):
                act = spec.add_actuator(); act.name = j.name; act.target = j.name; act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gainprm[0] = kp; act.biasprm[0] = 0; act.biasprm[1] = -kp; act.biasprm[2] = -kv
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED; act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.forcerange = [-eff, eff]; act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                j.armature = arm
                break
        else:
            raise ValueError(f"no actuator group for joint {j.name}")
    # touch sensors: feet (ankle_roll links) and torso, sites enclosing the colliders
    for body_name, site_size in (("left_ankle_roll_link", (0.12, 0.06, 0.04)), ("right_ankle_roll_link", (0.12, 0.06, 0.04)),
                                 ("torso_link", (0.15, 0.15, 0.25))):
        body = next(b for b in spec.bodies if b.name == body_name)
        s = body.add_site(); s.name = body_name + "_touch"; s.type = mujoco.mjtGeom.mjGEOM_BOX; s.size = list(site_size)
        s.pos = [0.03, 0.0, -0.02] if "ankle" in body_name else [0.0, 0.0, 0.15]
        sens = spec.add_sensor(); sens.name = body_name + "_touch"; sens.type = mujoco.mjtSensor.mjSENS_TOUCH
        sens.objtype = mujoco.mjtObj.mjOBJ_SITE; sens.objname = s.name
    # Isaac's initial state as a keyframe ("init"). qpos0 is left at the USD's zero pose: MuJoCo
    # measures joint angles from qpos0, so overwriting it would redefine the zero of every joint.
    m = spec.compile()
    q0 = m.qpos0.copy()
    q0[0:3] = INIT_POS; q0[3:7] = (1, 0, 0, 0)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for pat, val in INIT_JOINTS:
            if re.fullmatch(pat, name):
                q0[m.jnt_qposadr[j]] = val
    key = spec.add_key(); key.name = "init"; key.qpos = q0.tolist()
    m = spec.compile()
    info = {"spec": spec}
    return m, info


# --------------------------------------------------------------------------------------------------
# Warp kernels: observations, commands, rewards/terminations/resets

@wp.func
def quat_to_mat_wxyz(q: wp.vec4) -> wp.mat33:
    w = q[0]; x = q[1]; y = q[2]; z = q[3]
    return wp.mat33(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))


@wp.kernel
def g1_obs(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), default_q: wp.array(dtype=float),
           cmd: wp.array2d(dtype=float), last_action: wp.array2d(dtype=float), height_scan: wp.array2d(dtype=float),
           n_scan: int, noise_seed: int, step_idx: wp.array(dtype=int), obs: wp.array2d(dtype=float)):
    e = wp.tid()
    nj = qpos.shape[1] - 7
    q = wp.vec4(qpos[e, 3], qpos[e, 4], qpos[e, 5], qpos[e, 6])
    R = quat_to_mat_wxyz(q)
    Rt = wp.transpose(R)
    v_w = wp.vec3(qvel[e, 0], qvel[e, 1], qvel[e, 2])
    v_b = Rt * v_w                                   # base linear velocity in the body frame
    w_b = wp.vec3(qvel[e, 3], qvel[e, 4], qvel[e, 5])   # MuJoCo free-joint angular velocity is body-frame
    g_b = Rt * wp.vec3(0.0, 0.0, -1.0)
    rng = wp.rand_init(noise_seed, step_idx[0] * 7919 + e)
    obs[e, 0] = v_b[0] + wp.randf(rng, -0.1, 0.1); obs[e, 1] = v_b[1] + wp.randf(rng, -0.1, 0.1); obs[e, 2] = v_b[2] + wp.randf(rng, -0.1, 0.1)
    obs[e, 3] = w_b[0] + wp.randf(rng, -0.2, 0.2); obs[e, 4] = w_b[1] + wp.randf(rng, -0.2, 0.2); obs[e, 5] = w_b[2] + wp.randf(rng, -0.2, 0.2)
    obs[e, 6] = g_b[0] + wp.randf(rng, -0.05, 0.05); obs[e, 7] = g_b[1] + wp.randf(rng, -0.05, 0.05); obs[e, 8] = g_b[2] + wp.randf(rng, -0.05, 0.05)
    obs[e, 9] = cmd[e, 0]; obs[e, 10] = cmd[e, 1]; obs[e, 11] = cmd[e, 2]
    for i in range(nj):
        obs[e, 12 + i] = qpos[e, 7 + i] - default_q[7 + i] + wp.randf(rng, -0.01, 0.01)
        obs[e, 12 + nj + i] = qvel[e, 6 + i] + wp.randf(rng, -1.5, 1.5)
        obs[e, 12 + 2 * nj + i] = last_action[e, i]
    for k in range(n_scan):
        obs[e, 12 + 3 * nj + k] = wp.clamp(height_scan[e, k] + wp.randf(rng, -0.1, 0.1), -1.0, 1.0)


@wp.kernel
def g1_apply_action(action: wp.array2d(dtype=float), default_q: wp.array(dtype=float), scale: float,
                    last_action: wp.array2d(dtype=float), prev_action: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
    e = wp.tid()
    nj = ctrl.shape[1]
    for i in range(nj):
        a = wp.clamp(action[e, i], -100.0, 100.0)
        prev_action[e, i] = last_action[e, i]
        last_action[e, i] = a
        ctrl[e, i] = default_q[7 + i] + scale * a


@wp.kernel
def g1_commands(qpos: wp.array2d(dtype=float), cmd: wp.array2d(dtype=float), heading: wp.array(dtype=float),
                standing: wp.array(dtype=wp.bool), resample: wp.array(dtype=wp.bool), seed: int, step_idx: wp.array(dtype=int)):
    """Isaac UniformVelocityCommand with heading_command=True: lin_vel_x in [0,1], lin_vel_y in [-1,1],
    target heading in [-pi, pi]; ang_vel_z = clip(0.5 * wrap(heading - yaw), -1, 1); 2% standing envs."""
    e = wp.tid()
    if resample[e]:
        rng = wp.rand_init(seed + 3, step_idx[0] * 104729 + e)
        cmd[e, 0] = wp.randf(rng, 0.0, 1.0)
        cmd[e, 1] = wp.randf(rng, -1.0, 1.0)
        heading[e] = wp.randf(rng, -3.14159265, 3.14159265)
        standing[e] = wp.randf(rng, 0.0, 1.0) < 0.02
        resample[e] = False
    q = wp.vec4(qpos[e, 3], qpos[e, 4], qpos[e, 5], qpos[e, 6])
    R = quat_to_mat_wxyz(q)
    yaw = wp.atan2(R[1, 0], R[0, 0])
    err = heading[e] - yaw
    err = err - 6.2831853 * wp.floor((err + 3.14159265) / 6.2831853)
    cmd[e, 2] = wp.clamp(0.5 * err, -1.0, 1.0)
    if standing[e]:
        cmd[e, 0] = 0.0; cmd[e, 1] = 0.0; cmd[e, 2] = 0.0


@wp.kernel
def g1_reward_done(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), qacc: wp.array2d(dtype=float),
                   qfrc_actuator: wp.array2d(dtype=float), sensordata: wp.array2d(dtype=float),
                   site_xpos: wp.array2d(dtype=wp.vec3), cvel: wp.array2d(dtype=wp.spatial_vector),
                   cmd: wp.array2d(dtype=float), last_action: wp.array2d(dtype=float), prev_action: wp.array2d(dtype=float),
                   default_q: wp.array(dtype=float), jnt_range: wp.array2d(dtype=float),
                   group: wp.array(dtype=int), touch_adr: wp.vec3i, foot_site: wp.vec2i, foot_body: wp.vec2i,
                   air_time: wp.array2d(dtype=float), contact_time: wp.array2d(dtype=float), dt: float,
                   t: wp.array(dtype=int), max_t: int, step_idx: wp.array(dtype=int),
                   buf_rew: wp.array2d(dtype=float), buf_done: wp.array2d(dtype=float), reset_mask: wp.array(dtype=wp.bool),
                   resample: wp.array(dtype=wp.bool), resample_every: int,
                   ep_ret: wp.array(dtype=float), ep_len: wp.array(dtype=int), stats: wp.array(dtype=float), stats_i: wp.array(dtype=int),
                   terms: wp.array2d(dtype=float),
                   curriculum: int, level: wp.array(dtype=int), col: wp.array(dtype=int), origin_table: wp.array2d(dtype=float),
                   n_levels: int, n_cols: int, cell_size: float, episode_s: float, origins: wp.array2d(dtype=float),
                   seed: int):
    e = wp.tid()
    nj = last_action.shape[1]
    q = wp.vec4(qpos[e, 3], qpos[e, 4], qpos[e, 5], qpos[e, 6])
    R = quat_to_mat_wxyz(q)
    Rt = wp.transpose(R)
    v_w = wp.vec3(qvel[e, 0], qvel[e, 1], qvel[e, 2])
    w_b = wp.vec3(qvel[e, 3], qvel[e, 4], qvel[e, 5])
    w_w = R * w_b
    g_b = Rt * wp.vec3(0.0, 0.0, -1.0)
    # yaw-frame linear velocity (Isaac: rotate world velocity by inverse yaw)
    yaw = wp.atan2(R[1, 0], R[0, 0])
    cy = wp.cos(yaw); sy = wp.sin(yaw)
    vx = cy * v_w[0] + sy * v_w[1]
    vy = -sy * v_w[0] + cy * v_w[1]
    # 1. tracking
    lin_err = (cmd[e, 0] - vx) * (cmd[e, 0] - vx) + (cmd[e, 1] - vy) * (cmd[e, 1] - vy)
    r_lin = 1.0 * wp.exp(-lin_err / 0.25)
    ang_err = (cmd[e, 2] - w_w[2]) * (cmd[e, 2] - w_w[2])
    r_ang = 2.0 * wp.exp(-ang_err / 0.25)
    # 2. feet air time (positive biped): reward = sum over feet of clamp(min(air,contact) of the other..., simplified per Isaac:
    #    feet_air_time_positive_biped: in_mode_time = where(in_contact, contact_time, air_time); single_stance = sum(in_contact)==1;
    #    reward = min(where(single_stance, in_mode_time, 0), threshold); zero when |cmd| < 0.1
    cmd_norm = wp.sqrt(cmd[e, 0] * cmd[e, 0] + cmd[e, 1] * cmd[e, 1] + cmd[e, 2] * cmd[e, 2])
    in_contact0 = sensordata[e, touch_adr[0]] > 1.0
    in_contact1 = sensordata[e, touch_adr[1]] > 1.0
    if in_contact0:
        contact_time[e, 0] = contact_time[e, 0] + dt; air_time[e, 0] = 0.0
    else:
        air_time[e, 0] = air_time[e, 0] + dt; contact_time[e, 0] = 0.0
    if in_contact1:
        contact_time[e, 1] = contact_time[e, 1] + dt; air_time[e, 1] = 0.0
    else:
        air_time[e, 1] = air_time[e, 1] + dt; contact_time[e, 1] = 0.0
    n_contact = 0
    if in_contact0:
        n_contact += 1
    if in_contact1:
        n_contact += 1
    r_air = float(0.0)
    if n_contact == 1 and cmd_norm > 0.1:
        m0 = contact_time[e, 0] if in_contact0 else air_time[e, 0]
        m1 = contact_time[e, 1] if in_contact1 else air_time[e, 1]
        r_air = 0.25 * (wp.min(m0, 0.4) + wp.min(m1, 0.4))
    # 3. feet slide: foot xy speed while in contact
    r_slide = float(0.0)
    for f in range(2):
        cv = cvel[e, foot_body[f]]
        sp = wp.sqrt(cv[3] * cv[3] + cv[4] * cv[4])
        inc = in_contact0 if f == 0 else in_contact1
        if inc:
            r_slide += -0.1 * sp
    # 4. joint terms: limits (ankles), deviation groups, torques, accelerations
    r_lim = float(0.0); r_dev = float(0.0); r_tau = float(0.0); r_acc = float(0.0); r_rate = float(0.0)
    for i in range(nj):
        gi = group[i]          # 0 hip yaw/roll, 1 hip pitch/knee, 2 ankle, 3 torso, 4 arms, 5 fingers
        qi = qpos[e, 7 + i]
        dq = qi - default_q[7 + i]
        if gi == 2:
            lo = jnt_range[i, 0]; hi = jnt_range[i, 1]
            r_lim += -1.0 * (wp.max(lo - qi, 0.0) + wp.max(qi - hi, 0.0))
        if gi == 0 or gi == 3 or gi == 4:
            r_dev += -0.1 * wp.abs(dq)
        if gi == 5:
            r_dev += -0.05 * wp.abs(dq)
        tau = qfrc_actuator[e, 6 + i]
        if gi <= 2:
            r_tau += -1.5e-7 * tau * tau
        if gi <= 1:
            a = qacc[e, 6 + i]
            r_acc += -1.25e-7 * a * a
        da = last_action[e, i] - prev_action[e, i]
        r_rate += -0.005 * da * da
    # 5. orientation, angular velocity xy
    r_orient = -1.0 * (g_b[0] * g_b[0] + g_b[1] * g_b[1])
    r_wxy = -0.05 * (w_w[0] * w_w[0] + w_w[1] * w_w[1])
    # terminations: torso contact, time out; a non-finite state (solver blow-up) also ends the
    # episode and is counted separately (stats_i[3]) so it is reported, not hidden
    t[e] = t[e] + 1
    fell = sensordata[e, touch_adr[2]] > 1.0
    trunc = t[e] >= max_t
    blown = int(0)
    for i in range(7 + nj):
        if not wp.isfinite(qpos[e, i]) or wp.abs(qpos[e, i]) > 1000.0:
            blown = 1
    for i in range(6 + nj):
        if not wp.isfinite(qvel[e, i]) or wp.abs(qvel[e, i]) > 1000.0:     # 1000 rad/s (m/s) is not a robot any more
            blown = 1
    if blown == 1:
        r_lin = 0.0; r_ang = 0.0; r_air = 0.0; r_slide = 0.0; r_lim = 0.0; r_dev = 0.0; r_tau = 0.0; r_acc = 0.0
        r_rate = 0.0; r_orient = 0.0; r_wxy = 0.0
        fell = True
    done = fell or trunc
    r_term = float(0.0)
    if fell:
        r_term = -200.0
    r = (r_lin + r_ang + r_air + r_slide + r_lim + r_dev + r_tau + r_acc + r_rate + r_orient + r_wxy) * dt + r_term
    s = step_idx[0] - 1
    buf_rew[s, e] = r
    buf_done[s, e] = 1.0 if done else 0.0
    terms[e, 0] = r_lin; terms[e, 1] = r_ang; terms[e, 2] = r_air; terms[e, 3] = r_slide; terms[e, 4] = r_dev
    terms[e, 5] = r_orient; terms[e, 6] = r_rate; terms[e, 7] = r_term
    ep_ret[e] = ep_ret[e] + r
    ep_len[e] = ep_len[e] + 1
    reset_mask[e] = done
    resample[e] = resample[e] or (t[e] % resample_every == 0)
    if done and curriculum == 1:
        # Isaac terrain_levels_vel: up a level after walking more than half a cell, down a level
        # after walking less than half the commanded distance; at the top, a random level
        dx = qpos[e, 0] - origins[e, 0]; dy = qpos[e, 1] - origins[e, 1]
        walked = wp.sqrt(dx * dx + dy * dy)
        cmd_dist = wp.sqrt(cmd[e, 0] * cmd[e, 0] + cmd[e, 1] * cmd[e, 1]) * episode_s
        lv = level[e]
        if walked > 0.5 * cell_size:
            lv += 1
        elif walked < 0.5 * cmd_dist:
            lv -= 1
        if lv >= n_levels:
            rng2 = wp.rand_init(seed + 29, step_idx[0] * 9973 + e)
            lv = wp.randi(rng2, 0, n_levels)
        if lv < 0:
            lv = 0
        level[e] = lv
        k = lv * n_cols + col[e]
        origins[e, 0] = origin_table[k, 0]; origins[e, 1] = origin_table[k, 1]; origins[e, 2] = origin_table[k, 2]
    if done:
        wp.atomic_add(stats, 0, ep_ret[e])
        wp.atomic_add(stats_i, 0, ep_len[e])
        wp.atomic_add(stats_i, 1, 1)
        if fell:
            wp.atomic_add(stats_i, 2, 1)
        if blown == 1:
            wp.atomic_add(stats_i, 3, 1)
        ep_ret[e] = 0.0; ep_len[e] = 0; t[e] = 0
        air_time[e, 0] = 0.0; air_time[e, 1] = 0.0; contact_time[e, 0] = 0.0; contact_time[e, 1] = 0.0
        resample[e] = True


@wp.kernel
def g1_reset(reset_mask: wp.array(dtype=wp.bool), default_q: wp.array(dtype=float), origins: wp.array2d(dtype=float),
             seed: int, step_idx: wp.array(dtype=int), qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
             last_action: wp.array2d(dtype=float), prev_action: wp.array2d(dtype=float)):
    e = wp.tid()
    if not reset_mask[e]:
        return
    rng = wp.rand_init(seed + 11, step_idx[0] * 6151 + e)
    for i in range(qpos.shape[1]):
        qpos[e, i] = default_q[i]
    for i in range(qvel.shape[1]):
        qvel[e, i] = 0.0
    qpos[e, 0] = origins[e, 0] + wp.randf(rng, -0.5, 0.5)
    qpos[e, 1] = origins[e, 1] + wp.randf(rng, -0.5, 0.5)
    qpos[e, 2] = origins[e, 2] + default_q[2]
    yaw = wp.randf(rng, -3.14159265, 3.14159265)
    qpos[e, 3] = wp.cos(0.5 * yaw); qpos[e, 4] = 0.0; qpos[e, 5] = 0.0; qpos[e, 6] = wp.sin(0.5 * yaw)
    for i in range(last_action.shape[1]):
        last_action[e, i] = 0.0; prev_action[e, i] = 0.0


# --------------------------------------------------------------------------------------------------

class G1VelocityTask:
    """Isaac Lab velocity task shape for `metalsim.learn.ppo_warp.PPOWarp`."""

    def __init__(self, n, terrain: str = "flat", seed: int = 0, height_scan: bool | None = None, device="metal:0",
                 physics_dt: float = PHYSICS_DT):
        self.n, self.seed, self.device = n, seed, device
        self.terrain_kind = terrain
        self.hfield = None
        self.use_scan = (terrain != "flat") if height_scan is None else height_scan
        if terrain != "flat":
            from metalsim.learn.terrain import isaac_rough_terrain
            self.hfield = isaac_rough_terrain(seed=seed)
        self.model, self.info = build_g1_model(terrain, self.hfield, physics_dt=physics_dt)
        self.physics_dt = physics_dt
        m = self.model
        self.nj = m.nu
        self.decimation = int(round(CONTROL_DT / physics_dt))
        # contact capacity: 3 colliders x <= 4 kept contacts per pair; MuJoCo Warp's heightfield default
        # (256 per world) sizes GPU scratch by naconmax and exhausts memory at 4096 worlds
        self.sim = BatchSim(m, n, options=BatchSimOptions(substeps=self.decimation, njmax=256, nconmax=32,
                                                          solver_iterations=10, ls_iterations=20))
        self.max_t = int(EPISODE_S / CONTROL_DT)
        self.n_scan = 187 if self.use_scan else 0
        self.obs_dim = 12 + 3 * self.nj + self.n_scan
        self.act_dim = self.nj
        self.ctrl_lo = [-100.0] * self.nj; self.ctrl_hi = [100.0] * self.nj   # actions are unbounded targets (scaled in-kernel)
        self.needs_step_idx = True
        self.action_scratch = wp.zeros((n, self.nj), dtype=float, device=device)   # policy writes here; apply_action maps to ctrl
        z = lambda *s, dtype=float: wp.zeros(*s, dtype=dtype, device=device)
        self.default_q = wp.array(m.key_qpos[0].astype(np.float32), dtype=float, device=device)   # Isaac default joint pos
        self.obs = z((n, self.obs_dim)); self.cmd = z((n, 3)); self.heading = z(n); self.standing = z(n, dtype=wp.bool)
        self.resample = wp.array(np.ones(n, bool), dtype=wp.bool, device=device)
        self.last_action = z((n, self.nj)); self.prev_action = z((n, self.nj)); self.height_scan = z((n, max(self.n_scan, 1)))
        self.air_time = z((n, 2)); self.contact_time = z((n, 2)); self.t = z(n, dtype=int)
        self.ep_ret = z(n); self.ep_len = z(n, dtype=int); self.stats = z(4); self.stats_i = z(4, dtype=int)
        self.terms = z((n, 8))
        # joint groups for reward terms
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        def grp(nm):
            if re.fullmatch(r".*_hip_(yaw|roll)_joint", nm): return 0
            if re.fullmatch(r".*_hip_pitch_joint|.*_knee_joint", nm): return 1
            if re.fullmatch(r".*_ankle_.*", nm): return 2
            if nm == "torso_joint": return 3
            if re.fullmatch(r".*_(shoulder|elbow)_.*", nm): return 4
            return 5
        self.group = wp.array(np.array([grp(nm) for nm in names], np.int32), dtype=int, device=device)
        jr = np.array([m.jnt_range[m.actuator_trnid[i][0]] for i in range(m.nu)], np.float32)
        self.jnt_range = wp.array(jr, dtype=float, device=device)
        sadr = lambda nm: int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, nm)])
        self.touch_adr = wp.vec3i(sadr("left_ankle_roll_link_touch"), sadr("right_ankle_roll_link_touch"), sadr("torso_link_touch"))
        bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        self.foot_body = wp.vec2i(bid("left_ankle_roll_link"), bid("right_ankle_roll_link"))
        self.foot_site = wp.vec2i(0, 0)
        # env origins: flat -> a grid with 2.5 m spacing (Isaac env_spacing); rough -> terrain cell centers
        if terrain == "flat":
            side = int(math.ceil(math.sqrt(n)))
            o = np.array([[(i % side) * 2.5, (i // side) * 2.5, 0.0] for i in range(n)], np.float32)
            o[:, :2] -= o[:, :2].mean(0)
        else:
            # Isaac: envs start at random levels up to max_init_terrain_level (5) of the 10 rows and
            # a random terrain type (column); the curriculum then moves them between rows
            rr = np.random.default_rng(seed + 7)
            lv = rr.integers(0, min(5, self.hfield["num_rows"] - 1) + 1, n); cc = rr.integers(0, self.hfield["num_cols"], n)
            tab = self.hfield["origin_table"]()
            o = tab[lv, cc]
            self.origin_table = wp.array(tab.reshape(-1, 3), dtype=float, device=device)
            self.level = wp.array(lv.astype(np.int32), dtype=int, device=device); self.col = wp.array(cc.astype(np.int32), dtype=int, device=device)
        self.curriculum = 1 if terrain != "flat" else 0
        if terrain == "flat":
            self.origin_table = wp.zeros((1, 3), dtype=float, device=device); self.level = wp.zeros(n, dtype=int, device=device); self.col = wp.zeros(n, dtype=int, device=device)
        self.n_levels = self.hfield["num_rows"] if self.hfield else 1; self.n_cols = self.hfield["num_cols"] if self.hfield else 1
        self.cell_size = float(self.hfield["cell_size"]) if self.hfield else 8.0
        self.origins = wp.array(o, dtype=float, device=device)
        self.scanner = None
        if self.use_scan:
            from metalsim.learn.terrain import HeightScanner
            self.scanner = HeightScanner(self, self.hfield)
        self.sim.synchronize()
        # start from the initial pose everywhere
        self.pol_step = None

    # -- graph pieces (called inside PPOWarp's capture) ----------------------------------------------

    def launch_obs(self, step_idx=None):
        if self.scanner is not None:
            self.scanner.launch(step_idx)
        wp.launch(g1_commands, dim=self.n, inputs=[self.sim.d.qpos, self.cmd, self.heading, self.standing, self.resample,
                                                   self.seed, step_idx], device=self.device)
        wp.launch(g1_obs, dim=self.n, inputs=[self.sim.d.qpos, self.sim.d.qvel, self.default_q, self.cmd, self.last_action,
                                              self.height_scan, self.n_scan, self.seed, step_idx, self.obs], device=self.device)

    def launch_apply_action(self, action):
        wp.launch(g1_apply_action, dim=self.n, inputs=[action, self.default_q, ACTION_SCALE, self.last_action, self.prev_action,
                                                       self.sim.d.ctrl], device=self.device)

    def launch_reward_done_reset(self, pol, bufs):
        d = self.sim.d
        wp.launch(g1_reward_done, dim=self.n, inputs=[
            d.qpos, d.qvel, d.qacc, d.qfrc_actuator, d.sensordata, d.site_xpos, d.cvel, self.cmd, self.last_action, self.prev_action,
            self.default_q, self.jnt_range, self.group, self.touch_adr, self.foot_site, self.foot_body, self.air_time,
            self.contact_time, CONTROL_DT, self.t, self.max_t, pol.step_idx, bufs.rew, bufs.done, self.sim._reset_mask,
            self.resample, int(10.0 / CONTROL_DT), self.ep_ret, self.ep_len, self.stats, self.stats_i, self.terms,
            self.curriculum, self.level, self.col, self.origin_table, self.n_levels, self.n_cols, self.cell_size, EPISODE_S,
            self.origins, self.seed], device=self.device)
        import mujoco_warp as mjw
        mjw.reset_data(self.sim.m, d, reset=self.sim._reset_mask)
        wp.launch(g1_reset, dim=self.n, inputs=[self.sim._reset_mask, self.default_q, self.origins, self.seed, pol.step_idx,
                                                d.qpos, d.qvel, self.last_action, self.prev_action], device=self.device)
        # After a reset only the kinematics are needed before the next observation (body poses for
        # the height scan); everything else (sensors, accelerations, contact forces) is produced by
        # the next step itself. A full forward pass here cost ~20 ms per step at 4096 envs.
        mjw.kinematics(self.sim.m, d)

    def reset_all(self):
        """Host-driven initial reset (once)."""
        self.sim._reset_mask.fill_(True)
        idx = wp.zeros(1, dtype=int, device=self.device)
        import mujoco_warp as mjw
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.sim.m, self.sim.d, reset=self.sim._reset_mask)
            wp.launch(g1_reset, dim=self.n, inputs=[self.sim._reset_mask, self.default_q, self.origins, self.seed, idx,
                                                    self.sim.d.qpos, self.sim.d.qvel, self.last_action, self.prev_action], device=self.device)
            mjw.forward(self.sim.m, self.sim.d)
        self.sim.synchronize()

    def episode_stats(self):
        s = self.stats.numpy(); si = self.stats_i.numpy()
        c = int(si[1])
        self.last_termination_fraction = si[2] / c if c else 0.0   # episodes ended by torso contact (vs time-out)
        self.blown_up_episodes = int(si[3])                          # episodes ended by a non-finite state
        self.stats.zero_(); self.stats_i.zero_()
        return (s[0] / c if c else 0.0, si[0] / c if c else 0.0, c)


# --------------------------------------------------------------------------------------------------
# Isaac Lab benchmark_non_rl protocol: uniformly random actions in [-1, 1], wall time per env.step

def benchmark_step(task: G1VelocityTask, num_frames: int = 100, warmup: int = 10, capture: bool = True):
    """env.step = apply actions, physics (decimation substeps), observations/rewards/resets, all on
    the GPU; timed like Isaac's script (wall clock around each step call, mean over frames).
    Isaac's env.step is one fixed pipeline call; ours is one captured graph replay per step
    (``capture=False`` launches the same kernels eagerly, which adds Python launch overhead)."""
    n = task.n
    from metalsim.learn.warp_policy import RolloutBuffers, bump, zero_int
    class _Pol:  # minimal step index holder
        step_idx = wp.zeros(1, dtype=int, device=task.device)
    pol = _Pol()
    bufs = RolloutBuffers(1, n, task.obs_dim, task.act_dim)
    bufs.rew = wp.zeros((1, n), dtype=float, device=task.device); bufs.done = wp.zeros((1, n), dtype=float, device=task.device)
    action = wp.zeros((n, task.act_dim), dtype=float, device=task.device)
    t_action = tb.mps_tensor(action)
    task.reset_all()
    def body():
        wp.launch(zero_int, dim=1, inputs=[pol.step_idx], device=task.device)
        wp.launch(bump, dim=1, inputs=[pol.step_idx], device=task.device)
        task.launch_apply_action(action)
        task.sim.launch_step()
        task.launch_reward_done_reset(pol, bufs)
        task.launch_obs(pol.step_idx)
    graph = None
    if capture:
        with wp.ScopedCapture(device=task.device) as cap:
            body()
        graph = cap.graph
    def step():
        t_action.uniform_(-1.0, 1.0)                       # random actions (torch, as Isaac's script)
        v = task.sim.event.next_value(); tb.signal_event(task.sim.event, v); task.sim.wait(task.sim.event, v)
        if graph is not None:
            wp.capture_launch(graph)
        else:
            body()
    times = []
    for i in range(warmup + num_frames):
        t0 = time.perf_counter_ns()
        step()
        if i >= warmup:
            times.append(time.perf_counter_ns() - t0)
    task.sim.synchronize()
    # Isaac reports mean per-step wall time; with GPU-bound stepping the host blocks on queue depth,
    # so the mean converges to GPU time. Report both the per-call mean and the synchronized rate.
    t_ns = np.array(times, dtype=np.float64)
    return {"num_envs": n, "frames": num_frames, "mean_step_ms": t_ns.mean() / 1e6, "fps_isaac_style": n / (t_ns.mean() / 1e9)}


# --------------------------------------------------------------------------------------------------
# Isaac Lab's G1 PPO configuration (g1_rsl_rl_ppo_cfg.py): 24 steps/env, 5 epochs, 4 minibatches,
# lr 1e-3 adaptive (desired KL 0.01), gamma 0.99, lam 0.95, clip 0.2, entropy 0.008, ELU MLP
# 512-256-128 (rough) / 256-128-128 (flat), init noise std 1.0.

def g1_ppo_config(terrain: str, iterations: int, seed: int = 0):
    from metalsim.learn.ppo_warp import PPOWarpConfig
    return PPOWarpConfig(iterations=iterations, rollout=24, epochs=5, minibatches=4, lr=1e-3, gamma=0.99, lam=0.95,
                         clip=0.2, ent_coef=0.008, vf_coef=1.0, max_grad_norm=1.0, desired_kl=0.01, seed=seed,
                         hidden=(512, 256, 128) if terrain != "flat" else (256, 128, 128), log_every=1)


def train_g1(n=4096, terrain="flat", iterations=1500, seed=0, log_path=None, checkpoint=None, physics_dt=PHYSICS_DT):
    from metalsim.learn.ppo_warp import PPOWarp
    task = G1VelocityTask(n, terrain=terrain, seed=seed, physics_dt=physics_dt)
    algo = PPOWarp(task, g1_ppo_config(terrain, iterations, seed))
    f = open(log_path, "a") if log_path else None
    def log(msg):
        print(msg, flush=True)
        if f:
            f.write(msg + "\n"); f.flush()
    log(f"G1 {terrain} PPO: N={n} obs_dim {task.obs_dim} act_dim {task.act_dim} rollout 24 x {iterations} iterations, physics dt {physics_dt} (decimation {task.decimation})")
    def save(path, it):
        torch.save({"net": algo.net.state_dict(), "terrain": terrain, "n": n, "iterations": it, "obs_dim": task.obs_dim,
                    "act_dim": task.act_dim, "hidden": algo.cfg.hidden}, path)
    cb = (lambda it, a: save(checkpoint.replace(".pt", f"_it{it}.pt"), it) if it % 100 == 0 else None) if checkpoint else None
    algo.train(log=log, callback=cb)
    if checkpoint:
        save(checkpoint, iterations)
        log(f"saved policy to {checkpoint}")
    return algo


if __name__ == "__main__":
    import sys
    wp.config.quiet = True
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    terrain = sys.argv[2] if len(sys.argv) > 2 else "flat"
    if len(sys.argv) > 3 and sys.argv[3] == "train":
        train_g1(n, terrain, int(sys.argv[4]) if len(sys.argv) > 4 else 1500, log_path=sys.argv[5] if len(sys.argv) > 5 else None,
                 checkpoint=sys.argv[6] if len(sys.argv) > 6 else None, physics_dt=float(sys.argv[7]) if len(sys.argv) > 7 else PHYSICS_DT)
        sys.exit(0)
    task = G1VelocityTask(n, terrain=terrain)
    print(f"G1 ({terrain}): nbody {task.model.nbody} nv {task.model.nv} nu {task.model.nu} ngeom {task.model.ngeom} obs_dim {task.obs_dim}")
    r = benchmark_step(task, num_frames=100)
    # synchronized measurement over the same protocol
    t0 = time.perf_counter(); r2 = benchmark_step(task, num_frames=100); task.sim.synchronize(); dt = time.perf_counter() - t0
    # The headline is the synchronized rate: PhysX's fetchResults makes Isaac's per-call time the GPU
    # time, whereas our per-call mean is absorbed by the Metal queue for the first ~64 calls.
    print(f"N={n}: {110 * n / dt:,.0f} env-steps/s (110 steps, synchronized); per-call mean {r2['mean_step_ms']:.2f} ms "
          f"-> {r2['fps_isaac_style']:,.0f} env-steps/s is queue-absorbed and not comparable")
