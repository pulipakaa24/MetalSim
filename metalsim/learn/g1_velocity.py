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
* rewards on flat terrain: Isaac's G1FlatEnvCfg set (``reward_cfg="flat"``, see G1VelocityTask.__init__);
  on rough terrain the G1 rough weights: track_lin_vel_xy_yaw_frame_exp 1.0 (std 0.5), track_ang_vel_z_world_exp
  2.0 (std 0.5), feet_air_time_positive_biped 0.25 (threshold 0.4), feet_slide -0.1,
  dof_pos_limits (ankles) -1.0, joint_deviation_l1 hips(yaw/roll) -0.1, arms -0.1, fingers -0.05,
  torso -0.1, termination -200, flat_orientation_l2 -1.0, action_rate_l2 -0.005,
  dof_acc_l2 (hips, knees) -1.25e-7, dof_torques_l2 (hips, knees, ankles) -1.5e-7, ang_vel_xy_l2 -0.05
* terminations: time out; torso_link contact force > 1 N (flat set: max over the 15 ms contact-history window)
* resets: base xy in +-0.5 m, yaw in +-pi, default joint pose, zero velocities
* events not reproduced: startup friction randomization (0.8/0.6 applied deterministically),
  base mass / COM randomization (disabled in the G1 config anyway), pushes (disabled in G1 config)

Contact quantities come from MuJoCo touch sensors added to the feet and torso (sites enclosing
the colliders), evaluated by MuJoCo Warp each step.

Engines (``G1VelocityTask(engine=...)``): "mjwarp" (default; MuJoCo Warp, ``BatchSim``) or "newton"
(Newton XPBD, ``metalsim.physics.newton_backend.NewtonSim``: Isaac's actuator via ActuatorPD, joint
relaxation 0.4/0.4, ``newton_iterations`` at ``newton_dt``). NewtonSim exposes the same MuJoCo-layout
state arrays, so observation, reward, termination and reset kernels are shared verbatim; on Newton the
touch "sensors" are contact force magnitudes on the foot / torso colliders (XPBD update_contacts), qacc
is the joint velocity difference over the last 2.5 ms, and feet_slide reads the foot body's own world velocity
(on MuJoCo Warp derived from cvel at the foot's frame origin).
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
                standing: wp.array(dtype=wp.bool), resample: wp.array(dtype=wp.bool), seed: int, step_idx: wp.array(dtype=int),
                lin_y: float):
    """Isaac UniformVelocityCommand with heading_command=True: lin_vel_x in [0,1], lin_vel_y in [-lin_y, lin_y]
    (G1FlatEnvCfg 0.5; the rough task here keeps its historical 1.0),
    target heading in [-pi, pi]; ang_vel_z = clip(0.5 * wrap(heading - yaw), -1, 1); 2% standing envs."""
    e = wp.tid()
    if resample[e]:
        rng = wp.rand_init(seed + 3, step_idx[0] * 104729 + e)
        cmd[e, 0] = wp.randf(rng, 0.0, 1.0)
        cmd[e, 1] = wp.randf(rng, -lin_y, lin_y)
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
                   site_xpos: wp.array2d(dtype=wp.vec3), foot_vel: wp.array2d(dtype=wp.vec3),
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
                   seed: int, isaac_flat: int, torso_hist: wp.array(dtype=float),
                   use_sensor: int, sens_air: wp.array2d(dtype=float), sens_con: wp.array2d(dtype=float),
                   foot_hist: wp.array2d(dtype=float)):
    """Reward, termination and episode bookkeeping. ``isaac_flat`` = 1 selects Isaac's G1FlatEnvCfg reward set
    (see G1VelocityTask), 0 the G1RoughEnvCfg set as ported first. ``jnt_range`` holds the limits the
    dof_pos_limits term uses (soft limits for the flat set). ``torso_hist`` is the max torso touch force over
    the contact-history window of this control step (0 when the physics hook did not run). ``use_sensor`` = 1
    (flat set on MuJoCo Warp): the feet contact flags and in-mode times come from the ContactSensor
    (Isaac's current_air_time / current_contact_time, contact iff contact_time > 0, updated every physics
    substep), feet_slide's contact flag from the max foot force over the history (``foot_hist``), and the
    torso termination from ``torso_hist`` (max torso force over the history) alone."""
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
    w_ang = 2.0
    if isaac_flat == 1:
        w_ang = 1.0                                    # G1FlatEnvCfg: track_ang_vel_z_exp.weight = 1.0
    r_ang = w_ang * wp.exp(-ang_err / 0.25)
    # 2. feet air time (positive biped): reward = sum over feet of clamp(min(air,contact) of the other..., simplified per Isaac:
    #    feet_air_time_positive_biped: in_mode_time = where(in_contact, contact_time, air_time); single_stance = sum(in_contact)==1;
    #    reward = min(where(single_stance, in_mode_time, 0), threshold); zero when |cmd| < 0.1
    cmd_norm = wp.sqrt(cmd[e, 0] * cmd[e, 0] + cmd[e, 1] * cmd[e, 1] + cmd[e, 2] * cmd[e, 2])
    if isaac_flat == 1:
        cmd_norm = wp.sqrt(cmd[e, 0] * cmd[e, 0] + cmd[e, 1] * cmd[e, 1])   # Isaac: norm of command[:, :2]
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
    m0 = contact_time[e, 0] if in_contact0 else air_time[e, 0]
    m1 = contact_time[e, 1] if in_contact1 else air_time[e, 1]
    slide0 = in_contact0
    slide1 = in_contact1
    if use_sensor == 1:
        # Isaac feet_air_time_positive_biped on ContactSensor data; feet_slide: net_forces_w_history max > 1 N
        in_contact0 = sens_con[e, 0] > 0.0
        in_contact1 = sens_con[e, 1] > 0.0
        m0 = sens_con[e, 0] if in_contact0 else sens_air[e, 0]
        m1 = sens_con[e, 1] if in_contact1 else sens_air[e, 1]
        slide0 = foot_hist[e, 0] > 1.0
        slide1 = foot_hist[e, 1] > 1.0
    n_contact = 0
    if in_contact0:
        n_contact += 1
    if in_contact1:
        n_contact += 1
    r_air = float(0.0)
    if n_contact == 1 and cmd_norm > 0.1:
        if isaac_flat == 1:
            # feet_air_time_positive_biped: min over feet of the in-mode time, clamped at 0.4; flat weight 0.75
            r_air = 0.75 * wp.min(wp.min(m0, m1), 0.4)
        else:
            r_air = 0.25 * (wp.min(m0, 0.4) + wp.min(m1, 0.4))
    # 3. feet slide (Isaac mdp.feet_slide): |body_lin_vel_w[foot].xy| while the foot is in contact; the foot
    #    body's own world linear velocity (at its frame origin), filled per engine before this kernel
    r_slide = float(0.0)
    for f in range(2):
        fv = foot_vel[e, f]
        sp = wp.sqrt(fv[0] * fv[0] + fv[1] * fv[1])
        inc = slide0 if f == 0 else slide1
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
        a = qacc[e, 6 + i]
        if isaac_flat == 1:
            if gi <= 1:                                # G1FlatEnvCfg: dof_torques_l2 -2e-6 on hips + knees, dof_acc_l2 -1e-7
                r_tau += -2.0e-6 * tau * tau
                r_acc += -1.0e-7 * a * a
        else:
            if gi <= 2:
                r_tau += -1.5e-7 * tau * tau
            if gi <= 1:
                r_acc += -1.25e-7 * a * a
        da = last_action[e, i] - prev_action[e, i]
        r_rate += -0.005 * da * da
    # 5. orientation, angular velocity xy
    r_orient = -1.0 * (g_b[0] * g_b[0] + g_b[1] * g_b[1])
    r_wxy = -0.05 * (w_w[0] * w_w[0] + w_w[1] * w_w[1])
    r_vz = float(0.0)
    if isaac_flat == 1:
        r_wxy = -0.05 * (w_b[0] * w_b[0] + w_b[1] * w_b[1])   # Isaac ang_vel_xy_l2: root_ang_vel_b
        v_b = Rt * v_w
        r_vz = -0.2 * v_b[2] * v_b[2]                      # G1FlatEnvCfg: lin_vel_z_l2 -0.2 on root_lin_vel_b
    # terminations: torso contact, time out; a non-finite state (solver blow-up) also ends the
    # episode and is counted separately (stats_i[3]) so it is reported, not hidden
    t[e] = t[e] + 1
    fell = sensordata[e, touch_adr[2]] > 1.0
    if isaac_flat == 1:
        # illegal_contact: max over the contact sensor's history (3 physics steps of 5 ms in Isaac)
        fell = wp.max(sensordata[e, touch_adr[2]], torso_hist[e]) > 1.0
        if use_sensor == 1:
            fell = torso_hist[e] > 1.0
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
        r_rate = 0.0; r_orient = 0.0; r_wxy = 0.0; r_vz = 0.0
        fell = True
    done = fell or trunc
    r_term = float(0.0)
    if fell:
        r_term = -200.0
    # Isaac's RewardManager multiplies every term by dt, the termination penalty included (-200 * 0.02 = -4)
    r = (r_lin + r_ang + r_air + r_slide + r_lim + r_dev + r_tau + r_acc + r_rate + r_orient + r_wxy + r_vz + r_term) * dt
    # row of the rollout buffer; the modulo lets a caller keep a free-running step counter (fresh RNG
    # streams every step) with a one-row buffer (metalsim.learn.rslrl_adapter); PPOWarp's s < T is unchanged
    s = (step_idx[0] - 1) % buf_rew.shape[0]
    buf_rew[s, e] = r
    buf_done[s, e] = 1.0 if done else 0.0
    terms[e, 0] = r_lin; terms[e, 1] = r_ang; terms[e, 2] = r_air; terms[e, 3] = r_slide; terms[e, 4] = r_dev
    terms[e, 5] = r_orient; terms[e, 6] = r_rate; terms[e, 7] = r_term
    terms[e, 8] = r_vz; terms[e, 9] = r_wxy; terms[e, 10] = r_tau; terms[e, 11] = r_acc; terms[e, 12] = r_lim
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
def g1_foot_vel_mjwarp(cvel: wp.array2d(dtype=wp.spatial_vector), xpos: wp.array2d(dtype=wp.vec3),
                       subtree_com: wp.array2d(dtype=wp.vec3), foot_body: wp.vec2i, root_body: wp.vec2i,
                       foot_vel: wp.array2d(dtype=wp.vec3)):
    """World linear velocity of each foot body's frame origin from MuJoCo's cvel, which is the body's spatial
    velocity [rot; lin] expressed at the subtree COM of its tree root: v = v_lin + w x (xpos - subtree_com[root])."""
    e = wp.tid()
    for f in range(2):
        cv = cvel[e, foot_body[f]]
        w = wp.vec3(cv[0], cv[1], cv[2])
        v = wp.vec3(cv[3], cv[4], cv[5])
        foot_vel[e, f] = v + wp.cross(w, xpos[e, foot_body[f]] - subtree_com[e, root_body[f]])


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


@wp.kernel
def g1_contact_hist_max(hist: wp.array4d(dtype=float), foot_hist: wp.array2d(dtype=float), torso_hist: wp.array(dtype=float)):
    """Max net contact force norm over the ContactSensor history (N, T, B=3 [left foot, right foot, torso], 3)."""
    e = wp.tid()
    for b in range(3):
        mx = float(0.0)
        for t in range(hist.shape[1]):
            fx = hist[e, t, b, 0]; fy = hist[e, t, b, 1]; fz = hist[e, t, b, 2]
            mx = wp.max(mx, wp.sqrt(fx * fx + fy * fy + fz * fz))
        if b < 2:
            foot_hist[e, b] = mx
        else:
            torso_hist[e] = mx


@wp.kernel
def g1_record_timeout(step_idx: wp.array(dtype=int), buf_done: wp.array2d(dtype=float), terms: wp.array2d(dtype=float),
                      buf_timeout: wp.array2d(dtype=float)):
    """1 where this step ended the episode by the time limit (done and not a fall / blow-up)."""
    e = wp.tid()
    s = (step_idx[0] - 1) % buf_done.shape[0]
    to = float(0.0)
    if buf_done[s, e] > 0.5 and terms[e, 7] >= 0.0:
        to = 1.0
    buf_timeout[s, e] = to


# --------------------------------------------------------------------------------------------------

class G1VelocityTask:
    """Isaac Lab velocity task shape for `metalsim.learn.ppo_warp.PPOWarp`."""

    def __init__(self, n, terrain: str = "flat", seed: int | None = None, height_scan: bool | None = None, device="metal:0",
                 physics_dt: float = PHYSICS_DT, engine: str = "mjwarp", newton_iterations: int = 4, newton_dt: float = 0.00125,
                 newton_kw: dict | None = None,
                 reward_cfg: str | None = None):
        """``reward_cfg``: "flat" = Isaac's G1FlatEnvCfg (default on flat terrain): track_ang_vel_z 1.0,
        lin_vel_y in +-0.5, feet_air_time 0.75 x min over feet (xy command norm), lin_vel_z_l2 -0.2 and
        ang_vel_xy_l2 -0.05 in the body frame, dof_torques_l2 -2e-6 and dof_acc_l2 -1e-7 on hips + knees,
        dof_pos_limits on the soft limits (G1_CFG soft_joint_pos_limit_factor 0.9), torso-contact
        termination on the max force over the contact-history window (Isaac: 3 physics steps of 5 ms =
        the last 15 ms of the control step; here round(15 ms / physics_dt) substeps on MuJoCo Warp, the
        final substep only on Newton). "rough" = the G1RoughEnvCfg set as first ported (default on
        rough terrain, unchanged; runs before this port used it on flat terrain too).

        ``seed`` (default: 42 on rough terrain, Isaac's rsl_rl default seed, from which Isaac generates the
        terrain; 0 on flat) seeds the task's random streams and, on rough terrain, the terrain generator.
        Rough terrain assigns env i to terrain column floor(i / (n / num_cols)) as Isaac's
        TerrainImporter does and draws the initial level uniformly in [0, 5] with torch.randint (CPU
        generator seeded with ``seed``; Isaac draws it from the CUDA generator, so the draw itself differs)."""
        if seed is None:
            seed = 42 if terrain != "flat" else 0
        self.n, self.seed, self.device = n, seed, device
        self.terrain_kind = terrain
        self.hfield = None
        self.use_scan = (terrain != "flat") if height_scan is None else height_scan
        if terrain != "flat":
            from metalsim.learn.terrain import isaac_rough_terrain
            self.hfield = isaac_rough_terrain(seed=seed)
        self.engine = engine
        if engine not in ("mjwarp", "newton"):
            raise ValueError(f"engine must be 'mjwarp' or 'newton', not {engine!r}")
        self.model, self.info = build_g1_model(terrain, self.hfield, physics_dt=physics_dt)   # metadata source for both engines
        m = self.model
        self.nj = m.nu
        if engine == "newton":
            from metalsim.physics.newton_backend import NewtonSim
            self.physics_dt = newton_dt
            self.sim = NewtonSim(m, n, iterations=newton_iterations, dt=newton_dt, control_dt=CONTROL_DT, device=device,
                                 hfield=self.hfield, **(newton_kw or {}))
            self.decimation = self.sim.substeps
        else:
            self.physics_dt = physics_dt
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
        # per-term values of the last step (weights applied, before x dt): 0 track_lin, 1 track_ang, 2 feet_air_time,
        # 3 feet_slide, 4 joint_deviation (all groups), 5 flat_orientation, 6 action_rate, 7 termination,
        # 8 lin_vel_z, 9 ang_vel_xy, 10 dof_torques, 11 dof_acc, 12 dof_pos_limits
        self.terms = z((n, 13))
        self.reward_cfg = reward_cfg or ("flat" if terrain == "flat" else "rough")
        if self.reward_cfg not in ("flat", "rough"):
            raise ValueError(f"reward_cfg must be 'flat' or 'rough', not {self.reward_cfg!r}")
        self.isaac_flat = 1 if self.reward_cfg == "flat" else 0
        self.lin_vel_y = 0.5 if self.isaac_flat else 1.0
        self.torso_hist = z(n)
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
        if self.isaac_flat:     # Isaac joint_pos_limits reads soft_joint_pos_limits: mid +- 0.9 * half range
            mid = 0.5 * (jr[:, 0] + jr[:, 1]); half = 0.5 * (jr[:, 1] - jr[:, 0])
            jr = np.stack([mid - 0.9 * half, mid + 0.9 * half], 1).astype(np.float32)
        self.jnt_range = wp.array(jr, dtype=float, device=device)
        sadr = lambda nm: int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, nm)])
        self.touch_adr = wp.vec3i(sadr("left_ankle_roll_link_touch"), sadr("right_ankle_roll_link_touch"), sadr("torso_link_touch"))
        bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        self.foot_body = wp.vec2i(bid("left_ankle_roll_link"), bid("right_ankle_roll_link"))
        self.foot_root = wp.vec2i(int(m.body_rootid[self.foot_body[0]]), int(m.body_rootid[self.foot_body[1]]))
        self.foot_vel = z((n, 2), dtype=wp.vec3)     # foot body origin world linear velocity (feet_slide), per engine
        self.foot_site = wp.vec2i(0, 0)
        # env origins: flat -> a grid with 2.5 m spacing (Isaac env_spacing); rough -> terrain cell centers
        if terrain == "flat":
            side = int(math.ceil(math.sqrt(n)))
            o = np.array([[(i % side) * 2.5, (i // side) * 2.5, 0.0] for i in range(n)], np.float32)
            o[:, :2] -= o[:, :2].mean(0)
        else:
            # Isaac: envs start at random levels up to max_init_terrain_level (5) of the 10 rows and
            # terrain type (column) by env id; the curriculum then moves them between rows
            nr, nc = self.hfield["num_rows"], self.hfield["num_cols"]
            gen = torch.Generator().manual_seed(int(seed))
            lv = torch.randint(0, min(5, nr - 1) + 1, (n,), generator=gen).numpy()
            cc = torch.div(torch.arange(n), n / nc, rounding_mode="floor").to(torch.long).numpy()   # Isaac terrain_types
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
        # contact history window: Isaac's ContactSensor keeps 3 physics steps of 5 ms (15 ms); here the same
        # 15 ms, round(15 ms / dt) substeps (6 at 2.5 ms)
        self.hist_substeps = max(1, min(self.decimation, int(round(0.015 / self.physics_dt))))
        self.foot_hist = z((n, 2))
        self.contact = None
        self.use_sensor = 0
        if self.isaac_flat and engine == "newton":       # NewtonSim records the torso contact history itself
            self.sim.hist_out, self.sim.hist_substeps = self.torso_hist, self.hist_substeps
        if self.isaac_flat and engine == "mjwarp":
            # Isaac's contact_forces sensor (metalsim.sensors.contact.ContactSensor): net normal force per tracked
            # body with a substep history and air/contact times, updated after every substep (BatchSim substep
            # hook, so every sim.launch_step / captured step includes it); reset with the envs
            from metalsim.sensors.contact import ContactSensor
            self.contact = ContactSensor(self.sim, ["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"],
                                         history_length=self.hist_substeps, track_air_time=True, force_threshold=1.0)
            self.use_sensor = 1
        self._dummy2 = z((n, 2))
        self.sim.synchronize()
        # start from the initial pose everywhere
        self.pol_step = None

    # -- graph pieces (called inside PPOWarp's capture) ----------------------------------------------

    def launch_obs(self, step_idx=None):
        if self.scanner is not None:
            self.scanner.launch(step_idx)
        wp.launch(g1_commands, dim=self.n, inputs=[self.sim.d.qpos, self.cmd, self.heading, self.standing, self.resample,
                                                   self.seed, step_idx, self.lin_vel_y], device=self.device)
        wp.launch(g1_obs, dim=self.n, inputs=[self.sim.d.qpos, self.sim.d.qvel, self.default_q, self.cmd, self.last_action,
                                              self.height_scan, self.n_scan, self.seed, step_idx, self.obs], device=self.device)

    def launch_apply_action(self, action):
        wp.launch(g1_apply_action, dim=self.n, inputs=[action, self.default_q, ACTION_SCALE, self.last_action, self.prev_action,
                                                       self.sim.d.ctrl], device=self.device)

    def launch_reward_done_reset(self, pol, bufs):
        d = self.sim.d
        if self.engine == "newton":
            foot_vel = d.foot_vel                    # written by NewtonSim from the foot bodies' state
        else:
            wp.launch(g1_foot_vel_mjwarp, dim=self.n, inputs=[d.cvel, d.xpos, d.subtree_com, self.foot_body, self.foot_root],
                      outputs=[self.foot_vel], device=self.device)
            foot_vel = self.foot_vel
        if self.contact is not None:
            wp.launch(g1_contact_hist_max, dim=self.n, inputs=[self.contact._hist, self.foot_hist, self.torso_hist], device=self.device)
            sens_air, sens_con = self.contact._cur_air, self.contact._cur_con
        else:
            sens_air = sens_con = self._dummy2
        wp.launch(g1_reward_done, dim=self.n, inputs=[
            d.qpos, d.qvel, d.qacc, d.qfrc_actuator, d.sensordata, d.site_xpos, foot_vel, self.cmd, self.last_action, self.prev_action,
            self.default_q, self.jnt_range, self.group, self.touch_adr, self.foot_site, self.foot_body, self.air_time,
            self.contact_time, CONTROL_DT, self.t, self.max_t, pol.step_idx, bufs.rew, bufs.done, self.sim._reset_mask,
            self.resample, int(10.0 / CONTROL_DT), self.ep_ret, self.ep_len, self.stats, self.stats_i, self.terms,
            self.curriculum, self.level, self.col, self.origin_table, self.n_levels, self.n_cols, self.cell_size, EPISODE_S,
            self.origins, self.seed, self.isaac_flat, self.torso_hist, self.use_sensor, sens_air, sens_con, self.foot_hist],
            device=self.device)
        if self.engine == "newton":
            wp.launch(g1_reset, dim=self.n, inputs=[self.sim._reset_mask, self.default_q, self.origins, self.seed, pol.step_idx,
                                                    d.qpos, d.qvel, self.last_action, self.prev_action], device=self.device)
            self.sim.launch_reset()          # MuJoCo-layout reset state -> Newton joint coordinates -> FK (masked)
            return
        import mujoco_warp as mjw
        mjw.reset_data(self.sim.m, d, reset=self.sim._reset_mask)
        if self.contact is not None:     # Isaac resets the contact sensor (history, air/contact times) with the env
            self.contact.launch_reset(self.sim._reset_mask)
        # reset randomization keyed by the free-running counter when the caller has one (PPOWarp's rng_step)
        wp.launch(g1_reset, dim=self.n, inputs=[self.sim._reset_mask, self.default_q, self.origins, self.seed,
                                                getattr(pol, "rng_step", pol.step_idx),
                                                d.qpos, d.qvel, self.last_action, self.prev_action], device=self.device)
        # After a reset only the kinematics are needed before the next observation (body poses for
        # the height scan); everything else (sensors, accelerations, contact forces) is produced by
        # the next step itself. A full forward pass here cost ~20 ms per step at 4096 envs.
        mjw.kinematics(self.sim.m, d)

    def launch_timeouts(self, pol, bufs):
        """Time-out flags of this step into ``bufs.timeout`` (for PPO's bootstrapping on truncation)."""
        wp.launch(g1_record_timeout, dim=self.n, inputs=[pol.step_idx, bufs.done, self.terms, bufs.timeout], device=self.device)

    def reset_all(self):
        """Host-driven initial reset (once)."""
        self.sim._reset_mask.fill_(True)
        idx = wp.zeros(1, dtype=int, device=self.device)
        if self.engine == "newton":
            with wp.ScopedDevice(self.device):
                wp.launch(g1_reset, dim=self.n, inputs=[self.sim._reset_mask, self.default_q, self.origins, self.seed, idx,
                                                        self.sim.d.qpos, self.sim.d.qvel, self.last_action, self.prev_action], device=self.device)
                self.sim.launch_reset()
                self.sim.d.sensordata.zero_(); self.sim.d.qacc.zero_(); self.sim.d.qfrc_actuator.zero_()
            self.sim.synchronize()
            return
        import mujoco_warp as mjw
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.sim.m, self.sim.d, reset=self.sim._reset_mask)
            if self.contact is not None:
                self.contact.launch_reset(self.sim._reset_mask)
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


def train_g1(n=4096, terrain="flat", iterations=1500, seed=None, log_path=None, checkpoint=None, physics_dt=PHYSICS_DT,
             engine="mjwarp", newton_iterations=4, newton_dt=0.00125, newton_kw=None):
    from metalsim.learn.ppo_warp import PPOWarp
    task = G1VelocityTask(n, terrain=terrain, seed=seed, physics_dt=physics_dt, engine=engine,
                          newton_iterations=newton_iterations, newton_dt=newton_dt, newton_kw=newton_kw)
    seed = task.seed                     # None -> the task's default (42 rough, Isaac's; 0 flat)
    algo = PPOWarp(task, g1_ppo_config(terrain, iterations, seed))
    f = open(log_path, "a") if log_path else None
    def log(msg):
        print(msg, flush=True)
        if f:
            f.write(msg + "\n"); f.flush()
    eng = f"newton XPBD {newton_iterations} it {newton_kw or ''}" if engine == "newton" else "mjwarp"
    log(f"G1 {terrain} PPO: N={n} obs_dim {task.obs_dim} act_dim {task.act_dim} rollout 24 x {iterations} iterations, engine {eng}, "
        f"physics dt {task.physics_dt} (decimation {task.decimation}), seed {seed}")
    def save(path, it):
        torch.save({"net": algo.net.state_dict(), "terrain": terrain, "n": n, "iterations": it, "obs_dim": task.obs_dim,
                    "act_dim": task.act_dim, "hidden": algo.cfg.hidden, "engine": engine}, path)
    cb = (lambda it, a: save(checkpoint.replace(".pt", f"_it{it}.pt"), it) if it % 100 == 0 else None) if checkpoint else None
    from metalsim.learn.monitor import AnomalyMonitor
    mon = AnomalyMonitor(task, algo.cfg, log=log, path=(log_path + ".anomalies.jsonl") if log_path else None)
    algo.train(log=log, callback=cb, monitor=mon)
    if checkpoint:
        save(checkpoint, iterations)
        log(f"saved policy to {checkpoint}")
    return algo


if __name__ == "__main__":
    import sys
    wp.config.quiet = True
    # optional flags (any position): --engine mjwarp|newton, --newton_it N, --newton_dt S
    opts = {"--engine": "mjwarp", "--newton_it": "4", "--newton_dt": "0.00125", "--newton_limit_margin": "0.15", "--newton_kw": ""}
    for k in list(opts):
        if k in sys.argv:
            i = sys.argv.index(k); opts[k] = sys.argv[i + 1]; del sys.argv[i:i + 2]
    ekw = dict(engine=opts["--engine"], newton_iterations=int(opts["--newton_it"]), newton_dt=float(opts["--newton_dt"]))
    if ekw["engine"] == "newton":         # "none": keep the USD's revolute limits (for a Newton build that unwraps angles)
        lm = opts["--newton_limit_margin"]; ekw["newton_kw"] = {"limit_margin": None if lm.lower() == "none" else float(lm)}
        for kv in filter(None, opts["--newton_kw"].split(",")):   # e.g. drive=solver,joint_coloring=True,relaxation=0.8
            k, v = kv.split("="); ekw["newton_kw"][k] = v if k == "drive" else eval(v)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    terrain = sys.argv[2] if len(sys.argv) > 2 else "flat"
    if len(sys.argv) > 3 and sys.argv[3] == "train":
        train_g1(n, terrain, int(sys.argv[4]) if len(sys.argv) > 4 else 1500, log_path=sys.argv[5] if len(sys.argv) > 5 else None,
                 checkpoint=sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None,
                 physics_dt=float(sys.argv[7]) if len(sys.argv) > 7 else PHYSICS_DT, **ekw)
        sys.exit(0)
    task = G1VelocityTask(n, terrain=terrain, physics_dt=float(sys.argv[3]) if len(sys.argv) > 3 else PHYSICS_DT, **ekw)
    print(f"G1 ({terrain}, {task.engine}, physics dt {task.physics_dt}): nbody {task.model.nbody} nv {task.model.nv} nu {task.model.nu} ngeom {task.model.ngeom} obs_dim {task.obs_dim}")
    r = benchmark_step(task, num_frames=100)
    # synchronized measurement over the same protocol
    t0 = time.perf_counter(); r2 = benchmark_step(task, num_frames=100); task.sim.synchronize(); dt = time.perf_counter() - t0
    # The headline is the synchronized rate: PhysX's fetchResults makes Isaac's per-call time the GPU
    # time, whereas our per-call mean is absorbed by the Metal queue for the first ~64 calls.
    print(f"N={n}: {110 * n / dt:,.0f} env-steps/s (110 steps, synchronized); per-call mean {r2['mean_step_ms']:.2f} ms "
          f"-> {r2['fps_isaac_style']:,.0f} env-steps/s is queue-absorbed and not comparable")
