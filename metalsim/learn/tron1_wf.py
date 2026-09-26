"""TRON1 wheel-foot locomotion policy for *this* robot (with its sensor pack), trained on MetalSim.

Why: LimX's public WF policy is "meant for dev/sim reference" (LimX, tron1-rl-deploy-python issue
#1) and shuffles on the real robot. This task follows LimX's open wheel-foot recipe
(tron1-rl-isaacgym `wheelfoot_flat`, commit 43776fd) and adds what differs here:

* robot: LimX's WF_TRON1A MJCF with our sensor pack (`metalsim.tron1.sim.build_model`, nominal),
  tyres as the ellipsoid fit to the real tread (129.8 mm crown), joint friction
* control: 50 Hz policy, 200 Hz physics (decimation 4). Legs: q = default + 0.25 a, Kp 42 / Kd 2.5,
  80 N m; wheels: velocity 0.5 a through Kd 0.8, 40 N m (LimX `control`). PD runs inside the
  physics as MuJoCo affine actuators, recomputed every physics step as in LimX's loop
* action delay 0-20 ms per env at physics-step (5 ms) resolution (LimX `delay_ms_range`)
* observations exactly as LimX (28): gyro * 0.25, projected gravity, leg q - default (6),
  all joint velocities * 0.05 (8), last actions (8); LimX's uniform noise x 1.5; plus an IMU
  mounting offset (+-1.2 deg) and gyro bias. History of 10 frames + scaled command (3) -> actor
  input 283. (LimX feeds the history through a separate encoder; one MLP on the history is the
  same information, and the deploy code builds the same history.)
* domain randomization (LimX ranges): friction 0.2-1.6, base mass -0.5..+2 kg, base COM
  +-(3, 2, 3) cm, inertia x0.8-1.2, Kp / Kd / motor strength x0.8-1.2, default joint offsets
  +-0.05 rad, pushes up to 1 m/s every 7 s. Added: sensor pack 0.5-3 kg with its COM anywhere on
  the top plate, joint friction x0.3-2.5 (`SimParams.sample` ranges)
* rewards: LimX's scales (keep_balance, tracking, nominal foot position, leg symmetry, same-foot
  x/z, lin_vel_z, ang_vel_xy, torques, dof_acc, action rate / smoothness, dof limits,
  orientation, base height), minus `feet_distance` (our MJCF's wheels sit 0.297 m apart at abad 0,
  so LimX's 0.32-0.35 m window would force the legs to splay). Added for standing still (command
  exactly zero, 30 % of segments): base speed, yaw rate, leg joint speed, and distance / heading
  from where the zero command began. Rewards are scale * term * dt, as in legged_gym.
* commands: every 5 s, 30 % zero, 20 % small (|vx| <= 0.3 m/s, |wz| <= 0.2 rad/s), 50 % LimX's
  full range (vx in [-1, 1], wz in [-0.6, 0.6])
* termination: tilt over 60 deg or base below 0.3 m; time out 20 s

    python -m metalsim.learn.tron1_wf train [--n 8192] [--iterations 3000] [--run name]
"""
from __future__ import annotations

import json
import math
import os
import time

import mujoco
import numpy as np
import torch
import warp as wp

from metalsim.learn.g1_velocity import quat_to_mat_wxyz
from metalsim.physics.batch import BatchSim, BatchSimOptions

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTROL_DT, PHYSICS_DT, DECIM, EPISODE_S = 0.02, 0.005, 4, 20.0
KP, KD_LEG, KD_WHEEL, LEG_TAU, WHEEL_TAU = 42.0, 2.5, 0.8, 80.0, 40.0
SCALE_POS, SCALE_VEL = 0.25, 0.5
OBS_DIM, HIST = 28, 10
ACTOR_IN = OBS_DIM * HIST + 3
N_STATIC = 13                             # per-robot physics the critic sees (see _randomize_physics)
PRIV_DIM = 27 + N_STATIC + 12             # privileged block after the actor's 283 columns
CRITIC_IN = ACTOR_IN + PRIV_DIM
CMD_SCALE = (2.0, 2.0, 0.25)             # LimX obs_scales lin_vel, lin_vel, ang_vel
WHEELS = (3, 7)
BASE_HEIGHT_TARGET = 0.7664              # LimX base_height_target (0.6 + 0.1664)
TYRE_R = 0.1298
PUSH_EVERY = int(7.0 / CONTROL_DT)
RESAMPLE_EVERY = int(5.0 / CONTROL_DT)


EFFORT_LIMIT = "joint"      # "joint" (default since 2026-09-26) or "actuator" (archived)


def build_train_model():
    """LimX's WF MJCF + our sensor pack (nominal), with the joint PD as affine actuators."""
    from metalsim.tron1.realism import SimParams
    from metalsim.tron1.sim import build_model
    m = build_model(SimParams.nominal())
    m.opt.timestep = PHYSICS_DT
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    m.opt.iterations, m.opt.ls_iterations = 10, 20
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    for i in range(m.nu):
        wheel = i in WHEELS
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[i, :] = 0.0
        m.actuator_biasprm[i, :] = 0.0
        if wheel:                       # tau = Kd (ctrl - qdot): velocity target in ctrl
            m.actuator_gainprm[i, 0] = KD_WHEEL
            m.actuator_biasprm[i, 2] = -KD_WHEEL
        else:                           # tau = Kp (ctrl - q) - Kd qdot: position target in ctrl
            m.actuator_gainprm[i, 0] = KP
            m.actuator_biasprm[i, 1] = -KP
            m.actuator_biasprm[i, 2] = -KD_LEG
        lim = WHEEL_TAU if wheel else LEG_TAU
        # LimX clamps the total motor torque (MJCF <motor ctrlrange> +-80 / +-40, URDF effort, SDK), i.e. at the joint:
        # jnt_actfrcrange keeps the PD's implicit damping while clamped (engine_derivative.c drops a clamped actuator's
        # velocity derivative); EFFORT_LIMIT = "actuator" restores the archived actuator forcerange
        j = m.actuator_trnid[i][0]
        if EFFORT_LIMIT == "actuator":
            m.actuator_forcelimited[i] = 1
            m.actuator_forcerange[i] = (-lim, lim)
        else:
            m.actuator_forcelimited[i] = 0
            m.jnt_actfrclimited[j] = 1
            m.jnt_actfrcrange[j] = (-lim, lim)
        m.actuator_ctrllimited[i] = 0
    return m


# --------------------------------------------------------------------------------------------------
# kernels

BASE_HEIGHT_TARGET_W = wp.constant(BASE_HEIGHT_TARGET)
TYRE_R_W = wp.constant(TYRE_R)

@wp.func
def imu_rot(r: float, p: float) -> wp.mat33:
    cr = wp.cos(r); sr = wp.sin(r); cp = wp.cos(p); sp = wp.sin(p)
    Rx = wp.mat33(1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr)
    Ry = wp.mat33(cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp)
    return Rx * Ry


@wp.kernel
def wf_commands(cmd: wp.array2d(dtype=float), resample: wp.array(dtype=wp.bool), qpos: wp.array2d(dtype=float),
                anchor: wp.array2d(dtype=float), seed: int, step_idx: wp.array(dtype=int)):
    e = wp.tid()
    if not resample[e]:
        return
    rng = wp.rand_init(seed + 3, step_idx[0] * 104729 + e)
    u = wp.randf(rng)
    if u < 0.3:
        cmd[e, 0] = 0.0; cmd[e, 2] = 0.0
    elif u < 0.5:
        cmd[e, 0] = wp.randf(rng, -0.3, 0.3); cmd[e, 2] = wp.randf(rng, -0.2, 0.2)
    else:
        cmd[e, 0] = wp.randf(rng, -1.0, 1.0); cmd[e, 2] = wp.randf(rng, -0.6, 0.6)
    cmd[e, 1] = 0.0
    q = wp.vec4(qpos[e, 3], qpos[e, 4], qpos[e, 5], qpos[e, 6])
    R = quat_to_mat_wxyz(q)
    anchor[e, 0] = qpos[e, 0]; anchor[e, 1] = qpos[e, 1]; anchor[e, 2] = wp.atan2(R[1, 0], R[0, 0])
    resample[e] = False


@wp.kernel
def wf_obs(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), cmd: wp.array2d(dtype=float),
           last_action: wp.array2d(dtype=float), imu_off: wp.array2d(dtype=float), gyro_bias: wp.array2d(dtype=float),
           fresh: wp.array(dtype=wp.bool), hist: wp.array2d(dtype=float), seed: int, step_idx: wp.array(dtype=int),
           nz: float, anchor: wp.array2d(dtype=float), priv_static: wp.array2d(dtype=float), delay: wp.array(dtype=int),
           offset: wp.array2d(dtype=float), obs: wp.array2d(dtype=float)):
    e = wp.tid()
    q = wp.vec4(qpos[e, 3], qpos[e, 4], qpos[e, 5], qpos[e, 6])
    R = quat_to_mat_wxyz(q)
    M = imu_rot(imu_off[e, 0], imu_off[e, 1])          # IMU frame = body frame rotated by the mount error
    Mt = wp.transpose(M)
    g_b = Mt * (wp.transpose(R) * wp.vec3(0.0, 0.0, -1.0))
    w_b = Mt * wp.vec3(qvel[e, 3], qvel[e, 4], qvel[e, 5])
    rng = wp.rand_init(seed, step_idx[0] * 7919 + e)
    o = wp.vector(length=28, dtype=float)
    for k in range(3):
        o[k] = (w_b[k] + gyro_bias[e, k]) * 0.25 + nz * wp.randf(rng, -0.075, 0.075)      # 0.2 * 1.5 * 0.25
        o[3 + k] = g_b[k] + nz * wp.randf(rng, -0.075, 0.075)                            # 0.05 * 1.5
    for k in range(6):
        j = k
        if k >= 3:
            j = k + 1                                                                 # skip wheel_L
        o[6 + k] = qpos[e, 7 + j] + nz * wp.randf(rng, -0.015, 0.015)                     # default pose = 0
    for k in range(8):
        o[12 + k] = qvel[e, 6 + k] * 0.05 + nz * wp.randf(rng, -0.1125, 0.1125)          # 1.5 * 1.5 * 0.05
        o[20 + k] = last_action[e, k]
    if fresh[e]:
        for h in range(10):
            for k in range(28):
                hist[e, h * 28 + k] = o[k]
        fresh[e] = False
    else:
        for h in range(9):
            for k in range(28):
                hist[e, h * 28 + k] = hist[e, (h + 1) * 28 + k]
        for k in range(28):
            hist[e, 252 + k] = o[k]
    for i in range(280):
        obs[e, i] = wp.clamp(hist[e, i], -100.0, 100.0)
    obs[e, 280] = cmd[e, 0] * 2.0
    obs[e, 281] = cmd[e, 1] * 2.0
    obs[e, 282] = cmd[e, 2] * 0.25
    # ---- privileged block (critic only): true state and this robot's physics
    Rb = wp.transpose(R)
    v_b = Rb * wp.vec3(qvel[e, 0], qvel[e, 1], qvel[e, 2])
    g_t = Rb * wp.vec3(0.0, 0.0, -1.0)
    p = 283
    for k in range(3):
        obs[e, p + k] = v_b[k] * 2.0
        obs[e, p + 3 + k] = qvel[e, 3 + k] * 0.25
        obs[e, p + 6 + k] = g_t[k]
    obs[e, p + 9] = (qpos[e, 2] - BASE_HEIGHT_TARGET_W) * 5.0
    for k in range(6):
        j = k
        if k >= 3:
            j = k + 1
        obs[e, p + 10 + k] = qpos[e, 7 + j]
    for k in range(8):
        obs[e, p + 16 + k] = qvel[e, 6 + k] * 0.05
    dx = qpos[e, 0] - anchor[e, 0]; dy = qpos[e, 1] - anchor[e, 1]
    yaw = wp.atan2(R[1, 0], R[0, 0])
    dyaw = yaw - anchor[e, 2]
    dyaw = dyaw - 6.2831853 * wp.floor((dyaw + 3.14159265) / 6.2831853)
    obs[e, p + 24] = wp.clamp(wp.cos(yaw) * dx + wp.sin(yaw) * dy, -1.0, 1.0)
    obs[e, p + 25] = wp.clamp(-wp.sin(yaw) * dx + wp.cos(yaw) * dy, -1.0, 1.0)
    obs[e, p + 26] = dyaw
    for k in range(13):
        obs[e, p + 27 + k] = priv_static[e, k]
    q2 = p + 40
    obs[e, q2] = float(delay[e]) * 0.25
    obs[e, q2 + 1] = imu_off[e, 0] * 50.0
    obs[e, q2 + 2] = imu_off[e, 1] * 50.0
    for k in range(3):
        obs[e, q2 + 3 + k] = gyro_bias[e, k] * 100.0
    for k in range(6):
        j = k
        if k >= 3:
            j = k + 1
        obs[e, q2 + 6 + k] = offset[e, j] * 20.0


@wp.kernel
def wf_record_action(action: wp.array2d(dtype=float), last: wp.array2d(dtype=float), prev: wp.array2d(dtype=float),
                     prev2: wp.array2d(dtype=float)):
    e = wp.tid()
    for i in range(8):
        prev2[e, i] = prev[e, i]
        prev[e, i] = last[e, i]
        last[e, i] = wp.clamp(action[e, i], -100.0, 100.0)


@wp.kernel
def wf_ctrl(last: wp.array2d(dtype=float), applied: wp.array2d(dtype=float), delay: wp.array(dtype=int), sub: int,
            offset: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
    """Physics substep `sub` of 4: the new action takes effect after the env's delay (0-4 substeps)."""
    e = wp.tid()
    for i in range(8):
        a = applied[e, i]
        if sub >= delay[e]:
            a = last[e, i]
        if i == 3 or i == 7:
            ctrl[e, i] = 0.5 * a
        else:
            ctrl[e, i] = offset[e, i] + 0.25 * a
        if sub == 3:
            applied[e, i] = last[e, i]


@wp.kernel
def wf_push(t: wp.array(dtype=int), push_phase: wp.array(dtype=int), qvel: wp.array2d(dtype=float), seed: int,
            step_idx: wp.array(dtype=int), push_vel: float):
    e = wp.tid()
    if push_vel > 0.0 and (t[e] + push_phase[e]) % 350 == 349:
        rng = wp.rand_init(seed + 17, step_idx[0] * 3571 + e)
        qvel[e, 0] = qvel[e, 0] + wp.randf(rng, -push_vel, push_vel)
        qvel[e, 1] = qvel[e, 1] + wp.randf(rng, -push_vel, push_vel)


@wp.kernel
def wf_reward_done(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), qacc: wp.array2d(dtype=float),
                   qfrc_actuator: wp.array2d(dtype=float), xpos: wp.array2d(dtype=wp.vec3), wheel_body: wp.vec2i,
                   cmd: wp.array2d(dtype=float), anchor: wp.array2d(dtype=float), last: wp.array2d(dtype=float),
                   prev: wp.array2d(dtype=float), prev2: wp.array2d(dtype=float), jnt_range: wp.array2d(dtype=float),
                   prev_track: wp.array2d(dtype=float),
                   dt: float, t: wp.array(dtype=int), max_t: int, step_idx: wp.array(dtype=int),
                   buf_rew: wp.array2d(dtype=float), buf_done: wp.array2d(dtype=float), reset_mask: wp.array(dtype=wp.bool),
                   resample: wp.array(dtype=wp.bool), fresh: wp.array(dtype=wp.bool),
                   ep_ret: wp.array(dtype=float), ep_len: wp.array(dtype=int), stats: wp.array(dtype=float),
                   stats_i: wp.array(dtype=int), terms: wp.array2d(dtype=float)):
    e = wp.tid()
    q = wp.vec4(qpos[e, 3], qpos[e, 4], qpos[e, 5], qpos[e, 6])
    R = quat_to_mat_wxyz(q)
    Rt = wp.transpose(R)
    base = wp.vec3(qpos[e, 0], qpos[e, 1], qpos[e, 2])
    v_b = Rt * wp.vec3(qvel[e, 0], qvel[e, 1], qvel[e, 2])
    w_b = wp.vec3(qvel[e, 3], qvel[e, 4], qvel[e, 5])
    g_b = Rt * wp.vec3(0.0, 0.0, -1.0)
    # tracking (body frame, LimX)
    lin_err = (cmd[e, 0] - v_b[0]) * (cmd[e, 0] - v_b[0]) + (cmd[e, 1] - v_b[1]) * (cmd[e, 1] - v_b[1])
    tr_lin = wp.exp(-lin_err / 0.2)
    ang_err = (cmd[e, 2] - w_b[2]) * (cmd[e, 2] - w_b[2])
    tr_ang = wp.exp(-ang_err / 0.25)
    # potential-based tracking terms (LimX *_pb): change of the tracking reward per second
    pb_lin = float(0.0); pb_ang = float(0.0)
    if t[e] > 0:
        pb_lin = (tr_lin - prev_track[e, 0]) / dt
        pb_ang = (tr_ang - prev_track[e, 1]) / dt
    prev_track[e, 0] = tr_lin; prev_track[e, 1] = tr_ang
    cmd_norm2 = cmd[e, 0] * cmd[e, 0] + cmd[e, 1] * cmd[e, 1] + cmd[e, 2] * cmd[e, 2]
    # feet in the base frame
    fL = Rt * (xpos[e, wheel_body[0]] - base)
    fR = Rt * (xpos[e, wheel_body[1]] - base)
    nom = -(BASE_HEIGHT_TARGET_W - TYRE_R_W)
    r_nom = 0.5 * (wp.exp(-(nom - fL[2]) * (nom - fL[2]) / 0.005) + wp.exp(-(nom - fR[2]) * (nom - fR[2]) / 0.005)) \
        * wp.exp(-cmd_norm2 / 0.5)
    sym = wp.abs(fL[1]) - wp.abs(fR[1])
    r_sym = wp.exp(-sym * sym / 0.001)
    same_x = wp.abs(fL[0] - fR[0])
    same_z = (fL[2] - fR[2]) * (fL[2] - fR[2])
    tau2 = float(0.0); acc2 = float(0.0); rate2 = float(0.0); smooth2 = float(0.0); lim = float(0.0); legv2 = float(0.0)
    for i in range(8):
        tau = qfrc_actuator[e, 6 + i]
        tau2 += tau * tau
        a = qacc[e, 6 + i]
        acc2 += a * a
        d1 = last[e, i] - prev[e, i]
        rate2 += d1 * d1
        d2 = last[e, i] - 2.0 * prev[e, i] + prev2[e, i]
        smooth2 += d2 * d2
        if i != 3 and i != 7:
            lo = jnt_range[i, 0]; hi = jnt_range[i, 1]
            mid = 0.5 * (lo + hi); half = 0.5 * (hi - lo) * 0.95
            qi = qpos[e, 7 + i]
            lim += wp.max(mid - half - qi, 0.0) + wp.max(qi - mid - half, 0.0)
            legv2 += qvel[e, 6 + i] * qvel[e, 6 + i]
    h_err = base[2] - BASE_HEIGHT_TARGET_W
    # standing still (command exactly zero): stay where the zero command began
    r_still = float(0.0)
    if cmd_norm2 == 0.0:
        dx = base[0] - anchor[e, 0]; dy = base[1] - anchor[e, 1]
        yaw = wp.atan2(R[1, 0], R[0, 0])
        dyaw = yaw - anchor[e, 2]
        dyaw = dyaw - 6.2831853 * wp.floor((dyaw + 3.14159265) / 6.2831853)
        r_still = -2.0 * (v_b[0] * v_b[0] + v_b[1] * v_b[1]) - 0.5 * w_b[2] * w_b[2] - 0.02 * legv2 \
            - 1.0 * wp.min(dx * dx + dy * dy, 1.0) - 0.5 * dyaw * dyaw
    r = (1.0 + 4.0 * tr_lin + 2.0 * tr_ang + 1.0 * pb_lin + 0.2 * pb_ang + 4.0 * r_nom + 0.5 * r_sym
         - 50.0 * same_x - 100.0 * same_z - 0.3 * v_b[2] * v_b[2] - 0.3 * (w_b[0] * w_b[0] + w_b[1] * w_b[1])
         - 1.6e-4 * tau2 - 1.5e-7 * acc2 - 0.03 * rate2 - 0.03 * smooth2 - 2.0 * lim
         - 12.0 * (g_b[0] * g_b[0] + g_b[1] * g_b[1]) - 20.0 * h_err * h_err + r_still) * dt
    # terminations
    t[e] = t[e] + 1
    fell = g_b[2] > -0.5 or base[2] < 0.3
    blown = int(0)
    for i in range(15):
        if not wp.isfinite(qpos[e, i]):
            blown = 1
    for i in range(14):
        if not wp.isfinite(qvel[e, i]):
            blown = 1
    if blown == 1:
        r = 0.0
        fell = True
    trunc = t[e] >= max_t
    done = fell or trunc
    if fell:
        r = r - 5.0
    s = step_idx[0] - 1
    buf_rew[s, e] = r
    buf_done[s, e] = 1.0 if done else 0.0
    terms[e, 0] = tr_lin; terms[e, 1] = tr_ang; terms[e, 2] = r_nom; terms[e, 3] = r_still
    terms[e, 4] = wp.sqrt(v_b[0] * v_b[0] + v_b[1] * v_b[1]) if cmd_norm2 == 0.0 else -1.0
    terms[e, 5] = wp.sqrt(legv2); terms[e, 6] = h_err; terms[e, 7] = wp.sqrt(rate2)
    ep_ret[e] = ep_ret[e] + r
    ep_len[e] = ep_len[e] + 1
    reset_mask[e] = done
    resample[e] = resample[e] or (t[e] % 250 == 0)
    if done:
        wp.atomic_add(stats, 0, ep_ret[e])
        wp.atomic_add(stats_i, 0, ep_len[e])
        wp.atomic_add(stats_i, 1, 1)
        if fell:
            wp.atomic_add(stats_i, 2, 1)
        if blown == 1:
            wp.atomic_add(stats_i, 3, 1)
        ep_ret[e] = 0.0; ep_len[e] = 0; t[e] = 0
        resample[e] = True
        fresh[e] = True


@wp.kernel
def wf_reset(reset_mask: wp.array(dtype=wp.bool), origins: wp.array2d(dtype=float), init_z: float, seed: int,
             step_idx: wp.array(dtype=int), qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
             last: wp.array2d(dtype=float), prev: wp.array2d(dtype=float), prev2: wp.array2d(dtype=float),
             applied: wp.array2d(dtype=float), delay: wp.array(dtype=int), offset: wp.array2d(dtype=float),
             imu_off: wp.array2d(dtype=float), gyro_bias: wp.array2d(dtype=float), push_phase: wp.array(dtype=int),
             dr: float, delay_max: float):
    e = wp.tid()
    if not reset_mask[e]:
        return
    rng = wp.rand_init(seed + 11, step_idx[0] * 6151 + e)
    for i in range(qvel.shape[1]):
        qvel[e, i] = 0.0
    qpos[e, 0] = origins[e, 0] + wp.randf(rng, -0.5, 0.5)
    qpos[e, 1] = origins[e, 1] + wp.randf(rng, -0.5, 0.5)
    qpos[e, 2] = init_z
    yaw = wp.randf(rng, -3.14159265, 3.14159265)
    qpos[e, 3] = wp.cos(0.5 * yaw); qpos[e, 4] = 0.0; qpos[e, 5] = 0.0; qpos[e, 6] = wp.sin(0.5 * yaw)
    for i in range(8):
        qpos[e, 7 + i] = 0.0
        last[e, i] = 0.0; prev[e, i] = 0.0; prev2[e, i] = 0.0; applied[e, i] = 0.0
        offset[e, i] = 0.0
        if i != 3 and i != 7:
            offset[e, i] = dr * wp.randf(rng, -0.05, 0.05)
    delay[e] = int(wp.randf(rng, 0.0, delay_max + 0.999))
    imu_off[e, 0] = dr * wp.randf(rng, -0.0209, 0.0209)       # +-1.2 deg
    imu_off[e, 1] = dr * wp.randf(rng, -0.0209, 0.0209)
    for k in range(3):
        gyro_bias[e, k] = dr * wp.randf(rng, -0.009, 0.009)   # +-0.5 deg/s
    push_phase[e] = int(wp.randf(rng, 0.0, 349.0))


# --------------------------------------------------------------------------------------------------

class Tron1WFTask:
    """Shape expected by `metalsim.learn.ppo_warp.PPOWarp` (with `launch_physics`)."""

    def __init__(self, n, seed: int = 0, device="metal:0", randomize=True, noise=1.0, delay_max=4, push_vel=1.0):
        self.n, self.seed, self.device = n, seed, device
        # curriculum knobs (kernel arguments, so a captured graph must be re-captured after changing them)
        self.noise, self.delay_max, self.push_vel, self.dr = float(noise), float(delay_max), float(push_vel), 1.0 if randomize else 0.0
        self.model = m = build_train_model()
        fields = ("body_mass", "body_ipos", "body_inertia", "geom_friction", "actuator_gainprm", "actuator_biasprm",
                  "dof_frictionloss") if randomize else ()
        self.sim = BatchSim(m, n, options=BatchSimOptions(substeps=1, njmax=64, nconmax=16, solver_iterations=10,
                                                          ls_iterations=20, per_world_fields=fields))
        self.max_t = int(EPISODE_S / CONTROL_DT)
        self.obs_dim, self.actor_dim, self.act_dim = CRITIC_IN, ACTOR_IN, 8
        self.ctrl_lo = [-100.0] * 8; self.ctrl_hi = [100.0] * 8
        self.needs_step_idx = True
        z = lambda *s, dtype=float: wp.zeros(*s, dtype=dtype, device=device)
        self.action_scratch = z((n, 8))
        self.obs = z((n, CRITIC_IN)); self.priv_static = z((n, N_STATIC)); self.hist = z((n, 280)); self.cmd = z((n, 3)); self.anchor = z((n, 3))
        self.resample = wp.array(np.ones(n, bool), dtype=wp.bool, device=device)
        self.fresh = wp.array(np.ones(n, bool), dtype=wp.bool, device=device)
        self.last = z((n, 8)); self.prev = z((n, 8)); self.prev2 = z((n, 8)); self.applied = z((n, 8))
        self.delay = z(n, dtype=int); self.offset = z((n, 8)); self.imu_off = z((n, 2)); self.gyro_bias = z((n, 3))
        self.push_phase = z(n, dtype=int); self.prev_track = z((n, 2))
        self.t = z(n, dtype=int); self.ep_ret = z(n); self.ep_len = z(n, dtype=int)
        self.stats = z(4); self.stats_i = z(4, dtype=int); self.terms = z((n, 8))
        jr = np.array([m.jnt_range[m.actuator_trnid[i][0]] for i in range(m.nu)], np.float32)
        self.jnt_range = wp.array(jr, dtype=float, device=device)
        self.wheel_body = wp.vec2i(m.body("wheel_L_Link").id, m.body("wheel_R_Link").id)
        side = int(math.ceil(math.sqrt(n)))
        o = np.array([[(i % side) * 3.0, (i // side) * 3.0, 0.0] for i in range(n)], np.float32)
        o[:, :2] -= o[:, :2].mean(0)
        self.origins = wp.array(o, dtype=float, device=device)
        self.init_z = 0.93                       # legs straight at q = 0: base 0.91 m above the floor
        if randomize:
            self._randomize_physics(np.random.default_rng(seed))
        self.sim.synchronize()

    def _randomize_physics(self, rng):
        m, n, tm = self.model, self.n, self.sim.tm
        base, pack = m.body("base_Link").id, m.body("sensor_pack").id
        mass = np.tile(m.body_mass, (n, 1)); ipos = np.tile(m.body_ipos, (n, 1, 1)); inertia = np.tile(m.body_inertia, (n, 1, 1))
        mass[:, 1:] *= rng.uniform(0.9, 1.1, (n, 1))
        mass[:, base] += rng.uniform(-0.5, 2.0, n)
        mass[:, pack] = rng.uniform(0.5, 3.0, n)
        ipos[:, base] += rng.uniform(-1, 1, (n, 3)) * np.array([0.03, 0.02, 0.03])
        ipos[:, pack] = np.stack([rng.uniform(-0.05, 0.05, n), rng.uniform(-0.03, 0.03, n), rng.uniform(0.03, 0.12, n)], 1)
        inertia *= rng.uniform(0.8, 1.2, (n, 1, 1))
        fric = np.tile(m.geom_friction, (n, 1, 1)); fric[:, :, 0] = rng.uniform(0.2, 1.6, (n, 1))
        gain = np.tile(m.actuator_gainprm, (n, 1, 1)); bias = np.tile(m.actuator_biasprm, (n, 1, 1))
        kp = rng.uniform(0.8, 1.2, (n, 8)); kd = rng.uniform(0.8, 1.2, (n, 8)); st = rng.uniform(0.8, 1.2, (n, 8))
        for i in range(8):
            if i in WHEELS:
                gain[:, i, 0] = KD_WHEEL * kd[:, i] * st[:, i]; bias[:, i, 2] = -KD_WHEEL * kd[:, i] * st[:, i]
            else:
                gain[:, i, 0] = KP * kp[:, i] * st[:, i]; bias[:, i, 1] = -KP * kp[:, i] * st[:, i]
                bias[:, i, 2] = -KD_LEG * kd[:, i] * st[:, i]
        fl = np.tile(m.dof_frictionloss, (n, 1)); fl[:, 6:] *= rng.uniform(0.3, 2.5, (n, 8))
        # what the critic is told about this robot (roughly unit scale)
        static = np.stack([fric[:, 0, 0] - 0.9, mass[:, pack] - 1.5, *(ipos[:, pack] - [0.0, 0.0, 0.075]).T * 20,
                           mass[:, base] - m.body_mass[base], *(ipos[:, base] - m.body_ipos[base]).T * 30,
                           kp.mean(1) - 1, kd.mean(1) - 1, st.mean(1) - 1, fl[:, 6:].mean(1) / m.dof_frictionloss[6:].mean() - 1], 1)
        assert static.shape[1] == N_STATIC
        self.priv_static.assign(static.astype(np.float32))
        put = lambda name, a: getattr(tm, name).copy_(torch.as_tensor(a.astype(np.float32), device="mps"))
        put("body_mass", mass); put("body_ipos", ipos); put("body_inertia", inertia); put("geom_friction", fric)
        put("actuator_gainprm", gain); put("actuator_biasprm", bias); put("dof_frictionloss", fl)
        torch.mps.synchronize()
        v = self.sim.recompute_constants()
        self.sim.synchronize()

    # -- graph pieces --------------------------------------------------------------------------------
    def launch_obs(self, step_idx=None):
        wp.launch(wf_commands, dim=self.n, inputs=[self.cmd, self.resample, self.sim.d.qpos, self.anchor, self.seed, step_idx],
                  device=self.device)
        wp.launch(wf_obs, dim=self.n, inputs=[self.sim.d.qpos, self.sim.d.qvel, self.cmd, self.last, self.imu_off, self.gyro_bias,
                                              self.fresh, self.hist, self.seed, step_idx, self.noise, self.anchor,
                                              self.priv_static, self.delay, self.offset, self.obs], device=self.device)

    def launch_apply_action(self, action):
        wp.launch(wf_record_action, dim=self.n, inputs=[action, self.last, self.prev, self.prev2], device=self.device)

    def launch_physics(self, step_idx):
        wp.launch(wf_push, dim=self.n, inputs=[self.t, self.push_phase, self.sim.d.qvel, self.seed, step_idx, self.push_vel],
                  device=self.device)
        for sub in range(DECIM):
            wp.launch(wf_ctrl, dim=self.n, inputs=[self.last, self.applied, self.delay, sub, self.offset, self.sim.d.ctrl],
                      device=self.device)
            self.sim.launch_step()

    def launch_reward_done_reset(self, pol, bufs):
        d = self.sim.d
        wp.launch(wf_reward_done, dim=self.n, inputs=[
            d.qpos, d.qvel, d.qacc, d.qfrc_actuator, d.xpos, self.wheel_body, self.cmd, self.anchor, self.last, self.prev,
            self.prev2, self.jnt_range, self.prev_track, CONTROL_DT, self.t, self.max_t, pol.step_idx, bufs.rew, bufs.done,
            self.sim._reset_mask, self.resample, self.fresh, self.ep_ret, self.ep_len, self.stats, self.stats_i, self.terms],
            device=self.device)
        self._reset(pol.step_idx)

    def _reset(self, step_idx):
        import mujoco_warp as mjw
        d = self.sim.d
        mjw.reset_data(self.sim.m, d, reset=self.sim._reset_mask)
        wp.launch(wf_reset, dim=self.n, inputs=[self.sim._reset_mask, self.origins, self.init_z, self.seed, step_idx, d.qpos, d.qvel,
                                                self.last, self.prev, self.prev2, self.applied, self.delay, self.offset,
                                                self.imu_off, self.gyro_bias, self.push_phase, self.dr, self.delay_max],
                  device=self.device)
        mjw.forward(self.sim.m, d)

    def reset_all(self):
        self.sim._reset_mask.fill_(True)
        idx = wp.zeros(1, dtype=int, device=self.device)
        with wp.ScopedDevice(self.device):
            self._reset(idx)
        self.sim.synchronize()

    def episode_stats(self):
        s = self.stats.numpy(); si = self.stats_i.numpy()
        c = int(si[1])
        self.last_termination_fraction = si[2] / c if c else 0.0
        self.blown_up_episodes = int(si[3])
        self.stats.zero_(); self.stats_i.zero_()
        return (s[0] / c if c else 0.0, si[0] / c if c else 0.0, c)

    def term_means(self):
        t = self.terms.numpy()
        still = t[:, 4][t[:, 4] >= 0]
        return dict(track_lin=float(t[:, 0].mean()), track_ang=float(t[:, 1].mean()), nom_foot=float(t[:, 2].mean()),
                    still_rew=float(t[:, 3].mean()), still_speed=float(still.mean()) if len(still) else float("nan"),
                    leg_speed=float(t[:, 5].mean()), height_err=float(np.abs(t[:, 6]).mean()), act_rate=float(t[:, 7].mean()))


# --------------------------------------------------------------------------------------------------

def ppo_config(iterations, seed=0):
    from metalsim.learn.ppo_warp import PPOWarpConfig
    return PPOWarpConfig(iterations=iterations, rollout=24, epochs=5, minibatches=4, lr=1e-3, gamma=0.99, lam=0.95, clip=0.2,
                         ent_coef=0.01, vf_coef=1.0, max_grad_norm=1.0, desired_kl=0.01, seed=seed, hidden=(512, 256, 128),
                         log_every=10)


def export_onnx(net, path):
    """Actor mean only: input 283 (history 10 x 28, scaled command 3) -> 8 actions. The privileged
    critic columns are not part of the deployed network."""
    import copy
    actor = copy.deepcopy(net.actor).to("cpu").eval()
    x = torch.zeros(1, ACTOR_IN)
    torch.onnx.export(actor, x, path, input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False)


def train(n=8192, iterations=3000, seed=0, run=None, save_every=250, resume=None, easy=False):
    from metalsim.learn.ppo_warp import PPOWarp
    run = run or time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "runs", "tron1_wf", run)
    os.makedirs(out, exist_ok=True)
    task = Tron1WFTask(n, seed=seed, **(dict(randomize=False, noise=0.0, delay_max=0, push_vel=0.0) if easy else {}))
    task.reset_all()
    algo = PPOWarp(task, ppo_config(iterations, seed))
    if resume:
        ck = torch.load(resume, map_location="mps")
        algo.net.load_state_dict(ck["net"])
    logf = open(os.path.join(out, "train.log"), "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log(f"TRON1 WF PPO: N={n} obs {task.obs_dim} act {task.act_dim} rollout 24 x {iterations}; out {out}")
    t0 = time.perf_counter(); steps = 0
    for it in range(1, iterations + 1):
        algo.rollout()
        steps += 24 * n
        st = algo.update()
        if it % 10 == 0 or it == iterations:
            torch.mps.synchronize()
            ret, length, count = task.episode_stats()
            tm = task.term_means()
            sps = steps / (time.perf_counter() - t0)
            log(f"it {it:5d} steps {steps:11d} sps {sps:8,.0f} | ep_ret {ret:7.2f} ep_len {length:6.1f} (n={count}, fell "
                f"{task.last_termination_fraction:.2f}, blown {task.blown_up_episodes}) | lin {tm['track_lin']:.2f} ang "
                f"{tm['track_ang']:.2f} nom {tm['nom_foot']:.2f} still_v {tm['still_speed']:.3f} legv {tm['leg_speed']:.2f} "
                f"h {tm['height_err']:.3f} rate {tm['act_rate']:.2f} | kl {st['kl']:.4f} lr {algo.lr:.1e} std "
                f"{algo.net.log_std.exp().mean().item():.2f}")
        if it % save_every == 0 or it == iterations:
            torch.save({"net": algo.net.state_dict(), "it": it}, os.path.join(out, f"model_{it}.pt"))
            export_onnx(algo.net, os.path.join(out, "policy.onnx"))
    json.dump({"n": n, "iterations": iterations, "seed": seed}, open(os.path.join(out, "config.json"), "w"))
    return algo, out


if __name__ == "__main__":
    import argparse
    wp.config.quiet = True
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "bench"])
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--easy", action="store_true", help="no randomization, noise, delay or pushes (diagnostic)")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.n, a.iterations, a.seed, a.run, resume=a.resume, easy=a.easy)
    else:
        train(a.n, 20, a.seed, "bench")
