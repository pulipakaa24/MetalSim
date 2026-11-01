"""so101Sim's SO-101 cube-lift teacher in the filming tool: so101Sim's ``SO101LiftEnv`` (observations, actions, reset
distribution, success rule and 20 ms control loop; github.com/pulipakaa24/so101Sim, ``rl/native/so101_lift.py``) run
on MetalSim's MuJoCo Warp (``BatchSim``), driven by its PPO teacher extracted into plain torch weights.

so101Sim's loop, per 20 ms control step (``SO101LiftEnv.step``):

    a = clip(action, -1, 1) ; d.ctrl = lo + (a + 1) / 2 (hi - lo) ; 4 x mj_step (5 ms) ; obs, height, success

``SO101Lift`` runs it on ``BatchSim`` (n worlds of the scene, 4 substeps per step). The observation (21: arm joint
positions and velocities, box position, box minus gripper-site position, box linear velocity ``cvel[box][3:6]``) is
read after the step from the same fields so101Sim reads (``qpos`` / ``qvel`` after the integration; ``xpos``,
``site_xpos`` and ``cvel`` from the last substep's kinematics, before its integration, as in MuJoCo C's ``mj_step``),
in float64 on the host. The reset is so101Sim's ``reset(seed=...)``: gymnasium's ``np_random`` (``PCG64`` from the
seed, the same draws as ``np.random.default_rng(seed)``), drawn in so101Sim's order (box radius 0.15-0.26 m, bearing
+-0.7 rad, yaw, arm jitter +-0.08 rad on the five arm joints, box mass x 0.7-1.4, box friction x 0.7-1.3). Mass and
friction are per-world model fields here; like so101Sim, which writes ``m.body_mass`` without ``mj_setConst``, the
constants derived from the mass are not recomputed. Success as so101Sim's: the box centre above 0.10 m for 10
consecutive control steps; it does not end the episode. An episode ends at 300 steps (6 s) or when the box leaves the
0.45 m workspace radius (``terminated``).

The policy file (``extract_policy`` in ``runs/film/so101_lift/port/``): the SB3 ``MlpPolicy`` actor (21 -> 256 -> 256
-> 6, tanh) and the VecNormalize observation statistics, as plain tensors (``torch.load(weights_only=True)``); the
action is the actor's mean, clipped to [-1, 1], after so101Sim's eval normalisation
``clip((obs - mean) / sqrt(var + 1e-8), -10, 10)`` (``rl/native/eval_lift_native.py``).

Servo model (``adapter_kw["servo"]``, ``SERVOS``): "12v" (the default) runs the scene as it is, whose servos are the
12 V STS3215; "7v4" changes what the supply voltage changes for the standard follower's 7.4 V C001 (``apply_servo``).
"7v4_real" / "12v_real" model the servo under LeRobot's registers; "7v4_bam" is Rhoban BAM's identified 7.4 V servo with
its load-dependent gearbox friction, which reproduces how real followers hold against gravity (updated every physics
step: ``friction_hook`` on CPU MuJoCo, ``metalsim.physics.actuation.BamFrictionWarp`` in ``SO101Lift``'s step graph).

Controller (``adapter_kw["controller"]``, ``CONTROLLERS``): "policy" (the default) is the loop above; "deploy_expert"
runs ``so101_expert.DeployExpert`` (so101Sim's scripted IK plan streamed as absolute joint goals at LeRobot's 30 Hz) on
a "_real" servo: one physics step per BatchSim step, the experts act at their 30 Hz ticks on CPU kinematics of each
world's state, and the servo layer turns the held command into every physics step's target (``SO101Lift``).

    .venv/bin/python -m metalsim.render.film so101_lift --stage record              # PPO teacher, the scene's 12 V servos
    .venv/bin/python -m metalsim.render.film so101_lift_74v --stage record          # the same, 7.4 V servos
    .venv/bin/python -m metalsim.render.film so101_lift_smooth --stage record       # 7.4 V, the smooth retrain
    .venv/bin/python -m metalsim.render.film so101_lift_deploy --stage record       # DeployExpert on 7v4_bam
(the entries are registered in ``tasks.TASKS``; ``python -m metalsim.render.film.so101 <task> ...`` also works)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np

from metalsim.render.film.cfg import AssetRef, FilmCfg, FilmTask, LookCfg, ShotCfg, ViewerCfg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# so101Sim (the owner's project), at the commit whose env code the port follows; the teacher is so101Sim's run 1
# checkpoint (tracked there, rl/runs/lift_ppo_run1_backup) continued by its rl/native/continue_lift.py (run 2)
SO101SIM = dict(repo="https://github.com/pulipakaa24/so101Sim", commit="d4c2990",
                env="rl/native/so101_lift.py", train="rl/native/train_lift.py", continue_="rl/native/continue_lift.py",
                eval="rl/native/eval_lift_native.py", run1="rl/runs/lift_ppo_run1_backup")
SCENE = "assets/so101/scene_box_rl.xml"
POLICY = "runs/film/so101_lift/policy/so101_lift_ppo.pt"

LIFT_Z = 0.10           # m: success height of the box centre (spawn z 0.03)
SUCCESS_HOLD = 10       # control steps above LIFT_Z
CTRL_DT = 0.02
MAX_STEPS = 300
OOB_R = 0.45            # m: box beyond this radius ends the episode
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CONTROLLERS = ("policy", "deploy_expert")   # adapter_kw["controller"]: the PPO policy file, or so101_expert.DeployExpert

# -- servo models ------------------------------------------------------------------------------------------------
# The scene's servos (assets/so101/so101.xml, default class "sts3215", Menagerie = SO-ARM100's so101_new_calib.xml at
# aec17bbc): MuJoCo position actuators, kp 998.22, kv 2.731, forcerange +-2.94 N.m, from Gregory119's DC-motor
# derivation for the 12 V ST-3215-C047 (30 kg.cm stall, 45 rpm no load, 12 V; firmware P gain 16; SO-ARM100 PR #141,
# github.com/Gregory119/RBE501-RL-arm-project gymnasium_env/calc_actuator_model.py); joint damping 0.60, frictionloss
# 0.052, armature 0.028, in SO-ARM100's SO-101 MJCF since its first version (they match Rhoban BAM's MuJoCo export of its
# STS3215 7.4 V identification, damping = viscous friction + kt^2 / R, frictionloss = its Coulomb friction; bam/to_mujoco.py,
# bam/params/feetech_sts3215_7_4V). MuJoCo clamps the servo's whole force to the forcerange, so the actuator has no
# speed-dependent limit of its own; flat out, the servo force minus the joint damping is the line 2.94 - 0.60 w N.m,
# zero at 4.9 rad/s (the 12 V servo: 2.94 N.m stall, 4.72 rad/s no load).
KGCM = 0.0980665        # N.m per kg.cm
RPM = 2 * np.pi / 60    # rad/s per rpm
SERVO_BASE = dict(forcerange=2.94, damping=0.60)        # the scene's values that "7v4" replaces (checked)
SERVOS = {
    "12v": dict(label="STS3215 12 V (the scene as it is: Menagerie / SO-ARM100 PR #141)"),
    # the SO-101 follower's servo: STS3215 C001, 7.4 V, 1/345 (SO-ARM100 README, bill of materials). Feetech datasheet
    # "STS3215 200620a": stall torque 19.5 kg.cm @ 7.4 V, 52 rpm @ 7.4 V (no load); torque output is the motor's
    # voltage duty cycle, 1000 = 100 %. LeRobot's SO-101 driver (lerobot/robots/so_follower/so_follower.py, configure)
    # sets the gripper's Max_Torque_Limit to 500 (50 %).
    "7v4": dict(label="STS3215 C001 7.4 V (SO-101 follower), gripper at LeRobot's 50 % torque limit",
                stall_Nm=19.5 * KGCM, noload_rad_s=52 * RPM, gripper_torque_limit=0.5),
}


# -- the real servo under LeRobot's settings: "7v4_real", "12v_real" ----------------------------------------------------
# What the servo does (sources; "(inf.)" marks inference):
# * firmware position loop: duty = P x error x 0.166 x 1.00494 per rad, clipped at +-0.97 (Rhoban BAM, identified on
#   STS3215 7.4 V benches with an oscilloscope and a pendulum fit, P in {4, 8, 16, 32}, D = 0: bam/feetech/actuator.py
#   error_gain / max_pwm, bam/params/feetech_sts3215_7_4V/m1.json error_gain_ratio; bam/feetech/record.py writes P and
#   D = 0). A third-party STS memory table gives the same gain, "Kp scaled by 1/4" (0.25 permille duty per step per
#   unit = 0.163 per rad). LeRobot writes P = 16, I = 0, D = 32 (lerobot so_follower config_so_follower.py
#   position_*_coefficient; so_follower.py configure).
# * the loop tracks an internal target that moves toward the goal at most 5.29 rad/s (BAM max_velocity, m1), and
#   with LeRobot's Acceleration = 254 (feetech.py configure_motors) at most 254 x 100 step/s^2 = 39.0 rad/s^2 (unit:
#   third-party memory tables, 8.7 deg/s^2 = 100 step/s^2; that it applies in position mode with Goal_Velocity 0:
#   datasheet "Mode 0 ... support acceleration movement", and LeRobot's first SO-100 calibration lowered Acceleration
#   to 10 to slow its position moves; (inf.) the profile's shape).
# * DC motor: torque-speed line through the datasheet's stall torque and no-load speed (the brief's limits; 7.4 V C001
#   19.5 kg.cm, 52 rpm; 12 V C018 30 kg.cm, 0.222 s / 60 deg); back-EMF + viscous friction as joint damping, outside
#   the duty clip, as back-EMF acts. Coulomb friction and rotor armature: BAM m1 (frictionloss 0.0515, armature
#   0.0261; (inf.) the same gearbox at 12 V). BAM's own kt / R put the 7.4 V stall at 3.41 N.m, 1.8x the datasheet;
#   a replay fit on real SO-101 teleop data (Fontaine blog 2026-08-11, "fit from upstream") lands at kp 8.8, ±1.97 N.m.
# * latency: (a) servo side 1.05 ms (BAM m1 command_delay: bus write + firmware loop + current build-up; the motor's
#   L/R and the 16 / 24 kHz PWM are far below the 5 ms physics step); (c) host side: one sync_read of 6 servos and one
#   sync_write at 1 Mbps through a USB serial adapter, about 3 ms (inf.: protocol arithmetic, no published
#   measurement found).
# * not modelled: D = 32 (its scale needs the firmware's loop period, not published; bounded by a sensitivity run),
#   overload protection (LeRobot's gripper Overload_Torque 25 / Protection_Current 250: semantics disagree between
#   sources), the dead zone (1 step) and encoder quantization (1.5 mrad), backlash.
STS_STEP = 2 * np.pi / 4096                 # rad per encoder step
BAM_M1 = dict(error_gain=0.166, error_gain_ratio=1.0049367060648848, max_pwm=0.97, armature=0.02608291141307205,
              friction_base=0.05152047758105413, friction_viscous=0.05989161871380604, max_velocity=5.2883481319359,
              command_delay=0.0010469000118575051, kt=1.1776311631627974, R=2.478681934101461)
LEROBOT = dict(P_Coefficient=16, I_Coefficient=0, D_Coefficient=32, Acceleration=254, gripper_Max_Torque_Limit=500)
HOST_LATENCY = 0.003                        # s, (inf.) see above
_REAL_BASE = dict(gripper_torque_limit=LEROBOT["gripper_Max_Torque_Limit"] / 1000, P=LEROBOT["P_Coefficient"],
                  max_duty=BAM_M1["max_pwm"], coulomb=BAM_M1["friction_base"], armature=BAM_M1["armature"],
                  speed_limit=BAM_M1["max_velocity"], accel_limit=LEROBOT["Acceleration"] * 100 * STS_STEP,
                  servo_delay=BAM_M1["command_delay"], host_delay=HOST_LATENCY, d_cycle_s=0.0, line=True)
SERVOS["7v4_real"] = dict(label="STS3215 C001 7.4 V under LeRobot's registers (P 16, Acceleration 254, gripper 50 %), "
                                "BAM-identified firmware loop, datasheet torque-speed line, servo + host latency",
                          stall_Nm=19.5 * KGCM, noload_rad_s=52 * RPM, **_REAL_BASE)
SERVOS["12v_real"] = dict(label="STS3215 C018 12 V under LeRobot's registers, as 7v4_real",
                          stall_Nm=30.0 * KGCM, noload_rad_s=(np.pi / 3) / 0.222, **_REAL_BASE)

# -- "7v4_bam": the 7.4 V servo as Rhoban BAM identified it, load-dependent gearbox friction included -------------------
# Why (so101-sag, 2026-10-31; runs/readme_media/handoff/so101-sag.md): real SO-101 followers (two LeRobot so_follower
# teleop datasets; e = action - state is the follower's own Goal_Position - Present_Position) hold a gravity load with a
# friction band whose width grows with the load: after moving with gravity a joint stops 30-70 mrad short of where it
# stops after moving against it, at shoulder_lift, elbow_flex and wrist_flex alike. 7v4_real (constant Coulomb 0.0515,
# kp 5.26) has a +-9.8 mrad band centred on g / kp, so it sagged 3-6x more than the real shoulder (which mostly stops
# after lowering) and matched the elbow (which mostly stops after rising) by coincidence. Replaying the datasets'
# commands on the sim follower, median |sim - real| sag per load bin: shoulder_lift 93 -> 17 mrad, elbow_flex 9 -> 12
# (after moving with gravity 62 -> 14), wrist_flex 1.9 -> 1.1.
# What (every number from bam/params/feetech_sts3215_7_4V/m6.json at github.com/Rhoban/bam b9e1111, BAM's best-ranked
# friction model; m3 / m4 give the same sag within 3 mrad):
# * motor: full-duty torque kt x 7.4 V / R = 3.43 N.m; kp = 3.43 x P 16 x 0.166 x error_gain_ratio 1.162 = 10.58
#   N.m/rad; duty clip 0.97 (stall 3.32 N.m, 1.74x the datasheet's 19.5 kg.cm: open, see the hand-off); back-EMF
#   kt^2 / R 0.591 + viscous 0.028 as joint damping (no-load 5.29 rad/s, the datasheet's 52 rpm within 3 %);
#   armature 0.0216; firmware speed cap 5.10 rad/s; command delay 4.98 ms (+ the 3 ms host latency, as 7v4_real).
# * friction: BAM's M6 budget (Coulomb 0.053 + Stribeck + |0.244 tau_e - 0.045 tau_m| + |0.142 tau_e - 0.0001 tau_m| at
#   rest + a quadratic term), written to dof_frictionloss before every physics step from the previous step's motor and
#   external torques (metalsim.physics.actuation.BamFriction, BAM's bam/mujoco.py loop), with BAM's stiff friction
#   constraint (bam/mjlab.py). On MuJoCo Warp the same update is a Warp kernel after every physics substep inside the
#   step graph (actuation.BamFrictionWarp, per-world dof_frictionloss; SO101Lift wires it; so101-bam-gpu checked it
#   against the CPU update).
# * LeRobot's registers as 7v4_real (P 16, Acceleration 254, gripper Max_Torque_Limit 500 = 50 % duty: 1.71 N.m here).
BAM_M6 = dict(model="m6", kt=1.2753202369175398, R=2.753421214375944, error_gain_ratio=1.1616979929275297,
              armature=0.021564290129722483, max_velocity=5.095754044445978, command_delay=0.0049807526194816515,
              friction_base=0.0533134815909817, friction_stribeck=4.257322248151969e-05,
              load_friction_motor=0.044912012282132555, load_friction_external=0.24355483356616028,
              load_friction_motor_stribeck=0.00012448975632695385,
              load_friction_external_stribeck=0.1422905635963302, load_friction_motor_quad=0.007594943296670792,
              load_friction_external_quad=0.00340408727365082, dtheta_stribeck=0.371136775424206,
              alpha=9.93094944677977, friction_viscous=0.028160537042911545)
BAM_VIN = 7.4                                # V, bam/feetech/actuator.py STS3215Actuator (vin)
_M6_STALL = BAM_M6["kt"] * BAM_VIN / BAM_M6["R"] * BAM_M1["max_pwm"]
_M6_DAMPING = BAM_M6["kt"] ** 2 / BAM_M6["R"] + BAM_M6["friction_viscous"]
SERVOS["7v4_bam"] = dict(label="STS3215 7.4 V as Rhoban BAM's m6 identification (motor, firmware gain, load-dependent "
                               "Stribeck gearbox friction) under LeRobot's registers, servo + host latency",
                         stall_Nm=_M6_STALL, noload_rad_s=(_M6_STALL - BAM_M6["friction_base"]) / _M6_DAMPING,
                         **dict(_REAL_BASE, coulomb=BAM_M6["friction_base"], armature=BAM_M6["armature"],
                                speed_limit=BAM_M6["max_velocity"], servo_delay=BAM_M6["command_delay"],
                                error_gain_ratio=BAM_M6["error_gain_ratio"], friction=BAM_M6))


def apply_servo(m, servo: str = "12v", **over) -> dict:
    """Set the six arm servos of the MjModel ``m`` (in place) to ``SERVOS[servo]``; return what was applied.

    "12v" changes nothing. "7v4" changes what the supply voltage changes: the force limit becomes the C001's 7.4 V stall
    torque (1.912 N.m), and the joint damping becomes stall / no-load speed (1.912 / 5.445 = 0.351 N.m.s/rad), so the
    flat-out line (servo force minus damping) reaches zero at the datasheet's 52 rpm, as the 12 V model's reaches zero
    near its no-load speed. The gripper's force limit is 50 % of that (LeRobot's Max_Torque_Limit 500: half the duty
    cycle, i.e. half the drive voltage), with the same damping, so its line reaches zero at half the no-load speed, as a
    DC motor at half the voltage does. kp, kv, armature and frictionloss are kept: kp and kv come from the motor's
    current constants (stall and no-load current), which the C001 datasheet does not give, and act only within
    forcerange / kp (about 2 mrad) of the target; armature and friction are mechanical. No overload or over-current
    protection and no speed-dependent force limit are added (the 12 V model has neither).

    "7v4_real" / "12v_real" (the block above): kp, kv, forcerange, damping, frictionloss and armature from
    ``metalsim.physics.actuation.voltage_servo``, and a ``setpoint`` entry (command delay, firmware speed and
    acceleration limits) for ``servo_layer``, which the caller must run between its commands and ``d.ctrl``. ``over``
    replaces any of the servo's entries (ablations): e.g. ``accel_limit=np.inf``, ``host_delay=0``, ``servo_delay=0``,
    ``speed_limit=np.inf``, ``line=False`` (no back-EMF: joint damping = viscous friction only), ``d_cycle_s`` (model
    LeRobot's D = 32 assuming that firmware loop period; 0 = no D term), ``P``.

    "7v4_bam" (the block above) is built the same way from BAM's m6 file (its own ``error_gain_ratio``) and adds a
    ``friction`` entry: the arm dofs' friction constraint is stiffened (BAM's solref / solimp) and their
    ``dof_frictionloss`` set to the budget at rest; the caller must run ``friction_hook(applied, m).update(d)`` before
    every physics step (CPU MuJoCo), which rewrites it from the load; on MuJoCo Warp, ``friction_hook(applied, m, sim)``
    on a BatchSim built after this call with ``"dof_frictionloss"`` in ``per_world_fields`` (``SO101Lift`` does)."""
    import mujoco

    s = dict(SERVOS[servo], **over)
    oid = lambda t, n: mujoco.mj_name2id(m, t, n)
    act = np.array([oid(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS])
    dof = np.array([int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
    if "stall_Nm" in s:
        fr, dmp = m.actuator_forcerange[act], m.dof_damping[dof]
        if not (np.allclose(fr, [-SERVO_BASE["forcerange"], SERVO_BASE["forcerange"]])
                and np.allclose(dmp, SERVO_BASE["damping"]) and m.actuator_forcelimited[act].all()):
            raise ValueError(f"servo {servo!r}: the model's arm servos are not the 12 V base {SERVO_BASE} "
                             f"(forcerange {fr.tolist()}, damping {dmp.tolist()})")
    setpoint = None
    if "stall_Nm" in s and "P" in s:                                   # the real servo
        from metalsim.physics.actuation import voltage_servo

        duty_per_rad = s["P"] * BAM_M1["error_gain"] * s.get("error_gain_ratio", BAM_M1["error_gain_ratio"])
        kd = (LEROBOT["D_Coefficient"] / 8 * 1e-3 * s["d_cycle_s"] / STS_STEP) if s["d_cycle_s"] > 0 else 0.0
        for i, j in enumerate(ARM_JOINTS):
            v = voltage_servo(s["stall_Nm"], s["noload_rad_s"], duty_per_rad, s["max_duty"],
                              s["gripper_torque_limit"] if j == "gripper" else 1.0, s["coulomb"], kd)
            if not s["line"]:
                v["damping"] = (s.get("friction") or BAM_M1)["friction_viscous"]
            a = act[i]
            m.actuator_gainprm[a, 0] = v["kp"]
            m.actuator_biasprm[a, 1:3] = -v["kp"], -v["kv"]
            m.actuator_forcerange[a] = -v["forcerange"], v["forcerange"]
            m.actuator_forcelimited[a] = 1
            m.dof_damping[dof[i]] = v["damping"]
            m.dof_frictionloss[dof[i]] = v["frictionloss"]
            m.dof_armature[dof[i]] = s["armature"]
        if s.get("friction"):                                          # load-dependent friction (BAM)
            from metalsim.physics.actuation import BamFriction

            fh = BamFriction(m, dof, s["friction"])
            fh.stiffen()
            fh.reset()
        setpoint = dict(cmd_delay=float(s["servo_delay"] + s["host_delay"]), servo_delay=float(s["servo_delay"]),
                        host_delay=float(s["host_delay"]), speed_limit=float(s["speed_limit"]),
                        accel_limit=float(s["accel_limit"]), timestep=float(m.opt.timestep))
    elif "stall_Nm" in s:
        stall, w0 = float(s["stall_Nm"]), float(s["noload_rad_s"])
        lim = np.array([s["gripper_torque_limit"] if j == "gripper" else 1.0 for j in ARM_JOINTS])
        m.actuator_forcerange[act] = np.stack([-stall * lim, stall * lim], 1)
        m.dof_damping[dof] = stall / w0
    fr, dmp = m.actuator_forcerange[act, 1], m.dof_damping[dof]
    out = dict(servo=servo, label=s["label"], joints=list(ARM_JOINTS), forcerange_Nm=fr.round(4).tolist(),
               damping_Nms=dmp.round(4).tolist(), noload_line_rad_s=(fr / dmp).round(3).tolist(),
               kp=m.actuator_gainprm[act, 0].round(3).tolist(), kv=(-m.actuator_biasprm[act, 2]).round(4).tolist(),
               armature=m.dof_armature[dof].round(4).tolist(), frictionloss=m.dof_frictionloss[dof].round(4).tolist())
    if over:
        out["overrides"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in over.items()}
    if setpoint is not None:
        out["setpoint"] = setpoint
        out["actuators"] = act.tolist()
    if s.get("friction"):
        p = s["friction"]
        out["friction"] = dict(params=dict(p), dofs=dof.tolist(), emf=p["kt"] ** 2 / p["R"],
                               solref=m.dof_solref[dof[0]].tolist(), solimp=m.dof_solimp[dof[0]].tolist())
    return out


def servo_layer(applied: dict, n: int):
    """The ``SetpointLayer`` (n worlds) that ``applied`` (``apply_servo``'s result) needs between commands and
    ``d.ctrl``, or None for the servo options without one. Reset it with the arm's joint positions before stepping."""
    sp = applied.get("setpoint")
    if sp is None:
        return None
    from metalsim.physics.actuation import SetpointLayer, SetpointParams

    return SetpointLayer(n, len(applied["actuators"]), sp["timestep"],
                         SetpointParams(cmd_delay=sp["cmd_delay"], speed_limit=sp["speed_limit"],
                                        accel_limit=sp["accel_limit"]))


def friction_hook(applied: dict, m, sim=None):
    """The ``BamFriction`` that ``applied`` (``apply_servo``'s result on ``m``) needs before every physics step
    (``hook.update(d)``; ``hook.reset()`` after a reset of the data), or None for the servo options without one.

    With ``sim`` (a ``BatchSim`` of ``m`` with ``"dof_frictionloss"`` in ``per_world_fields``): the same update on the
    GPU, ``BamFrictionWarp``, registered as a substep hook (it runs inside every step; reset worlds get the budget at
    rest); after every host write of the state, write ``d.ctrl`` (the servos' holding targets), ``sim.forward()``, then
    call its ``update()``: the budget takes the motor torque from the last forward pass (``SO101Lift.reset``)."""
    fr = applied.get("friction")
    if fr is None:
        return None
    if sim is not None:
        from metalsim.physics.actuation import BamFrictionWarp

        return BamFrictionWarp(sim, fr["dofs"], fr["params"], fr["emf"])
    from metalsim.physics.actuation import BamFriction

    return BamFriction(m, fr["dofs"], fr["params"], fr["emf"])


def load_policy(path: str) -> dict:
    """The extracted teacher (``format`` so101_lift_sb3_mlp_v1) as float32 / float64 numpy arrays."""
    import torch

    P = torch.load(path, weights_only=True)
    if P.get("format") != "so101_lift_sb3_mlp_v1":
        raise ValueError(f"{path}: not an so101_lift policy file (format {P.get('format')!r})")
    out = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in P.items() if k != "pi"}
    out["pi"] = {k: v.numpy() for k, v in P["pi"].items()}
    return out


def policy_act(P: dict, obs: np.ndarray) -> np.ndarray:
    """(N, 6) float32 deterministic actions for (N, 21) float64 observations: so101Sim's eval normalisation in float64,
    then the actor in float32 (SB3 casts the observation to float32), clipped to [-1, 1] (SB3 ``predict``)."""
    import torch

    W = P["pi"]
    n = np.clip((obs - P["obs_mean"]) / np.sqrt(P["obs_var"] + P["obs_eps"]), -P["clip_obs"], P["clip_obs"])
    with torch.no_grad():
        x = torch.as_tensor(n.astype(np.float32))
        x = torch.tanh(x @ torch.as_tensor(W["mlp_extractor.policy_net.0.weight"]).T
                       + torch.as_tensor(W["mlp_extractor.policy_net.0.bias"]))
        x = torch.tanh(x @ torch.as_tensor(W["mlp_extractor.policy_net.2.weight"]).T
                       + torch.as_tensor(W["mlp_extractor.policy_net.2.bias"]))
        a = x @ torch.as_tensor(W["action_net.weight"]).T + torch.as_tensor(W["action_net.bias"])
    return np.clip(a.numpy(), -1.0, 1.0).astype(np.float32)


def reset_draw(seed: int, home: np.ndarray, box_qadr: int, arm_qadr: np.ndarray, randomize: bool = True):
    """so101Sim ``SO101LiftEnv.reset(seed=seed)``'s start state: (qpos, mass scale, friction scale)."""
    q = home.astype(np.float64).copy()
    if not randomize:
        q[box_qadr:box_qadr + 3] = (0.22, 0.0, 0.03)
        return q, 1.0, 1.0
    rng = np.random.default_rng(seed)        # gymnasium.utils.seeding.np_random(seed): Generator(PCG64(SeedSequence(seed)))
    r = rng.uniform(0.15, 0.26)
    th = rng.uniform(-0.7, 0.7)
    yaw = rng.uniform(-np.pi, np.pi)
    q[box_qadr:box_qadr + 3] = (r * np.cos(th), r * np.sin(th), 0.03)
    q[box_qadr + 3:box_qadr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    q[arm_qadr[:5]] += rng.uniform(-0.08, 0.08, 5)
    mass = rng.uniform(0.7, 1.4)
    fric = rng.uniform(0.7, 1.3)
    return q, mass, fric


class SO101Lift:
    """so101Sim's lift env on MuJoCo Warp, ``num_envs`` worlds; env ``e`` starts from so101Sim's reset with seed
    ``film.seed + e`` (``film.seed`` 10000: eval_lift_native.py's seeds). One frame per control step (50 fps), the state
    after its 4 physics steps. ``adapter_kw``: ``scene`` (default ``SCENE``), ``randomize`` (default True), ``servo``
    ("12v", the default: the scene as it is; "7v4", "7v4_real", "12v_real": ``apply_servo``), ``servo_over`` (its
    overrides). The "_real" servos run ``servo_layer`` on the host and feed each physics step's target to the GPU
    (``metalsim.physics.actuation.CtrlPlanFeeder``); "7v4_bam" also updates its load-dependent friction after every
    physics step on the GPU (``friction_hook`` -> ``BamFrictionWarp``, per-world ``dof_frictionloss``).

    ``controller`` "deploy_expert" (a "_real" servo only; no policy file) replaces the policy with one
    ``so101_expert.DeployExpert`` per world, wired as ``runs/readme_media/scratch/so101-expert/eval_metalsim.py``: BatchSim
    steps one 5 ms physics step at a time (4 per frame); before physics step k, if k is a control tick (round(j * 200 /
    30), j = 0, 1, ...), each live world's qpos / qvel go to a CPU MjData of the model, ``mj_forward`` runs, and that
    world's expert returns its six absolute goals, held until the next tick; every physics step the servo layer turns the
    held goals into ``d.ctrl``. Frames, success and episode end are the policy loop's."""

    backend = "mujoco"

    def __init__(self, task, film, checkpoint, viewer=None):
        import mujoco
        import warp as wp

        from metalsim.physics.batch import BatchSim, BatchSimOptions

        wp.config.quiet = True
        self.wp, self.mujoco = wp, mujoco
        akw = dict(getattr(task, "adapter_kw", None) or {})
        self.scene = os.path.join(ROOT, akw.get("scene", SCENE))
        self.randomize = bool(akw.get("randomize", True))
        self.controller = akw.get("controller", "policy")
        if self.controller not in CONTROLLERS:
            raise ValueError(f"adapter_kw controller {self.controller!r}: not one of {CONTROLLERS}")
        expert = self.controller == "deploy_expert"
        if expert:                                           # scripted: no policy file
            self.checkpoint, self.P = None, None
        else:
            self.checkpoint = checkpoint or os.path.join(ROOT, POLICY)
            if not os.path.exists(self.checkpoint):
                raise FileNotFoundError(f"{self.checkpoint} missing: extract it with runs/film/so101_lift/port/extract_policy.py")
            self.P = load_policy(self.checkpoint)
        n = self.num_envs = int(film.num_envs)
        self.seed0 = int(film.seed)
        t0 = time.perf_counter()
        self.model = m = mujoco.MjModel.from_xml_path(self.scene)
        self.servo = apply_servo(m, akw.get("servo", "12v"), **dict(akw.get("servo_over") or {}))  # before BatchSim
        if expert and "setpoint" not in self.servo:
            raise ValueError(f"controller deploy_expert needs a '_real' servo (firmware profile and P loop the expert's "
                             f"goals are streamed to), not {self.servo['servo']!r}")
        self.n_sub = max(1, int(round(CTRL_DT / m.opt.timestep)))
        # the expert's 30 Hz ticks fall inside a 20 ms frame: one physics step per BatchSim step, stepped n_sub times
        pwf = ("body_mass", "geom_friction") + (("dof_frictionloss",) if "friction" in self.servo else ())
        self.sim = BatchSim(m, n, options=BatchSimOptions(substeps=1 if expert else self.n_sub, njmax=512,
                                                          per_world_fields=pwf))
        # "7v4_bam": BAM's load-dependent friction, a Warp kernel after every physics substep (None for the others)
        self.friction = friction_hook(self.servo, m, self.sim)
        # "_real" servos: command delay + firmware profile between the command and d.ctrl, per physics step
        self.layer = servo_layer(self.servo, n)
        self.feeder = None
        if self.layer is not None:
            if not expert:
                from metalsim.physics.actuation import CtrlPlanFeeder

                self.feeder = CtrlPlanFeeder(self.sim, self.n_sub)
            self.servo_act = np.array(self.servo["actuators"])
        self.frame_dt = float(m.opt.timestep) * self.n_sub
        rng = m.actuator_ctrlrange.copy()
        lim = m.actuator_ctrllimited.astype(bool).reshape(-1)
        self.lo, self.hi = np.where(lim, rng[:, 0], -np.pi), np.where(lim, rng[:, 1], np.pi)
        oid = lambda t, s: mujoco.mj_name2id(m, t, s)
        self.grip_site = oid(mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.box_body = oid(mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_geom = oid(mujoco.mjtObj.mjOBJ_GEOM, "box")
        bj = oid(mujoco.mjtObj.mjOBJ_JOINT, "box")
        self.box_qadr = int(m.jnt_qposadr[bj]) if bj >= 0 else 6
        self.arm_qadr = np.array([int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
        self.arm_vadr = np.array([int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS])
        self.mass0 = float(m.body_mass[self.box_body])
        self.fric0 = m.geom_friction[self.box_geom].copy()
        self.home = m.qpos0.copy()
        if expert:
            from metalsim.render.film import so101_expert as X

            ids = dict(grip_site=self.grip_site, box_body=self.box_body, box_qadr=self.box_qadr,
                       arm_qadr=self.arm_qadr, arm_vadr=self.arm_vadr)
            self.experts = [X.DeployExpert(m, ids) for _ in range(n)]
            self.kins = [mujoco.MjData(m) for _ in range(n)]     # each world's measured state, fresh kinematics
            dt = float(m.opt.timestep)
            self.ticks = {int(round(j / X.CTRL_HZ / dt)) for j in range(int(MAX_STEPS * CTRL_DT * X.CTRL_HZ) + 1)}
        self.build_s = time.perf_counter() - t0
        self.ckpt_iter = None
        if expert:
            self.ckpt_format = self.ckpt_source = None
        else:
            self.ckpt_format = "so101_lift_sb3_mlp_v1 (plain torch tensors from SB3 PPO model.zip + VecNormalize vecnorm.pkl)"
            self.ckpt_source = dict(source="so101Sim lift teacher (SB3 PPO), extracted",
                                    source_url=SO101SIM["repo"], so101sim_commit=SO101SIM["commit"],
                                    sb3_model_zip_sha256=self.P["source"]["model_zip_sha256"],
                                    sb3_vecnorm_pkl_sha256=self.P["source"]["vecnorm_pkl_sha256"],
                                    num_timesteps=self.P["source"]["num_timesteps"])
        self.overflow = None

    # -- state -------------------------------------------------------------------------------------

    def _read(self):
        d = self.sim.d
        f = lambda a: a.numpy().astype(np.float64)          # (n, ...) with vector fields expanded (vec3, spatial)
        qpos, qvel = f(d.qpos), f(d.qvel)
        box = f(d.xpos)[:, self.box_body]
        grip = f(d.site_xpos)[:, self.grip_site]
        bvel = f(d.cvel)[:, self.box_body, 3:6]
        return qpos, qvel, box, grip, bvel

    def _obs(self, qpos, qvel, box, grip, bvel):
        return np.concatenate([qpos[:, self.arm_qadr], qvel[:, self.arm_vadr], box, box - grip, bvel], 1)

    def reset(self):
        n, sim = self.num_envs, self.sim
        sim.reset()
        sim.synchronize()
        q = np.zeros((n, self.model.nq))
        self.mass_scale, self.fric_scale = np.ones(n), np.ones(n)
        for e in range(n):
            q[e], self.mass_scale[e], self.fric_scale[e] = reset_draw(self.seed0 + e, self.home, self.box_qadr,
                                                                      self.arm_qadr, self.randomize)
        bm = sim.m.body_mass.numpy()
        bm[:, self.box_body] = self.mass0 * self.mass_scale
        sim.m.body_mass.assign(bm)
        gf = sim.m.geom_friction.numpy()
        gf[:, self.box_geom] = self.fric0[None] * self.fric_scale[:, None]
        sim.m.geom_friction.assign(gf)
        sim.set_state(q, np.zeros((n, self.model.nv)))      # host write, then forward (so101Sim: mj_forward)
        if self.friction is not None:
            # the first step's friction budget from this state with the servos powered on holding it: the budget takes
            # the motor torque from the last forward pass, which after reset_data ran at ctrl 0 (a spurious full-clip
            # torque toward 0; one step of it shifts where the joint sticks inside its friction band)
            c = np.zeros((n, self.model.nu), np.float32)
            c[:, self.servo_act] = q[:, self.arm_qadr]
            with self.wp.ScopedDevice(sim.device):
                sim.d.ctrl.assign(c)
            sim.forward()
            self.friction.update()
        sim.synchronize()
        self.start_qpos = self._read()[0]
        self.state = self._read()
        if self.layer is not None:                           # the servos power on holding where they are
            q0 = np.zeros((n, self.model.nu))
            q0[:, self.servo_act] = self.start_qpos[:, self.arm_qadr]
            self.layer.reset(q0)
        self.t = 0
        self.held = np.zeros(n, np.int64)
        self.succeeded = np.zeros(n, bool)
        self.success_t = np.full(n, -1, np.int64)
        self.hmax = np.zeros(n)
        self.alive = np.ones(n, bool)
        self.end_height = np.zeros(n)
        self.acc_sq = np.zeros(n)
        self.acc_n = np.zeros(n)
        self.prev_qvel = self.state[1][:, self.arm_vadr].copy()
        self.last = None
        if self.controller == "deploy_expert":                # the command starts where the arm is (servo power-on)
            self.meas_q, self.meas_qd = self.state[0], self.state[1]
            self.cmd = self.state[0][:, self.arm_qadr].copy()
            for e, x in enumerate(self.experts):
                x.reset(self.state[0][e, self.arm_qadr])
            self.ctrl_buf = np.zeros((n, self.model.nu), np.float32)
            self.nph = 0                                     # physics steps taken

    def _advance_policy(self):
        """One control step of so101Sim's loop: the policy's action from the last state, then n_sub physics steps."""
        wp, sim = self.wp, self.sim
        obs = self._obs(*self.state)
        a = policy_act(self.P, obs)
        ctrl = self.lo + 0.5 * (a.astype(np.float64) + 1.0) * (self.hi - self.lo)
        with wp.ScopedDevice(sim.device):
            if self.layer is not None:
                self.feeder.load(self.layer.plan(ctrl, self.n_sub))
            else:
                sim.d.ctrl.assign(ctrl.astype(np.float32))
            sim.step()
            sim.synchronize()
        return a, ctrl

    def _advance_expert(self):
        """One frame (n_sub physics steps) of the deployment-style expert; returns (held goals, last step's targets)."""
        wp, sim, mj, m = self.wp, self.sim, self.mujoco, self.model
        dt = float(m.opt.timestep)
        for _ in range(self.n_sub):
            if self.nph in self.ticks:
                for e in np.flatnonzero(self.alive):
                    k = self.kins[e]
                    k.qpos[:], k.qvel[:] = self.meas_q[e], self.meas_qd[e]
                    mj.mj_forward(m, k)
                    self.cmd[e] = self.experts[e].act(k, self.nph * dt)
            tg = self.layer.tick(self.cmd)
            self.ctrl_buf[:, self.servo_act] = tg
            with wp.ScopedDevice(sim.device):
                sim.d.ctrl.assign(self.ctrl_buf)
                sim.step()
            sim.synchronize()
            self.meas_q = sim.d.qpos.numpy().astype(np.float64)
            self.meas_qd = sim.d.qvel.numpy().astype(np.float64)
            self.nph += 1
        return self.cmd.copy(), tg

    def step(self):
        t0 = time.perf_counter()
        a, ctrl = self._advance_expert() if self.controller == "deploy_expert" else self._advance_policy()
        self.state = qpos, qvel, box, grip, bvel = self._read()
        sim_s = time.perf_counter() - t0
        self.t += 1
        height = box[:, 2]
        lifted = height > LIFT_Z
        al = self.alive
        self.held = np.where(lifted, self.held + 1, 0)
        new = al & (self.held >= SUCCESS_HOLD) & ~self.succeeded
        self.success_t[new] = self.t - 1                     # eval_lift_native.py's t (0-based control step)
        self.succeeded |= al & (self.held >= SUCCESS_HOLD)
        self.hmax[al] = np.maximum(self.hmax[al], height[al])
        v = qvel[:, self.arm_vadr]
        acc = (v - self.prev_qvel) / self.frame_dt
        self.prev_qvel = v.copy()
        self.acc_sq[al] += (acc[al] ** 2).sum(1)
        self.acc_n[al] += 1
        oob = np.linalg.norm(box[:, :2], axis=1) > OOB_R
        done = (oob | (self.t >= MAX_STEPS)) & al
        self.end_height[done] = height[done]
        self.alive &= ~done
        self.last = dict(action=a, ctrl=ctrl, box=box, grip=grip, height=height)
        snap = dict(qpos=qpos)
        info = dict(terminated=oob & done, success=self.succeeded.copy(), success_t=self.success_t.copy(),
                    hmax=self.hmax.copy(), height=height.copy(), lifted=lifted.copy())
        if self.controller == "deploy_expert":
            info["retries"] = np.array([x.retries for x in self.experts], np.int64)
        return snap, done | ~al, info, dict(sim_s=sim_s)

    # -- selection and provenance ----------------------------------------------------------------------

    def smoothness(self):
        """(N,) RMS arm joint acceleration over the first episode (rad/s^2; finite difference of the six joint
        velocities per 20 ms control step): lower is smoother."""
        return np.sqrt(self.acc_sq / np.maximum(self.acc_n * len(ARM_JOINTS), 1))

    def summary(self, ends) -> str:
        self.overflow = self.sim.check_overflow()
        s = self.smoothness()
        if self.controller == "deploy_expert":
            rows = [f"{self.seed0 + e}:{'S' if self.succeeded[e] else '-'}"
                    + (f"@{self.success_t[e] * CTRL_DT:.2f}s" if self.success_t[e] >= 0 else "")
                    + f"/{self.hmax[e]:.3f}m/{s[e]:.1f}/r{self.experts[e].retries}" for e in range(self.num_envs)]
            ok = self.succeeded.sum()
            return (f"[record] first episodes, seed:success[@time to lift]/max height/RMS joint acc rad/s^2/retries: "
                    + " ".join(rows) + f"\n[record] deploy expert: {ok}/{self.num_envs} success, "
                    f"{int((self.succeeded & (self.end_height > LIFT_Z)).sum())} still up at the end, "
                    f"{sum(x.retries for x in self.experts)} retries")
        rows = [f"{self.seed0 + e}:{'S' if self.succeeded[e] else '-'}"
                + (f"@{self.success_t[e] * CTRL_DT:.2f}s" if self.success_t[e] >= 0 else "")
                + f"/{self.hmax[e]:.3f}m/{s[e]:.1f}" for e in range(self.num_envs)]
        return "[record] first episodes, seed:success[@time to lift]/max height/RMS joint acc rad/s^2: " + " ".join(rows)

    def rank(self, first):
        """Successful episodes that still hold the box above 0.10 m when the episode ends first, then the smoothest
        (lowest RMS arm joint acceleration), then the seed. No other preference."""
        n = self.num_envs
        ok = self.succeeded & (self.end_height > LIFT_Z)
        s = self.smoothness()
        return [k[-1] for k in sorted((not ok[e], not self.succeeded[e], s[e], e) for e in range(n))]

    def twin_spec(self):
        """The simulated scene with a tabletop look: the floor plane (z = 0, the surface the arm and box stand on)
        becomes a finite 0.9 m x 0.7 m wooden table top (4 cm slab, top at z = 0) on four legs, and a studio floor
        0.75 m below; render-only geoms (no collision), the robot and box are the simulated model's."""
        mj = self.mujoco
        spec = mj.MjSpec.from_file(self.scene)
        spec.meshdir = os.path.join(os.path.dirname(self.scene), spec.meshdir or "")
        for g in list(spec.geoms):
            if g.name == "floor":
                spec.delete(g)
        for t in list(spec.textures):
            if t.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(t)
        wb = spec.worldbody
        box_t = mj.mjtGeom.mjGEOM_BOX
        top = dict(cx=0.10, cy=0.0, hx=0.45, hy=0.35, hz=0.02)
        wb.add_geom(name="table_top", type=box_t, size=[top["hx"], top["hy"], top["hz"]],
                    pos=[top["cx"], top["cy"], -top["hz"]], contype=0, conaffinity=0, rgba=[0.6, 0.45, 0.3, 1])
        leg_h = 0.73
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            wb.add_geom(name=f"table_leg_{i}", type=box_t, size=[0.025, 0.025, leg_h / 2],
                        pos=[top["cx"] + sx * (top["hx"] - 0.05), top["cy"] + sy * (top["hy"] - 0.05),
                             -2 * top["hz"] - leg_h / 2], contype=0, conaffinity=0, rgba=[0.2, 0.2, 0.2, 1])
        wb.add_geom(name="studio_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                    pos=[0, 0, -2 * top["hz"] - leg_h], contype=0, conaffinity=0, rgba=[0.5, 0.5, 0.5, 1])
        return spec

    def _controller_meta(self) -> dict:
        """What the "deploy_expert" controller is, for rec.json (it stands where the policy file's provenance is)."""
        import hashlib

        from metalsim.render.film import provenance
        from metalsim.render.film import so101_expert as X

        with open(X.__file__, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        return dict(name="deploy_expert", cls="metalsim.render.film.so101_expert.DeployExpert",
                    source=provenance.relpath(X.__file__), source_sha256=sha,
                    plan=f"so101Sim {SO101SIM['repo']} rl/native/scripted_expert.py at {SO101SIM['commit']}: its "
                         "waypoints, damped-least-squares IK and phases",
                    command="absolute joint goals at 30 Hz, held between ticks (LeRobot SOFollower.send_action); the "
                            "servo layer (command delay + firmware profile) turns them into each physics step's target",
                    ctrl_hz=X.CTRL_HZ, ticks="before physics step round(j * 200 / 30), j = 0, 1, ...",
                    vmax_rad_s=X.VMAX.tolist(), amax_rad_s2=X.AMAX.tolist(), t_settle_s=X.T_SETTLE,
                    gripper_open_close_rad=[X.GRIP_OPEN, X.GRIP_CLOSE],
                    perception="cube pose from the simulator state at each tick (not a camera)",
                    gravity_compensation="none (LeRobot has none for the SO arms)")

    def meta(self):
        from metalsim.render.film import provenance

        m = self.model
        expert = self.controller == "deploy_expert"
        if expert:
            head = dict(controller=self._controller_meta())
        else:
            head = dict(policy=provenance.relpath(self.checkpoint), policy_format=self.ckpt_format,
                        policy_source=self.ckpt_source, policy_check=self.P.get("check"),
                        policy_arch="SB3 MlpPolicy actor: 21 -> 256 -> 256 -> 6 (tanh), mean action clipped to "
                                    "[-1, 1]; VecNormalize obs statistics, clip 10")
        out = dict(**head,
                   so101sim=SO101SIM, scene=provenance.relpath(self.scene), servo=self.servo, env_build_s=self.build_s,
                   env=dict(ctrl_dt=CTRL_DT, substeps=self.n_sub, max_steps=MAX_STEPS, lift_z=LIFT_Z,
                            success_hold=SUCCESS_HOLD, oob_radius=OOB_R, randomize=self.randomize,
                            seeds=[self.seed0, self.seed0 + self.num_envs - 1]),
                   physics=("MuJoCo Warp BatchSim (metalsim.physics.batch), one 5 ms physics step per BatchSim step, "
                            "4 per 20 ms frame; before each, at the expert's 30 Hz ticks, a CPU MjData copy of each "
                            "world's qpos / qvel with mj_forward for the expert; every physics step the servo layer's "
                            "target into ctrl; box mass and friction per world (no set_const, as so101Sim)") if expert
                   else ("MuJoCo Warp BatchSim (metalsim.physics.batch), so101Sim's control loop: ctrl from the "
                         "action, 4 physics steps of 5 ms; box mass and friction per world (no set_const, as "
                         "so101Sim)"),
                   model=dict(timestep=float(m.opt.timestep), integrator=int(m.opt.integrator), cone=int(m.opt.cone),
                              solver=int(m.opt.solver), iterations=int(m.opt.iterations),
                              ls_iterations=int(m.opt.ls_iterations), impratio=float(m.opt.impratio), nq=int(m.nq),
                              nv=int(m.nv), nu=int(m.nu)),
                   capacity_overflow=self.overflow,
                   actions=("the expert's absolute joint goals (deterministic)" if expert
                            else "the teacher's mean action (deterministic)"),
                   done=f"300 control steps (6 s) or box beyond {OOB_R} m (terminated); success does not end it")
        if hasattr(self, "succeeded"):
            out["first_episode"] = dict(
                seeds=list(range(self.seed0, self.seed0 + self.num_envs)), success=self.succeeded.tolist(),
                success_t_s=[round(float(t) * CTRL_DT, 3) if t >= 0 else None for t in self.success_t],
                max_height_m=self.hmax.round(4).tolist(), end_height_m=self.end_height.round(4).tolist(),
                rms_joint_acc=self.smoothness().round(2).tolist(), mass_scale=self.mass_scale.round(4).tolist(),
                friction_scale=self.fric_scale.round(4).tolist(),
                ranking="success and still lifted at the end, then lowest RMS arm joint acceleration, then seed")
            if expert:
                out["first_episode"]["retries"] = [x.retries for x in self.experts]
                out["first_episode"]["events"] = [[list(ev) for ev in x.events] for x in self.experts]
        return out


# -- the film entry ------------------------------------------------------------------------------------

PH_CC0 = "CC0 1.0 (Poly Haven; fetched, not committed)"
FETCH_PH = "scripts/fetch_polyhaven_assets.sh train"
KIT_FOVY_16_9 = 36.0
# Fixed cameras in the env frame (the arm's base at the origin, reaching along +x), from the -y side: the box stays in
# view from the approach through the hold, where the arm folds back and holds it out to the +y side (seen from +y, the
# folded arm hides the box). Kit's lens (36 deg vertical).
SO101_VIEWER = ViewerCfg(eye=(0.62, -0.48, 0.36), lookat=(0.12, 0.03, 0.10), origin_type="env", fovy=KIT_FOVY_16_9,
                         resolution=(1280, 720), smooth_s=0.0)
SO101_README_VIEWER = ViewerCfg(eye=(0.55, -0.42, 0.32), lookat=(0.12, 0.03, 0.11), origin_type="env",
                                fovy=KIT_FOVY_16_9, resolution=(640, 480), smooth_s=0.0)

PORT = "runs/film/so101_lift/port"
SO101_NOTES = f"""
**The policy.** so101Sim's state-based lift teacher (<{SO101SIM['repo']}>, the owner's project, commit
`{SO101SIM['commit']}`): stable-baselines3 PPO, `MlpPolicy` with two 256-unit tanh layers for actor and critic, 12
`SubprocVecEnv` workers with `VecNormalize`, on CPU MuJoCo (`rl/native/train_lift.py`). so101Sim trained it in two
runs (`docs/AGENT_CONTEXT.md` section 6): run 1, 0 to 3 M steps, plateaued at 13 % success because success ended the
episode, so hovering just under the lift height paid more than lifting; run 2, 3 to 8 M steps from run 1's checkpoint
with a non-terminating success and a per-step bonus while the box is up (`rl/native/continue_lift.py`), reached 98 %
within 300 k steps, and so101Sim reports a deterministic 100/100 (mean lift height 0.17 m, 0.95 s to lift). Run 2's
output is not published: it is in neither so101Sim's git nor its LFS, and not on this machine. Run 1's checkpoint is
tracked there (`rl/runs/lift_ppo_run1_backup`, 3,004,416 steps, training success 0.13 at its end). **Run 2 was reproduced here** with
so101Sim's own code on the CPU and no tracked file changed: the run 1 checkpoint copied to the untracked
`rl/runs/lift_ppo` its scripts read, then `continue_lift.py` with its default 5 M extra steps (8.2 min, 12 workers),
then `eval_lift_native.py 100`: **100/100, mean lift height 0.170 m (lowest 0.148 m), 0.95 s to lift**, the documented
figures (training success 0.94 at 3.10 M steps and 0.95 to 1.00 from then on; `{PORT}/repro/`: `run.json`, `progress.txt`,
`eval_lift_native.log`). The film plays that checkpoint (8,005,632 steps; SB3 `model.zip` sha256 `1ddf1bc3...`,
`vecnorm.pkl` `ce78ad44...`) through plain torch weights: `{PORT}/extract_policy.py` (run in so101Sim's environment)
writes the actor's three layers and VecNormalize's observation mean and variance as tensors to
`{POLICY}` (sha256 `9dfdea71...`, 296,754 bytes; `torch.load(weights_only=True)`, no stable-baselines3 needed); on 4,096
observations its output differs from SB3's `predict(deterministic=True)` by at most 2.0e-6. The action is the actor's
mean clipped to [-1, 1], after so101Sim's eval normalisation, clip((obs - mean) / sqrt(var + 1e-8), -10, 10). The
repository's `.gitignore` ignores `*.pt` files unless they are listed; without the weights file, rebuild it from
so101Sim (the reproduction above, then `extract_policy.py`; `{PORT}/repro/run.json` lists the commands).

**so101Sim's task on MuJoCo Warp.** `metalsim/render/film/so101.py`, `SO101Lift`: so101Sim's `SO101LiftEnv`
(`rl/native/so101_lift.py`) on `metalsim.physics.batch.BatchSim`, 16 worlds. Per 20 ms control step the action sets
each position servo's target, a = -1 to 1 across the actuator's control range, then 4 physics steps of 5 ms. The
21 observations are so101Sim's: the six joint positions and velocities, the box position, the box minus the gripper
frame site, and the box's linear velocity, read after the step from the same MuJoCo fields (joint state after the
integration; the box and site positions and the box velocity from the last substep's kinematics, as MuJoCo's `mj_step`
leaves them). The reset is so101Sim's `reset(seed=...)`, the same random draws in the same order (box 0.15 to 0.26 m
from the base within +-0.7 rad, random yaw; +-0.08 rad on the five arm joints; box mass x 0.7 to 1.4 and friction x
0.7 to 1.3, set per world without recomputing derived constants, as so101Sim does); over seeds 10000 to 10099 the start
states equal so101Sim's to float32 rounding (3e-8). Success is so101Sim's: the box centre above 0.10 m for 10 control
steps in a row (it does not end the episode); an episode ends at 300 steps (6 s) or when the box leaves a 0.45 m
radius. The scene is `assets/so101/scene_box_rl.xml` (MuJoCo Menagerie's SO-101 with the box and floor). so101Sim's
own `scene_box_rl.xml` is not published (the README's "known gaps"); compiled with so101Sim's MuJoCo 3.10.0, every
numeric model array of the two is equal except the floor plane's render size, the box's colour, the box's position
before the reset overwrites it, a keyframe and a skybox (`{PORT}/scene_diff.py`, output in `repro/scene_diff.txt`).
MetalSim compiles it with MuJoCo 3.14.0, which recentres the meshes differently from 3.10.0 (mesh geom frames differ
by up to 1.3e-3 in quaternion, mesh vertices by up to 46 um; inertia frames by 3e-6).

**Against so101Sim's own CPU run.** `{PORT}/reference.py` runs so101Sim's `SO101LiftEnv` unchanged on CPU MuJoCo
3.10.0 with the extracted weights and `eval_lift_native.py`'s seeds and loop (`reference.json`; with the weights it
gives so101Sim's own eval figures). `{PORT}/compare.py` runs the film's adapter on the same 100 seeds, one MuJoCo
Warp world each (`compare.json`, a GPU-queue job).

| 100 episodes, seeds 10000-10099 | so101Sim, CPU MuJoCo 3.10.0 | MetalSim, MuJoCo Warp | MetalSim, three re-runs |
|---|---|---|---|
| success (box above 0.10 m for 10 steps) | 100 | 98 (seeds 10021, 10027: the box tipped onto its side at the grasp) | 98, 97, 98 |
| box still above 0.10 m when the episode ends (6 s) | 99 | 94 | 92, 94, 94 |
| mean of the highest box height | 0.170 m | 0.166 m | 0.166, 0.165, 0.167 m |
| time to lift, mean over successes | 0.95 s | 0.91 s | 0.94, 0.90, 0.93 s |
| same outcome, seed by seed | | 98 of 100 | 98, 97, 98 (10021 and 10027 each time, 10098 once) |

The GPU rollout is not bitwise repeatable from run to run (float atomics accumulate in arrival order; `docs/GAPS.md`),
so each MetalSim figure is one draw. The re-runs (2026-10-31, MuJoCo Warp fork `8eb51fe637` on Warp `4127c48183`;
`{PORT}/compare.py` twice and `runs/film/so101_lift_74v/port/compare_servo.py 12v` once; `runs/readme_media/handoff/revalidate.md`) put
MetalSim at **97 to 98 successes in 100 over three runs** (92 to 94 boxes still up at 6 s); the first column's run was
on the earlier fork checkout `07a51a63`.

The trajectories agree until the jaws touch the box: on seeds 10000 to 10004 the box positions differ by less than
1 mm up to 0.26 to 0.36 s (first jaw contact: 0.24 s, median), then by centimetres (median over the 100 seeds: 1.3 cm
at 0.5 s, 2.4 cm at 1 s, 3.8 cm at 6 s). The policy amplifies float differences before that, too: its actions are
mostly at +-1 (realism check below), so a pre-activation near zero flips a joint's command from one end of its range
to the other (seed 10002: a difference of 1.91 in one action within the first 10 steps, before contact). For scale,
so101Sim's CPU run itself, restarted with the start state rounded to float32 or with 1e-7 / 1e-6 rad of noise on the
arm joints (11 runs of 100 episodes, `{PORT}/perturbed/perturbed.json`), gives 99 to 100 successes and 95 to 99
boxes still up at 6 s. MetalSim's 97 to 98 and 92 to 94 (four runs) are at or just below the low end of that spread; whether float32 contact
on the GPU makes the grasp slightly less reliable or this is the spread is not settled. MuJoCo Warp also flags its
solver-iteration caps (the model's 10 iterations and 20 line-search iterations) during the run: the iteration cap in
12 of the 100 worlds and the line-search cap in all of them, at some step (`compare.json` `capacity_overflow`; warnings,
no contact or row was dropped). CPU MuJoCo reached its 10-iteration cap on 0.03 % of control steps (the last physics
step of each, 20 episodes; an experiment in the scratch area, not kept).

**Realism check: beyond the standard SO-101 follower.** Measured on so101Sim's CPU run (`{PORT}/realism.py`,
`realism.json`; 100 episodes, every 5 ms physics step). The SO-101 follower uses six Feetech STS3215 servos, the 7.4 V
1/345 C001 in The Robot Studio's bill of materials (<https://github.com/TheRobotStudio/SO-ARM100>); a 12 V version
is the alternative. C001 at 7.4 V: 19.5 kg.cm = 1.91 N.m stall torque, 52 rpm = 5.45 rad/s no-load speed (Feetech
datasheet "STS3215 200620a"); 12 V (C018): 30 kg.cm = 2.94 N.m, 0.222 s/60 deg = 4.72 rad/s (vendor listings). The
scene's actuators are Menagerie's: position servos with kp 998.22, kv 2.731 and a 2.94 N.m force limit, and joint
damping 0.6 N.m.s/rad. Servo force minus damping is a motor line from 2.94 N.m at rest to zero at 4.9 rad/s, so the
model is the 12 V servo. A DC servo at full voltage is bounded by |tau / tau_stall + omega / omega_0| <= 1 (all four
quadrants); the table counts physics steps outside that bound by more than 2 %, with tau the servo force minus
damping.

| joint | peak speed | episode peak (median) | steps at the 2.94 N.m limit | outside the 12 V C018 bound | outside the 7.4 V C001 bound (p99 ratio) |
|---|---|---|---|---|---|
| shoulder pan | 5.01 rad/s | 4.00 rad/s | 99.7 % | 2.5 % | 99.4 % (1.84) |
| shoulder lift | 5.14 rad/s | 4.45 rad/s | 98.4 % | 3.8 % | 98.1 % (1.83) |
| elbow flex | 5.40 rad/s | 5.14 rad/s | 99.8 % | 3.5 % | 98.6 % (1.80) |
| wrist flex | 5.18 rad/s | 5.01 rad/s | 19.6 % | 6.5 % | 16.3 % (1.76) |
| wrist roll | 4.99 rad/s | 4.79 rad/s | 100 % | 6.5 % | 96.2 % (1.85) |
| gripper | 5.65 rad/s | 4.77 rad/s | 100 % | 6.7 % | 96.4 % (1.67) |

- **Speeds** are within what the servos turn at: above the 12 V no-load speed on at most 1.4 % of steps, above the
  7.4 V C001's 5.45 rad/s once (the gripper, 5.65 rad/s).
- **Torques** are the 12 V servo's, flat out: five joints sit at the 2.94 N.m limit for 98 to 100 % of the episode,
  and the motion stays within the 12 V motor line to about 4 % (p99 ratio 1.03 to 1.04). The standard 7.4 V follower
  cannot supply it: on five joints it would need 1.67 to 1.85 times its servos' bound (p99) for 96 to 99 % of the
  episode.
- **The policy is bang-bang.** 80 % of all actions are at +-1 (the ends of each joint's command range); the median
  command jump is 1.96 rad per 20 ms step. While it holds the box, the arm joints' commands switch between the two ends
  of their range on 68 % of steps (median over episodes), close to a 25 Hz square wave, and the joints move at a
  median 0.87 rad/s while the arm holds the box up: the servos reverse at full torque nearly every step. No servo controller,
  current limit, backlash or overload protection is modelled; a real STS3215 driven like this runs at stall current
  and its overload protection is likely to cut in.
- **Gripper and contact.** The gripper's command is at one end of its range on 97 % of steps (open on the approach,
  fully closed from the grasp on); it holds the 4 cm box at 0.33 rad, so it squeezes at the full 2.94 N.m for the whole
  hold (106 N summed normal force on the box, median). LeRobot's
  SO-101 driver caps the real gripper at 50 % torque, 25 % when overloaded
  (`lerobot/robots/so_follower/so_follower.py`, `configure`: `Max_Torque_Limit` 500, `Overload_Torque` 25). The box is
  held only by the jaws (12 contacts, median; no contact with the floor or other links during the hold) and ends up
  lying on its side in them (largest tilt during the hold: 84 deg, median). The deepest box penetration per episode
  is 3.2 mm (median), 10.6 mm at most; in three episodes inspected at 3 s the moving jaw's collision mesh carried
  25 to 55 N per contact, at 1.7 to 3.4 mm penetration.
- **Lift.** The box rises at up to 0.86 m/s (median of episode peaks; 1.26 m/s at most) with about 2 g of vertical
  acceleration, and 99 of 100 boxes are still up at 6 s (seed 10014 is dropped after it counted as a success).

Verdict: plausible kinematics, but not the standard SO-101. The motion is what a 12 V SO-101 could do only by running
every servo at its limit with a full-range command switching up to 25 times a second, and with a gripper squeezing
twice as hard as LeRobot lets the real one. The 7.4 V follower in the bill of materials could not follow it. so101Sim
reports 5/5 zero-shot in Unreal (the same MuJoCo physics in URLab), and nothing has run on the physical arm. The film
shows what the simulation does; it is not evidence that the policy works on hardware.

**How the env was chosen.** 16 worlds, seeds 10000 to 10015 (`film.seed` 10000, the first of `eval_lift_native.py`'s
seeds); `envs="auto"` ranks them (`SO101Lift.rank`): episodes that succeeded and still hold the box above 0.10 m at
6 s first, then the smoothest, by the RMS of the six joints' accelerations (finite difference of the joint velocities
per control step), then the seed. In this take all 16 succeeded and still held the box at 6 s; the RMS joint
accelerations span only 67.3 to 70.3 rad/s^2 (every episode chatters), and seed 10013 (world 13) is the smoothest:
lifted at 0.76 s, box up to 0.162 m (`rec.json` `adapter.first_episode`). That list is the top four of the ranking;
both shots film the first.

**Camera and look.** Two fixed cameras in the env frame with Kit's lens (36 deg vertical), on the -y side of the arm,
so the box stays in view through the hold (the arm folds back and holds it out to its +y side; from +y the arm hides
it): `hero`, 1280x720, eye (0.62, -0.48, 0.36) m, looking at (0.12, 0.03, 0.10) m; `readme`, 640x480 (4:3, for the
README grid), eye (0.55, -0.42, 0.32), looking at (0.12, 0.03, 0.11). The render twin is the simulated scene with the
floor plane (z = 0, what the arm and box stand on) replaced by a 0.9 x 0.7 m wooden table top, 4 cm thick, on four
legs, with a grey studio floor 0.75 m below (render-only geoms; `SO101Lift.twin_spec`); Poly Haven's `wood_table_001`
on the top at its real size (1.5 m per image, `ground_tile_m`) and in its own colours, the box in so101Sim's green
(0.1, 0.9, 0.15), the legs dark metal (0.12, metallic 0.6). The film tool writes these look colours into the geoms' own
`rgba` (`render.build_scene`): the twin gives the box (0, 1, 0), the legs 0.2 and the top (0.6, 0.45, 0.3), and MuJoCo
draws a geom's own `rgba` over its material's (times the texture on the top).

**Rendering.** Tier 2 (the path tracer), 64 samples per pixel (16 passes x 4), 4 bounces, Open Image Denoise, ACES
tone map, sRGB, lit by the Poly Haven studio HDRI as an importance-sampled dome. One frame per 20 ms control step
(50 fps), 300 frames.

**README clip.** [so101_lift.webp](so101_lift.webp) is the `readme` shot from 0 to 5 s (frames 0 to 249), resampled
from 50 to 30 fps and written as MuJoCo Warp writes its README clips: 320x240 (Lanczos), lossy animated WebP from
Pillow at quality 70, 33 ms per frame, looping; 150 frames. Command, from the repository root after the render stage:
`python -m metalsim.render.film.encode webp runs/film/so101_lift/readme/frames docs/renders/so101_lift.webp --start 0
--seconds 5 --src-fps 50 --fps 30`.

**This take, and where its timings come from.** Recorded on 2026-10-31 on the 14-inch MacBook Pro (Mac16,6), at
metalsim `ed66d8ef` plus this film's uncommitted files (hence `+dirty` in the pins); rendered and encoded again on
2026-11-01 on the same machine, at metalsim `68f695b2` plus uncommitted files, from the same recording,
after the film tool started laying the wood at its real size and drawing the look's colours on geoms with their own
`rgba` (the frames rendered on 2026-10-31 had the wood 1.33 times too large and tinted (0.6, 0.45, 0.3), the box in
(0, 1, 0) and the legs at 0.2). This was not a timing run; quote nothing on this page as MetalSim's speed (MetalSim's
timing runs are made on the 16-inch through the GPU queue's timing kind; throughput figures are in `docs/PARITY.md`).
Power, as the stages stamped it: on AC power at every stamp (charging at 88 % when the record stage wrote `rec.json` at
14:53 on 2026-10-31; 100 % and finishing its charge at the shots' first frames, 00:34 and 00:36 on 2026-11-01, local
time), with pmset's power-mode setting at High Power (`powermode 2`: the configured setting, which does not show that
the machine ran in High Power Mode). Record and render ran as GPU-queue jobs of kind render, one at a time, into
`runs/film/so101_lift`. The entry is registered in `tasks.py` (`python -m metalsim.render.film so101_lift`, as below);
`python -m metalsim.render.film.so101 <film-tool args>` also runs the film tool with it. A new recording is another
take: the GPU rollout is not bitwise repeatable.
"""

PORT74 = "runs/film/so101_lift_74v/port"
SO101_74V_SECTION = f"""
## 7.4 V servo model

The realism check above finds the motion beyond the standard SO-101 follower's servos. Here the same policy, unchanged
and not retrained, runs on a model of those servos, and the `so101_lift_74v` take films it
(`python -m metalsim.render.film.so101 so101_lift_74v`; adapter option `servo="7v4"`; the default, `"12v"`, is the
scene as it is and everything above).

**The scene's servos and where their numbers come from.** `assets/so101/so101.xml`, class `sts3215` (Menagerie's copy
of SO-ARM100's `so101_new_calib.xml` at `aec17bbc`):

- position actuators, kp 998.22, kv 2.731, force limit +-2.94 N.m: Gregory119's DC-motor derivation for the 12 V
  ST-3215-C047 (30 kg.cm stall and 45 rpm at 12 V, servo P gain 16;
  <https://github.com/Gregory119/RBE501-RL-arm-project>, `gymnasium_env/calc_actuator_model.py` and its README),
  merged by SO-ARM100 PR #141 (2025-11-27), which replaced kp 17.8 and a 3.35 N.m limit;
- joint damping 0.60 N.m.s/rad, frictionloss 0.052 N.m, armature 0.028 kg.m^2: in SO-ARM100's SO-101 MJCF since its
  first version (2025-05-19). They match Rhoban BAM's MuJoCo export of its identified 7.4 V STS3215
  (<https://github.com/Rhoban/bam>, `bam/to_mujoco.py`: damping = viscous friction + kt^2 / R, frictionloss = Coulomb
  friction; `bam/params/feetech_sts3215_7_4V/m1.json`: 0.060 + 1.178^2 / 2.479 = 0.62, 0.052, armature 0.026), as
  do the replaced kp 17.8 and 3.35 N.m (the export's kp and vin kt / R: 18 and 3.5 with today's parameters). That
  origin is inferred from the numbers; no file states it.

MuJoCo clamps the servo's whole force (kp x error - kv x speed) to the force limit, so the actuator has no
speed-dependent limit of its own. Flat out, servo force minus joint damping is the line 2.94 - 0.60 w N.m, zero at
4.9 rad/s.

**What changes for 7.4 V** (`apply_servo` in `metalsim/render/film/so101.py`):

| | 12 V (the scene) | 7.4 V | source |
|---|---|---|---|
| force limit, five arm joints | 2.94 N.m | 1.912 N.m | Feetech datasheet "STS3215 200620a": 19.5 kg.cm at 7.4 V; the follower uses the 7.4 V C001 (SO-ARM100 README, bill of materials) |
| gripper force limit | 2.94 N.m | 0.956 N.m | LeRobot's SO-101 driver sets the gripper's `Max_Torque_Limit` to 500, i.e. 50 % (`lerobot/robots/so_follower/so_follower.py`, `configure`); the datasheet: torque output is the motor's voltage duty cycle, 1000 = 100 % |
| joint damping, six joints | 0.60 N.m.s/rad | 0.351 N.m.s/rad | 1.912 N.m / 5.445 rad/s (datasheet: 52 rpm at 7.4 V), so the flat-out line ends at the datasheet's no-load speed, as the 12 V model's ends at 4.9 rad/s |
| flat-out line reaches zero at | 4.9 rad/s | 5.45 rad/s; gripper 2.72 rad/s | half the duty cycle halves stall torque and no-load speed alike |
| kp, kv, armature, frictionloss | 998.22, 2.731, 0.028, 0.052 | unchanged | kp and kv need the motor's stall and no-load currents, which the C001 datasheet does not give; they act only within force limit / kp (2 mrad) of the target |

Nothing is added that the 12 V model lacks: no overload or over-current protection (LeRobot sets `Overload_Torque`
25 %), current limit, backlash or servo P loop. The damping is the one judgement call. In a DC motor the line's slope
(kt^2 / R) does not change with voltage, and BAM puts it at 0.62; but with the datasheet's stall torque, 0.60 would end
the line at 3.2 rad/s, below both the datasheet's 5.45 rad/s and BAM's 5.3 rad/s. The sensitivity row below keeps 0.60.

**Results**, 100 episodes, seeds 10000 to 10099, the policy unchanged:

- CPU: so101Sim's env on MuJoCo 3.10.0 (`{PORT74}/reference_servo.py`, which runs `{PORT}/reference.py` with the
  servo model applied to the env's MjModel);
- MetalSim: the film's adapter on MuJoCo Warp, one world per seed (`{PORT74}/compare_servo.py`, which runs
  `{PORT}/compare.py`; GPU queue);
- tables: `{PORT74}/analysis.py` (`analysis.json`).

| 100 episodes | 12 V, CPU | 7.4 V, CPU | 12 V, MetalSim | 7.4 V, MetalSim |
|---|---|---|---|---|
| success (box above 0.10 m for 10 steps) | 100 | 77 | 98 | 78 (re-runs: 73, 78) |
| box still above 0.10 m at 6 s | 99 | 70 | 94 | 65 (66, 71) |
| mean of the highest box height | 0.170 m | 0.148 m | 0.167 m | 0.147 m (0.140, 0.148 m) |
| time to lift, mean over successes | 0.95 s | 1.62 s | 0.97 s | 1.58 s (1.63, 1.64 s) |
| lifted, then dropped by 6 s | 1 | 8 | 4 | 13 |
| box never above 6 cm | 0 | 21 | 1 | 22 |
| same outcome as CPU, seed by seed | | | 98 | 81 (86, 81) |

The 12 V MetalSim column is a re-run with the servo forces recorded (another GPU take); the first run, in the
comparison above, gave 98, 94, 0.166 m and 0.91 s. Each MetalSim figure is one GPU draw (not bitwise repeatable). The
7.4 V re-runs in brackets are two more runs of `{PORT74}/compare_servo.py 7v4` on MuJoCo Warp `8eb51fe637` (the
column's run was on `07a51a63`; `runs/readme_media/handoff/revalidate.md`): **73 to 78 successes in 100 over two
re-runs** (78 in the first run; 75 in the next section's table, a fourth run). At 12 V the re-runs give 97 to 98 over
three runs (the comparison above).

Fraction of the episode each servo spends at its force limit:

| joint | 12 V, CPU | 7.4 V, CPU | 12 V, MetalSim | 7.4 V, MetalSim |
|---|---|---|---|---|
| shoulder pan | 99.7 % | 90.8 % | 98.9 % | 90.5 % |
| shoulder lift | 98.4 % | 96.9 % | 97.0 % | 95.0 % |
| elbow flex | 99.8 % | 99.9 % | 99.7 % | 99.8 % |
| wrist flex | 19.6 % | 53.4 % | 20.0 % | 55.0 % |
| wrist roll | 100 % | 100 % | 100 % | 100 % |
| gripper | 100 % | 98.6 % | 99.5 % | 98.4 % |

CPU: every 5 ms physics step (`{PORT}/realism.py`; `{PORT74}/realism_7v4.json`). MetalSim: the last physics step of
each 20 ms control step.

| CPU, 100 episodes | success | still up at 6 s | highest box, mean | time to lift |
|---|---|---|---|---|
| 7.4 V | 77 | 70 | 0.148 m | 1.62 s |
| 7.4 V, damping kept at 0.60 (sensitivity) | 76 | 74 | 0.139 m | 1.91 s |
| 7.4 V, gripper at the full 1.91 N.m (diagnostic, not the follower's setting) | 98 | 96 | 0.171 m | 1.03 s |

**What the failures look like** (CPU, `analysis.json`; MetalSim's counts are in the table):

- The arm is strong enough. With the gripper at full torque (diagnostic row), the 7.4 V arm lifts the box in 98
  episodes, and in failed episodes the gripper still rises to 0.20 to 0.29 m.
- The gripper is what fails. At half the torque and half the speed it opens slowly. When the jaws first touch the box
  (0.24 s, median, as at 12 V), the gripper is open 0.47 rad, against 0.90 rad at 12 V (median). The jaws land on the
  box instead of around it and tip it over: tilted beyond 60 deg while still on the table in 76 episodes, against 43
  at 12 V.
- In 21 episodes the box never rises above 6 cm. It lies on its side on the table, and the policy hovers over it,
  gripper open, and does not pick it up.
- When it does lift the box, the gripper squeezes with a quarter of the force: 27 N summed normal force during the
  hold (median), against 106 N. 8 boxes (13 on MetalSim) slip out of the jaws in the chattering hold before 6 s.
- So the failures come from the gripper's 50 % limit, LeRobot's setting, not from the 7.4 V arm.
- The policy stays bang-bang (66 % of actions at +-1, against 80 %). The joints move more gently: RMS joint
  acceleration 41 to 50 rad/s^2 in the take's 16 worlds, against 67 to 70 at 12 V.

**The take** (`runs/film/so101_lift_74v/`). It uses the so101_lift take's 16 worlds, look, cameras and shots, and
films world 13 (seed 10013), the 12 V take's env, whatever happens in it.

- 12 of the 16 first episodes succeed. In 10007, 10010, 10011 and 10012 the box never rises above 5 cm.
- World 13 succeeds; it is one of the 77 %, not a failure:
  - the gripper comes down over the box only part open;
  - the jaws straddle the box at about 0.4 s and close on it by 0.6 s;
  - the arm lifts the box from about 0.85 s; success is counted at 1.10 s, against 0.76 s at 12 V;
  - the arm holds the box at 0.14 to 0.15 m to the end, hanging tilted in the jaws (at 12 V it lies on its side).
- Clip: [so101_lift_74v.webp](so101_lift_74v.webp), made with the same command and window as so101_lift.webp (`--start
  0 --seconds 5 --src-fps 50 --fps 30`).
- Hero: [so101_lift_74v.mp4](so101_lift_74v.mp4) (0.8 MB) and `so101_lift_74v_still{{1,2,3}}.jpg` (frames 15, 30
  and 150).
- Recorded and first rendered on the 14-inch MacBook Pro (Mac16,6), 2026-10-31, on AC power and charging at every
  stamp (97 to 98 %, 15:17 to 15:19), powermode 2. The frames here were rendered again from that recording on
  2026-11-01 (00:37 to 00:39, AC power, 100 % and finishing its charge, powermode 2), at metalsim `68f695b2` plus
  uncommitted files, with the wood at its real size and the look's colours, as so101_lift's ("This
  take" above). This was not a timing run.

**Verdict.** On a model of the standard follower's servos, the unchanged policy succeeds 77 times in 100 on CPU (73 to
78 on MetalSim over four runs) instead of 100. It takes 1.6 s instead of 0.95 s to lift, and the box is still up at 6 s
in 65 to 71 episodes. The cause is the weak, slow gripper under LeRobot's 50 % limit. This is still a simulation: the servo's P
loop, current limit, overload protection and backlash are not modelled, and nothing has run on the arm.
"""

# Section "Smooth 7.4 V retrain" (the so101-smooth stream; numbers from runs/film/so101_lift_smooth/eval/table.md).
SO101_SMOOTH_SECTION = r"""
## Smooth 7.4 V retrain

The owner's goal for this cell is a realistic control style on the standard follower's 7.4 V servos, and a realistic
simulation that fails beats a working, unrealistic one. So so101Sim's teacher was retrained on the 7.4 V servo model
(`apply_servo(m, "7v4")`, section above) with smoothness penalties: two variants, then stop. The `so101_lift_smooth`
take films the second (`python -m metalsim.render.film.so101 so101_lift_smooth`). **Neither variant reaches a realistic
control style.**

**How control style is measured.** Torque saturation does not separate a smooth controller from a bang-bang one on
these servos: so101Sim's scripted expert (`rl/native/scripted_expert.py`, damped-least-squares IK with each joint
target moving at most 0.06 rad per step; the controller that generated so101Sim's demonstrations) sits at the force
limit 39 to 90 % of the time at 7.4 V, because the servo's proportional band is only force limit / kp = 1.9 mrad. Two
measures do separate them, both on the action the env applies (normalised to [-1, 1]):

- action change per step, the mean of |a(t) - a(t-1)| over the six joints (scripted_expert.py's "mean |dctrl|/tick");
- actions at the limit, the share of action components with |a| > 0.99.

The expert at 7.4 V gives 0.0139 and 0.9 %; the original policy gives 0.68 and 66 %.

**Training, both variants.** The run-2 teacher continued for 10 M steps on the 7.4 V model, with everything else
so101Sim's: `SO101LiftEnv`'s reward, reset and loop, 12 workers, VecNormalize continued from run 2, and run 2's PPO
settings (`runs/film/so101_lift_smooth/train/train_smooth.py`, `smooth_env.py`; run in so101Sim's environment, about
16 minutes each on CPU; logs in `train/runs/<variant>/`). The penalties are Isaac Lab's lift task's
(IsaacLab `source/isaaclab_tasks/isaaclab_tasks/contrib/lift/lift_env_cfg.py`, `RewardsCfg` `action_rate` and
`joint_vel`, `CurriculumCfg`):

- `action_rate_l2`, the sum over the action of (a(t) - a(t-1))^2;
- `joint_vel_l2`, the sum over the joints of qdot^2;
- weights 1e-4 each, raised to 1e-1 after 10,000 of the 36,000 environment steps of its rsl_rl run
  (`config/franka/agents/rsl_rl_ppo_cfg.py`: 1500 iterations x 24 steps); here after 27.8 % of the 10 M steps.

**Variant 1, `isaac_scaled`: no effect.** Isaac Lab's weights times 9/37 (so101Sim's largest task reward per step over
Isaac Lab's), penalising the action after the [-1, 1] clip. The action change per step rose from 0.68 to 0.76 and the
actions at the limit from 66 to 78 %. SB3's exploration noise for wrist flex and wrist roll grew from 0.92 and 2.4 to 16 and 103.
The cause (`runs/film/so101_lift_smooth/eval/raw_actions.py`): the deterministic action is the policy's Gaussian mean
clipped to [-1, 1], and that mean lies outside [-1, 1] on 45 to 86 % of steps per joint for the original policy (median
|mean| 20 for variant 1's wrist flex). A penalty on the clipped action cannot see a mean beyond the clip, nor noise
beyond it. Isaac Lab penalises the raw action: its `ActionManager.process_action` stores the action as the policy
emits it (`source/isaaclab/isaaclab/managers/action_manager.py`), and its rsl_rl wrapper clips nothing unless
`clip_actions` is set (`source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`, default `None`; the lift task does not
set it). Isaac Lab files read at `main` = `6bb65480`.

**Variant 2, `isaac_unscaled_raw`, the filmed one.** Isaac Lab's weights unscaled, on the raw policy output as Isaac
Lab computes them. SB3 passes the Gaussian sample through unclipped (action space widened to +-1000); so101Sim's
`step` still clips it to [-1, 1] for the servo targets, so the arm sees what it saw before. The other candidate,
LeRobot's `max_relative_target`, was not used, for three reasons:

- it is off by default on the SO-101 follower (`None`, `lerobot/robots/so_follower/config_so_follower.py`, lerobot
  0.6.1);
- typical values are 5 degrees (the old Aloha default, `lerobot/common/robot_devices/robots/configs.py` at
  `3c0a209f`) or 10 to 15 degrees (Koch, in LeRobot PR #373's first revision);
- it clamps each goal to the present position +- the limit (`lerobot/robots/utils.py`, `ensure_safe_goal_position`).
  5 degrees is 87 mrad, 45 times the servo's 1.9 mrad proportional band, so the policy's output would be unchanged
  and the servo torque would still switch between its limits.

**Results**, 100 episodes, seeds 10000 to 10099, 7.4 V unless marked, CPU | MetalSim:

| controller | success | box up at 6 s | time to lift | action change per step | actions at the limit | servo at force limit, arm mean / gripper | peak joint speed |
|---|---|---|---|---|---|---|---|
| scripted expert | 20 \| 36 | 17 \| 29 | 4.13 \| 4.20 s | 0.0139 \| 0.0142 | 0.9 \| 0.7 % | 70 / 90 \| 59 / 95 % | 5.19 \| 5.16 rad/s |
| original policy | 77 \| 75 | 70 \| 68 | 1.62 \| 1.50 s | 0.682 \| 0.696 | 65.7 \| 66.3 % | 88 / 99 \| 88 / 98 % | 5.87 \| 6.33 rad/s |
| variant 1, `isaac_scaled` | 77 \| 77 | 75 \| 73 | 1.51 \| 1.65 s | 0.756 \| 0.761 | 77.5 \| 77.0 % | 85 / 99 \| 86 / 100 % | 5.98 \| 6.23 rad/s |
| variant 2, `isaac_unscaled_raw` | 86 \| 92 | 83 \| 85 | 1.36 \| 1.34 s | 0.348 \| 0.355 | 72.4 \| 72.3 % | 72 / 100 \| 72 / 100 % | 6.15 \| 5.96 rad/s |
| original policy, 12 V | 100 \| 98 | 99 \| 97 | 0.95 \| 0.95 s | 0.991 \| 0.982 | 79.6 \| 79.2 % | 83 / 100 \| 83 / 100 % | 5.65 \| 5.53 rad/s |
| scripted expert, 12 V | 95 \| 98 | 86 \| 94 | 1.60 \| 1.37 s | 0.0051 \| 0.0043 | 0.2 \| 0.1 % | 24 / 90 \| 14 / 94 % | 5.39 \| 5.00 rad/s |

How each column is measured:

- **CPU:** so101Sim's env on MuJoCo 3.10.0 with the servo model applied (`eval/eval_cpu.py`). Force and speed are
  read at every 5 ms physics step.
- **MetalSim:** the film's adapter on MuJoCo Warp, one world per seed (`eval/eval_metalsim.py`, GPU queue, MuJoCo
  Warp fork `8eb51fe637`, 2026-10-31 16:13 to 16:14). Force and speed are read at the end of each 20 ms control step.
  The expert runs here on CPU kinematics of each world's state.
- **Tables:** `eval/table.py`, giving `eval/table.md` and `eval/results.json`. The same jobs run at 15:58 gave
  successes within 3 of these: MetalSim is not bitwise repeatable from run to run.
- The section above's MetalSim column for the unchanged policy (78 and 65 at 7.4 V) was measured at 15:17, on
  the earlier fork checkout `07a51a63` (moved to `8eb51fe6` at 15:47, `upstream/mujoco_warp` reflog), with
  `compare_servo.py`; this table's 75 and 68 are from `8eb51fe637`.

**Verdict.**

- **Neither variant reaches a realistic control style.**
- Variant 2 halves the original's action change per step, from 0.68 to 0.35. Its joint targets no longer flip every
  step: the median target change is 0 rad, against 0.67 rad. Instead the policy holds its targets at the joint limits
  and lets the saturated servos make the motion.
- That is still 25 times the expert's action change, and 72 % of its actions sit at +-1, against 0.9 % for the expert.
  The elbow still flips (0.89 per step).
- It costs no success. Variant 2 succeeds more often than the original on the 7.4 V model: 86 against 77 on CPU, 92
  against 75 on MetalSim. It lifts sooner (1.36 s against 1.62 s) and higher (0.185 m against 0.148 m mean highest
  box). The box never rises above 6 cm in 11 episodes, against 21.
- The only realistic controller here is so101Sim's scripted expert, and on the 7.4 V model with LeRobot's 50 %
  gripper limit it succeeds 20 times in 100 on CPU (36 on MetalSim). In 77 of its CPU episodes the box never rises
  above 6 cm.

**The take** (`runs/film/so101_lift_smooth/`): the so101_lift take's 16 worlds, look, cameras and shots, with the
variant-2 weights (`runs/film/so101_lift_smooth/policy/so101_lift_smooth_ppo.pt`, sha256 `4593a0ed...`; extraction
checked against SB3's predict clipped to [-1, 1]: 1.7e-6).

- All 16 first episodes succeed. Their RMS joint acceleration is 28.8 to 41.2 rad/s^2, against 67 to 70 at 12 V and
  41 to 50 for the unchanged policy at 7.4 V.
- The ranking films world 8 (seed 10008), lifted at 0.86 s:
  - the gripper comes down over the box with the jaws open, the fixed jaw meets the box and tips it (0.3 s);
  - the jaws close around it (0.6 s);
  - the arm swings the box up and back over its base by about 1.2 s and holds it there, at about 0.21 m and upright
    in the jaws, to the end;
  - during the hold the wrist and box still turn back and forth by a few degrees from frame to frame.
- Clip: [so101_lift_smooth.webp](so101_lift_smooth.webp), the same command and window as so101_lift.webp.
- Hero: [so101_lift_smooth.mp4](so101_lift_smooth.mp4) (0.7 MB) and `so101_lift_smooth_still{1,2,3}.jpg` (frames
  15, 30 and 150).
- Recorded and first rendered on the 14-inch MacBook Pro (Mac16,6), 2026-10-31, 16:14 to 16:17, with MuJoCo Warp at
  `8eb51fe637`, **on battery** (58 to 55 %, discharging), powermode 2. The frames here were rendered again from that
  recording on 2026-11-01 (00:39 to 00:42, AC power, 100 % and finishing its charge, powermode 2), at metalsim
  `68f695b2` plus uncommitted files, with the wood at its real size and the look's colours, as
  so101_lift's ("This take" above). This was not a timing run.

Still a simulation: the servo's P loop, current limit, overload protection and backlash are not modelled, and nothing
has run on the arm.
"""

SO101_SERVO_SECTION = r"""
## Servo model under LeRobot's settings (7v4_real, 7v4_bam)

The owner watched the 7.4 V model above and the motion is wrong: each joint rings about every target, and the arm
moves faster than the servos can. The cause is the scene's actuator, not the policies. `apply_servo(m, "7v4_real")`
(and `"12v_real"`) models the STS3215 as LeRobot drives it; the old options are unchanged. `"7v4_bam"` (the last part of
this section) is the same servo under the same registers with every motor and friction number from Rhoban BAM's
identification, including the gearbox friction that grows with the load. It holds a gravity load far closer to real
followers than 7v4_real does (the shoulder still sags 1.6-2.4 times as much as theirs), and it is the servo of the
`so101_lift_deploy` film.

**What the real servo does.**

- **The firmware loop is soft.** The STS3215 sets its PWM duty to P x position error, about 0.167 duty per rad per
  unit of P, clipped at 0.97. That is Rhoban BAM's identification of the 7.4 V STS3215 with an oscilloscope and a
  pendulum, at P = 4, 8, 16 and 32 with D = 0 (<https://github.com/Rhoban/bam>, `bam/feetech/actuator.py`
  `error_gain` 0.166 and `max_pwm` 0.97, `bam/params/feetech_sts3215_7_4V/m1.json` `error_gain_ratio` 1.005,
  `bam/feetech/record.py`). A third-party STS memory table gives the same gain, "Kp scaled by 1/4": 0.25 per mille of
  duty per encoder step, 0.163 per rad. LeRobot writes P 16, I 0 and D 32 (`so_follower/config_so_follower.py`
  `position_*_coefficient`, `so_follower.py` `configure`).
- **What that means as torque.** At P 16 a 0.06 rad error gives 16 % duty, 0.32 N.m on the 7.4 V servo. The scene's
  kp of 998 N.m/rad comes from Gregory119's derivation, which reads P 16 as 16 V per degree of error, with an assumed
  1 ohm winding. That is about 190 times too stiff, and with only 3.3 N.m.s/rad of damping it rings. SO-ARM100's
  SO-101 model before PR #141 had kp 17.8, which matches BAM's own MuJoCo export at the factory P of 32 (18). That
  match is inferred from the numbers only.
- **The firmware moves its own target.** The loop tracks an internal target that moves toward the goal at no more than
  5.29 rad/s (BAM `max_velocity`, m1). LeRobot also writes Acceleration 254 (`motors/feetech/feetech.py`
  `configure_motors`). At 100 step/s^2 per unit (third-party memory tables) that is 39 rad/s^2.

  Two things support that this register acts in position mode when Goal_Velocity is 0. The datasheet says "Mode 0 ...
  support acceleration movement". LeRobot's first SO-100 calibration (commit 07e87163) wrote Acceleration 10 "to lower
  the acceleration of the motors" before position moves. That it acts here, and the shape of the profile, are
  inference: no source states either.
- **The motor.** The torque-speed line runs through the datasheet's stall torque and no-load speed, friction
  included:

  | | 7.4 V C001 | 12 V C018 |
  |---|---|---|
  | stall torque | 19.5 kg.cm | 30 kg.cm |
  | no-load speed | 52 rpm | 0.222 s/60 deg |

  The back-EMF and viscous friction are a joint damping that acts outside the duty clip, as back-EMF does. MuJoCo then
  produces the four-quadrant line itself. The scene's actuator clamps its whole force, back-EMF term included, so its
  line came from the joint damping alone. Coulomb friction 0.0515 N.m and armature 0.026 kg.m^2 are BAM m1's.

  BAM's own kt/R put the 7.4 V stall at 3.41 N.m, 1.8 times the datasheet. A replay fit on real SO-101 teleop
  episodes (Fontaine blog, 2026-08-11) lands at kp 8.8 and +-1.97 N.m. That is between this model's 5.26 (datasheet
  torque) and BAM's 9.38.
- **Latency.**
  - Servo side: 1.05 ms (BAM m1 `command_delay`). This covers the bus write, the firmware loop and the current
    build-up. The motor's L/R and the 16 or 24 kHz PWM are far below the 5 ms physics step, so torque build-up has no
    model of its own.
  - Host side: 3 ms, for one sync_read and one sync_write of six servos at 1 Mbps through a USB serial adapter. This
    is inference from protocol arithmetic; no published measurement was found.

  The env keeps its 50 Hz control. LeRobot records at 30 fps, which would change the task, not the servo.
- **Not modelled.**
  - D 32: its scale needs the firmware's loop period, which is unpublished. The sensitivity run below bounds it.
  - The gripper's overload protection (LeRobot's Overload_Torque 25 and Protection_Current 250): the datasheet,
    LeRobot's comment and the third-party tables disagree on what the registers do.
  - The 1-step dead zone, the 1.5 mrad encoder step and backlash.

| | kp N.m/rad | force limit N.m | joint damping | firmware target limits | latency |
|---|---|---|---|---|---|
| old 7.4 V (`7v4`) | 998.22, kv 2.731 | 1.912, gripper 0.956 | 0.351 | none | none |
| **`7v4_real`** | 5.26, kv 0 | 1.912, gripper 0.986 (50 % duty) | 0.342 | 5.29 rad/s, 39.0 rad/s^2 | 1.05 + 3 ms |
| `12v_real` | 8.09, kv 0 | 2.942, gripper 1.517 | 0.613 | 5.29 rad/s, 39.0 rad/s^2 | 1.05 + 3 ms |

The layer between the command and `d.ctrl` is `metalsim/physics/actuation.py` (`SetpointLayer`): a fractional command
delay, then the firmware profile. It is open loop in the command, so `SO101Lift` plans each 20 ms period's four targets
on the host and feeds them to the GPU with a substep hook (`CtrlPlanFeeder`). Against CPU MuJoCo with the same layer
and a fixed command schedule, the GPU joint trajectories agree to 8e-7 rad. Holding the first target for all four steps
instead differs by up to 0.09 rad.

**Step response.** The orchestrator's protocol, reproduced for the old model within 4 points of overshoot: so101Sim's
scene, every joint at qpos0, targets equal to positions, 1 s settling, then one joint's target +0.06 rad (the expert's
largest step per tick). Overshoot and crossings are against the final value, with a 0.5 % hysteresis. "Sag" is the
gravity offset before the step.

| servo | joint | overshoot | crossings | settle to 2 % | 90 % at | sag | peak speed / acc, 0.06 rad | peak, 0.5 rad step |
|---|---|---|---|---|---|---|---|---|
| old 12 V | shoulder_lift | 52 % | 6 | 190 ms | 40 ms | 1 mrad | 2.31 rad/s, 96 rad/s^2 | 3.78, 121 |
| old 12 V | elbow_flex | 52 % | 5 | 170 ms | 40 ms | 0 | 2.47, 113 | 3.81, 144 |
| old 12 V | wrist_flex | 33 % | 5 | 120 ms | 35 ms | 0 | 2.49, 130 | 4.39, 180 |
| old 7.4 V | shoulder_lift | 89 % | 6 | 315 ms | 50 ms | 1 mrad | 2.09, 68 | 3.37, 77 |
| old 7.4 V | elbow_flex | 72 % | 6 | 240 ms | 45 ms | 0 | 2.26, 78 | 3.48, 99 |
| old 7.4 V | wrist_flex | 52 % | 5 | 190 ms | 45 ms | 0 | 2.32, 87 | 4.31, 117 |
| 12v_real | shoulder_lift | 0 % | 0 | 275 ms | 235 ms | 79 mrad | 0.35, 6 | 3.04, 28 |
| 12v_real | elbow_flex | 0 % | 0 | 250 ms | 215 ms | 61 mrad | 0.38, 7 | 3.05, 29 |
| 12v_real | wrist_flex | 0 % | 0 | 240 ms | 200 ms | 13 mrad | 0.45, 9 | 3.09, 31 |
| **7v4_real** | shoulder_lift | 0 % | 0 | 310 ms | 275 ms | 121 mrad | 0.31, 5 | 2.81, 22 |
| **7v4_real** | elbow_flex | 0 % | 0 | 280 ms | 245 ms | 96 mrad | 0.31, 5 | 2.95, 26 |
| **7v4_real** | wrist_flex | 0 % | 0 | 255 ms | 220 ms | 21 mrad | 0.41, 7 | 3.24, 30 |

Two published hardware figures check the soft loop:

- lerobot issue #3400 measures a mean settle error of 63 encoder steps (97 mrad) at P 16, against 26.7 at P 32. The
  model's sag is 21-121 mrad, inversely proportional to P. The old model's is 1 mrad.
- The firmware profile caps the target's acceleration at 39 rad/s^2. The model's peak joint accelerations on a
  0.5 rad step are 22-30 rad/s^2; the old model's are 77-180.

**Ablation**, cumulative, then one piece at a time off the full model. The step columns give overshoot and settle;
the last column is the wrist on a 0.5 rad step. The task columns are CPU runs over seeds 10000-10099: so101Sim's
original PPO policy, trained on the old model, and its velocity reversals per second (arm joints).

| 7v4_real variant | lift step | elbow step | wrist step | wrist 0.5 rad: speed, acc | PPO success | PPO reversals /s |
|---|---|---|---|---|---|---|
| firmware gain + motor line only | 0.4 %, 270 ms | 0.3 %, 235 ms | 0.3 %, 215 ms | 3.54, 60 | 86 | 6-32 |
| + (a) servo latency 1.05 ms | 0.4 %, 270 ms | 0.3 %, 240 ms | 0.3 %, 220 ms | 3.54, 60 | 79 | 6-28 |
| + (b) firmware speed + acceleration limits | 0.4 %, 310 ms | 0.3 %, 280 ms | 0.2 %, 255 ms | 3.24, 30 | 11 | 2.0-5.5 |
| + (c) host latency 3 ms (= 7v4_real) | 0.4 %, 310 ms | 0.3 %, 280 ms | 0.2 %, 255 ms | 3.24, 30 | 6 | 2.1-5.9 |
| without (d), the back-EMF line | 33 %, 580 ms | 44 %, 510 ms | 49 %, 460 ms | 5.04, 62 | 1 | 3.0-5.5; peak 16-18 rad/s |
| without the acceleration limit | 0.4 %, 275 ms | 0.3 %, 245 ms | 0.3 %, 225 ms | 3.48, 40 | 24 | 4.1-6.6 |
| D 32 at an assumed 10 ms loop | 0.0 %, 320 ms | 0.0 %, 285 ms | 0.0 %, 265 ms | 3.03, 28 | | |
| BAM's kt/R (stall 3.41 N.m) | 0.7 %, 250 ms | 0.5 %, 230 ms | 0.0 %, 215 ms | 3.31, 33 | | |
| host latency 10 ms | 0.3 %, 320 ms | 0.3 %, 290 ms | 0.2 %, 265 ms | 3.24, 30 | 9 | 2.3-6.2 |

What each piece contributes:

- **The firmware gain** removes the ringing.
- **(d), the back-EMF line,** removes the rest of the overshoot. It also keeps speeds physical: without it the PPO
  policy reaches 16-18 rad/s.
- **(b), the firmware profile,** halves the peak acceleration of large steps. It also turns the PPO policy's 25 Hz
  command flipping (23-32 reversals per second on gains alone) into slow swings.
- **(a) and (c), the latencies,** change nothing visible at these time scales.
- **The D term** at any plausible loop period, and BAM's stronger motor, change the step by at most a few points and
  50 ms.

**Joint-motion smoothness** of every 5 ms physics step, CPU, seeds 10000-10099, per joint (pan / lift / elbow / wrist
flex / wrist roll / gripper):

- RMS acceleration and RMS jerk, by finite differences of qvel;
- velocity reversals per second, counted between stretches faster than 0.05 rad/s;
- the RMS joint velocity above 5 Hz, from an FFT per episode. The task's motion lies below a few Hz; the 5 Hz edge is
  a judgement.

The expert on `7v4_real` is the reference (no published values exist).

| controller | servo | success | RMS acc rad/s^2 | RMS jerk 10^3 rad/s^3 | reversals /s | velocity RMS > 5 Hz, rad/s |
|---|---|---|---|---|---|---|
| scripted expert | old 7.4 V | 20 | 35 / 47 / 47 / 55 / 49 / 19 | 7.0 / 8.4 / 8.5 / 9.9 / 10 / 2.3 | 13 / 7.8 / 8.1 / 9.7 / 13 / 3.0 | 0.42 / 0.57 / 0.54 / 0.72 / 0.51 / 0.27 |
| scripted expert | **7v4_real** | 0 | 4.9 / 8.5 / 11 / 8.5 / 2.7 / 4.1 | 1.4 / 2.5 / 3.3 / 2.5 / 0.4 / 0.2 | 0.19 / 0.29 / 0.34 / 0.82 / 0.21 / 0.17 | 0.03 / 0.05 / 0.07 / 0.05 / 0.02 / 0.03 |
| original PPO | old 7.4 V | 77 | 54 / 55 / 49 / 45 / 60 / 16 | 10 / 10 / 9.1 / 8.4 / 11 / 2.1 | 27 / 26 / 22 / 11 / 27 / 2.5 | 0.49 / 0.50 / 0.49 / 0.48 / 0.55 / 0.19 |
| original PPO | **7v4_real** | 6 | 60 / 77 / 68 / 61 / 36 / 21 | 16 / 21 / 19 / 16 / 6.2 / 3.2 | 3.6 / 5.9 / 5.6 / 4.3 / 2.1 / 2.2 | 0.42 / 0.54 / 0.46 / 0.46 / 0.28 / 0.19 |
| isaac_scaled | old 7.4 V | 77 | 57 / 51 / 48 / 36 / 61 / 16 | 11 / 9.9 / 9.2 / 7.0 / 11 / 2.2 | 27 / 26 / 25 / 7.1 / 29 / 3.0 | 0.54 / 0.45 / 0.45 / 0.38 / 0.56 / 0.17 |
| isaac_scaled | **7v4_real** | 15 | 54 / 72 / 67 / 57 / 37 / 22 | 14 / 19 / 18 / 15 / 6.1 / 3.8 | 3.4 / 5.2 / 5.8 / 4.8 / 2.5 / 2.1 | 0.39 / 0.51 / 0.47 / 0.44 / 0.27 / 0.21 |
| isaac_unscaled_raw | old 7.4 V | 86 | 51 / 37 / 51 / 37 / 41 / 15 | 9.2 / 7.6 / 10 / 7.4 / 7.1 / 2.3 | 16 / 8.6 / 29 / 5.5 / 3.9 / 3.1 | 0.58 / 0.37 / 0.46 / 0.39 / 0.45 / 0.17 |
| isaac_unscaled_raw | **7v4_real** | 4 | 56 / 68 / 61 / 56 / 35 / 22 | 15 / 18 / 16 / 14 / 6.1 / 4.1 | 3.1 / 5.0 / 4.7 / 4.6 / 2.4 / 2.0 | 0.39 / 0.48 / 0.41 / 0.41 / 0.26 / 0.20 |

What the table shows:

- **Old model.** The expert rings: 8-13 reversals per second and 0.4-0.7 rad/s above 5 Hz, at RMS accelerations of
  35-55 rad/s^2. That is what the owner saw. The "smoothness" retrain changed none of it.
- **New model, expert.** Reversals fall 15-60-fold, high-frequency velocity about 10-fold, and RMS acceleration
  4-18-fold.
- **New model, PPO policies.** All three were trained on the old servo; on the new one they no longer chatter at
  25 Hz (2-6 reversals per second). They still drive the joints at 35-77 rad/s^2 RMS, 7-15 times the expert, and
  succeed 4-15 times in 100.

Their peak speeds exceed the 5.45 rad/s no-load speed on 8-14 % of steps (6.2-8.4 rad/s), where the old model's do
on 0.1 %. With gravity and contacts off, bang-bang commands still reach 5.6-6.0 rad/s on the coupled joints, while
wrist roll alone stops at 5.44 (`speed_check.py`). The motor line holds per joint; gravity and the other joints'
inertial coupling drive a joint past it, as they would back-drive a real servo.

**The scripted expert on 7v4_real: 0/100**, seeds 10000-10099, on CPU and on MetalSim, with the same outcome on all
100 seeds. On 80 of them it never leaves the approach phase:

- so101Sim's expert sets each target to the measured joint position plus at most 0.06 rad;
- at P 16 that is at most 0.32 N.m, and holding the arm needs 0.39-0.57 N.m at the shoulder (`qfrc_bias`, the
  `pickup` keyframe and qpos0);
- so the arm sinks until the gripper is near the table, then creeps into a folded pose where gravity needs less.

The real follower under LeRobot (I = 0) would do the same with this expert. The 12 V model also gives 0/100. On
7v4_bam, whose kp is 10.58 N.m/rad, 0.06 rad gives 0.63 N.m; so101Sim's expert has not been run on it. A
deployment-style version of the expert, which streams absolute goals as LeRobot does, succeeds on 7v4_real and on
7v4_bam (next section).

A diagnostic variant builds each target on its previous command instead of the measured position, capped at the
0.363 rad where P 16 saturates the duty. It is not so101Sim's controller (`expert_variants.py`). It reaches the lift
phase in 45/100 episodes but succeeds in only 3. Its jaws are still open (1.0 rad) when the expert's fixed 0.3 s grasp
phase ends, because the gripper servo is slower than the old model's.

**The CPU-vs-MetalSim gap on the old model (20 vs 36) is the evaluation harness, not the simulator.** so101Sim's
expert on CPU reads `env.data` after `mj_step`, so its kinematics (TCP, Jacobian, box pose) are one 5 ms physics step
older than `qpos`. MetalSim's evaluation runs `mj_forward` on each world's state first. With `mj_forward` before the
expert on CPU too, the old model gives 38/100 against MetalSim's 34/100 (this run). The two now agree on 66 seeds,
against 56 before. The stale kinematics matter on the old model because the ringing joints move fast. On 7v4_real both
backends give 0/100.

Preview, MuJoCo's renderer, real time, frame time from `d.time`:

- `runs/readme_media/scratch/ik_preview/expert_7v4_real.mp4`: seeds 10010 and 10000;
- `expert_old_vs_real.mp4`: seed 10010, old model left (lifts at 3.36 s), 7v4_real right.

Scripts and outputs: `runs/readme_media/scratch/so101-servo/` (`stepresp.py`, `eval_cpu.py`, `eval_metalsim.py`,
`gpu_feeder_check.py`, `speed_check.py`, `phases.py`, `smoothness.py`). That directory is not tracked.

### 7v4_bam: BAM's identified servo, with load-dependent gearbox friction

**Why.** Real followers hold a gravity load differently from 7v4_real. Two LeRobot teleoperation datasets of SO-101
followers were measured (`hungho77/so101_left_arm_pick_red_cube_into_pot` and `IlyaGusev/so101_pick_gum`). The steady
error there is action minus state, the follower's own Goal_Position minus Present_Position, so the leader's calibration
does not enter it. A joint stops much nearer its goal after moving with gravity than after moving against it:

| joint | after moving with gravity | after moving against gravity |
|---|---|---|
| shoulder_lift | 5-28 mrad | 26-98 mrad |
| elbow_flex | 7-18 mrad | 55-82 mrad |

That is a static friction band that widens with the load. 7v4_real's band is a constant +-9.8 mrad around gravity / kp
5.26. Its shoulder, which in the datasets mostly stops after lowering, sagged 5-6 times more than the real ones; its
elbow, which mostly stops after rising, agreed only by that coincidence.

**What.** `apply_servo(m, "7v4_bam")` takes every motor and friction number from BAM's m6 identification of the
7.4 V STS3215 (<https://github.com/Rhoban/bam> at `b9e1111`, `bam/params/feetech_sts3215_7_4V/m6.json`; BAM's
best-ranked friction model; paper arXiv 2410.08650). Nothing was fitted to the sag data.

| | 7v4_real | 7v4_bam |
|---|---|---|
| full-duty torque | 1.971 N.m (datasheet stall / 0.97) | 3.43 N.m (kt x 7.4 V / R) |
| kp, N.m/rad (P 16) | 5.26 | 10.58 (m6's `error_gain_ratio` 1.162) |
| force limit (duty clip 0.97), N.m | 1.912, gripper 0.986 | 3.32, gripper 1.71 (LeRobot's 50 %) |
| joint damping (back-EMF + viscous) | 0.342 | 0.619 (kt^2 / R 0.591 + 0.028) |
| no-load speed | 5.45 rad/s (datasheet, 52 rpm) | 5.29 rad/s |
| gearbox friction | constant Coulomb 0.0515 N.m (m1) | m6's load-dependent budget (below) |
| armature | 0.0261 | 0.0216 |
| firmware speed cap | 5.29 rad/s | 5.10 rad/s |
| servo-side delay | 1.05 ms | 4.98 ms |
| registers (P 16, Acceleration 254 = 39 rad/s^2, gripper 500), 3 ms host latency | yes | the same |

**The friction.** BAM's m6 budget per joint is Coulomb 0.053 N.m, a Stribeck term, and terms in the motor torque tau_m and
the external torque tau_e (gravity, contacts, joint limits): |0.244 tau_e - 0.045 tau_m|, |0.142 tau_e - 0.0001 tau_m|
at rest, and a quadratic term (`bam/model.py` `compute_frictions`; our copy equals it exactly on 20,000 random inputs).

- It is written to `dof_frictionloss` before every physics step from the previous step's tau_m and tau_e, as BAM's own
  MuJoCo loop does (`bam/mujoco.py`).
- The friction constraint is BAM's stiff one (solref -5e4, -2e2; solimp 0.99 0.9999 0.001 0.5 2; `bam/mjlab.py`).
  With MuJoCo's default soft one, a held shoulder creeps 11 mrad in 2.5 s.
- On CPU MuJoCo the update is `metalsim.physics.actuation.BamFriction` (`friction_hook(applied, m)`, `update(d)` before
  every `mj_step`).
- On MuJoCo Warp it is `BamFrictionWarp`: a Warp kernel over (world, arm joint), run after every physics substep inside
  BatchSim's step graph, writing a per-world `dof_frictionloss`. `SO101Lift` wires it. Against the CPU update (8
  seeds, 40 control steps) the joint positions agree to 1.9e-6 rad (`tests/test_bam_friction.py` has the kernel and
  GPU checks).
- At reset, the first budget comes from a forward pass with the servos powered on, holding the start pose.

**Against the real arms.** The datasets' own commands were replayed on the simulated follower: free air, each episode
starting from the real follower's state. Median |sim - real| steady error per load bin, mrad:

| joint | 7v4_real | 7v4_bam |
|---|---|---|
| shoulder_lift | 93 | 17 |
| elbow_flex | 9.3 (after moving with gravity: 62) | 12 (14) |
| wrist_flex | 1.9 | 1.1 |

In the heaviest fifth of the shoulder's steady samples, the simulated shoulder sags 1.6 and 2.4 times as much as the
real ones in the two datasets (7v4_real: 5.3 and 6.7 times).

**Open.**

- **The residual is not fitted, on purpose.** The shoulder still sags 1.6-2.4 times real. The real elbow's band is about
  1.7 times wider, and without load the real band is 1.5-3 times wider. That is consistent with more friction on the
  real units than on BAM's one bench servo, and with the MJCF's CAD masses (not weighed); teleop data cannot separate
  the two.
- **The stall torque is unresolved: 3.32 N.m from BAM's kt / R, against the datasheet's 1.91 (19.5 kg.cm).** The sag
  data constrain only the small-signal kp. With BAM's kp and damping kept and the clip set to the datasheet's value, the
  deployment-style expert (next section) still succeeds 100/100 on CPU but lifts later on every seed (median time to
  lift 2.52 s against 2.18) and retries 6 times against 1; its sag in the lift phase is unchanged. A bench test would
  settle it: block a joint, step its goal, read the force at a known lever.
- No 12 V version: BAM has no 12 V identification of the STS3215.

Scripts and outputs: `runs/readme_media/scratch/so101-sag/` (`real_dir.py`, `static_band.py`, `replay.py`,
`compare.py`, `hold_test.py`, `check_budget.py`, `eval_cpu_sag.py`, `table.md`) and
`runs/readme_media/scratch/so101-bam-gpu/` (`gpu_bam_check.py`, `eval_metalsim_bam.py`, `table.md`). Those
directories are not tracked.
"""
SO101_DEPLOY_SECTION = r"""
## Deployment-style scripted expert on 7v4_bam

The section above leaves no realistic controller on the standard follower's servos as LeRobot sets them: the PPO
policies still drive the joints at 35-77 rad/s^2 RMS and succeed 4-15 times in 100 on `7v4_real`, and so101Sim's
scripted expert, which sets each target at most 0.06 rad beyond the *measured* joint position, sinks under gravity and
succeeds 0 times. The `so101_lift_deploy` take films a deployment-style version of that expert, `DeployExpert`
(`metalsim/render/film/so101_expert.py`), on `7v4_bam`, the servo model with BAM's identified motor and load-dependent
gearbox friction (the last part of the section above): `python -m metalsim.render.film so101_lift_deploy` (adapter
options `servo="7v4_bam"`, `controller="deploy_expert"`). The same expert also runs on `7v4_real`
(`servo="7v4_real"`); its numbers stay below for comparison, and its earlier take was replaced by this one. **It is
scripted, not learned**, and it reads the cube's pose from the simulator, not from a camera.

**How LeRobot commands a real SO-101** (LeRobot at `bf31dd79`):

- every control tick the host writes *absolute* goal positions: `SOFollower.send_action`
  (`src/lerobot/robots/so_follower/so_follower.py`) calls `bus.sync_write("Goal_Position", ...)`, and
  `max_relative_target` is `None` by default (`config_so_follower.py`), so nothing clamps a goal to the present
  position;
- at 30 per second: `DatasetRecordConfig.fps` (`lerobot-record`, `configs/dataset.py`) and the rollout's `fps`
  (`lerobot-rollout`, `rollout/configs.py`) both default to 30;
- the servo's own loop does the rest (P 16, I 0, D 32, Acceleration 254, gripper `Max_Torque_Limit` 500: the section
  above);
- nothing in LeRobot compensates gravity on the SO arms (its only gravity compensation is the Unitree G1's, off by
  default);
- LeRobot's own scripted move, `return_to_initial_position` (`rollout/strategies/core.py`), streams absolute targets
  interpolated from the present position, open loop: the precedent for streaming a planned trajectory.

**What the expert keeps and what changes.** It keeps so101Sim's plan (`rl/native/scripted_expert.py` at `d4c2990`):
the waypoints (10 cm above the cube; 8 mm above its centre to grasp; 0.20 m over the grasp point to lift), the
damped-least-squares IK on the point between the jaws (lambda 1e-4, five arm joints, wrist roll aligned with the cube's
yaw), the phases (approach, descend, grasp, lift, and a retry when the cube is still below 7 cm) and their tolerances.
Its IK step equals so101Sim's bit for bit (4,000 iterations over 100 seeds compared; `check_port.py`). What changes is
the command:

| | so101Sim's expert | DeployExpert | source |
|---|---|---|---|
| command | measured joint position + one IK step, at most 0.06 rad | absolute goal per joint, held between ticks | `send_action` |
| rate | 50 Hz (the env's 20 ms step) | 30 Hz: a tick before physics step round(k x 200 / 30) | LeRobot's 30 fps |
| waypoint goal | one IK step from the measured, sagging pose | the same IK iterated to convergence on the kinematic model, warm-started from the last goal, re-solved every tick for the cube's pose | LeRobot's IK pipeline is kinematic too |
| path to the goal | none (the 0.06 rad cap) | straight line in joint space, synchronised trapezoid, re-targeted every tick | `return_to_initial_position` |
| speed and acceleration limits | | 2.72 / 1.85 / 1.53 / 2.26 / 2.67 rad/s; 21 / 14 / 15 / 28 / 35 rad/s^2 (pan to wrist roll) | 7v4_real's torque budget (below) |
| gripper | open 1.4, close -0.15 rad | the same, as steps | so101Sim; LeRobot's 50 % torque limit is in the servo model |
| next phase | measured TCP within tolerance | the same, or 0.31 s after the command arrived | 7v4_real's 2 % settle time at the shoulder |
| grasp | 0.3 s | at least 0.3 s, then once the jaws stop (below 0.05 rad/s); timeout 0.88 s | |
| gravity compensation | none | none | LeRobot has none |

The limits come from the `7v4_real` servo model. R = 1.912 - 0.0515 - g_max N.m is the torque left after Coulomb
friction and the largest gravity torque on the task's paths (100 seeds, the box at 1.4 times its mass in hand). Half of
it goes to speed (R / 2b, b = 0.342 N.m.s/rad) and half to acceleration (R / 2M, M the largest inertia, armature
included). That split is the time-optimal trapezoid under that budget for moves of 0.3-0.7 rad, the size of the task's
descend and lift moves. None reaches the firmware's caps (5.29 rad/s, 39 rad/s^2). **No parameter was tuned**: the
first full run on 7v4_real gave its CPU row below. Derivation and checks: `runs/readme_media/scratch/so101-expert/`
(`limits.py`, `check_port.py`). **On 7v4_bam the expert is unchanged**, limits and settle time included: they were
derived for 7v4_real's weaker 1.91 N.m clip, so on 7v4_bam's 3.32 N.m (open: the section above) they are conservative.

**In the film tool** (`SO101Lift` with `controller="deploy_expert"`):

- BatchSim steps one 5 ms physics step at a time, four per 20 ms frame.
- Before physics step k, if k is a control tick, each live world's joint state is copied to a CPU `MjData` of the
  model, `mj_forward` gives fresh kinematics, and that world's expert returns its six goals.
- Every physics step the servo layer (command delay, then the firmware's speed and acceleration profile;
  `metalsim/physics/actuation.py`) turns the held goals into `d.ctrl`.
- On 7v4_bam, BAM's friction update runs after every physics step inside BatchSim's step graph (`BamFrictionWarp`,
  per-world `dof_frictionloss`); after the reset's state write, the first budget comes from a forward pass with the
  servos holding the start pose.
- Frames, success and the episode's end are the policy films'.

This is so101-expert's evaluation loop (`runs/readme_media/scratch/so101-expert/eval_metalsim.py`; with the friction,
so101-bam-gpu's `eval_metalsim_bam.py`). On 7v4_real, with a CPU MuJoCo stand-in for BatchSim, the film adapter
reproduces that script's CPU run exactly: the same outcome, retries and phase-change times on all 100 seeds, and the same
arm joint positions at every frame. The option leaves the policy films unchanged: their adapter output is bit for bit
the same as before (60 frames x 4 worlds for each of `so101_lift`, `so101_lift_74v` and `so101_lift_smooth`). Script and
output: `runs/readme_media/scratch/so101-film/cpu_check.py`, `cpu_check.json`.

**Results**, seeds 10000-10099, one world per seed. One run each; CPU MuJoCo is deterministic, each MetalSim run is one
GPU draw:

| run | servo | success | box up at 6 s | time to lift, median (mean) | retries | not lifted |
|---|---|---|---|---|---|---|
| CPU MuJoCo 3.10.0, so101Sim's env (so101-sag `eval_cpu_sag.py`) | 7v4_bam | 100 | 100 | 2.19 s (2.23) | 1 | |
| MetalSim, so101-bam-gpu's evaluation (`eval_metalsim_bam.py`) | 7v4_bam | 100 | 100 | 2.18 s (2.23) | 1 | |
| **MetalSim, this take's own recording (`rec.json`)** | **7v4_bam** | 100 | 100 | 2.18 s (2.23) | 1 | |
| CPU MuJoCo 3.14.0, the film's scene and reset (`eval_metalsim_bam.py --cpu-dry`) | 7v4_bam | 100 | 100 | 2.18 s (2.23) | 1 | |
| CPU MuJoCo 3.10.0, so101Sim's env (`eval_cpu.py`) | 7v4_real | 100 | 100 | 2.46 s (2.61) | 6 | |
| MetalSim, so101-expert's evaluation (`eval_metalsim.py`) | 7v4_real | 98 | 98 | 2.46 s (2.66) | 12 | 10032, 10064 |
| MetalSim, the earlier 7v4_real take's recording (replaced) | 7v4_real | 98 | 98 | 2.46 s (2.66) | 12 | 10032, 10064 |
| CPU MuJoCo 3.14.0, the film's scene and reset (`eval_metalsim.py --cpu-dry`) | 7v4_real | 97 | 97 | 2.46 s (2.64) | 10 | 10032, 10043, 10095 |

- **On 7v4_bam every run agrees on every seed.** The take's own 100 worlds equal so101-bam-gpu's MetalSim evaluation
  seed for seed: the same outcome, time to lift, retries and phase-change times on all 100. Against CPU 3.10 and CPU
  3.14 the outcome and retries are the same on all 100 seeds, and the time to lift is within 0.04 s on every seed
  (identical on 71 and on 86). The one retry is seed 10024 in every run. The take ran on MuJoCo Warp `787a5b3a37` (the
  fork took EPA collision fixes between the two runs), the evaluation on `93e5ce18e7`, both on Warp `52c09cdd82`; the
  rollout is still not bitwise repeatable.
- **On 7v4_real the backends differed on a few seeds**: against CPU 3.10, MetalSim gave the same outcome on 98 seeds
  and time to lift within 0.1 s on 84 of the 98. Its failures, 10032 and 10064, lose the box in a grasp that slips,
  retry, and run out of time; on CPU they lift. On CPU MuJoCo 3.14.0, seed 10095 hits "NaN, Inf or huge value in QVEL"
  on the box's free joint at 2.8 s (not investigated).
- **7v4_bam lifts sooner and retries less**: median time to lift 2.18 s against 2.46 s; 1 retry in 100 against 6-12.
  10th / 90th percentile of time to lift: 2.06 / 2.36 s on CPU 3.10 and on MetalSim (7v4_real: 2.24 / 2.80 s and
  2.24 / 2.90 s).
- MuJoCo Warp flagged its line-search iteration cap (`LS_ITERATIONS`, a convergence warning) in all 100 of the take's
  worlds and its solver iteration cap (`ITERATIONS`) in 2, as in the evaluation; 7v4_real's runs flag `LS_ITERATIONS`
  only. No capacity overflow, so no contact or constraint row was dropped.

**Joint motion and holding**, every 5 ms physics step, MetalSim evaluations, per joint (pan / lift / elbow / wrist flex /
wrist roll / gripper; the measures of the section above):

| | 7v4_real | 7v4_bam |
|---|---|---|
| RMS acceleration, rad/s^2 | 2.1 / 7.2 / 5.5 / 6.3 / 3.6 / 7.6 | 2.3 / 5.0 / 4.5 / 4.5 / 3.2 / 7.6 |
| RMS jerk, 10^3 rad/s^3 | 0.33 / 1.33 / 1.09 / 1.04 / 0.16 / 0.47 | 0.46 / 0.34 / 0.60 / 0.29 / 0.09 / 0.57 |
| velocity reversals per second | 0.39 / 0.81 / 0.53 / 0.59 / 0.32 / 0.60 | 0.44 / 0.72 / 0.18 / 0.38 / 0.01 / 0.17 |
| velocity RMS above 5 Hz, rad/s | 0.017 / 0.046 / 0.035 / 0.039 / 0.013 / 0.087 | 0.019 / 0.023 / 0.031 / 0.021 / 0.018 / 0.083 |
| steady error in the lift phase (box in hand), median, mrad (no gripper) | 5 / 32 / 126 / 46 / 9 | 4 / 6 / 46 / 11 / 1 |
| joint speed, 99th percentile, rad/s (no gripper) | 1.32 / 2.48 / 1.83 / 2.20 / 2.73 | 1.27 / 2.17 / 1.77 / 1.98 / 2.66 |
| lag, mean \|command(t) - position(t + 1)\|, deg (no gripper) | 0.90 / 3.86 / 5.68 / 3.14 / 1.73 | 0.62 / 2.25 / 2.96 / 1.73 / 1.11 |

- On 7v4_bam the motion is as smooth or smoother: jerk at the shoulder lift is a quarter of 7v4_real's, and the elbow,
  wrist roll and gripper reverse direction a third as often or less. Two measures are higher: the pan's jerk (0.46
  against 0.33) and wrist roll's velocity above 5 Hz (0.018 against 0.013 rad/s).
- The original PPO policy on 7v4_real: 36-77 rad/s^2 RMS and 6 successes in 100.
- Real SO-101 followers in the two LeRobot datasets of the section above (`hungho77/so101_left_arm_pick_red_cube_into_pot`,
  50 episodes; `IlyaGusev/so101_pick_gum`, 104 of its 155 episodes): joint speeds 1.1-2.6 rad/s at the 99th
  percentile, and a mean lag of 1.94 and 2.53 deg over the five joints. 7v4_bam's is 1.73 deg (7v4_real's 3.06).
- On 7v4_bam, shoulder lift and elbow use more than the datasheet's 1.912 N.m stall torque on about 3 % of physics
  steps, in all 100 episodes (peaks 2.30 and 2.36 N.m), within the model's 3.32 N.m clip (open, section above). The
  gripper sits at its 1.71 N.m limit on 87 % of steps.

**The take** (`runs/film/so101_lift_deploy/`). The so101_lift take's look, cameras and shots; 100 worlds, seeds 10000 to
10099, so the recording itself gives every seed's outcome (the "this take" row above). It films world 51, seed 10051.

- **Why seed 10051: it is a typical episode**, and the seed of the replaced 7v4_real take. On 7v4_bam it lifts at
  2.26 s in this take and in every other run: the 68th to 72nd percentile, inside the interquartile range (2.10-2.30 s;
  median 2.18 s). It needs no retry; only seed 10024 retries on 7v4_bam.
- What happens. The phase changes are in `rec.json`, `adapter.first_episode.events`, world 51:
  - the arm starts straight out over the table and folds down toward the box;
  - at 0.80 s the measured jaw point is within 2 cm of the approach waypoint and the jaws are aligned with the box,
    and the arm descends;
  - at 1.17 s the jaws close; the box tips in the closing jaws while still on the table, 14 deg by the time the lift
    starts;
  - at 1.74 s the jaws have stopped, and the arm lifts;
  - success (the box above 0.10 m for 10 control steps) is counted at 2.26 s;
  - the box rises to 0.19 m and hangs tilted in the jaws, 43 deg from upright, still 0.19 m up at 6 s.
- Clip: [so101_lift_deploy.webp](so101_lift_deploy.webp), the `readme` shot from 0 to 5 s, made with the same command
  and window as so101_lift.webp (0.69 MB, 150 frames, PSNR 34.5 dB against the source frames).
- Hero: [so101_lift_deploy.mp4](so101_lift_deploy.mp4) (0.7 MB) and `so101_lift_deploy_still{1,2,3}.jpg` (frames 60,
  110 and 199: the open jaws coming down over the box, the box rising, the hold).
- Recorded on 2026-10-31 in one GPU-queue job of kind render (21:51 to 21:54), on the 14-inch MacBook Pro (Mac16,6),
  with a first render. At every stage stamp (21:51:30 to 21:53:14) and at the job's start and end it was on AC power
  and charging (79 %), powermode 2. Software: MuJoCo Warp `787a5b3a37`, Warp `52c09cdd82`, MuJoCo 3.14.0, metalsim
  `ed66d8ef` plus uncommitted changes. This was not a timing run.
- The frames here were rendered again from that recording on 2026-11-01 (00:31 to 00:34, a GPU-queue job of kind
  render; AC power, 100 % and finishing its charge at both shots' stamps, powermode 2), at metalsim `68f695b2` plus
  uncommitted files, after the film tool started drawing the look's colours on geoms with their own
  `rgba` (`render.build_scene`): the first render drew the box in the twin's (0, 1, 0), the table legs at 0.2 and the
  wood times (0.6, 0.45, 0.3).
- Rendering: tier 2, 64 samples per pixel, 4 bounces, Open Image Denoise, the studio HDRI. The table's wood texture is
  at its real size (Poly Haven's 1.5 m tile, `ground_tile_m`) and in its own colours.

**Against the 7v4_real take** (the same seed, plan, cameras and frames; `runs/readme_media/scratch/so101-film-bam/`):

- Seed 10051 goes through the same phases at nearly the same times on both servo models (descend 0.80 s, grasp
  1.17 s, lift 1.74 s against 1.77 s) and lifts at 2.26 s against 2.34 s.
- **The arm droops less in the hold.** The takes' recorded joint angles match the evaluations' runs within 8e-5 rad, so
  those runs' commands give the takes' steady error. In this seed's hold (3-6 s), for shoulder lift / elbow / wrist
  flex it is 18 / 123 / 48 mrad on 7v4_real and 0.3 / 45 / 12 mrad on 7v4_bam. The commands differ by at most 9 mrad on those
  joints, so the two poses differ by about the sag: the elbow by 72-75 mrad (4.2 deg), the wrist flex by 44-46 mrad,
  the shoulder lift by 12 mrad.
- On 7v4_real the box rises to 0.20 m and then sinks to 0.17 m as the arm settles; on 7v4_bam it rises to 0.19 m and
  stays there (0.188 m at 6 s). Side by side at frames 149 and 299, the jaws and the box sit visibly higher on 7v4_bam
  and the jaws point more upward; the base and the shoulder look the same.
- **The box tilts more**: 43.5 deg from upright at 6 s, against 35.5 deg in the 7v4_real take; 14 deg against 6 deg
  when the lift starts.

**The grasp and the box in hand** (CPU MuJoCo 3.14.0, the film's scene and reset, 100 seeds, lift phase after 3 s;
`grasp_cpu.py`; on 7v4_bam it gives the evaluations' outcomes, 100/100, median 2.18 s, 1 retry; on 7v4_real 99/100
with 11 retries, against the `--cpu-dry` run's 97 and 10: this harness passes float64 states to the expert where
`--cpu-dry` passes float32, and 7v4_real's slipping grasps are sensitive to small differences):

| | 7v4_real | 7v4_bam |
|---|---|---|
| gripper actuator force in the hold | 0.99 N.m (its limit) | 1.71 N.m (its limit) |
| gripper gearbox friction in the hold | 0.05 N.m | 0.57 N.m (load-dependent) |
| sum of jaw contact normal forces, median | 39 N | 42 N |
| deepest jaw penetration into the box, median / largest | 0.94 / 1.81 mm | 0.94 / 1.57 mm |
| box tilt when the lift starts, median | 5 deg | 11 deg |
| box tilt at 6 s, median / 90th percentile / largest | 29 / 42 / 47 deg | 40 / 50 / 57 deg |
| box moving in the jaws from 3 s to 6 s, median / 90th percentile | 2.5 / 11.7 mm, 6.7 / 15 deg | 1.9 / 2.9 mm, 3.4 / 8.1 deg |

- **The stronger gripper does not squeeze much harder.** Its motor torque is 1.71 N.m against 0.99. On 7v4_bam,
  though, the gripper's own gearbox friction depends on the load too, and with the jaws stopped on the box after
  closing about 0.57 N.m of it holds against the motor, so roughly 1.14 N.m reaches the jaws, against 0.94 on 7v4_real
  (inferred from the friction budget). The measured sum of the jaws' contact forces is 42 N against 39 N, and the
  jaws press 0.94 mm into the box's collision shape at the median on both models. With 7v4_bam's gripper limited to
  7v4_real's 0.986 N.m, its friction holds 0.34 N.m and the sum falls to 25 N, as that reading predicts (ablation,
  `grasp_cpu_bam_grip0986.json`).
- **No visible interpenetration or slip in the take.** The jaws' rendered meshes reach at most 0.6 mm into the box in
  either take (each mesh vertex tested against the box in the recorded poses; `jaw_depth.py`). Where a jaw tip seems
  to overlap the box in a frame, it is in front of the box's face. From 3 s to 6 s the box moves 0.9 mm and turns
  1.7 deg in the jaws (7v4_real take: 2.9 mm, 2.7 deg).
- **The box turns with the hand; it does not swing in the jaws.** Shoulder lift, elbow flex and wrist flex turn about
  parallel axes, and the expert's position-only IK lets the hand pitch as it lifts: from the lift start to 6 s the hand
  pitches by a median 35.8 deg (this seed: 45.7 deg) over the 15 cm lift, and the box's tilt in that plane changes by
  39.2 deg. In the jaws the box turns a median 8 deg (this seed: 4.6 deg), about 5 deg of it MuJoCo's documented slow
  slippage of soft contacts (0.1 deg with noslip, which MuJoCo Warp does not have). Contact settings (condim, torsional
  and sliding friction, solref, impratio, cone) move the 6 s median by at most about 5 deg. Real SO-101 teleop lifts
  pitch the hand the same way: a median 45 deg over 13-18 cm lifts (hungho77's 50 episodes) and 17 deg over 6-12 cm
  (IlyaGusev's 88). So the tilt's size is realistic for this lift. 7v4_real's 29 deg comes from a smaller pitch change
  (31 deg; its elbow and wrist sag more) and a smaller tip while the jaws close (4 against 10.5 deg). It is not the
  stronger gripper: limited to 7v4_real's 0.986 N.m, 7v4_bam still gives 38 deg at 6 s. (CPU MuJoCo 3.14.0, the film's
  scene, 100 seeds; `runs/readme_media/handoff/so101-tilt.md`.)

**Caveats.**

- **Scripted, not learned.** The cube's pose comes from the simulator state at every tick, not from a camera:
  perception is not modelled.
- **The shoulder still sags more than real arms do.** On the real followers' replayed commands 7v4_bam cuts the
  shoulder's steady error from 93 to 17 mrad, but in the heaviest fifth of steady samples its shoulder sags 1.6-2.4
  times as much as theirs (7v4_real: 5-7 times), and the elbow's friction band is narrower than real. Not fitted, on
  purpose (section above).
- **The stall torque is unresolved**: 7v4_bam's 3.32 N.m (BAM's kt / R) against the datasheet's 1.91 N.m. It changes
  this task's timing, not its success: with the datasheet's clip the expert still lifts 100/100 on CPU, later on every
  seed (median 2.52 s against 2.18), with 6 retries. The gripper's 50 % limit follows it (1.71 against 0.99 N.m).
- **The grasp is less realistic than the lift.** The box leans 40 deg at 6 s (median of 100 seeds on CPU; this take
  43.5 deg) because the hand pitches, as real lifts do (above); no real arm has held this box, so that tilt has not
  been measured on one. Two parts are not like a real operator's grasp, both from so101Sim's grasp waypoints (the
  `pickup` keyframe's grasp point and `GRASP_DZ`), kept as they are:
  - the hand comes down 26-28 deg off vertical, where real operators grasp at 4-13 deg, so the box sits about 28 deg
    to the jaws;
  - the closing jaw pushes the box a median 6.6 mm across the table and tips it 10.5 deg before the lift.
- The servo model's inferred parts carry over (section above): the firmware profile's shape, the 3 ms host latency, and
  D 32, which is not modelled.
- Nothing has run on the arm.

Reproduce (main venv; the record and render stages are GPU-queue jobs):

```bash
python -m metalsim.render.film so101_lift_deploy --stage record
python -m metalsim.render.film so101_lift_deploy --stage render
python -m metalsim.render.film so101_lift_deploy --stage encode --shots hero
python -m metalsim.render.film.encode webp runs/film/so101_lift_deploy/readme/frames docs/renders/so101_lift_deploy.webp --start 0 --seconds 5 --src-fps 50 --fps 30
```

Sources: so101-expert's evaluations and measurements in `runs/readme_media/scratch/so101-expert/` (`eval_cpu.py`,
`eval_metalsim.py`, `limits.py`, `check_port.py`, `sag.py`, `real_sag_model.py`, `sim_lag.py`; outputs in `cpu/`,
`metalsim/`, `logs/`); 7v4_bam's in `runs/readme_media/scratch/so101-sag/` and `runs/readme_media/scratch/so101-bam-gpu/`
(the section above); this take's comparison and measurements in `runs/readme_media/scratch/so101-film-bam/`
(`compare_take.py`, `take_measure.py`, `droop.py`, `grasp_cpu.py`, and their JSON outputs); the tilt's decomposition,
contact ablations and real teleop hand pitch in `runs/readme_media/scratch/so101-tilt/` (`tilt_probe.py`,
`real_pitch.py`, `table.md`; summary and sources in `runs/readme_media/handoff/so101-tilt.md`). Those directories are
not tracked.
"""
SO101_NOTES = SO101_NOTES + SO101_74V_SECTION + SO101_SMOOTH_SECTION + SO101_SERVO_SECTION + SO101_DEPLOY_SECTION

SO101_LIFT = FilmTask(
    name="so101_lift",
    adapter="metalsim.render.film.so101:SO101Lift",
    venv="main",
    checkpoint=POLICY,
    adapter_kw={},
    title="SO-101 cube lift",
    summary=("The Robot Studio's SO-101 arm grasps a 4 x 4 x 6 cm box and lifts it: so101Sim's state-based cube-lift "
             "task and its PPO teacher (trained on CPU MuJoCo in so101Sim), played on metalsim (MuJoCo Warp on Metal) "
             "through so101Sim's own observation, action and control loop. The realism check below finds the motion "
             "beyond what the standard SO-101 follower's servos can do. Rendered on the 14-inch MacBook Pro (Mac16,6), "
             "not in a timing run, so its Performance table is not a timing figure."),
    notes=SO101_NOTES,
    viewer=SO101_VIEWER,
    # stills: the grasp (0.3 s), the lift (0.6 s) and the hold (3 s)
    film=FilmCfg(num_envs=16, envs="auto", grid=1, seed=10_000, max_seconds=6.0, hold_s=0.0, passes=16, spp=4,
                 bounces=4, stills=(0.05, 0.1, 0.5), output_filename_prefix="so101_lift"),
    look=LookCfg(hdri="hdri/train/poly_haven_studio.hdr", sun=False, ground_texture="tex/train/wood_table_001.png",
                 ground_geoms=("table_top",), ground_tile_m=1.5,   # wood_table_001: 1500 x 1500 mm (Poly Haven)
                 materials={r"^box$": dict(rgba=(0.1, 0.9, 0.15, 1.0), roughness=0.5, specular=0.3),
                            r"^table_leg_": dict(rgba=(0.12, 0.12, 0.12, 1.0), roughness=0.4, metallic=0.6),
                            r"^studio_floor$": dict(rgba=(0.35, 0.35, 0.36, 1.0), roughness=0.8)}),
    assets=[
        AssetRef("SO-101 arm MJCF and STL meshes (MuJoCo Menagerie robotstudio_so101; assets/so101/so101.xml)",
                 "Google DeepMind MuJoCo Menagerie, from The Robot Studio's SO-ARM100",
                 "https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101", "Apache-2.0", "committed"),
        AssetRef("Lift scene: the arm, a 4 x 4 x 6 cm box on a free joint, the floor (assets/so101/scene_box_rl.xml; "
                 "physics equal to so101Sim's scene_box_rl.xml)", "this repository / so101Sim",
                 "assets/so101/scene_box_rl.xml", "Apache-2.0", "committed"),
        AssetRef("Task, reset distribution and control loop (SO101LiftEnv), ported", "so101Sim (pulipakaa24)",
                 f"{SO101SIM['repo']}/blob/{SO101SIM['commit']}/{SO101SIM['env']}",
                 "no licence file published; the owner's project, used with the owner's approval", ""),
        AssetRef("Policy: so101Sim's PPO lift teacher (run 1 + run 2, 8 M steps), extracted to plain torch weights",
                 "so101Sim (pulipakaa24), trained on CPU MuJoCo", POLICY,
                 "no licence file published; the owner's project, used with the owner's approval",
                 "runs/film/so101_lift/port/extract_policy.py"),
        AssetRef("Studio HDRI (poly_haven_studio, 2k)", "Poly Haven", "https://polyhaven.com/a/poly_haven_studio",
                 PH_CC0, FETCH_PH),
        AssetRef("Table texture (wood_table_001, 1k diffuse)", "Poly Haven", "https://polyhaven.com/a/wood_table_001",
                 PH_CC0, FETCH_PH),
    ],
    shots=[ShotCfg("hero", 1), ShotCfg("readme", 1, SO101_README_VIEWER)],
    doc="docs/renders/SO101_LIFT.md",
)

# The same policy, seeds, look, cameras and shots with the standard follower's 7.4 V servos (``apply_servo``), filming
# the so101_lift take's env (world 13, seed 10013) for a direct comparison, whatever happens in it: a box knocked out
# of the workspace ends the episode, and that is filmed too. No doc stage: its section is in SO101_LIFT.md.
SO101_LIFT_74V = replace(
    SO101_LIFT,
    name="so101_lift_74v",
    adapter_kw={"servo": "7v4"},
    title="SO-101 cube lift, 7.4 V servos",
    summary=("The so101_lift film's policy, seeds and cameras with the standard SO-101 follower's 7.4 V STS3215 "
             "servos (1.91 N.m stall, 52 rpm; gripper at LeRobot's 50 % torque limit) instead of the scene's 12 V "
             "ones; the policy is unchanged. See docs/renders/SO101_LIFT.md, section \"7.4 V servo model\"."),
    notes="",
    film=replace(SO101_LIFT.film, envs=[13], allow_early_end=True, output_filename_prefix="so101_lift_74v"),
    doc=None,
)

# so101Sim's teacher (run 2) continued 10 M steps on the 7.4 V servo model with Isaac Lab's lift-task smoothness
# penalties and their curriculum, Isaac Lab's weights unscaled and applied to the raw policy output as Isaac Lab applies
# them (variant 2 of section "Smooth 7.4 V retrain" of SO101_LIFT.md; scripts in runs/film/so101_lift_smooth/train/).
# It halves the original's action change per step but does not reach a realistic control style (that section). The
# so101_lift take's 16 worlds, seeds, look, cameras and shots; the filmed world is the adapter's ranking ("auto").
SMOOTH_POLICY = "runs/film/so101_lift_smooth/policy/so101_lift_smooth_ppo.pt"
SO101_LIFT_SMOOTH = replace(
    SO101_LIFT,
    name="so101_lift_smooth",
    checkpoint=SMOOTH_POLICY,
    adapter_kw={"servo": "7v4"},
    title="SO-101 cube lift, 7.4 V retrain with smoothness penalties",
    summary=("so101Sim's lift teacher continued 10 M steps on the standard SO-101 follower's 7.4 V servos (gripper at "
             "LeRobot's 50 % torque limit) with Isaac Lab's lift-task action-rate and joint-velocity penalties, played "
             "on metalsim with the so101_lift film's seeds and cameras. It halves the original policy's action change per "
             "step but still commands the joint limits on most steps, so it is not a realistic control style. See "
             "docs/renders/SO101_LIFT.md, section \"Smooth 7.4 V retrain\"."),
    notes="",
    film=replace(SO101_LIFT.film, allow_early_end=True, output_filename_prefix="so101_lift_smooth"),
    assets=[a if a.url != POLICY else
            AssetRef("Policy: so101Sim's PPO lift teacher (run 2) continued 10 M steps on the 7.4 V servo model with "
                     "Isaac Lab's lift-task smoothness penalties (weights unscaled, on the raw policy output), extracted "
                     "to plain torch weights",
                     "so101Sim (pulipakaa24) teacher, retrained here on CPU MuJoCo", SMOOTH_POLICY,
                     "no licence file published; the owner's project, used with the owner's approval",
                     "runs/film/so101_lift_smooth/train/train_smooth.py isaac_unscaled_raw, "
                     "runs/film/so101_lift/port/extract_policy.py")
            for a in SO101_LIFT.assets],
    doc=None,
)

# so101-expert's deployment-style scripted expert (``so101_expert.DeployExpert``) on Rhoban BAM's identified servo model
# (7v4_bam: m6 motor, firmware gain and load-dependent gearbox friction, under LeRobot's registers), as LeRobot drives a
# real SO-101 follower: absolute joint goals at 30 Hz, the servos' own P loop and firmware profile, no gravity
# compensation. No policy file. 100 worlds = seeds 10000-10099, the expert's MetalSim eval (so101-bam-gpu), so the take's
# own rec.json (all_first_episodes, adapter.first_episode) checks every seed's outcome against it; the film shows world
# 51 (seed 10051: no retry, inside the eval's interquartile time to lift). The so101_lift look, cameras and shots. Its
# section is in SO101_LIFT.md ("Deployment-style scripted expert on 7v4_bam"); no doc stage. "7v4_real" stays a servo
# option (its take was not kept).
SO101_LIFT_DEPLOY = replace(
    SO101_LIFT,
    name="so101_lift_deploy",
    checkpoint=None,
    adapter_kw={"servo": "7v4_bam", "controller": "deploy_expert"},
    title="SO-101 cube lift, deployment-style scripted expert on BAM's identified servo model",
    summary=("so101Sim's scripted IK lift plan streamed as absolute joint goals at LeRobot's 30 Hz (DeployExpert) to a "
             "model of the standard SO-101 follower's 7.4 V STS3215 servos under LeRobot's registers (7v4_bam: Rhoban "
             "BAM's m6 identification, load-dependent gearbox friction updated every physics step, P 16, firmware "
             "speed and acceleration limits, gripper at 50 % duty), on metalsim (MuJoCo Warp on Metal). Scripted, not "
             "learned; the cube pose comes from the simulator. See docs/renders/SO101_LIFT.md, section "
             "\"Deployment-style scripted expert on 7v4_bam\"."),
    notes="",
    # stills (hero, seed 10051; frames 60, 110, 199): the open jaws coming down over the box (1.2 s), the box rising
    # (2.2 s), the hold (4.0 s)
    film=replace(SO101_LIFT.film, num_envs=100, envs=[51], allow_early_end=True, stills=(0.2, 0.367, 0.667),
                 output_filename_prefix="so101_lift_deploy"),
    assets=[a if a.url != POLICY else
            AssetRef("Controller: DeployExpert, so101Sim's scripted IK lift plan (waypoints, IK, phases) streamed as "
                     "absolute joint goals at 30 Hz; no learned policy, no policy file",
                     "this repository (metalsim/render/film/so101_expert.py), porting so101Sim's "
                     "rl/native/scripted_expert.py", "metalsim/render/film/so101_expert.py",
                     "Apache-2.0 (this repository); the ported plan: so101Sim, no licence file published, the owner's "
                     "project, used with the owner's approval", "")
            for a in SO101_LIFT.assets],
    doc=None,
)

SO101_TASKS = (SO101_LIFT, SO101_LIFT_74V, SO101_LIFT_SMOOTH, SO101_LIFT_DEPLOY)   # registered in tasks.TASKS


def main(argv=None):
    """The film tool (``python -m metalsim.render.film``) with the SO-101 entries (``SO101_TASKS``) registered in this
    process (``tasks.TASKS`` registers them too): ``python -m metalsim.render.film.so101 [task] [film-tool args]`` (task
    default ``so101_lift``)."""
    from metalsim.render.film import __main__ as film_main
    from metalsim.render.film import tasks

    for t in SO101_TASKS:
        tasks.TASKS.setdefault(t.name, t)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = [SO101_LIFT.name] + argv
    return film_main.main(argv)


if __name__ == "__main__":
    sys.exit(main())
