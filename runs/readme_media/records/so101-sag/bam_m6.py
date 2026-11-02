"""Rhoban BAM's STS3215 7.4 V friction models, as code-faithful copies, plus MetalSim servo parameters derived from them.

Source: github.com/Rhoban/bam at b9e1111 (v1.0.3), cloned for so101-servo:
  * bam/model.py:125-208 Model.compute_frictions (the friction budget tau_fm of M1..M6, at the joint, N.m);
  * bam/params/feetech_sts3215_7_4V/m{1..6}.json (identified parameters);
  * bam/feetech/actuator.py:33-44, 146-155 (vin 7.4 V, error_gain 0.166, max_pwm 0.97; duty = P * err * 0.166 * ratio);
  * bam/actuator.py:317-342 (torque = kt V / R - kt^2 / R * w);
  * bam/mujoco.py:170-203 MujocoController.update (external torque = -qfrc_bias + qfrc_constraint - own dof-friction
    force; motor torque = qfrc_actuator; the budget is written to dof_frictionloss, the viscous term to dof_damping;
    every physics step, from the previous step's data).
"""
import json
import os

import numpy as np

PDIR = "/private/tmp/claude-501/so101-servo-src/bam/bam/params/feetech_sts3215_7_4V"
VIN, ERROR_GAIN, MAX_PWM = 7.4, 0.166, 0.97           # bam/feetech/actuator.py:35-43


def load(model: str) -> dict:
    with open(os.path.join(PDIR, f"{model}.json")) as f:
        return json.load(f)


FLAGS = dict(m1=dict(), m2=dict(stribeck=True), m3=dict(load=True), m4=dict(load=True, stribeck=True),
             m5=dict(load=True, stribeck=True, directional=True),
             m6=dict(load=True, stribeck=True, directional=True, quadratic=True))      # bam/model.py:256-281


def budget(p: dict, tau_m, tau_e, dtheta):
    """bam/model.py Model.compute_frictions, frictionloss part (vectorised). tau_m motor torque, tau_e external torque
    (gravity + constraints, same sign convention: net = tau_m + tau_e), dtheta joint velocity."""
    f = FLAGS[p["model"]]
    tau_m, tau_e, dtheta = (np.asarray(x, float) for x in (tau_m, tau_e, dtheta))
    if f.get("directional"):
        gb = np.abs(tau_e * p["load_friction_external"] - tau_m * p["load_friction_motor"])
        if f.get("stribeck"):
            gbs = np.abs(tau_e * p["load_friction_external_stribeck"] - tau_m * p["load_friction_motor_stribeck"])
    else:
        gb = np.abs(tau_e - tau_m)
    if f.get("stribeck"):
        sc = np.exp(-(np.abs(dtheta / p["dtheta_stribeck"]) ** p["alpha"]))
    fl = np.full(np.broadcast(tau_m, tau_e, dtheta).shape, p["friction_base"], float)
    if f.get("load"):
        fl = fl + (gb if f.get("directional") else p["load_friction_base"] * gb)
    if f.get("stribeck"):
        fl = fl + sc * p["friction_stribeck"]
        if f.get("load"):
            fl = fl + (gbs * sc if f.get("directional") else p["load_friction_stribeck"] * gb * sc)
            if f.get("quadratic"):
                en = np.sign(tau_e) != np.sign(tau_m)
                dm = np.abs(tau_e) < np.abs(tau_m)
                de = np.abs(tau_e) > np.abs(tau_m)
                fl = fl + sc * (dm * p["load_friction_external_quad"] * np.abs(tau_e) ** 2
                                + de * p["load_friction_motor_quad"] * np.abs(tau_m) ** 2) * en
    return fl


def motor(p: dict, P: float = 16.0) -> dict:
    """Small-signal and limit constants of BAM's STS3215 actuator at firmware gain P (LeRobot: 16), joint side.
    full = kt V / R (torque at 100 % duty, zero speed); kp = full * P * 0.166 * error_gain_ratio (N.m/rad);
    stall = full * max_pwm (the duty clip); emf = kt^2 / R (back-EMF damping, N.m.s/rad)."""
    full = p["kt"] * VIN / p["R"]
    return dict(full=full, kp=full * P * ERROR_GAIN * p["error_gain_ratio"], stall=full * MAX_PWM,
                emf=p["kt"] ** 2 / p["R"], viscous=p["friction_viscous"], coulomb=p["friction_base"],
                armature=p["armature"], max_velocity=p["max_velocity"], command_delay=p["command_delay"],
                duty_per_rad=P * ERROR_GAIN * p["error_gain_ratio"],
                noload=(full * MAX_PWM - p["friction_base"]) / (p["kt"] ** 2 / p["R"] + p["friction_viscous"]))


def static_band(kp: float, g_abs: float, fric) -> tuple:
    """Sag s (rad, > 0 = beyond the goal the way gravity pulls) range in which a P servo (motor torque kp * s) holds
    still against gravity |g| with friction budget fric(tau_m, tau_e) at zero speed: |tau_m + tau_e| <= fric."""
    s = np.linspace(-0.3, 0.6, 90001)
    tm, te = kp * s, -g_abs * np.ones_like(s)
    ok = np.abs(tm + te) <= fric(tm, te)
    return (float(s[ok].min()), float(s[ok].max())) if ok.any() else (np.nan, np.nan)
