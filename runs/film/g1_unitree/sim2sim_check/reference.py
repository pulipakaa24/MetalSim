# SPDX-License-Identifier: BSD-3-Clause
# The control loop below is unitree_rl_gym's deploy/deploy_mujoco/deploy_mujoco.py (commit 276801e4), Copyright (c)
# 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD-3-Clause (licenses/BSD-3-Clause-Unitree.txt).
"""The CPU reference of the g1_unitree film (docs/renders/G1_UNITREE.md): Unitree's deploy_mujoco.py loop, unchanged, headless, on CPU MuJoCo, for 8 s.

The loop body below is deploy/deploy_mujoco/deploy_mujoco.py (unitree_rl_gym @ 276801e4) line for line, with the
viewer, viewer.sync() and the wall-clock sleep removed and `while ... < simulation_duration` replaced by a fixed
number of physics steps. Recording (base position per physics step; qpos, feet and floor contacts at each policy
step) only reads the state. Usage: .venv/bin/python runs/film/g1_unitree/sim2sim_check/reference.py <out_dir> [seconds] (CPU only;
this take: out_dir runs/readme_media/scratch/g1-unitree/ref, 8 s; reference.json here is its output)
"""
import json
import os
import sys

import mujoco
import numpy as np
import torch
import yaml

from metalsim.render.film import record as REC
from metalsim.render.film import tasks
from metalsim.render.film.unitree import (UNITREE_FLOOR, unitree_contacts, unitree_feet, unitree_foot_geoms,
                                          unitree_paths)

out_dir = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
os.makedirs(out_dir, exist_ok=True)
P = unitree_paths(tasks.UNITREE_RL_GYM)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


with open(P["config"], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    policy_path = P["policy"]
    xml_path = P["scene"]

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    cmd = np.array(config["cmd_init"], dtype=np.float32)

# define context variables
action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()
obs = np.zeros(num_obs, dtype=np.float32)

counter = 0

# Load robot model
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

# load policy
policy = torch.jit.load(policy_path)

# recording (reads only)
n_steps = int(round(seconds / simulation_dt))
foot_geoms = unitree_foot_geoms(m)
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, UNITREE_FLOOR)
d_fk = mujoco.MjData(m)
base = np.zeros((n_steps + 1, 3)); base[0] = d.qpos[:3]
snaps, ankles, actions = [], [], []

for _ in range(n_steps):
    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
    d.ctrl[:] = tau
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    counter += 1
    if counter % control_decimation == 0:
        # Apply control signal here.

        # create observation
        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        quat = d.qpos[3:7]
        omega = d.qvel[3:6]

        qj = (qj - default_angles) * dof_pos_scale
        dqj = dqj * dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * ang_vel_scale

        period = 0.8
        count = counter * simulation_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = cmd * cmd_scale
        obs[9 : 9 + num_actions] = qj
        obs[9 + num_actions : 9 + 2 * num_actions] = dqj
        obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
        obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        # policy inference
        action = policy(obs_tensor).detach().numpy().squeeze()
        # transform action to target_dof_pos
        target_dof_pos = action * action_scale + default_angles

        # --- recording: the frame the film would take (state after the 10th step) ---
        q = d.qpos.copy()[None]
        feet, fp, ank = unitree_feet(m, d_fk, q, foot_geoms)
        touch = unitree_contacts(d.contact.geom[: d.ncon], np.zeros(d.ncon, int), 1, foot_geoms, floor)
        snaps.append(dict(qpos=q, feet=feet, feet_footprint=fp, feet_contact=touch, cmd=cmd.astype(np.float64)[None]))
        ankles.append(ank[0]); actions.append(np.array(action, np.float32))
    base[counter] = d.qpos[:3]

frame_dt = simulation_dt * control_decimation
g = REC.gait_metrics(snaps, np.array([-1]), np.array([False]), frame_dt)
print("\n".join(REC.gait_lines(g, [0])))
A = np.array(ankles)                      # (F, 2, 3) ankle-roll link origins
k0 = int(round(1.0 / frame_dt))
C = np.stack([s["feet_contact"][0] for s in snaps])
swing_peak = [float(np.percentile(A[k0:, f, 2][~C[k0:, f]], 95)) for f in range(2)]
res = dict(source="unitree_rl_gym deploy_mujoco.py loop, CPU MuJoCo " + mujoco.__version__, seconds=seconds,
           sim_dt=simulation_dt, decimation=control_decimation, cmd=cmd.tolist(),
           base_at=dict((f"{t:.0f}s", base[int(round(t / simulation_dt))].round(5).tolist()) for t in (1, 2, 3, 4, 5, 6, 7, 8)
                        if int(round(t / simulation_dt)) <= n_steps),
           mean_base_z=float(base[int(1 / simulation_dt):, 2].mean()),
           ankle_origin_swing_p95_m=swing_peak, gait=g)
np.savez(os.path.join(out_dir, "reference.npz"), base=base, qpos=np.concatenate([s["qpos"] for s in snaps]),
         feet=np.concatenate([s["feet"] for s in snaps]), contact=C, ankles=A, actions=np.array(actions))
json.dump(res, open(os.path.join(out_dir, "reference.json"), "w"), indent=1, default=float)
print("base at 5 s", res["base_at"].get("5s"), "| base at 8 s", res["base_at"].get("8s"), "| mean z", round(res["mean_base_z"], 4),
      "| ankle-origin swing p95", [round(x, 4) for x in swing_peak])
