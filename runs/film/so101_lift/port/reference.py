"""The CPU reference of the so101_lift film (docs/renders/SO101_LIFT.md): so101Sim's own SO101LiftEnv, unchanged, on CPU
MuJoCo, driven by the extracted plain-torch teacher (extract_policy.py), with eval_lift_native.py's loop and seeds.

Per episode (seed 10000 + ep, 300 control steps of 20 ms, deterministic actions) it records the state at every 20 ms
control step, and at every 5 ms physics step (through ``mujoco.set_mjcb_control``, a read-only callback MuJoCo calls
inside each ``mj_step`` before the actuator forces; the env's code is not touched) the arm's joint positions, velocities
and controls, from which the analysis computes each position servo's force as MuJoCo does (kp (ctrl - q) - kv qdot,
clamped to the forcerange). Contacts of the
box are read after each control step. Outputs: ``reference.json`` (success, lift height and time as
eval_lift_native.py prints them, plus the realism measurements) and ``reference.npz`` (trajectories; local only).

Usage (so101Sim's environment, from its urlab_bridge/ directory; CPU only):
  uv run --no-sync python <MetalSim>/runs/film/so101_lift/port/reference.py <so101Sim root> <policy.pt> <out_dir> [episodes]
"""
import json
import os
import sys
import time

import mujoco
import numpy as np
import torch

root, policy_path, out_dir = sys.argv[1:4]
EPISODES = int(sys.argv[4]) if len(sys.argv) > 4 else 100
sys.path.insert(0, os.path.join(root, "rl/native"))
from so101_lift import LIFT_Z, SCENE, SO101LiftEnv  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
P = torch.load(policy_path, weights_only=True)
W = P["pi"]
mean, var = P["obs_mean"].numpy(), P["obs_var"].numpy()


def act(obs):
    n = np.clip((obs - mean) / np.sqrt(var + P["obs_eps"]), -P["clip_obs"], P["clip_obs"])
    x = torch.as_tensor(n[None].astype(np.float32))
    with torch.no_grad():
        x = torch.tanh(x @ W["mlp_extractor.policy_net.0.weight"].T + W["mlp_extractor.policy_net.0.bias"])
        x = torch.tanh(x @ W["mlp_extractor.policy_net.2.weight"].T + W["mlp_extractor.policy_net.2.bias"])
        a = x @ W["action_net.weight"].T + W["action_net.bias"]
    return np.clip(a.numpy()[0], -1.0, 1.0)


env = SO101LiftEnv()
m, d = env.model, env.data
names = lambda t, i: mujoco.mj_id2name(m, t, i)
box_geom = env._box_geom
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
# the gripper's colliders: every geom of the fixed-jaw body ("gripper") and the moving jaw (boxes, tip spheres and the
# unnamed collision meshes of class collision_gripper_mesh)
gripper_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("gripper", "moving_jaw_so101_v1")}
jaw = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in gripper_bodies}
sub = []                     # per physics step: (qpos arm 6, qvel arm 6, ctrl 6), the state the step starts from
recording = [False]


def cb(m_, d_):
    if recording[0]:
        sub.append(np.concatenate([d_.qpos[env._arm_qadr], d_.qvel[env._arm_vadr], d_.ctrl]))


mujoco.set_mjcb_control(cb)
T = 300
F = dict(qpos=np.zeros((EPISODES, T + 1, m.nq)), qvel=np.zeros((EPISODES, T + 1, m.nv)),
         action=np.zeros((EPISODES, T, m.nu), np.float32), ctrl=np.zeros((EPISODES, T, m.nu)),
         box=np.zeros((EPISODES, T + 1, 3)), grip=np.zeros((EPISODES, T + 1, 3)), height=np.zeros((EPISODES, T)),
         box_contacts=np.zeros((EPISODES, T, 4), np.int32),      # gripper geoms, floor, other arm geoms, total
         box_normal_force=np.zeros((EPISODES, T)), max_pen=np.zeros((EPISODES, T)),
         mass_scale=np.zeros(EPISODES), friction_scale=np.zeros(EPISODES), ends=np.zeros(EPISODES, np.int32))
SUB = np.full((EPISODES, T * env._n_sub, 18), np.nan)
eps = []
t0 = time.time()
for ep in range(EPISODES):
    obs, _ = env.reset(seed=10_000 + ep)
    F["mass_scale"][ep] = m.body_mass[env._box_body] / env._box_mass0
    F["friction_scale"][ep] = m.geom_friction[box_geom][0] / env._box_friction0[0]
    F["qpos"][ep, 0], F["qvel"][ep, 0] = d.qpos, d.qvel
    F["box"][ep, 0], F["grip"][ep, 0] = env._box_pos(), env._grip_pos()
    hmax, success_t, sub = 0.0, None, []
    recording[0] = True
    for t in range(T):
        a = act(obs)
        obs, r, term, trunc, info = env.step(a)
        F["action"][ep, t], F["ctrl"][ep, t] = a, d.ctrl
        F["qpos"][ep, t + 1], F["qvel"][ep, t + 1] = d.qpos, d.qvel
        F["box"][ep, t + 1], F["grip"][ep, t + 1] = env._box_pos(), env._grip_pos()
        F["height"][ep, t] = info["height"]
        c = np.zeros(4, np.int32)
        fn, pen = 0.0, 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if box_geom not in (g1, g2):
                continue
            o = g2 if g1 == box_geom else g1
            c[0 if o in jaw else 1 if o == floor else 2] += 1
            c[3] += 1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            fn += f6[0]
            pen = min(pen, float(con.dist))
        F["box_contacts"][ep, t], F["box_normal_force"][ep, t], F["max_pen"][ep, t] = c, fn, pen
        hmax = max(hmax, info["height"])
        if info["is_success"] and success_t is None:
            success_t = t
        if term or trunc:
            break
    recording[0] = False
    F["ends"][ep] = t
    SUB[ep, :len(sub)] = np.array(sub)
    lifted_end = bool(info["height"] > LIFT_Z)
    eps.append(dict(seed=10_000 + ep, success=success_t is not None, success_t=success_t, hmax=hmax,
                    final_height=float(info["height"]), lifted_at_end=lifted_end, terminated=bool(term), end=t))
mujoco.set_mjcb_control(None)
wall = time.time() - t0

succ = [e for e in eps if e["success"]]
summary = dict(episodes=EPISODES, successes=len(succ), success_rate=len(succ) / EPISODES,
               max_height_mean=float(np.mean([e["hmax"] for e in eps])), max_height_min=float(np.min([e["hmax"] for e in eps])),
               time_to_lift_mean_s=float(np.mean([e["success_t"] for e in succ]) * 0.02) if succ else None,
               lifted_at_end=int(sum(e["lifted_at_end"] for e in eps)), terminated=int(sum(e["terminated"] for e in eps)),
               note="time_to_lift as eval_lift_native.py: the control step at which is_success first holds (10 ticks "
                    "above 0.10 m), times 0.02 s; max height = the largest box-centre height reached")
np.savez_compressed(os.path.join(out_dir, "reference.npz"), **F, sub=SUB,
                    actuator=np.stack([m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1], m.actuator_biasprm[:, 2],
                                       m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]], 1),
                    joint_damping=m.dof_damping[env._arm_vadr], joint_frictionloss=m.dof_frictionloss[env._arm_vadr])
out = dict(script="runs/film/so101_lift/port/reference.py", env="so101Sim rl/native/so101_lift.py SO101LiftEnv (unchanged)",
           scene=os.path.relpath(SCENE, root), mujoco=mujoco.__version__, torch=str(torch.__version__),
           policy=dict(file=os.path.basename(policy_path), source=P["source"], check=P["check"]),
           seeds=[10_000, 10_000 + EPISODES - 1], wall_s=wall, summary=summary, episodes=eps)
json.dump(out, open(os.path.join(out_dir, "reference.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
