"""CPU evaluation for the so101-servo stream (so101-smooth's eval_cpu.py, plus the servo layer and smoothness
metrics): so101Sim's own SO101LiftEnv (unchanged; imported from so101Sim through sys.path) on CPU MuJoCo, with a
servo option of metalsim/render/film/so101.py attached by servo_cpu.attach (the "_real" options also run the
command delay + firmware profile at every physics step), driven by either

* a policy file in the film's format (``so101_lift_sb3_mlp_v1``: extract_policy.py's output), with
  eval_lift_native.py's normalisation and the deterministic action, as runs/film/so101_lift/port/reference.py; or
* ``expert``: so101Sim's scripted expert (rl/native/scripted_expert.py ``Expert``, its defaults: DQ_MAX 0.06 rad per
  tick, gripper 1.4 / -0.15 rad), the controller that generated the ACT demos (gen_lerobot_demos.py). Its ctrl is
  turned into the env's action with scripted_expert.norm_ctrl and passed through env.step, so success is the env's.

Seeds 10000 + ep (eval_lift_native.py's), 300 control steps, success does not end the episode. At every 5 ms physics
step a read-only mjcb_control callback records the arm's q, qdot and ctrl (as reference.py). Writes
<out_dir>/reference.json (reference.py's summary and per-episode fields), reference.npz and metrics.json
(metrics.control_style).

Usage (so101Sim's venv, CPU only):
  <so101Sim>/urlab_bridge/.venv/bin/python eval_cpu.py <so101Sim root> <policy.pt | expert> <servo> <out_dir> [episodes]
      [--over '{"accel_limit": Infinity}'] [--fresh-kin]

--fresh-kin (expert only): mj_forward before the expert reads the state, so its kinematics (xpos, site_xpos, Jacobian)
match qpos, as on MetalSim (eval_metalsim.py); without it the expert reads env.data after mj_step, whose kinematics
lag qpos by one 5 ms physics step. smooth.json: smoothness.joint_motion of every physics step.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METALSIM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
args = [a for a in sys.argv[1:]]
OVER, FRESH = {}, False
if "--over" in args:
    i = args.index("--over"); OVER = json.loads(args[i + 1]); del args[i:i + 2]
CMDBASE = "--cmdbase" in args
if CMDBASE:
    args.remove("--cmdbase")
if "--fresh-kin" in args:
    args.remove("--fresh-kin"); FRESH = True
root, controller, servo, out_dir = args[0:4]
EPISODES = int(args[4]) if len(args) > 4 else 100
for k in ("EXPERT_DQ", "EXPERT_GRIP_OPEN"):
    if k in os.environ:
        raise SystemExit(f"{k} is set: the expert must run with its defaults")
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.append(METALSIM)
sys.path.insert(1, HERE)
sys.path.insert(2, os.path.join(HERE, "..", "so101-smooth"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from so101_lift import LIFT_Z, SCENE, SO101LiftEnv  # noqa: E402

from metrics import control_style, row  # noqa: E402 (so101-smooth's)
from servo_cpu import attach  # noqa: E402
from smoothness import joint_motion  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
env = SO101LiftEnv()
applied = attach(env, servo, **OVER)
m, d = env.model, env.data

if controller == "expert":
    import scripted_expert as SE

    if CMDBASE:
        from expert_variants import CmdBaseExpert
        ex = CmdBaseExpert(env)
    else:
        ex = SE.Expert(env)
    src = dict(controller="so101Sim rl/native/scripted_expert.py Expert (defaults)" + (
               " as expert_variants.CmdBaseExpert (diagnostic: targets built on the previous command)" if CMDBASE else ""), DQ_MAX=SE.DQ_MAX,
               GRIP_OPEN=SE.GRIP_OPEN, GRIP_CLOSE=SE.GRIP_CLOSE)

    def reset_ctrl():
        ex.reset()

    def act(obs):
        if FRESH:                                        # mj_forward runs mjcb_control: don't record it
            was, recording[0] = recording[0], False
            mujoco.mj_forward(m, d)
            recording[0] = was
        return SE.norm_ctrl(m, ex.act())
else:
    import torch

    P = torch.load(controller, weights_only=True)
    W = P["pi"]
    mean, var = P["obs_mean"].numpy(), P["obs_var"].numpy()
    src = dict(controller="policy", file=os.path.relpath(os.path.abspath(controller), METALSIM)
               if os.path.abspath(controller).startswith(METALSIM) else os.path.basename(controller),
               source=P["source"], check=P["check"])

    def reset_ctrl():
        pass

    def act(obs):                                            # reference.py's act
        n = np.clip((obs - mean) / np.sqrt(var + P["obs_eps"]), -P["clip_obs"], P["clip_obs"])
        x = torch.as_tensor(n[None].astype(np.float32))
        with torch.no_grad():
            x = torch.tanh(x @ W["mlp_extractor.policy_net.0.weight"].T + W["mlp_extractor.policy_net.0.bias"])
            x = torch.tanh(x @ W["mlp_extractor.policy_net.2.weight"].T + W["mlp_extractor.policy_net.2.bias"])
            a = x @ W["action_net.weight"].T + W["action_net.bias"]
        return np.clip(a.numpy()[0], -1.0, 1.0)

sub = []
recording = [False]


def cb(m_, d_):
    if recording[0]:
        sub.append(np.concatenate([d_.qpos[env._arm_qadr], d_.qvel[env._arm_vadr], d_.ctrl]))


mujoco.set_mjcb_control(cb)
T, NS = 300, env._n_sub
F = dict(qpos=np.zeros((EPISODES, T + 1, m.nq)), qvel=np.zeros((EPISODES, T + 1, m.nv)),
         action=np.zeros((EPISODES, T, m.nu), np.float32), ctrl=np.zeros((EPISODES, T, m.nu)),
         box=np.zeros((EPISODES, T + 1, 3)), grip=np.zeros((EPISODES, T + 1, 3)), height=np.zeros((EPISODES, T)),
         mass_scale=np.zeros(EPISODES), friction_scale=np.zeros(EPISODES), ends=np.zeros(EPISODES, np.int32))
SUB = np.full((EPISODES, T * NS, 18), np.nan)
eps = []
t0 = time.time()
for ep in range(EPISODES):
    obs, _ = env.reset(seed=10_000 + ep)
    reset_ctrl()
    F["mass_scale"][ep] = m.body_mass[env._box_body] / env._box_mass0
    F["friction_scale"][ep] = m.geom_friction[env._box_geom][0] / env._box_friction0[0]
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
        hmax = max(hmax, info["height"])
        if info["is_success"] and success_t is None:
            success_t = t
        if term or trunc:
            break
    recording[0] = False
    F["ends"][ep] = t
    SUB[ep, :len(sub)] = np.array(sub)
    eps.append(dict(seed=10_000 + ep, success=success_t is not None, success_t=success_t, hmax=hmax,
                    final_height=float(info["height"]), lifted_at_end=bool(info["height"] > LIFT_Z),
                    terminated=bool(term), end=t))
mujoco.set_mjcb_control(None)
wall = time.time() - t0

# servo force at every physics step, as MuJoCo computes it (realism.py)
A = np.stack([m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1], m.actuator_biasprm[:, 2],
              m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]], 1)
q, qd, u = SUB[..., 0:6], SUB[..., 6:12], SUB[..., 12:18]
tau = np.clip(A[:, 0] * u + A[:, 1] * q + A[:, 2] * qd, A[:, 3], A[:, 4])
sub_ok = ~np.isnan(qd[..., 0])
sat = np.abs(tau) >= A[:, 4] - 1e-9
st = np.array([e["success_t"] if e["success_t"] is not None else -1 for e in eps])
M = control_style(F["action"], F["ctrl"], F["ends"], sat, sub_ok, qd, sub_ok, st,
                  np.array([e["lifted_at_end"] for e in eps]), np.array([e["hmax"] for e in eps]))
succ = [e for e in eps if e["success"]]
summary = dict(episodes=EPISODES, successes=len(succ), success_rate=len(succ) / EPISODES,
               max_height_mean=float(np.mean([e["hmax"] for e in eps])), max_height_min=float(np.min([e["hmax"] for e in eps])),
               time_to_lift_mean_s=float(np.mean([e["success_t"] for e in succ]) * 0.02) if succ else None,
               lifted_at_end=int(sum(e["lifted_at_end"] for e in eps)), terminated=int(sum(e["terminated"] for e in eps)))
np.savez_compressed(os.path.join(out_dir, "reference.npz"), **F, sub=SUB, actuator=A,
                    joint_damping=m.dof_damping[env._arm_vadr], joint_frictionloss=m.dof_frictionloss[env._arm_vadr])
meta = dict(script="runs/readme_media/scratch/so101-servo/eval_cpu.py", controller=src, servo=applied,
            env="so101Sim rl/native/so101_lift.py SO101LiftEnv (unchanged)", scene=os.path.relpath(SCENE, root),
            mujoco=mujoco.__version__, seeds=[10_000, 10_000 + EPISODES - 1], wall_s=wall)
json.dump(dict(**meta, summary=summary, episodes=eps), open(os.path.join(out_dir, "reference.json"), "w"), indent=1)
json.dump(dict(**meta, metrics=M), open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
SM = joint_motion(SUB[..., 0:6], SUB[..., 6:12], m.opt.timestep)
json.dump(dict(**meta, over=OVER, fresh_kin=FRESH, cmdbase=CMDBASE, summary=summary, smoothness=SM),
          open(os.path.join(out_dir, "smooth.json"), "w"), indent=1)
print(json.dumps(summary))
print(row(f"{os.path.basename(controller)} {servo}", M))
