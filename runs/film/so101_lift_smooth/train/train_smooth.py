"""Continue so101Sim's lift teacher (run 2, rl/runs/lift_ppo, 100/100 at 12 V) on the 7.4 V servo model with Isaac Lab's
lift-task smoothness penalties and their curriculum (smooth_env.py). Everything else is so101Sim's: SO101LiftEnv's
reward, reset and loop, train_lift.py's 12 SubprocVecEnv workers (seed 1000 + rank, Monitor), VecNormalize continued
from run 2's statistics, and the PPO settings stored in run 2's model.zip (lr 3e-4, n_steps 512, batch 1024, 10
epochs, gamma 0.99, GAE 0.95, clip 0.2, ent 0.005, [256, 256]), continued as continue_lift.py does
(reset_num_timesteps=False). Run 2's files are only read.

Curriculum (Isaac Lab contrib/lift/lift_env_cfg.py CurriculumCfg: ``modify_reward_weight`` sets both weights from
-1e-4 to -1e-1 once ``common_step_counter`` > 10000; its rsl_rl config runs 1500 iterations x 24 steps = 36000
steps, so the switch comes at 27.8 % of training). Here: the initial weights for the first 27.8 % of this run's
steps, the final weights after. The weights of a variant are Isaac Lab's times ``scale`` (see VARIANTS).

Logs (in <so101Sim>/rl/runs/lift_ppo_smooth/<variant>/): progress.txt (every 24k steps: the last 100 training
episodes, stochastic actions), evals.txt (every 1M steps and at the end: 20 deterministic episodes, seeds 20000-20019,
7.4 V), checkpoints ckpt_<steps>/, and model.zip + vecnorm.pkl at the end.

Usage (so101Sim's venv): .venv/bin/python train_smooth.py <so101Sim root> <variant> [total_steps]
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METALSIM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
root, variant = sys.argv[1:3]
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.append(METALSIM)
sys.path.insert(1, HERE)
import numpy as np  # noqa: E402
from smooth_env import INFO_KEYS, SmoothLiftEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

ISAAC = dict(w_initial=1e-4, w_final=1e-1, switch_common_steps=10_000, total_common_steps=1500 * 24,
             source="IsaacLab3 contrib/lift/lift_env_cfg.py RewardsCfg action_rate/joint_vel, CurriculumCfg; "
                    "config/franka/agents/rsl_rl_ppo_cfg.py max_iterations 1500, num_steps_per_env 24")
# Isaac Lab's largest task reward per step: reaching 1 + lifting 15 + goal tracking 16 + fine goal tracking 5 = 37
# (every term times dt, which cancels in the ratios); so101Sim's: reach 1 + lift 5 + held bonus 3 = 9. The reaching
# terms are the same function (1 - tanh(d / 0.1), weight 1).
VARIANTS = {
    "isaac_scaled": dict(scale=9 / 37, note="Isaac Lab's weights times so101Sim / Isaac Lab largest task reward per step"),
    "isaac_unscaled": dict(scale=1.0, note="Isaac Lab's weights as they are (matched to the identical reaching term)"),
    # variant 2 (chosen after variant 1, see the so101-smooth hand-off): Isaac Lab's weights as they are, on the raw
    # policy output as Isaac Lab computes action_rate_l2 (smooth_env.py ``raw``); the arm still gets the clipped action
    "isaac_unscaled_raw": dict(scale=1.0, raw=True,
                               note="Isaac Lab's weights as they are, action_rate_l2 on the raw (unclipped) policy "
                                    "output as Isaac Lab computes it; the env clips to [-1, 1] for the servo targets"),
}
V = VARIANTS[variant]
RAW = bool(V.get("raw", False))
W0 = ISAAC["w_initial"] * V["scale"]
W1 = ISAAC["w_final"] * V["scale"]
SWITCH = int(round(TOTAL * ISAAC["switch_common_steps"] / ISAAC["total_common_steps"]))
SRC = os.path.join(root, "rl", "runs", "lift_ppo")
RUN = os.path.join(root, "rl", "runs", "lift_ppo_smooth", variant + ("_smoke" if TOTAL < 1_000_000 else ""))
N_ENVS = 12


def make_env(rank):
    def _f():
        env = SmoothLiftEnv(servo="7v4", w_rate=W0, w_vel=W0, raw=RAW)
        env.reset(seed=1000 + rank)
        return Monitor(env, info_keywords=INFO_KEYS)
    return _f


def det_eval(model, venv, n=20):
    """n deterministic episodes on seeds 20000.. at 7.4 V with the current observation statistics."""
    env = SmoothLiftEnv(servo="7v4", raw=RAW)
    out = dict(success=0, up_end=0, act_rate=[], sat_arm=[], sat_grip=[], peak_v=[], ttl=[])
    for k in range(n):
        obs, _ = env.reset(seed=20_000 + k)
        succ_t = None
        for t in range(300):
            a, _ = model.predict(venv.normalize_obs(obs[None]), deterministic=True)
            obs, r, term, trunc, info = env.step(a[0])
            if info["is_success"] and succ_t is None:
                succ_t = t
            if term or trunc:
                break
        out["success"] += succ_t is not None
        out["up_end"] += info["lifted_end"] > 0
        out["ttl"] += [succ_t * 0.02] if succ_t is not None else []
        for key, ik in (("act_rate", "ep_act_rate"), ("sat_arm", "ep_sat_arm"), ("sat_grip", "ep_sat_grip"),
                        ("peak_v", "ep_peak_v")):
            out[key].append(info[ik])
    return dict(success=out["success"], up_end=out["up_end"], n=n,
                ttl=round(float(np.mean(out["ttl"])), 3) if out["ttl"] else None,
                act_rate=round(float(np.mean(out["act_rate"])), 4), sat_arm=round(float(np.mean(out["sat_arm"])), 4),
                sat_grip=round(float(np.mean(out["sat_grip"])), 4),
                peak_v_median=round(float(np.median(out["peak_v"])), 3))


class Curriculum(BaseCallback):
    def __init__(self, start):
        super().__init__()
        self.start, self.switched, self.next_eval = start, False, 0
        self.t0 = time.time()
        self.f = open(os.path.join(RUN, "progress.txt"), "a", buffering=1)
        self.fe = open(os.path.join(RUN, "evals.txt"), "a", buffering=1)

    def steps(self):
        return self.num_timesteps - self.start

    def save(self, tag):
        d = os.path.join(RUN, f"ckpt_{tag}")
        os.makedirs(d, exist_ok=True)
        self.model.save(os.path.join(d, "model"))
        self.model.get_env().save(os.path.join(d, "vecnorm.pkl"))

    def evaluate(self):
        r = det_eval(self.model, self.model.get_env())
        self.fe.write(json.dumps(dict(steps=self.steps(), w=self.model.get_env().get_attr("w", [0])[0], **r)) + "\n")

    def _on_step(self):
        s = self.steps()
        if s >= self.next_eval:
            self.evaluate()
            if self.next_eval > 0:
                self.save(f"{s // 1_000_000}M")
            self.next_eval += 1_000_000
        if not self.switched and s >= SWITCH:
            self.save("switch")
            self.model.get_env().env_method("set_weights", W1, W1)
            self.switched = True
            self.f.write(f"# switch at steps={s}: weights -> {W1:.4g}\n")
        if self.n_calls % 2000 == 0:
            buf = self.model.ep_info_buffer
            if buf:
                g = lambda k: float(np.mean([e.get(k, np.nan) for e in buf]))
                fps = int(s / max(1e-9, time.time() - self.t0))
                self.f.write(f"steps={s} success={g('is_success'):.2f} up_end={g('lifted_end'):.2f} "
                             f"ep_rew={g('r'):.1f} task_r={g('ep_task_r'):.2f} pen={g('ep_pen'):.3f} "
                             f"rate_l2={g('ep_rate_l2'):.2f} vel_l2={g('ep_vel_l2'):.2f} act_rate={g('ep_act_rate'):.3f} "
                             f"raw_l2={g('ep_raw_l2'):.2f} sat_arm={g('ep_sat_arm'):.3f} sat_grip={g('ep_sat_grip'):.3f} peak_v={g('ep_peak_v'):.2f} "
                             f"w={self.model.get_env().get_attr('w', [0])[0][0]:.3g} "
                             f"std={np.exp(self.model.policy.log_std.detach().numpy()).round(2).tolist()} fps={fps}\n")
        return True


if __name__ == "__main__":
    if os.path.exists(RUN):
        raise SystemExit(f"{RUN} exists: a variant is trained once")
    os.makedirs(RUN)
    for f in ("model.zip", "vecnorm.pkl"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(RUN, f"start_{f}"))
    cfg = dict(variant=variant, total_steps=TOTAL, switch_at_steps=SWITCH, w_initial=W0, w_final=W1, servo="7v4",
               start=dict(dir="rl/runs/lift_ppo (run 2)"), isaac=ISAAC, variant_cfg=V, n_envs=N_ENVS,
               script="runs/film/so101_lift_smooth/train/train_smooth.py", started=time.ctime())
    json.dump(cfg, open(os.path.join(RUN, "config.json"), "w"), indent=1)
    set_random_seed(0)
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecNormalize.load(os.path.join(SRC, "vecnorm.pkl"), venv)
    venv.training = True
    # raw: the stored action space is [-1, 1]; SB3 clips samples to the model's action space, so it is replaced by the
    # env's wider one (same policy network; only the clip bound changes)
    # (VecNormalize.load keeps the pickled wrapper's own action_space, run 2's [-1, 1]: set it from the env too)
    if RAW:
        venv.action_space = venv.venv.action_space
    custom = dict(action_space=venv.action_space) if RAW else None
    model = PPO.load(os.path.join(SRC, "model"), env=venv, device="cpu", custom_objects=custom)
    start = int(model.num_timesteps)
    cb = Curriculum(start)
    model.learn(total_timesteps=TOTAL, callback=cb, reset_num_timesteps=False)
    cb.evaluate()
    model.save(os.path.join(RUN, "model"))
    venv.save(os.path.join(RUN, "vecnorm.pkl"))
    cfg.update(finished=time.ctime(), num_timesteps=int(model.num_timesteps))
    json.dump(cfg, open(os.path.join(RUN, "config.json"), "w"), indent=1)
    print("saved to", RUN)
