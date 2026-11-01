"""Extract so101Sim's SB3 lift teacher into plain torch weights, so MetalSim plays it without stable_baselines3.

Reads so101Sim's run directory (rl/runs/lift_ppo: model.zip from SB3 PPO.save, vecnorm.pkl from VecNormalize.save) and
writes one ``torch.save`` file of plain tensors, strings and numbers (loadable with ``torch.load(weights_only=True)``):

* ``pi``: the actor, SB3 ``MlpPolicy`` with ``net_arch=[256, 256]`` (so101Sim rl/native/train_lift.py): Linear 21->256,
  Tanh, Linear 256->256, Tanh, Linear 256->6 (``mlp_extractor.policy_net.{0,2}``, ``action_net``), float32;
* ``obs_mean`` / ``obs_var`` (float64), ``obs_eps``, ``clip_obs``: VecNormalize's observation statistics, applied as
  so101Sim rl/native/eval_lift_native.py applies them: clip((obs - mean) / sqrt(var + eps), -clip, clip);
* ``log_std`` (unused by the deterministic film), ``source``: file hashes, SB3 / torch versions, run steps.

The deterministic action is the actor's mean, clipped to the action space [-1, 1] (SB3 ``predict``). The script checks
the plain forward against ``PPO.predict(deterministic=True)`` on 4096 normalised observations drawn around the
statistics and writes the largest difference into the output's ``check``.

Usage (so101Sim's environment, from its urlab_bridge/ directory):
  uv run --no-sync python <MetalSim>/runs/film/so101_lift/port/extract_policy.py <so101Sim>/rl/runs/lift_ppo <out.pt>
"""
import hashlib
import json
import os
import pickle
import sys
import zipfile

import numpy as np
import torch

run, out = sys.argv[1], sys.argv[2]
import stable_baselines3
from stable_baselines3 import PPO

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
model_zip, vecnorm = os.path.join(run, "model.zip"), os.path.join(run, "vecnorm.pkl")
model = PPO.load(os.path.join(run, "model"), device="cpu")
with open(vecnorm, "rb") as f:
    vn = pickle.load(f)
rms = vn.obs_rms
sd = model.policy.state_dict()
pi = {k: sd[k].detach().clone().float() for k in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
                                                   "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
                                                   "action_net.weight", "action_net.bias")}
with zipfile.ZipFile(model_zip) as z:
    data = json.loads(z.read("data"))
progress = os.path.join(run, "progress.txt")
last = open(progress).read().strip().splitlines()[-1] if os.path.exists(progress) else None


def plain(obs_norm32):
    x = torch.as_tensor(obs_norm32)
    x = torch.tanh(x @ pi["mlp_extractor.policy_net.0.weight"].T + pi["mlp_extractor.policy_net.0.bias"])
    x = torch.tanh(x @ pi["mlp_extractor.policy_net.2.weight"].T + pi["mlp_extractor.policy_net.2.bias"])
    return torch.clamp(x @ pi["action_net.weight"].T + pi["action_net.bias"], -1.0, 1.0).numpy()


rng = np.random.default_rng(0)
obs = rms.mean + np.sqrt(rms.var) * rng.standard_normal((4096, len(rms.mean))) * 1.5
n = np.clip((obs - rms.mean) / np.sqrt(rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
ref, _ = model.predict(n, deterministic=True)
mine = plain(n.astype(np.float32))
check = dict(samples=4096, max_abs_diff=float(np.abs(ref - mine).max()), against="PPO.predict(deterministic=True)")
print("check", check)
blob = dict(
    format="so101_lift_sb3_mlp_v1",
    pi=pi, activation="tanh", log_std=sd["log_std"].detach().clone().float(),
    obs_mean=torch.as_tensor(np.asarray(rms.mean, np.float64)), obs_var=torch.as_tensor(np.asarray(rms.var, np.float64)),
    obs_eps=float(vn.epsilon), clip_obs=float(vn.clip_obs), obs_count=float(rms.count),
    action_low=-1.0, action_high=1.0,
    source=dict(model_zip_sha256=sha(model_zip), vecnorm_pkl_sha256=sha(vecnorm), num_timesteps=int(data["num_timesteps"]),
                progress_last=last, stable_baselines3=str(stable_baselines3.__version__), torch=str(torch.__version__),
                policy_class=data["policy_class"]["__module__"] + "." + data["policy_class"].get("__qualname__", "")
                if isinstance(data.get("policy_class"), dict) else str(data.get("policy_class"))),
    check=check)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
torch.save(blob, out)
print("wrote", out, "sha256", sha(out), "num_timesteps", data["num_timesteps"])
