"""extract_policy.py's check compares its [-1, 1]-clamped forward with PPO.predict(deterministic=True), which clips to
the model's own action space. For the raw variant that space is [-1000, 1000], so the check reports the clip, not an
extraction error. This repeats the check with predict's output clipped to [-1, 1] (what so101Sim's env applies).
Usage (so101Sim venv): python check_extract_raw.py <run dir> <extracted.pt>"""
import pickle
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

run, pt = sys.argv[1:3]
model = PPO.load(run + "/model", device="cpu")
vn = pickle.load(open(run + "/vecnorm.pkl", "rb"))
P = torch.load(pt, weights_only=True)
W = P["pi"]
rng = np.random.default_rng(0)
rms = vn.obs_rms
obs = rms.mean + np.sqrt(rms.var) * rng.standard_normal((4096, len(rms.mean))) * 1.5
n = np.clip((obs - rms.mean) / np.sqrt(rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
ref, _ = model.predict(n, deterministic=True)
x = torch.as_tensor(n.astype(np.float32))
x = torch.tanh(x @ W["mlp_extractor.policy_net.0.weight"].T + W["mlp_extractor.policy_net.0.bias"])
x = torch.tanh(x @ W["mlp_extractor.policy_net.2.weight"].T + W["mlp_extractor.policy_net.2.bias"])
mine = torch.clamp(x @ W["action_net.weight"].T + W["action_net.bias"], -1, 1).numpy()
print("model action space", model.action_space, "max |clip(predict) - plain|", float(np.abs(np.clip(ref, -1, 1) - mine).max()),
      "unclipped predict |.|max", float(np.abs(ref).max()))
