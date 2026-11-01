"""Diagnostic: the policy's deterministic output BEFORE the [-1, 1] clip (the Gaussian mean), on so101Sim's env at a
servo model, 20 episodes (seeds 10000-10019). Tests whether the mean sits far outside [-1, 1], where a penalty on the
clipped action (smooth_env.py, variant 1) cannot see it. Usage (so101Sim venv):
  python raw_actions.py <so101Sim root> <policy.pt> <servo> [episodes]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METALSIM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
root, pol, servo = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 20
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.append(METALSIM)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402

from metalsim.render.film.so101 import apply_servo  # noqa: E402

P = torch.load(pol, weights_only=True)
W = P["pi"]
mean, var = P["obs_mean"].numpy(), P["obs_var"].numpy()


def raw(obs):
    n = np.clip((obs - mean) / np.sqrt(var + P["obs_eps"]), -P["clip_obs"], P["clip_obs"])
    x = torch.as_tensor(n[None].astype(np.float32))
    with torch.no_grad():
        x = torch.tanh(x @ W["mlp_extractor.policy_net.0.weight"].T + W["mlp_extractor.policy_net.0.bias"])
        x = torch.tanh(x @ W["mlp_extractor.policy_net.2.weight"].T + W["mlp_extractor.policy_net.2.bias"])
        return (x @ W["action_net.weight"].T + W["action_net.bias"]).numpy()[0]


env = SO101LiftEnv()
apply_servo(env.model, servo)
R = []
for ep in range(N):
    obs, _ = env.reset(seed=10_000 + ep)
    for t in range(300):
        u = raw(obs)
        R.append(u)
        obs, r, term, trunc, info = env.step(np.clip(u, -1, 1))
        if term or trunc:
            break
R = np.abs(np.array(R))
out = dict(policy=os.path.basename(pol), servo=servo, episodes=N, steps=len(R),
           log_std=P["log_std"].exp().numpy().round(3).tolist() if "log_std" in P else None,
           frac_abs_gt1=R.__gt__(1).mean(0).round(3).tolist(), median_abs=np.median(R, 0).round(2).tolist(),
           p90_abs=np.percentile(R, 90, 0).round(2).tolist(), frac_abs_gt3=R.__gt__(3).mean(0).round(3).tolist())
print(json.dumps(out))
