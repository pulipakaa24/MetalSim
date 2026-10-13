"""Franka-Pour env trajectory hashes (rigid_deterministic=True, captured sim steps, seeded random actions): body and
particle state hashed every policy step; compare two runs' JSON with --compare.
    franka_bitwise.py NUM_ENVS STEPS OUT.json [--compare OTHER.json]"""
import hashlib
import json
import sys

import numpy as np
import torch
import warp as wp

N = int(sys.argv[1]); STEPS = int(sys.argv[2]); OUT = sys.argv[3]
CMP = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--compare" else None
wp.config.quiet = True
from metalsim.learn import franka_pour as P  # noqa: E402

pcfg = P.S.PourPhysicsConfig(num_envs=N, rigid_deterministic=True)
env = P.FrankaPourEnv(P.PourTaskConfig(num_envs=N, seed=0, physics=pcfg))
env.reset_all()
g = torch.Generator(device=env.device).manual_seed(0)
hashes = []
for k in range(STEPS):
    a = torch.rand(env.num_envs, 8, device=env.device, generator=g) * 2 - 1
    env.step(a)
    ph = env.physics
    h = hashlib.sha1()
    for arr in (ph.s0.body_q, ph.s0.body_qd, ph.s0.joint_q, ph.s0.particle_q, ph.s0.particle_qd):
        if arr is not None:
            h.update(np.ascontiguousarray(arr.numpy()).tobytes())
    hashes.append(h.hexdigest())
env.physics.check_status()
res = {"num_envs": N, "steps": STEPS, "hashes": hashes}
if CMP:
    ref = json.load(open(CMP))["hashes"]
    eq = [a == b for a, b in zip(hashes, ref)]
    res["equal_steps"] = sum(eq)
    res["first_difference"] = eq.index(False) if False in eq else None
    print("bitwise equal steps", sum(eq), "of", len(eq), "first difference", res["first_difference"])
json.dump(res, open(OUT, "w"), indent=1)
print("done", OUT)
