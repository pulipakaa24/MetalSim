"""Diagnose the scripted expert on a servo option: phase, TCP error, arm sag per control step (one seed)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
root, servo, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
sys.path.insert(0, os.path.join(root, "rl", "native")); sys.path.append("~/MetalSim"); sys.path.insert(1, HERE)
import numpy as np, mujoco
from so101_lift import SO101LiftEnv
import scripted_expert as SE
from servo_cpu import attach
env = SO101LiftEnv(); attach(env, servo, **over); m, d = env.model, env.data
ex = SE.Expert(env); obs, _ = env.reset(seed=seed); ex.reset()
for t in range(300):
    ctrl = ex.act(); ph = ex.phase
    obs, r, term, trunc, info = env.step(SE.norm_ctrl(m, ctrl))
    q = d.qpos[env._arm_qadr]
    if t % 15 == 0 or t < 3:
        cube = d.xpos[env._box_body]; tcp = ex._tcp()
        print(f"t {t:3d} {ph:8s} |tcp-cube| {np.linalg.norm(tcp-cube):.3f} tcpz {tcp[2]:.3f} cube z {cube[2]:.3f} "
              f"cmd-q {np.round(ctrl - q, 3)} ")
