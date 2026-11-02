"""Which phases so101Sim's expert reaches, over seeds, on a servo option (failure-mode census). so101Sim venv.
  python phases.py <so101Sim> <servo> <n> [--cmdbase]"""
import os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
root, servo, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); CB = "--cmdbase" in sys.argv
sys.path.insert(0, os.path.join(root, "rl", "native")); sys.path.append("~/MetalSim"); sys.path.insert(1, HERE)
import numpy as np
from so101_lift import SO101LiftEnv
import scripted_expert as SE
from servo_cpu import attach
from expert_variants import CmdBaseExpert
env = SO101LiftEnv(); attach(env, servo); m, d = env.model, env.data
ex = (CmdBaseExpert if CB else SE.Expert)(env)
reach, tcpz_min, grip_at_lift, succ = collections.Counter(), [], [], 0
for s in range(N):
    env.reset(seed=10000 + s); ex.reset(); seen = set(); zmin = 1; gl = None; ok = False
    for t in range(300):
        c = ex.act(); seen.add(ex.phase)
        if ex.phase == "lift" and gl is None: gl = float(d.qpos[env._arm_qadr[5]])
        if ex.phase == "approach": zmin = min(zmin, float(ex._tcp()[2]))
        _, _, term, trunc, info = env.step(SE.norm_ctrl(m, c)); ok |= info["is_success"]
        if term: break
    succ += ok
    reach[max(seen, key=["approach", "descend", "grasp", "lift"].index)] += 1
    tcpz_min.append(zmin); grip_at_lift.append(gl)
g = [x for x in grip_at_lift if x is not None]
print(servo, "cmdbase" if CB else "expert", "success", succ, "furthest phase", dict(reach),
      "approach tcp z min median %.3f" % np.median(tcpz_min), "gripper q at lift start median", np.round(np.median(g), 2) if g else None)
