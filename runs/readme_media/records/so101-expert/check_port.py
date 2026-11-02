"""Port check: DeployExpert's kinematics against so101Sim's scripted_expert.Expert (so101Sim venv, MuJoCo 3.10).
TCP / jaw constants equal Expert.__init__'s calibration; one IK iteration (DeployExpert._ik_iter on a state) equals
Expert._ik_step on the same state (align on and off), over 100 seeds x 20 random arm poses."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); MS = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(sys.argv[1], "rl", "native")); sys.path.append(MS)
import mujoco, numpy as np
import scripted_expert as SE
from so101_lift import SO101LiftEnv
import metalsim.render.film.so101_expert as X
env = SO101LiftEnv(); m, d = env.model, env.data
ex = SE.Expert(env)
assert np.array_equal(ex.tcp_local, X.TCP_LOCAL) and np.array_equal(ex.jaw_local, X.JAW_LOCAL), "calibration differs"
ids = dict(grip_site=env._grip_site, box_body=env._box_body, box_qadr=env._box_qadr, arm_qadr=env._arm_qadr, arm_vadr=env._arm_vadr)
dx = X.DeployExpert(m, ids, vmax=np.ones(5), amax=np.ones(5))
rng = np.random.default_rng(1); worst = 0.0; n = 0
for seed in range(10000, 10100):
    env.reset(seed=seed)
    for _ in range(20):
        d.qpos[env._arm_qadr[:5]] = rng.uniform(m.actuator_ctrlrange[:5, 0], m.actuator_ctrlrange[:5, 1]) * 0.6
        mujoco.mj_forward(m, d)
        tgt = d.xpos[env._box_body] + rng.uniform(-0.05, 0.15, 3)
        for align in (True, False):
            ex._align = align
            q_ref, e_ref = ex._ik_step(tgt)
            dx.pd.qpos[:] = d.qpos; mujoco.mj_forward(m, dx.pd)
            q, e = dx._ik_iter(tgt, align)
            worst = max(worst, float(np.max(np.abs(q - q_ref))), abs(e - e_ref)); n += 1
print(f"calibration identical; {n} IK iterations, max |dq - dq_ref| / |de| = {worst:.3e}")
