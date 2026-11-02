"""7v4_real numbers the deployment expert's speed / acceleration limits are checked against (so101Sim venv)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); MS = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(sys.argv[1], "rl", "native")); sys.path.append(MS)
import mujoco, numpy as np
from so101_lift import SO101LiftEnv
from metalsim.render.film.so101 import apply_servo
env = SO101LiftEnv(); m, d = env.model, env.data
ap = apply_servo(m, "7v4_real")
print({k: ap[k] for k in ("forcerange_Nm", "damping_Nms", "kp", "frictionloss", "armature")}, ap["setpoint"])
va = env._arm_vadr
G, M = [], []
for seed in range(10000, 10100):
    env.reset(seed=seed)
    for q in (d.qpos.copy(),):
        d.qvel[:] = 0; mujoco.mj_forward(m, d)
        G.append(d.qfrc_bias[va].copy())
        Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, Mf); M.append(np.diag(Mf)[va])
# gravity over the arm's whole joint range for the expert's poses: sample uniformly in ctrlrange (arm only)
rng = np.random.default_rng(0); Gr = []
lo, hi = m.actuator_ctrlrange[:5, 0], m.actuator_ctrlrange[:5, 1]
for _ in range(20000):
    d.qpos[env._arm_qadr[:5]] = rng.uniform(lo, hi); d.qvel[:] = 0; mujoco.mj_forward(m, d)
    Gr.append(d.qfrc_bias[va].copy())
G, M, Gr = np.abs(np.array(G)), np.array(M), np.abs(np.array(Gr))
fr, b, c = np.array(ap["forcerange_Nm"]), np.array(ap["damping_Nms"]), np.array(ap["frictionloss"])
np.set_printoptions(precision=3, suppress=True)
print("gravity |tau| at start poses max", G.max(0), "\n over ctrlrange p99", np.percentile(Gr, 99, 0), "max", Gr.max(0))
print("M_jj (incl armature) start poses", M.mean(0))
for name, g in (("start-pose max gravity", G.max(0)), ("ctrlrange p99 gravity", np.percentile(Gr, 99, 0))):
    v = (fr - c - g) / b
    print(name, "-> line speed where gravity uses the rest of the torque (rad/s):", v)
    print("   accel the motor can add at 3 rad/s over gravity (rad/s^2):", (fr - c - g - b * 3.0) / M.mean(0))
