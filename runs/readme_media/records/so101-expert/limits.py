"""Planning limits VMAX / AMAX for metalsim/render/film/so101_expert.py from the 7v4_real servo and the task's poses.

so101Sim venv. For seeds 10000-10099 (so101Sim's reset): the start pose, then the DeployExpert IK chain on the kinematic
model: approach (cube + 10 cm, yaw aligned), descend (cube + 8 mm, aligned), lift ((cube xy), 0.20 m). Along the
straight joint-space segments start -> approach -> descend -> lift (11 samples each): gravity torque per joint
(qfrc_bias at rest; on the lift segment plus the box at its heaviest, 1.4 x mass, hung at the TCP: J^T m g) and the
joint-space inertia diagonal (armature included). R = forcerange - Coulomb - g_max; VMAX = R / 2b, AMAX = R / 2M_max,
capped by the firmware profile (speed_limit, accel_limit). Also prints D = M R / b^2 (the move for which the half
split is time-optimal) and the segments' joint distances. Writes limits.json.
  <so101Sim>/urlab_bridge/.venv/bin/python limits.py <so101Sim root>
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); MS = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(sys.argv[1], "rl", "native")); sys.path.append(MS)
import mujoco, numpy as np
from so101_lift import SO101LiftEnv
from metalsim.render.film.so101 import apply_servo
import metalsim.render.film.so101_expert as X

env = SO101LiftEnv(); m, d = env.model, env.data
ap = apply_servo(m, "7v4_real")
ids = dict(grip_site=env._grip_site, box_body=env._box_body, box_qadr=env._box_qadr, arm_qadr=env._arm_qadr, arm_vadr=env._arm_vadr)
ex = X.DeployExpert(m, ids, vmax=np.ones(5), amax=np.ones(5))
va, qa = env._arm_vadr[:5], env._arm_qadr[:5]
kd = mujoco.MjData(m)
G, Mx, dist, ikerr = [], [], [], []
mbox = env._box_mass0 * 1.4
for seed in range(10000, 10100):
    env.reset(seed=seed)
    cube = d.xpos[env._box_body].copy()
    q0 = d.qpos[qa].copy()
    qa_, ea = ex.solve(d, cube + [0, 0, X.APPROACH_DZ], True, q0)
    qd_, ed = ex.solve(d, cube + [0, 0, X.GRASP_DZ], True, qa_)
    ql_, el = ex.solve(d, np.r_[cube[:2], X.LIFT_Z], False, qd_)
    ikerr.append((ea, ed, el))
    dist.append(np.abs([qa_ - q0, qd_ - qa_, ql_ - qd_]))
    for k, (a, b) in enumerate(((q0, qa_), (qa_, qd_), (qd_, ql_))):
        for s in np.linspace(0, 1, 11):
            kd.qpos[:] = d.qpos; kd.qpos[qa] = a + s * (b - a); kd.qvel[:] = 0
            mujoco.mj_forward(m, kd)
            g = kd.qfrc_bias[va].copy()
            if k == 2:
                tcp = ex.tcp(kd); jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, kd, jp, None, tcp, ex.site_body)
                g += jp[:, va].T @ np.array([0, 0, mbox * 9.81])
            Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, kd, Mf)
            G.append(np.abs(g)); Mx.append(np.diag(Mf)[va])
G, Mx, dist, ikerr = np.array(G), np.array(Mx), np.array(dist), np.array(ikerr)
fr, b, c = np.array(ap["forcerange_Nm"][:5]), np.array(ap["damping_Nms"][:5]), np.array(ap["frictionloss"][:5])
sp = ap["setpoint"]
g, M = G.max(0), Mx.max(0)
R = fr - c - g
V = np.minimum(R / (2 * b), sp["speed_limit"]); A = np.minimum(R / (2 * M), sp["accel_limit"])
D = M * R / b ** 2
np.set_printoptions(precision=3, suppress=True)
print("box mass max", mbox, "IK residual max (m)", ikerr.max(0))
print("g_max", g, "\nM_max", M, "\nR", R, "\nVMAX", V, "\nAMAX", A, "\nD(opt half split)", D)
print("joint distance per segment median\n", np.median(dist, 0), "\n p90\n", np.percentile(dist, 90, 0))
json.dump(dict(script="runs/readme_media/scratch/so101-expert/limits.py", servo=ap, box_mass_max=mbox, g_max=g.tolist(),
               M_max=M.tolist(), R=R.tolist(), VMAX=V.tolist(), AMAX=A.tolist(), D_half_split_opt=D.tolist(),
               ik_residual_max=ikerr.max(0).tolist(), dist_median=np.median(dist, 0).tolist(),
               dist_p90=np.percentile(dist, 90, 0).tolist()), open(os.path.join(HERE, "limits.json"), "w"), indent=1)
