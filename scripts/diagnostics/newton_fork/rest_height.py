"""Newton fork item 5: the ~1 cm rest-height offset of the standing G1 (Newton 0.69-0.70 m vs MuJoCo 0.71 m).
Both engines, same USD, same Isaac gains, CPU. Variants:
  isaac : Isaac's init state (root z 0.74: feet 18.4 mm above the ground) and gains; the robot lands and slowly
          pitches forward (MuJoCo too), so the height at 0.25-0.5 s mixes landing and falling.
  quasi : feet starting on the ground (root z lowered by 18.4 mm), ankle kp 200 / kd 5 so the robot stands still;
          after 1.5 s: pelvis z, lowest foot point (penetration), pelvis height above the sole (FK) and leg joints.
usage: python scripts/diagnostics/newton_fork/rest_height.py [--it 4] [--dt_ms 1.25] [--kw k=v ...] [--drive actuator|solver]"""
import argparse, numpy as np, mujoco, warp as wp, newton
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import newton_backend as nb
import metalsim.interop.warp_metal as wm

class _E:
    def __init__(self, *x): self.v = 0
    def next_value(self): self.v += 1; return self.v
wm.SharedEvent = _E
ap = argparse.ArgumentParser(); ap.add_argument("--it", type=int, default=4); ap.add_argument("--dt_ms", type=float, default=1.25)
ap.add_argument("--kw", nargs="*", default=[]); ap.add_argument("--T", type=float, default=1.5); ap.add_argument("--relax", type=float, default=0.4); ap.add_argument("--mjdt_ms", type=float, default=2.5)
a = ap.parse_args()
skw = {}
import inspect
if "joint_armature_inertia" in inspect.signature(newton.solvers.SolverXPBD.__init__).parameters:
    skw["joint_armature_inertia"] = "none"
for kv in a.kw:
    k, v = kv.split("="); skw[k] = eval(v)
DROP = 0.0184259                              # lowest foot point above the ground at Isaac's init pose (both models)
LEG = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_hip_roll_joint", "left_ankle_roll_joint"]


def mj_run(quasi):
    m, _ = build_g1_model("flat", physics_dt=a.mjdt_ms * 1e-3); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_qpos[0][7:]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    if quasi:
        d.qpos[2] -= DROP
        for i, n in enumerate(names):
            if "ankle" in n:
                m.actuator_gainprm[i, 0] = 200.0; m.actuator_biasprm[i, 1] = -200.0; m.actuator_biasprm[i, 2] = -5.0
                m.actuator_forcerange[i] = [-300, 300]
    zs = []
    for k in range(int(a.T / 0.25)):
        for _ in range(int(round(0.25 / m.opt.timestep))): mujoco.mj_step(m, d)
        zs.append(d.qpos[2])
    feet = [g for g in range(m.ngeom) if "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
    low = min(((m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
                @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()) for g in feet)
    q = {n: d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] for i, n in enumerate(names)}
    return zs, low, q


def newton_run(quasi):
    m, _ = build_g1_model("flat", physics_dt=0.0025)
    sim = nb.NewtonSim(m, 1, iterations=a.it, dt=a.dt_ms * 1e-3, device="cpu", relaxation=a.relax, solver_kw=skw or None)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    q0 = m.key_qpos[0].astype(np.float32).copy()
    if quasi:
        q0[2] -= DROP
        kp = sim.actuator.kp.numpy(); kd = sim.actuator.kd.numpy(); eff = sim.actuator.effort.numpy()
        dof = sim.dof_of.numpy()
        for i, n in enumerate(names):
            if "ankle" in n:
                kp[dof[i]] = 200.0; kd[dof[i]] = 5.0; eff[dof[i]] = 300.0
        sim.actuator.kp.assign(kp); sim.actuator.kd.assign(kd); sim.actuator.effort.assign(eff)
    sim.d.qpos.assign(q0[None]); sim.d.qvel.zero_(); sim._reset_mask.fill_(True); sim.launch_reset()
    zs = []
    for k in range(int(a.T / 0.25)):
        for _ in range(int(round(0.25 / 0.02))): sim.launch_step()
        zs.append(float(sim.d.qpos.numpy()[0, 2]))
    M = sim.model; bq = sim.s0.body_q.numpy(); sb = M.shape_body.numpy(); lab = [l.split("/")[-1] for l in M.body_label]
    low = 1e9
    def rot(qq, v):
        x, y, z, w = qq; u = np.array([x, y, z]); return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)
    for s in range(M.shape_count):
        if sb[s] >= 0 and "ankle_roll" in lab[sb[s]]:
            p = bq[sb[s]]; X = M.shape_transform.numpy()[s]; h = M.shape_scale.numpy()[s]
            for c in [np.array([i * h[0], j * h[1], k * h[2]]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]:
                low = min(low, (p[:3] + rot(p[3:], X[:3] + rot(X[3:], c)))[2])
    qp = sim.d.qpos.numpy()[0]
    q = {n: qp[7 + i] for i, n in enumerate(names)}
    return zs, low, q


print(f"newton at {newton.__file__}; XPBD {a.it} it / {a.dt_ms} ms, relaxation {a.relax}, solver kwargs {skw}; MuJoCo C {a.mjdt_ms} ms")
for variant in ("isaac", "quasi"):
    zm, lm, qm = mj_run(variant == "quasi"); zn, ln, qn = newton_run(variant == "quasi")
    print(f"| {variant} | MuJoCo C | z every 0.25 s {np.round(zm, 4).tolist()} | lowest foot point {lm * 1e3:+.2f} mm | "
          + ", ".join(f"{n.replace('_joint', '')} {qm[n]:+.4f}" for n in LEG) + " |")
    print(f"| {variant} | Newton XPBD | z every 0.25 s {np.round(zn, 4).tolist()} | lowest foot point {ln * 1e3:+.2f} mm | "
          + ", ".join(f"{n.replace('_joint', '')} {qn[n]:+.4f}" for n in LEG) + " |", flush=True)
