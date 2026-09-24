"""Why Newton XPBD's joint drives looked dead, and how faithful each drive model is against MuJoCo C.

  A. One-joint pendulum (revolute, kp 200 / kd 5, target 0.5 rad): the drive DOES engage; State.joint_q
     is simply never written by XPBD (maximal coordinates; read joints with eval_ik). XPBD's compliance
     drive is biased: its gravity sag scales with 1/iterations instead of tau/kp. The PD actuator
     (metalsim.physics.newton_backend.ActuatorPD, joint_f) gives tau/kp at any iteration count, but only
     with equal linear/angular joint relaxation (Newton's 0.7/0.4 default mis-transmits joint torques).
  B. G1 with the base fixed 1 m up: joint angles after 2 s vs the exact static PD equilibrium
     kp (q* - q) = g(q) (gravity torques from MuJoCo C's model of the same USD).
  C. G1 floating PD hold from Isaac's default pose (MuJoCo C also falls, by ~1.4 s: Isaac's 20 Nm/rad
     ankles cannot hold the default pose): pelvis-height timeline and joint deviation vs MuJoCo C.
  D. Metal vs CPU for the same Newton setup.

usage: python scripts/diagnostics/newton_xpbd_drives.py [A|B|C|D ...]"""
import sys, time, numpy as np, warp as wp, newton, mujoco
wp.config.quiet = True
from metalsim.physics.newton_backend import G1XPBD, ActuatorPD
from metalsim.learn.g1_velocity import build_g1_model

PARTS = sys.argv[1:] or ["A", "B", "C", "D"]
H = 0.0025


def pendulum(iters, mode, dev="metal:0", T=2.0, relax=0.4):
    with wp.ScopedDevice(dev):
        b = newton.ModelBuilder(); b.gravity = -9.81
        link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
        b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
        j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                                 target_ke=200.0 if mode == "xpbd" else 0.0, target_kd=5.0 if mode == "xpbd" else 0.0)
        b.add_articulation([j]); b.joint_target_mode = [newton.JointTargetMode.POSITION]; b.joint_target_q = [0.5]
        m = b.finalize(); solver = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=relax, joint_angular_relaxation=0.4)
        s0, s1 = m.state(), m.state(); c = m.control()
        act = ActuatorPD(m, [200.0], [5.0], [1e6], [0.5], dt=H) if mode == "ipd" else None
        for _ in range(int(T / H)):
            if act: act.apply(s0, c)
            s0.clear_forces(); solver.step(s0, s1, c, None, H); s0, s1 = s1, s0
        q = wp.zeros(1, dtype=float); qd = wp.zeros(1, dtype=float); newton.eval_ik(m, s0, q, qd)
        mass = m.body_mass.numpy()[0]; com = m.body_com.numpy()[0]
        return float(s0.joint_q.numpy()[0]), float(q.numpy()[0]), mass, com


def mj_model():
    m, _ = build_g1_model("flat", physics_dt=H)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    return m, names


def newton_names(sim):
    m = sim.model; nj = m.joint_count // sim.n_envs; qs = m.joint_q_start.numpy(); qds = m.joint_qd_start.numpy()
    return {m.joint_label[j].split("/")[-1]: (int(qs[j]), int(qds[j])) for j in range(nj) if qds[j + 1] - qds[j] == 1}


GROUPS = {"legs": ("hip", "knee", "torso"), "ankles": ("ankle",), "arms": ("shoulder", "elbow"),
          "hands": ("zero", "one", "two", "three", "four", "five", "six")}


def group_of(name):
    return next(g for g, keys in GROUPS.items() if any(k in name for k in keys))


if "A" in PARTS:
    print("A. pendulum (6 kg, COM 0.208 m from the pivot), target 0.5 rad, kp 200 kd 5, 2 s at 2.5 ms, joint relaxation 0.4/0.4")
    tau_g = None
    for mode, its in (("xpbd", (1, 2, 4, 8, 16)), ("ipd", (1, 4, 16))):
        for it in its:
            jq, q, mass, com = pendulum(it, mode)
            tau_g = mass * 9.81 * com[0] * np.cos(q)
            print(f"  {mode:4s} it {it:2d}: State.joint_q {jq:.3f} (never written) | eval_ik q {q:.4f} | sag {q - 0.5:+.4f} rad "
                  f"(exact PD: tau_g/kp = {tau_g / 200:+.4f}) -> effective stiffness {tau_g / (q - 0.5):.0f} Nm/rad", flush=True)
    for relax in (0.7, 0.4):
        _, q, mass, com = pendulum(4, "ipd", relax=relax); tau_g = mass * 9.81 * com[0] * np.cos(q)
        print(f"  ipd it 4, linear/angular relaxation {relax}/0.4: sag {q - 0.5:+.4f} (exact {tau_g / 200:+.4f})")
    jq, qc, _, _ = pendulum(4, "xpbd", "cpu"); _, qm, _, _ = pendulum(4, "xpbd")
    print(f"  Metal vs CPU (xpbd it 4): {qm:.6f} vs {qc:.6f}")

if "B" in PARTS:
    print("B. G1 fixed base 1 m up, after 2 s: joint angle minus the exact static PD equilibrium, max |dq| per group [rad]")
    # exact static reference: fixed base at rest => kp (q* - q) = g(q) per joint (clipped at the effort limit);
    # g = MuJoCo's generalized gravity force at zero velocity. Fixed-point iteration on q.
    m, names = mj_model(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 1.0; d.qvel[:] = 0
    kp = m.actuator_gainprm[:, 0]; qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]; dadr = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    tgt = m.key_qpos[0][qadr]; lim = m.actuator_forcerange[:, 1]
    for _ in range(200):
        mujoco.mj_forward(m, d); d.qpos[qadr] = tgt - np.clip(d.qfrc_bias[dadr], -lim, lim) / kp
    mujoco.mj_forward(m, d)
    mj_q = {names[j]: d.qpos[m.jnt_qposadr[j]] for j in range(1, m.njnt)}
    key = {names[j]: m.key_qpos[0][m.jnt_qposadr[j]] for j in range(1, m.njnt)}
    sag = {g: max(abs(mj_q[n] - key[n]) for n in mj_q if group_of(n) == g) for g in GROUPS}
    print(f"  static equilibrium's sag from the targets (max per group): " + ", ".join(f"{g} {v:.3f}" for g, v in sag.items()))
    for drive, it, h in (("xpbd", 4, H), ("xpbd", 8, H), ("xpbd", 16, H), ("xpbd", 8, H / 2), ("ipd", 1, H), ("ipd", 2, H), ("ipd", 4, H), ("ipd", 8, H)):
        sim = G1XPBD(1, iterations=it, dt=h, drive=drive, floating=False, z0=1.0, armature_inertia="iso" if drive == "ipd" else False)
        sim.step(int(2.0 / h)); q, qd = sim.joint_state(); nn = newton_names(sim)
        err = {g: max(abs(q[0, nn[n][0]] - mj_q[n]) for n in mj_q if group_of(n) == g and n in nn) for g in GROUPS}
        print(f"  {drive:4s} it {it:2d} dt {h*1e3:.2f} ms: " + ", ".join(f"{g} {v:.3f}" for g, v in err.items())
              + f" | max |qd| {np.abs(qd).max():.2f}", flush=True)

if "C" in PARTS:
    print("C. G1 floating PD hold from Isaac's init state (root z 0.74, both engines): pelvis z every 0.25 s; joint deviation vs MuJoCo C at 0.5 s")
    m, names = mj_model(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.ctrl[:] = m.key_qpos[0][7:]
    zs, mjq05 = [], None
    for k in range(8):
        for _ in range(100): mujoco.mj_step(m, d)
        zs.append(d.qpos[2])
        if k == 1: mjq05 = {names[j]: d.qpos[m.jnt_qposadr[j]] for j in range(1, m.njnt)}
    print(f"  MuJoCo C          : z {np.round(zs, 3).tolist()}")
    for drive, it, h, relax in (("xpbd", 4, H, None), ("xpbd", 8, H, None), ("xpbd", 16, H, None), ("xpbd", 8, H, 0.4),
                                ("ipd", 1, H, 0.4), ("ipd", 2, H, 0.4), ("ipd", 4, H, 0.4), ("ipd", 8, H, 0.4), ("ipd", 4, H / 2, 0.4)):
        kw = {} if relax else {"solver_kw": {"joint_linear_relaxation": 0.7, "joint_angular_relaxation": 0.4}}
        sim = G1XPBD(4, iterations=it, dt=h, drive=drive, armature_inertia="iso" if drive == "ipd" else False, **kw); nn = newton_names(sim)
        zs, err = [], None
        for k in range(8):
            sim.step(int(0.25 / h)); zs.append(float(sim.pelvis_z()[0]))
            if k == 1:
                q, _ = sim.joint_state()
                err = {g: max(abs(q[0, nn[n][0]] - mjq05[n]) for n in mjq05 if group_of(n) == g and n in nn) for g in GROUPS}
        tag = f"{drive} it {it:2d} dt {h*1e3:.2f} rlx {'0.4/0.4' if relax else '0.7/0.4'}"
        print(f"  {tag:18s}: z {np.round(zs, 3).tolist()} | dq@0.5s " + ", ".join(f"{g} {v:.3f}" for g, v in err.items()), flush=True)

if "D" in PARTS:
    print("D. Metal vs CPU, G1 floating ipd it 4, 0.5 s")
    out = {}
    for dev in ("metal:0", "cpu"):
        sim = G1XPBD(1, iterations=4, drive="ipd", device=dev); t0 = time.time(); sim.step(200)
        out[dev] = sim.s0.body_q.numpy(); q, _ = sim.joint_state(); out[dev + "q"] = q
    print(f"  max |body_q diff| {np.abs(out['metal:0'] - out['cpu']).max():.2e}, max |joint q diff| {np.abs(out['metal:0q'] - out['cpuq']).max():.2e} rad")
