"""Joint drives inside SolverXPBD (Newton fork item 3) vs MetalSim's ActuatorPD, CPU device.
  A. pendulum (6 kg box, COM 0.25 m from the pivot), target 0.5 rad, ke 200 kd 5, 2 s: effective stiffness
     tau_g / sag at 1..16 iterations (exact: 200).
  B. G1, base fixed 1 m up, Isaac gains, 2 s: max |q - q_eq| per joint group against the exact static PD equilibrium
     (kp (q* - q) = g(q), clipped at the effort limit; MuJoCo's gravity force at rest, fixed-point iteration), as
     scripts/diagnostics/newton_xpbd_drives.py part B.
drive "solver" = the model's joint_target_ke/kd solved by SolverXPBD (joint_drive_mode = --mode on the fork, the
compliance drive upstream); "ipd" = ActuatorPD through Control.joint_f (armature added isotropically, as NewtonSim).

usage: python scripts/diagnostics/newton_fork/drive_equilibrium.py [A] [B] [--mode pd|compliance] [--dt_ms 2.5]
       [--its 1,2,4,8,16] [--arm iso|none|solver] [--relax 0.4]"""
import argparse, inspect, numpy as np, warp as wp, newton, mujoco
wp.config.quiet = True
from metalsim.physics.newton_backend import G1XPBD, ActuatorPD
from metalsim.learn.g1_velocity import build_g1_model

ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="*", default=["A", "B"])
ap.add_argument("--mode", default="pd"); ap.add_argument("--dt_ms", type=float, default=2.5)
ap.add_argument("--its", default="1,2,4,8,16"); ap.add_argument("--arm", default="none"); ap.add_argument("--relax", type=float, default=0.4)
ap.add_argument("--ipd", action="store_true", help="also run ActuatorPD")
ap.add_argument("--kw", nargs="*", default=[])
a = ap.parse_args(); H = a.dt_ms * 1e-3; ITS = [int(x) for x in a.its.split(",")]
FORK = "joint_drive_mode" in inspect.signature(newton.solvers.SolverXPBD.__init__).parameters
SKW = {"joint_drive_mode": a.mode} if FORK else {}
for kv in a.kw:
    k, v = kv.split("="); SKW[k] = eval(v)
DEV = "cpu"
print(f"newton at {newton.__file__}; fork: {FORK}; solver kwargs {SKW}; dt {a.dt_ms} ms; relaxation {a.relax}")


def pendulum(iters, mode, T=2.0):
    with wp.ScopedDevice(DEV):
        b = newton.ModelBuilder(); b.gravity = -9.81
        link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
        b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
        on = mode == "solver"
        j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                                 target_ke=200.0 if on else 0.0, target_kd=5.0 if on else 0.0)
        b.add_articulation([j]); b.joint_target_q = [0.5]
        m = b.finalize()
        S = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=a.relax, joint_angular_relaxation=a.relax, **SKW)
        s0, s1 = m.state(), m.state(); c = m.control()
        act = None if on else ActuatorPD(m, [200.0], [5.0], [1e6], [0.5], dt=H)
        for _ in range(int(T / H)):
            if act: act.apply(s0, c)
            s0.clear_forces(); S.step(s0, s1, c, None, H); s0, s1 = s1, s0
        q = wp.zeros(1, dtype=float); qd = wp.zeros(1, dtype=float); newton.eval_ik(m, s0, q, qd)
        q = float(q.numpy()[0]); tau_g = float(m.body_mass.numpy()[0]) * 9.81 * float(m.body_com.numpy()[0][0]) * np.cos(q)
        return tau_g / (q - 0.5), float(qd.numpy()[0])


GROUPS = {"legs": ("hip", "knee", "torso"), "ankles": ("ankle",), "arms": ("shoulder", "elbow"),
          "hands": ("zero", "one", "two", "three", "four", "five", "six")}
group_of = lambda name: next(g for g, keys in GROUPS.items() if any(k in name for k in keys))


def equilibrium():
    m, _ = build_g1_model("flat", physics_dt=H)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] = 1.0; d.qvel[:] = 0
    kp = m.actuator_gainprm[:, 0]; qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]; dadr = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    tgt = m.key_qpos[0][qadr]; lim = m.actuator_forcerange[:, 1]
    for _ in range(200):
        mujoco.mj_forward(m, d); d.qpos[qadr] = tgt - np.clip(d.qfrc_bias[dadr], -lim, lim) / kp
    mujoco.mj_forward(m, d)
    return {names[j]: d.qpos[m.jnt_qposadr[j]] for j in range(1, m.njnt)}


def g1(drive, it):
    arm = {"iso": "iso", "none": False, "solver": False}[a.arm] if drive == "solver" else "iso"
    kw = dict(SKW)
    if FORK and drive == "solver" and a.arm == "solver":
        kw["joint_armature_inertia"] = True
    sim = G1XPBD(1, iterations=it, dt=H, drive="xpbd" if drive == "solver" else "ipd", floating=False, z0=1.0, device=DEV,
                 armature_inertia=arm, relaxation=a.relax, solver_kw=kw if drive == "solver" else None)
    sim.step(int(2.0 / H)); q, qd = sim.joint_state()
    M = sim.model; qs = M.joint_q_start.numpy(); qds = M.joint_qd_start.numpy()
    nn = {M.joint_label[j].split("/")[-1]: int(qs[j]) for j in range(M.joint_count) if qds[j + 1] - qds[j] == 1}
    err = {g: max(abs(q[0, nn[n]] - EQ[n]) for n in EQ if group_of(n) == g and n in nn) for g in GROUPS}
    return err, float(np.abs(qd).max()) if np.isfinite(qd).all() else float("nan")


if "A" in a.parts:
    print("A. pendulum ke 200 kd 5: effective stiffness [N m/rad] (exact 200)")
    for mode in (["solver", "ipd"] if a.ipd else ["solver"]):
        print(f"  {mode:6s} " + "  ".join(f"it {it}: {pendulum(it, mode)[0]:.1f}" for it in ITS), flush=True)
if "B" in a.parts:
    EQ = equilibrium()
    print("B. G1 fixed base: max |q - q_eq| per group [rad] after 2 s")
    for drive in (["solver", "ipd"] if a.ipd else ["solver"]):
        for it in ITS:
            err, qdm = g1(drive, it)
            print(f"  {drive:6s} it {it:2d}: " + ", ".join(f"{g} {v:.4f}" for g, v in err.items()) + f" | max |qd| {qdm:.3f}", flush=True)
