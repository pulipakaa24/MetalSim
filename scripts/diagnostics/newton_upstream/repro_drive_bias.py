"""SolverXPBD: the joint position drive (target_ke / target_kd) is not a spring of stiffness target_ke, and its
effective stiffness depends on the iteration count. One revolute pendulum (6 kg, COM 0.208 m from the pivot)
held at target 0.5 rad by target_ke 200 N m/rad, target_kd 5, dt 2.5 ms, 2 s: the static sag under gravity
should be tau_gravity / ke at every iteration count. For comparison, the same PD law applied as a torque via
Control.joint_f with backward-Euler damping gives tau / ke exactly.
Requires only newton and warp; runs on the CPU device."""
import numpy as np, warp as wp, newton

wp.config.quiet = True
DEV, DT = "cpu", 0.0025


def run(iters, mode, lin=0.4, ang=0.4):
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, -9.81)
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    drive = mode == "xpbd"
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                             target_ke=200.0 if drive else 0.0, target_kd=5.0 if drive else 0.0)
    b.add_articulation([j]); b.joint_target_mode = [newton.JointTargetMode.POSITION]; b.joint_target_q = [0.5]
    m = b.finalize(DEV)
    S = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=lin, joint_angular_relaxation=ang)
    s0, s1, c = m.state(), m.state(), m.control()
    q = wp.zeros(1, dtype=float, device=DEV); qd = wp.zeros(1, dtype=float, device=DEV)
    I = float(m.body_inertia.numpy()[0][1, 1]) + float(m.body_mass.numpy()[0]) * float(m.body_com.numpy()[0][0]) ** 2
    for _ in range(int(2.0 / DT)):
        if not drive:        # PD as a joint torque, damping backward-Euler on the pivot inertia
            newton.eval_ik(m, s0, q, qd)
            e = 0.5 - float(q.numpy()[0]); v = float(qd.numpy()[0])
            c.joint_f = wp.array([200.0 * e - 5.0 * v / (1.0 + 5.0 * DT / I)], dtype=float, device=DEV)
        s0.clear_forces(); S.step(s0, s1, c, None, DT); s0, s1 = s1, s0
    newton.eval_ik(m, s0, q, qd)
    qv = float(q.numpy()[0])
    tau_g = float(m.body_mass.numpy()[0]) * 9.81 * float(m.body_com.numpy()[0][0]) * np.cos(qv)
    return qv - 0.5, tau_g / 200.0, tau_g / (qv - 0.5)


print("static sag [rad] of a pendulum held by a PD drive, ke 200 N m/rad (exact: tau_gravity / ke)")
for mode, its in (("xpbd", (1, 2, 4, 8, 16)), ("joint_f PD", (1, 4, 16))):
    for it in its:
        sag, exact, k = run(it, mode)
        print(f"  {mode:10s} iterations {it:2d}: sag {sag:+.4f} (exact {exact:+.4f}) -> effective stiffness {k:6.0f} N m/rad")
sag, exact, k = run(4, "xpbd", 0.7, 0.4)
print(f"  xpbd with the default relaxation 0.7/0.4, iterations 4: sag {sag:+.4f} (exact {exact:+.4f}) -> {k:.0f} N m/rad")
