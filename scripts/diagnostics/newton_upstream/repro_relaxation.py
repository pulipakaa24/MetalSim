"""SolverXPBD: unequal joint_linear_relaxation / joint_angular_relaxation (defaults 0.7 / 0.4) give wrong
articulated dynamics. One revolute pendulum (6 kg, COM 0.208 m from the pivot), no drive, dt 2.5 ms: angular
acceleration over the first 50 ms under (a) a 1 N m joint torque (Control.joint_f), no gravity, and (b)
gravity only, compared with the analytic value (inertia about the pivot) and SolverFeatherstone.
Requires only newton and warp; runs on the CPU device."""
import numpy as np, warp as wp, newton

wp.config.quiet = True
DEV, DT, T = "cpu", 0.0025, 0.05


def model(g):
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, g)
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0))
    b.add_articulation([j])
    return b.finalize(DEV)


def qdd(solver, g, tau):
    m = model(g); S = solver(m)
    s0, s1, c = m.state(), m.state(), m.control()
    c.joint_f = wp.array([tau], dtype=float, device=DEV)
    for _ in range(int(T / DT)):
        s0.clear_forces(); S.step(s0, s1, c, None, DT); s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device=DEV); qd = wp.zeros(1, dtype=float, device=DEV)
    newton.eval_ik(m, s0, q, qd)
    mass = float(m.body_mass.numpy()[0]); r = float(m.body_com.numpy()[0][0])
    I = float(m.body_inertia.numpy()[0][1, 1]) + mass * r * r
    return 2.0 * float(q.numpy()[0]) / T ** 2, I, mass * 9.81 * r


solvers = [("Featherstone", lambda m: newton.solvers.SolverFeatherstone(m, angular_damping=0.0))]
for it in (2, 8):
    for lin, ang in ((0.7, 0.4), (0.4, 0.4), (0.7, 0.7), (1.0, 1.0)):
        solvers.append((f"XPBD it {it} relaxation lin {lin} / ang {ang}" + (" (defaults)" if (lin, ang) == (0.7, 0.4) else ""),
                        lambda m, it=it, lin=lin, ang=ang: newton.solvers.SolverXPBD(
                            m, iterations=it, angular_damping=0.0, joint_linear_relaxation=lin, joint_angular_relaxation=ang)))
print("measured / analytic angular acceleration (a small-angle finite difference over 50 ms reads ~1.05 when exact)")
for name, sv in solvers:
    a_t, I, tg = qdd(sv, 0.0, 1.0)
    a_g, *_ = qdd(sv, -9.81, 0.0)
    print(f"  {name:48s} joint torque {a_t * I:.3f}   gravity {a_g * I / tg:.3f}")
