# SolverXPBD: joint relaxation factors. A 6 kg box hinged 0.25 m from its COM (revolute to the world, axis y), no drive,
# 20 steps of 2.5 ms from rest, CPU. Angular acceleration / analytic (inertia about the pivot); N semi-implicit
# steps from rest give q_N = a dt^2 N (N + 1) / 2, so an exact solver reads 1.000.
import numpy as np, warp as wp, newton

def response(solver_kw, gravity, torque, dt=2.5e-3, n=20):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0))
    b.add_articulation([j]); m = b.finalize("cpu")
    solver = newton.solvers.SolverXPBD(m, angular_damping=0.0, **solver_kw)
    s0, s1, c = m.state(), m.state(), m.control()
    c.joint_f.assign(np.array([torque], np.float32))
    for _ in range(n):
        s0.clear_forces(); solver.step(s0, s1, c, None, dt); s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device="cpu"); qd = wp.zeros(1, dtype=float, device="cpu"); newton.eval_ik(m, s0, q, qd)
    a = 2.0 * float(q.numpy()[0]) / (dt * dt * n * (n + 1))
    mass, r = float(m.body_mass.numpy()[0]), float(m.body_com.numpy()[0][0])
    inertia = float(m.body_inertia.numpy()[0][1, 1]) + mass * r * r
    return a * inertia / (torque if torque else mass * 9.81 * r)

for it in (2, 8):
    for lin, ang in ((0.7, 0.4), (0.4, 0.4), (1.0, 1.0)):
        kw = dict(iterations=it, joint_linear_relaxation=lin, joint_angular_relaxation=ang)
        print(f"{it} it, relaxation lin {lin} / ang {ang}: joint torque {response(kw, 0.0, 1.0):.3f}, gravity {response(kw, -9.81, 0.0):.3f}")
