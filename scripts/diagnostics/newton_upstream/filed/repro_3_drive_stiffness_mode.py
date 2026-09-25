# SolverXPBD: joint position drive stiffness vs iteration count. 6 kg box hinged 0.25 m from its COM, drive target
# 0.5 rad, target_ke 200 N m/rad, target_kd 5, gravity, dt 2.5 ms, 3 s, CPU. Effective stiffness = tau_gravity / sag
# (should be 200 at every iteration count).
import sys, numpy as np, warp as wp, newton

KW = {"joint_drive_mode": sys.argv[1]} if len(sys.argv) > 1 else {}

def stiffness(iters, dt=2.5e-3):
    b = newton.ModelBuilder()
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                             target_ke=200.0, target_kd=5.0, target_pos=0.5)
    b.add_articulation([j]); m = b.finalize("cpu")
    solver = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4, **KW)
    s0, s1, c = m.state(), m.state(), m.control()
    for _ in range(int(3.0 / dt)):
        s0.clear_forces(); solver.step(s0, s1, c, None, dt); s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device="cpu"); qd = wp.zeros(1, dtype=float, device="cpu"); newton.eval_ik(m, s0, q, qd)
    q = float(q.numpy()[0])
    tau_g = float(m.body_mass.numpy()[0]) * 9.81 * float(m.body_com.numpy()[0][0]) * np.cos(q)
    return tau_g / (q - 0.5)

for it in (1, 2, 4, 8, 16):
    print(f"{it:2d} it: effective stiffness {stiffness(it):7.1f} N m/rad (target_ke 200)")
