"""Newton fork item 6: two-box stack resting on the ground, 2 s at 1 ms, 4 iterations, CPU: SolverXPBD.body_contact_force
(exact per-body net contact force) vs update_contacts per-contact forces summed per body, against m g."""
import numpy as np, warp as wp, newton
wp.config.quiet=True
with wp.ScopedDevice("cpu"):
    b = newton.ModelBuilder()
    b.add_ground_plane()
    b1 = b.add_body(xform=wp.transform((0,0,0.1), wp.quat_identity()))
    b.add_shape_box(b1, hx=0.2, hy=0.2, hz=0.1)
    b2 = b.add_body(xform=wp.transform((0.05,0,0.25), wp.quat_identity()))
    b.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=3000.0))
    m = b.finalize(); m.request_contact_attributes("force")
    S = newton.solvers.SolverXPBD(m, iterations=4, body_contact_forces=True)
    s0, s1, c = m.state(), m.state(), m.control(); cp = newton.CollisionPipeline(m); ct = cp.contacts()
    for _ in range(2000):
        s0.clear_forces(); cp.collide(s0, ct); S.step(s0, s1, c, ct, 1/1000); s0, s1 = s1, s0
    S.update_contacts(ct)
    mass = m.body_mass.numpy(); g = 9.81
    bf = S.body_contact_force.numpy()
    n = int(ct.rigid_contact_count.numpy()[0]); f = ct.force.numpy()[:n, :3]; sh0 = ct.rigid_contact_shape0.numpy()[:n]; sh1 = ct.rigid_contact_shape1.numpy()[:n]; sb = m.shape_body.numpy()
    per = np.zeros((m.body_count, 3))
    for k in range(n):
        a_, b_ = sb[sh0[k]] if sh0[k]>=0 else -1, sb[sh1[k]] if sh1[k]>=0 else -1
        if a_ >= 0: per[a_] += f[k]
        if b_ >= 0: per[b_] -= f[k]
    for i in range(m.body_count):
        print(f"body {i} m g {mass[i]*g:.3f} | solver body_contact_force total {np.round(bf[i][:3],3)} normal {np.round(bf[i][3:],3)} | sum of update_contacts per contact {np.round(per[i],3)}")
