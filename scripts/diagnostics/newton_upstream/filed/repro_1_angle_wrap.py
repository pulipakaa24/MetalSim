# SolverXPBD: revolute limit range beyond +-pi. One link on a revolute joint (axis y) to the world, zero gravity,
# spun at 3 rad/s towards its upper limit (lower limit -0.5 rad), dt 1.25 ms, 1.5 s, CPU. Energy must not grow.
import warp as wp, newton

def run(upper, iters):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.15, 0, 0), wp.quat_identity()), hx=0.15, hy=0.03, hz=0.03)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                             limit_lower=-0.5, limit_upper=upper)
    b.add_articulation([j]); b.joint_qd = [3.0]
    m = b.finalize("cpu")
    solver = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
    s0, s1, c = m.state(), m.state(), m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    q = wp.zeros(1, dtype=float, device="cpu"); qd = wp.zeros(1, dtype=float, device="cpu"); peak = 0.0
    for _ in range(1200):
        solver.step(s0, s1, c, None, 1.25e-3); s0, s1 = s1, s0
        newton.eval_ik(m, s0, q, qd); peak = max(peak, abs(float(qd.numpy()[0])))
    return peak, (float(qd.numpy()[0]) / 3.0) ** 2, float(q.numpy()[0])

for upper in (3.0, 3.1, 3.4):
    for iters in (4, 16):
        peak, e, q = run(upper, iters)
        print(f"upper {upper} rad, {iters:2d} it: peak |qd| {peak:10.1f} rad/s, KE end/start {e:12.3f}, final q {q:+.3f} rad")
