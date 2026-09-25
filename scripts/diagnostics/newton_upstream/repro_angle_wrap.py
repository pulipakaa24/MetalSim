"""SolverXPBD: a revolute joint whose limit range reaches +-pi gets a huge corrective impulse when its angle crosses pi.
One revolute link (zero gravity, no drive) spun at 3 rad/s towards an upper limit; the limit should stop it (energy
lost or kept, never gained). With the upper limit at 3.0 rad (< pi) it stops there; with 3.4 rad (> pi, e.g. the
Unitree G1 elbow pitch 3.421) the angle read by the solver wraps from +pi to -pi, i.e. ~2 pi below the lower limit.
Requires only newton and warp; runs on the CPU device."""
import numpy as np, warp as wp, newton

wp.config.quiet = True
DEV, DT = "cpu", 0.00125


def run(upper, iters=4, w0=3.0, T=1.5):
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, 0.0)
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.15, 0, 0), wp.quat_identity()), hx=0.15, hy=0.03, hz=0.03)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                             limit_lower=-0.5, limit_upper=upper)
    b.add_articulation([j]); b.joint_qd = [w0]
    m = b.finalize(DEV)
    S = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4, angular_damping=0.0)
    s0, s1, c = m.state(), m.state(), m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    q = wp.zeros(1, dtype=float, device=DEV); qd = wp.zeros(1, dtype=float, device=DEV)
    I = float(m.body_inertia.numpy()[0][1, 1]) + float(m.body_mass.numpy()[0]) * float(m.body_com.numpy()[0][0]) ** 2
    e0 = 0.5 * I * w0 ** 2; peak = 0.0; qmax = -1e9
    for _ in range(int(T / DT)):
        S.step(s0, s1, c, None, DT); s0, s1 = s1, s0
        newton.eval_ik(m, s0, q, qd)
        peak = max(peak, abs(float(qd.numpy()[0]))); qmax = max(qmax, float(q.numpy()[0]))
    return peak, 0.5 * I * float(qd.numpy()[0]) ** 2 / e0, float(q.numpy()[0])


print("revolute link, zero gravity, spun at 3 rad/s towards the upper limit (lower limit -0.5 rad), 1.5 s, dt 1.25 ms")
for upper in (3.0, 3.1, 3.4):
    for it in (4, 16):
        peak, ratio, qf = run(upper, it)
        print(f"  upper limit {upper:.1f} rad, {it:2d} iterations: peak |qd| {peak:8.1f} rad/s, final energy / initial {ratio:10.3f}, final q {qf:+.3f}")
