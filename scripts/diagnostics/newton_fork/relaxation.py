"""Joint relaxation consistency (Newton fork item 2). One revolute pendulum (6 kg box, COM 0.25 m from the pivot, axis y),
no drive, dt 2.5 ms, 20 steps from rest: (a) 1 N m joint torque (Control.joint_f), no gravity; (b) gravity only.
Measured / analytic angular acceleration, using the discrete-exact relation of N semi-implicit Euler steps from rest,
q_N = a dt^2 N (N + 1) / 2 (so an exact solver reads 1.000; the upstream draft used q = a T^2 / 2 and read 1.05 when
exact). Also prints the static sag of the pendulum held by Control.joint_f PD (ke 200, kd 5) against tau_g / ke.

usage: python scripts/diagnostics/newton_fork/relaxation.py   (either venv; CPU)"""
import inspect, numpy as np, warp as wp, newton
wp.config.quiet = True
DEV, DT, N = "cpu", 0.0025, 20
LEGACY = "joint_legacy_relaxation" in inspect.signature(newton.solvers.SolverXPBD.__init__).parameters


def model(g):
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, g)
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0))
    b.add_articulation([j])
    return b.finalize(DEV)


def qdd(mk, g, tau):
    m = model(g); S = mk(m)
    s0, s1, c = m.state(), m.state(), m.control()
    c.joint_f = wp.array([tau], dtype=float, device=DEV)
    for _ in range(N):
        s0.clear_forces(); S.step(s0, s1, c, None, DT); s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device=DEV); qd = wp.zeros(1, dtype=float, device=DEV)
    newton.eval_ik(m, s0, q, qd)
    mass = float(m.body_mass.numpy()[0]); r = float(m.body_com.numpy()[0][0])
    I = float(m.body_inertia.numpy()[0][1, 1]) + mass * r * r
    return 2.0 * float(q.numpy()[0]) / (DT * DT * N * (N + 1)), I, mass * 9.81 * r


def sag(mk, ke=200.0, kd=5.0, T=3.0):
    m = model(-9.81); S = mk(m)
    s0, s1, c = m.state(), m.state(), m.control()
    q = wp.zeros(1, dtype=float, device=DEV); qd = wp.zeros(1, dtype=float, device=DEV)
    for _ in range(int(T / DT)):
        newton.eval_ik(m, s0, q, qd)
        c.joint_f.assign(np.array([-ke * q.numpy()[0] - kd * qd.numpy()[0]], np.float32))
        s0.clear_forces(); S.step(s0, s1, c, None, DT); s0, s1 = s1, s0
    newton.eval_ik(m, s0, q, qd)
    mass = float(m.body_mass.numpy()[0]); r = float(m.body_com.numpy()[0][0]); qs = float(q.numpy()[0])
    return qs, mass * 9.81 * r * np.cos(qs) / ke        # equilibrium: ke q = tau_g(q) = m g r cos q (q about +y, COM along +x)


rows = [("Featherstone", lambda m: newton.solvers.SolverFeatherstone(m, angular_damping=0.0))]
for it in (2, 8):
    cfgs = [(0.7, 0.4, None), (0.4, 0.4, None), (1.0, 1.0, None)]
    if LEGACY:
        cfgs.insert(1, (0.7, 0.4, True))
    for lin, ang, leg in cfgs:
        kw = {} if leg is None else {"joint_legacy_relaxation": True}
        name = f"XPBD it {it} lin {lin} / ang {ang}" + (" legacy" if leg else "") + (" (defaults)" if (lin, ang, leg) == (0.7, 0.4, None) else "")
        rows.append((name, lambda m, it=it, lin=lin, ang=ang, kw=kw: newton.solvers.SolverXPBD(
            m, iterations=it, angular_damping=0.0, joint_linear_relaxation=lin, joint_angular_relaxation=ang, **kw)))
print(f"newton at {newton.__file__}; legacy switch available: {LEGACY}")
print("| solver | torque response (1.000 = exact) | gravity response | PD hold ke 200: sag / exact |")
print("|---|---|---|---|")
for name, mk in rows:
    a_t, I, tg = qdd(mk, 0.0, 1.0)
    a_g, *_ = qdd(mk, -9.81, 0.0)
    try:
        qs, qe = sag(mk); sg = f"{qs:+.4f} / {qe:+.4f} rad ({qs / qe:.3f})"
    except Exception as e:
        sg = f"n/a ({e!r:.40})"
    print(f"| {name} | {a_t * I:.3f} | {a_g * I / tg:.3f} | {sg} |", flush=True)
