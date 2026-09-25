### Summary

With the default `SolverXPBD(joint_linear_relaxation=0.7, joint_angular_relaxation=0.4)`, a single revolute pendulum responds to a joint torque (`Control.joint_f`) 1.36× faster than the analytic hinge, and to gravity at 0.78× of it. The result does not depend on the iteration count. With equal factors, or with `SolverFeatherstone`, the response is exact. The error carries over to everything that acts through joints, for example PD controllers built on `joint_f`, which settle at the wrong static equilibrium.

### Reproducer

Newton `45458023` (1.7.0.dev0), `pip install "newton @ git+https://github.com/newton-physics/newton@45458023f24629c1feffb1f336594808f537db14"` (pulls warp-lang 1.17.0); CPU device.

```python
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
```

Output (warp-lang 1.17.0, CPU):

```
2 it, relaxation lin 0.7 / ang 0.4: joint torque 1.362, gravity 0.776
2 it, relaxation lin 0.4 / ang 0.4: joint torque 1.005, gravity 0.997
2 it, relaxation lin 1.0 / ang 1.0: joint torque 1.000, gravity 1.000
8 it, relaxation lin 0.7 / ang 0.4: joint torque 1.363, gravity 0.778
8 it, relaxation lin 0.4 / ang 0.4: joint torque 1.002, gravity 1.000
8 it, relaxation lin 1.0 / ang 1.0: joint torque 1.000, gravity 1.000
```

### Cause

In `solve_body_joints`, one positional constraint impulse `d_lambda` is applied with two different factors:
- the linear part is scaled by `linear_relaxation`;
- its moment about each body's COM (`angular_p = -cross(r_p, n)`, `angular_c = cross(r_c, n)`) is scaled by `angular_relaxation`.

The locations are [L1728-L1731](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1728-L1731) for DISTANCE joints and [L1884-L1887](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1884-L1887) for the positional rows of all other joints. The applied force therefore carries 0.4 / 0.7 = 57 % of its own moment. For a body hanging on a pivot, the pivot's reaction moment is under-transmitted: a joint torque rotates the body as if about a point between its COM and the pivot, and gravity is partly absorbed. More iterations repeat the same inconsistency.

### Suggested fix

Scale both parts of a positional row by `joint_linear_relaxation`; rotational rows keep `joint_angular_relaxation`.

Doing this changes stability on multi-joint bodies, because a body's joint corrections are summed. On a floating 44-body humanoid at 4 it / 1.25 ms:
- consistent rows with a linear factor of 0.7 inject energy (kinetic energy peaks at ×1.2 in 2 s with no external input);
- 0.6 and below are stable.

So the default linear factor would have to come down (the reference implementation uses 0.5), and a switch can keep the former behaviour.

Reference implementation, with tests: https://github.com/pulipakaa24/newton/commit/fda6658a5654bc739e301ac9d8d13d4906623b4c (fork branch `metalsim`; it adds `joint_legacy_relaxation`). With it, the reproducer reads joint torque 1.001 / 0.999 and gravity 0.999 / 1.000 at lin 0.7 / ang 0.4 (2 / 8 it). Happy to open a PR if wanted.

### System Information

- Newton: 1.7.0.dev0 @ 45458023f24629c1feffb1f336594808f537db14
- Warp: warp-lang 1.17.0 (PyPI)
- Python: 3.12
- OS: macOS (arm64)
- Device: CPU (the solver code is device-independent)
