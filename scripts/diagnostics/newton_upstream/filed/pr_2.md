## Description

Fixes #4314.

**Problem.** With the default `SolverXPBD(joint_linear_relaxation=0.7, joint_angular_relaxation=0.4)`, a revolute pendulum responds to a joint torque (`Control.joint_f`) at 1.36× the analytic hinge and to gravity at 0.78×, at any iteration count. PD controllers built on `joint_f` settle at the wrong static equilibrium (0.57× the exact sag).

**Cause.** In `solve_body_joints`, a positional row's impulse `d_lambda` is applied with two factors:
- the linear part is scaled by `linear_relaxation`;
- its moment about each body's COM is scaled by `angular_relaxation`.

The locations are [L1728-L1731](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1728-L1731) for DISTANCE joints and [L1884-L1887](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1884-L1887) for all other joints. The applied force then carries 0.4/0.7 of its own moment.

**Fix.**
- Both parts of a positional row now use `joint_linear_relaxation`; rotational rows keep `joint_angular_relaxation`.
- Default change: with consistent rows, a linear factor of 0.7 injects energy into floating multi-joint bodies, because a body's joint corrections are summed. On a 44-body humanoid (4 it / 1.25 ms, zero gravity, no drives, random joint velocities), kinetic energy peaks at ×1.95 within 2 s at 0.7; at 0.6 and 0.5 the ratio stays ≤ 0.999 throughout (main at its 0.7/0.4 defaults: 1.000). The default `joint_linear_relaxation` therefore becomes 0.5. Since this changes results for existing scenes, maintainers may prefer a different default; the consistency fix itself does not depend on it.
- `joint_legacy_relaxation=True` restores the former scaling.

Independent of #4313's fix (#4316); the PD-drive PR builds on both.

## Checklist

- [x] New or existing tests cover these changes
- [x] The documentation is up to date with these changes
- [x] For user-facing changes, a fragment has been added by following the
      [changelog fragment instructions](https://github.com/newton-physics/newton/blob/main/changelog/README.md)

## Test plan

New `newton/tests/test_solver_xpbd_joints.py` (CPU device), 2 tests; both fail on main and pass with this PR:
- `test_joint_relaxation_transmits_torque_and_gravity`: the default and unequal factors both give the analytic response within 1 %;
- `test_joint_legacy_relaxation_switch`.

```
uv run --extra dev -m newton.tests -k test_solver_xpbd_joints
```

Regression check: `test_solver_xpbd` (including `test_articulation_contact_drift`), `test_joint_limits`, `test_joint_drive`, `test_body_velocity`, `test_control_force`, `test_ik`, `test_joint_damping`, `test_physics_verification`, `test_kinematic_links` and `test_rigid_contact`. Run with warp-lang 1.17.0 on CPU, they show the same results as main (the only errors are modules skipped for missing optional dependencies). `pre-commit` hooks pass.

## Bug fix

**Steps to reproduce:** run the script below on main (CPU). N semi-implicit steps from rest give q_N = a dt² N (N+1) / 2, so an exact solver reads 1.000.

**Minimal reproduction:**

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

| setting | main: joint torque / gravity | this PR |
|---|---|---|
| 2 it, lin 0.7 / ang 0.4 (former defaults) | **1.362 / 0.776** | 1.001 / 0.999 |
| 8 it, lin 0.7 / ang 0.4 | **1.363 / 0.778** | 0.999 / 1.000 |
| 2 it, lin 0.4 / ang 0.4 | 1.005 / 0.997 | 1.005 / 0.997 |
| 2 it, lin 1.0 / ang 1.0 | 1.000 / 1.000 | 1.000 / 1.000 |
| 2 it, lin 0.7 / ang 0.4, `joint_legacy_relaxation=True` | – | 1.362 / 0.776 |
