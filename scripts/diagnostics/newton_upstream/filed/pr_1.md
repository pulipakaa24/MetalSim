## Description

Fixes #4313.

**Problem.** `SolverXPBD` measures the angle of a joint with one rotational DOF as a principal value in (−π, π]. A hinge whose limit range extends beyond ±π, or that overshoots a limit close to π, therefore reads an angle about 2π away from the true one when it crosses π, and receives a "limit correction" of that size. A single link spun into a 3.4 rad limit reaches 746,531 rad/s (kinetic energy ×6e10 in 1.5 s). `eval_ik` reports such angles on either branch depending on the quaternion sign of the bodies.

**Cause.** In `solve_body_joints`, the joint-frame quaternions are flipped into one hemisphere ([L1897](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1897)), and the twist and swing angles come from `2 asin` / `2 acos` ([L1913](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1913), [L1939](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1939)). The limits and targets are compared against these principal values.

**Fix.**
- For joints with one rotational DOF, the angle is measured within π of a reference inside the range. The reference is the middle of a limit range narrower than 2π; otherwise it is the position target (unlimited joints). A new kernel, `compute_joint_angle_references`, bakes the static reference of a limited joint into a solver copy of `joint_X_c` (the child frame rotated by −reference about the axis) and the reference is added back to the angular error. There is no per-iteration cost, the scheme is stateless (safe across resets), and the copy is refreshed by `notify_model_changed` for joint and joint-DOF property changes.
- `eval_ik` reports limited revolute and single-rotational-DOF D6 angles on the same branch (`joint_angle_reference`, `wrap_angle_near` in `sim/articulation.py`).
- Small-swing rescaling (related): `theta / sin(theta/2)` tends to 2, but swings below ~0.02 rad were left in quaternion space, which is half the angle ([L1934-L1945](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1934-L1945)). Hard rows are unaffected, because error and gradient scale together. A compliant row, however, read half its error there. On main, a compliance drive therefore settles at a different stiffness when its target is 0: 613 vs 1240 N·m/rad at 16 iterations for target 0 vs 0.5 in the reproducer of #4315. The reference above would otherwise have moved that zone from angle 0 to the target. With the rescale fixed, both targets read 1267 N·m/rad.

## Checklist

- [x] New or existing tests cover these changes
- [x] The documentation is up to date with these changes
- [x] For user-facing changes, a fragment has been added by following the
      [changelog fragment instructions](https://github.com/newton-physics/newton/blob/main/changelog/README.md)

## Test plan

New `newton/tests/test_solver_xpbd_joints.py` (CPU device), 4 tests; all fail on main and pass with this PR:
- `test_revolute_limit_beyond_pi_does_not_gain_energy`
- `test_revolute_eval_ik_angle_beyond_pi`
- `test_revolute_hold_beyond_pi`
- `test_revolute_unlimited_drive_takes_short_way`

```
uv run --extra dev -m newton.tests -k test_solver_xpbd_joints
```

Regression check: `test_solver_xpbd`, `test_joint_limits`, `test_joint_drive`, `test_body_velocity`, `test_control_force`, `test_ik`, `test_joint_damping`, `test_physics_verification`, `test_kinematic_links` and `test_rigid_contact`. Run with warp-lang 1.17.0 on CPU, they show the same results as main; the only errors are modules skipped for missing optional dependencies (mujoco, trimesh, scipy) in that environment. `pre-commit` hooks pass.

## Bug fix

**Steps to reproduce:** run the script below on main (CPU).

**Minimal reproduction:**

```python
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
```

| upper limit, iterations | main: peak \|qd\|, KE end/start, final q | this PR |
|---|---|---|
| 3.0 rad, 4 it | 3.0 rad/s, 0.024, +2.767 | 3.0 rad/s, 0.029, +2.747 |
| 3.1 rad, 4 it | 11.3 rad/s, 0.396, **−0.344** (thrown to the lower limit) | 3.0 rad/s, 0.027, +2.874 |
| 3.1 rad, 16 it | 14.6 rad/s, 0.009, **−0.438** | 3.0 rad/s, 0.000, +3.083 |
| 3.4 rad, 4 it | **746,531 rad/s, 6.2e10**, +5.721 | 3.0 rad/s, 0.027, +3.224 |
| 3.4 rad, 16 it | **491,630 rad/s, 2.7e10**, +5.792 | 3.0 rad/s, 0.000, +3.383 |

On a floating 44-body humanoid with a 3.42 rad elbow limit (zero gravity, no drives, random initial joint velocities, 4 it / 1.25 ms), kinetic energy after 2 s goes from ×924.9 to ×0.598.
