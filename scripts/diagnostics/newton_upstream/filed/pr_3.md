## Description

Fixes #4315.

**Depends on #4316 and #4317.** This branch is stacked on both. Its first two commits are those PRs; please review the last commit, "Add opt-in PD joint drives to SolverXPBD". It uses the angle reference of #4316, and the explicit spring part is only transmitted correctly with #4317's consistent rows.

**Problem.** The XPBD joint drive (`joint_target_ke`/`kd`) does not behave as a spring of stiffness `ke`. For `ke` 200 N·m/rad, the effective stiffness of a pendulum is 71, 144, 291, 593 and 1240 N·m/rad at 1, 2, 4, 8 and 16 iterations, and `joint_effort_limit` is ignored.

**Cause.** The drive is folded into the joint rows as compliance `1/ke` with damping `kd/ke` (`compute_angular_correction` / `compute_positional_correction`, [L2378-L2454](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L2378-L2454)). Its multiplier is never accumulated: `lambda_in = 0.0` on every iteration ([L1703](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1703), [L1864](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1864), [L2067-L2071](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L2067-L2071)). Each iteration therefore adds a fresh impulse, and at steady state the stiffness is about `iterations × relaxation × ke`. This is the joint-drive counterpart of #2933.

**Change (opt-in; the default keeps the current behaviour).** `SolverXPBD(joint_drive_mode=...)` takes one of three values:
- `"compliance"` (default): unchanged.
- `"pd"`: `ke` and `kd` are the stiffness and damping at any iteration count, and the drive force is clamped at `joint_effort_limit`.
  - The spring `ke (q* − q)` is taken from the start-of-step state, which gives exact static equilibria.
  - A fraction `1 / ((1 + kd dt w)(1 + (ke dt² w)²))` of the spring is applied as a joint force before the solve (`w` is the inverse inertia or mass along the axis).
  - The rest rides in a backward-Euler damper row, `impulse = spring − dt kd q̇`, solved by a new kernel `solve_joint_drive_rows` on the velocity after the joint's hard rows of the same pass.
  - Light, heavily damped links (`kd dt w >> 1`, e.g. fingers) therefore follow the damper instead of receiving an explicit velocity kick; a closed-form damper on the joint's own inertia moves them several times too fast. Very light links stay stable.
- `"implicit"`: backward-Euler PD rows with the impulse accumulated over the iterations, warm-started with the spring.

Under `"pd"` and `"implicit"`, D6 joints with several rotational DOFs keep compliance rows for those DOFs.

Related options and outputs:
- `joint_drive_relaxation` (default 1.0) scales the drive rows.
- `SolverXPBD.joint_drive_force` reports the applied drive force per DOF.
- `joint_armature_inertia` (`"none"` default, `"isotropic"`, `"axis"`) adds `Model.joint_armature` to the child body's inertia, used for integration and constraints.

The drive kernels run only when a drive mode other than `"compliance"` is selected. `solve_body_joints` itself only skips the drive DOFs in that case.

## Checklist

- [x] New or existing tests cover these changes
- [x] The documentation is up to date with these changes
- [x] For user-facing changes, a fragment has been added by following the
      [changelog fragment instructions](https://github.com/newton-physics/newton/blob/main/changelog/README.md)

## Test plan

`newton/tests/test_solver_xpbd_joints.py` (CPU device) gains 6 tests; they fail on main (options missing) and pass here:
- `test_joint_drive_stiffness_is_ke_at_any_iteration_count` ("pd" at 1/4/16 it, "implicit" at 4/16 it);
- `test_joint_drive_compliance_mode_is_legacy`;
- `test_joint_drive_effort_limit`;
- `test_joint_armature_inertia`;
- `test_joint_drive_light_damped_chain` (overdamped 3-link chain of 8 g links follows the first-order creep within 10 %);
- `test_joint_drive_gains_set_after_construction` (also checks `joint_drive_force`).

```
uv run --extra dev -m newton.tests -k test_solver_xpbd_joints
```

Regression check with warp-lang 1.17.0 on CPU: `test_solver_xpbd`, `test_joint_limits`, `test_joint_drive`, `test_body_velocity`, `test_control_force`, `test_ik`, `test_joint_damping`, `test_physics_verification`, `test_kinematic_links`, `test_rigid_contact`, and the other XPBD users: `test_admm_coupled_solver`, `test_body_force`, `test_conveyor_forces`, `test_coupled_solver`, `test_heightfield`, `test_import_urdf`, `test_import_mjcf`, `test_joint_controllers`, `test_mesh_backface`, `test_rigid_friction_ramp`, `test_runtime_gravity`, `test_shapes_no_bounce`, `test_speculative_contacts`, `test_up_axis`, `test_collision_pipeline`. Same results as main; the only errors are modules skipped for missing optional dependencies. `pre-commit` hooks pass.

## Bug fix

**Steps to reproduce:** run on main (CPU); on this branch pass `pd` or `implicit` as the argument.

**Minimal reproduction:**

```python
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
```

Effective stiffness [N·m/rad] for `target_ke` 200:

| iterations | main | this branch, default ("compliance") | "pd" | "implicit" |
|---|---|---|---|---|
| 1 | 71.3 | 73.3 | 199.8 | 199.8 |
| 2 | 144.4 | 147.7 | 199.8 | 199.8 |
| 4 | 290.7 | 296.6 | 199.7 | 199.7 |
| 8 | 593.0 | 605.3 | 200.3 | 199.6 |
| 16 | 1239.6 | 1266.7 | 199.5 | 199.7 |

The small change in the default column comes from #4316's small-swing rescale; see that PR.

Measured on a 44-body humanoid (Isaac Lab-style gains: kp 20–200, kd 2–10, armature 0.001–0.01 added isotropically) at 4 it / 1.25 ms with `"pd"`:
- Fixed base, max deviation from the exact static PD equilibrium: legs 0.0019 rad, arms 0.0083 rad at 4 iterations (0.0094 / 0.034 at 1 iteration). The compliance drive at 4 iterations, 2.5 ms: 0.073 / 0.42 rad.
- Zero-gravity step response (+0.3 rad targets), joint travel at 50 ms relative to MuJoCo with the same model and gains, median per group: legs 1.01, arms 1.07, hands 1.03; every hand joint within 11 %.
- 3σ random targets, 1024 envs × 400 control steps, CPU: 0 envs blew up (non-finite or > 1000 rad/s; peak joint speed 270 rad/s), against 17 with a PD actuator (same gains, implicit damping) applied through `Control.joint_f`.

## New feature / API change

```python
solver = newton.solvers.SolverXPBD(model, iterations=4, joint_drive_mode="pd", joint_armature_inertia="isotropic")
# joint_target_ke / joint_target_kd are now the stiffness / damping at any iteration count;
# joint_effort_limit bounds the drive force; solver.joint_drive_force holds the applied drive force per DOF.
```
