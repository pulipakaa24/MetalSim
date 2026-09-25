### Summary

A revolute joint held by an XPBD position drive (`target_ke` 200 N·m/rad, `target_kd` 5) does not behave as a spring of stiffness `target_ke`. Its effective stiffness, gravity torque divided by static sag, grows linearly with the iteration count: 71, 144, 291, 593 and 1240 N·m/rad at 1, 2, 4, 8 and 16 iterations. The drive also ignores `joint_effort_limit`. On articulated robots with light, heavily damped links, the effect varies per link, so drive-held poses depend on the solver settings. This is the joint-drive counterpart of #2933 (springs, bending and tets).

### Reproducer

Newton `45458023` (1.7.0.dev0), `pip install "newton @ git+https://github.com/newton-physics/newton@45458023f24629c1feffb1f336594808f537db14"` (pulls warp-lang 1.17.0); CPU device. Joint relaxation 0.4/0.4 isolates this from the relaxation issue.

```python
# SolverXPBD: joint position drive stiffness vs iteration count. 6 kg box hinged 0.25 m from its COM, drive target
# 0.5 rad, target_ke 200 N m/rad, target_kd 5, gravity, dt 2.5 ms, 3 s, CPU. Effective stiffness = tau_gravity / sag
# (should be 200 at every iteration count).
import numpy as np, warp as wp, newton

def stiffness(iters, dt=2.5e-3):
    b = newton.ModelBuilder()
    link = b.add_link(xform=wp.transform((0, 0, 1), wp.quat_identity()), mass=1.0)
    b.add_shape_box(link, xform=wp.transform((0.25, 0, 0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((0, 0, 1), wp.quat_identity()), axis=(0, 1, 0),
                             target_ke=200.0, target_kd=5.0, target_pos=0.5)
    b.add_articulation([j]); m = b.finalize("cpu")
    solver = newton.solvers.SolverXPBD(m, iterations=iters, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
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

Output (warp-lang 1.17.0, CPU):

```
 1 it: effective stiffness    71.3 N m/rad (target_ke 200)
 2 it: effective stiffness   144.4 N m/rad (target_ke 200)
 4 it: effective stiffness   290.7 N m/rad (target_ke 200)
 8 it: effective stiffness   593.0 N m/rad (target_ke 200)
16 it: effective stiffness  1239.6 N m/rad (target_ke 200)
```

### Cause

The drive is folded into the joint rows of `solve_body_joints` as compliance `alpha = 1/ke` with damping `gamma = kd/ke`, through `compute_angular_correction` / `compute_positional_correction` ([L2421-L2454](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L2421-L2454), [L2378-L2418](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L2378-L2418)). The multiplier is never accumulated: `lambda_in = 0.0` on every iteration ([L1703](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1703), [L1864](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1864), and the `0.0` argument at [L2067-L2071](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L2067-L2071)).

Each iteration therefore adds a fresh impulse of about `relaxation * err / (alpha/dt + (dt + gamma) * w)`, and it persists as a velocity. At steady state these impulses balance gravity, which makes the stiffness about `iterations * relaxation * ke`, reduced on light links (large inverse inertia `w`) by the `(ke dt² + kd dt) w` term. In standard XPBD the multiplier is accumulated within a step, and the fixed point `C + alpha~ lambda = 0` does not depend on the iteration count.

### Suggested fix

Either accumulate the drive multiplier per DOF across the iterations of a step, or solve the drive as an implicit PD law. The reference implementation (opt-in `joint_drive_mode`) does the latter:
- the spring `ke (q* − q)` is taken from the start-of-step state, which gives exact static equilibria at any iteration count;
- a backward-Euler damper row, solved Gauss-Seidel after the joint's hard rows, enforces `impulse = spring − dt kd q̇` on the velocity it sees;
- the total is clamped at `joint_effort_limit`;
- `joint_armature` can be added to the child inertia.

A closed-form damper on the joint's own inertia alone is not enough: a light, heavily damped link (`kd dt w >> 1`, e.g. fingers) then moves several times too fast.

Reference implementation, with tests: https://github.com/pulipakaa24/newton/commit/fda6658a5654bc739e301ac9d8d13d4906623b4c and https://github.com/pulipakaa24/newton/commit/9b0901d36f01f8761cf52a5113f890e31c8d6d1b (damping on light links); fork branch `metalsim`. With it, the reproducer reads 199.8, 199.8, 199.7, 200.3 and 199.5 N·m/rad at 1-16 iterations. On a 44-body humanoid in a zero-gravity step response, joint travel at 50 ms is within 10 % of MuJoCo's for every hand joint. Happy to open a PR if wanted.

### System Information

- Newton: 1.7.0.dev0 @ 45458023f24629c1feffb1f336594808f537db14
- Warp: warp-lang 1.17.0 (PyPI)
- Python: 3.12
- OS: macOS (arm64)
- Device: CPU (the solver code is device-independent)
