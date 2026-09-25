### Summary

`SolverXPBD` measures the angle of a revolute joint as a principal value in (−π, π]: the joint-frame quaternions are forced into one hemisphere and the swing-twist decomposition uses `2 asin`/`2 acos`. A revolute joint whose limit range extends beyond ±π (e.g. an elbow with limits −0.23 … 3.42 rad), or one that overshoots a limit close to π, therefore reads an angle about 2π away from the true one when it crosses π. The solver treats that as a limit violation of almost 2π and corrects it in one step, so the joint's kinetic energy grows by orders of magnitude. More iterations do not help. `newton.eval_ik` has a related ambiguity: its revolute angle depends on the quaternion branch of the bodies.

### Reproducer

Newton `45458023` (1.7.0.dev0), `pip install "newton @ git+https://github.com/newton-physics/newton@45458023f24629c1feffb1f336594808f537db14"` (pulls warp-lang 1.17.0); CPU device.

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

Output (warp-lang 1.17.0, CPU):

```
upper 3.0 rad,  4 it: peak |qd|        3.0 rad/s, KE end/start        0.024, final q +2.767 rad
upper 3.0 rad, 16 it: peak |qd|        3.0 rad/s, KE end/start        0.023, final q +2.771 rad
upper 3.1 rad,  4 it: peak |qd|       11.3 rad/s, KE end/start        0.396, final q -0.344 rad
upper 3.1 rad, 16 it: peak |qd|       14.6 rad/s, KE end/start        0.009, final q -0.438 rad
upper 3.4 rad,  4 it: peak |qd|   746531.5 rad/s, KE end/start 61923253388.028, final q +5.721 rad
upper 3.4 rad, 16 it: peak |qd|   491629.7 rad/s, KE end/start 26855527736.816, final q +5.792 rad
```

With a 3.1 rad limit, the overshoot past π throws the link to the lower limit (final q −0.34 rad). With 3.4 rad, the motion explodes. On a floating 44-body humanoid with a 3.42 rad elbow limit (zero gravity, no drives, random initial joint velocities, 4 it / 1.25 ms), kinetic energy grows ×925 in 2 s.

### Cause

`solve_body_joints`, angular part ([kernels.py L1894-L1945](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1894-L1945)):
- the hemisphere flip at [L1897](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1897);
- `err_0 = 2 asin(qtwist.x)` at [L1913](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1913);
- the swing rescale `theta = 2 acos(qswing.w)` at [L1939](https://github.com/newton-physics/newton/blob/45458023f24629c1feffb1f336594808f537db14/newton/_src/solvers/xpbd/kernels.py#L1939).

These are principal values, and the limits and targets are compared against them with no reference to the range.

### Suggested fix

For joints with a single rotational DOF, measure the angle relative to a reference inside its range:
- the reference is the middle of a limit range narrower than 2π, otherwise the drive target;
- rotate the child joint frame by −reference about the axis before the decomposition, then add the reference back to the error.

This is stateless, so it is safe across resets, and it keeps the gradients consistent. Every angle in the range, plus overshoots up to π − (upper − lower)/2, then reads correctly. `eval_ik` can report limited hinges on the same branch.

Reference implementation, with tests: https://github.com/pulipakaa24/newton/commit/f844a4e67f738592c1eb3e5722d3327071fa1965 (fork branch `metalsim`). With it, the reproducer gives peak |qd| 3.0 rad/s and KE end/start ≤ 0.03 for every case (final q +3.224 / +3.383 rad at the 3.4 rad limit), and the humanoid's energy ratio after 2 s is 0.598. Happy to open a PR if wanted.

### System Information

- Newton: 1.7.0.dev0 @ 45458023f24629c1feffb1f336594808f537db14
- Warp: warp-lang 1.17.0 (PyPI)
- Python: 3.12
- OS: macOS (arm64)
- Device: CPU (the solver code is device-independent)
