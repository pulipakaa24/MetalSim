# SolverXPBD: joint position drives (`target_ke` / `target_kd`) are not springs of stiffness `target_ke`; their stiffness scales with the iteration count

**Newton** 1.7.0.dev0 (`newton-physics/newton@45458023f24629c1feffb1f336594808f537db14`), **Warp** 1.18.0.dev; CPU and Metal give the same numbers.

## Summary

A revolute pendulum held by an XPBD position drive with `target_ke` 200 N m/rad settles with a gravity sag that
scales as 1/iterations instead of `tau_gravity / ke`: the effective stiffness is 72, 145, 291, 593 and 1240 N m/rad at
1, 2, 4, 8 and 16 iterations (relaxation 0.4/0.4 to isolate this from the relaxation issue). On a humanoid (Unitree G1,
Isaac Lab gains) the effect depends on the link: light, heavily damped links (hands, kd 10 on ~1e-6 kg m^2 links)
come out far softer than `ke`, heavy ones stiffer; joint angles with the base fixed differ from the exact static
PD equilibrium by 0.03-0.4 rad at 4-16 iterations. The same PD law applied as a torque through `Control.joint_f`
(damping integrated backward-Euler) is exact at any iteration count.

## Reproducer

`python repro_drive_bias.py` (newton + warp only, CPU):

```
static sag [rad] of a pendulum held by a PD drive, ke 200 N m/rad (exact: tau_gravity / ke)
  xpbd       iterations  1: sag +0.1369 (exact +0.0493) -> effective stiffness     72 N m/rad
  xpbd       iterations  2: sag +0.0714 (exact +0.0516) -> effective stiffness    145 N m/rad
  xpbd       iterations  4: sag +0.0363 (exact +0.0527) -> effective stiffness    291 N m/rad
  xpbd       iterations  8: sag +0.0180 (exact +0.0533) -> effective stiffness    593 N m/rad
  xpbd       iterations 16: sag +0.0086 (exact +0.0536) -> effective stiffness   1240 N m/rad
  joint_f PD iterations  1: sag +0.0522 (exact +0.0522) -> effective stiffness    200 N m/rad
  joint_f PD iterations 16: sag +0.0522 (exact +0.0522) -> effective stiffness    200 N m/rad
```

## Cause

The drive is folded into the angular joint rows of `solve_body_joints` as compliance `alpha = 1 / ke` with damping
`gamma = kd / ke` (`compute_angular_correction`, kernels.py 2421-2454):

```python
delta_lambda = -(err + alpha * lambda_in + gamma * derr)
delta_lambda /= (dt + gamma) * denom + alpha / dt
```

with `lambda_in = 0.0` on every iteration (lines 1703, 1864: no multiplier is accumulated across iterations), and
the result is applied by `apply_body_deltas` as a **velocity** change that persists into the next iteration and
step. Each iteration therefore adds an impulse of roughly `relaxation * err / (alpha / dt + (dt + gamma) * denom)`,
i.e. a restoring torque of about `iterations * relaxation * ke / (1 + (ke dt^2 + kd dt) * denom)`. The equilibrium is
where these impulses balance gravity, so the stiffness grows linearly with the iteration count and shrinks by the
`(ke dt^2 + kd dt) * denom` term on light links (large inverse inertia `denom`) with damping. In standard XPBD the
multiplier is accumulated (`lambda_in` = the running sum) and the fixed point is `C + alpha~ lambda = 0`, which is
independent of the iteration count.

## Suggested fix

Accumulate the drive multiplier per joint axis across iterations within a substep (reset per substep) and use it as
`lambda_in`, with compliance `alpha / dt^2` in position units, so the converged state satisfies the compliant
constraint exactly; or document that the drive's stiffness is iteration-dependent and point users to
`Control.joint_f` for calibrated PD. (Related: `joint_target_mode` is documented as unsupported, and `target_kd`
enters only through `gamma` above.)

## Workaround

Disable the drive (`target_ke = target_kd = 0`) and apply `tau = kp (q* - q) - kd qd / (1 + kd dt / I)` through
`Control.joint_f` each substep (`q, qd` from `newton.eval_ik`, I the joint-space inertia), with equal joint
relaxation factors (see the relaxation issue).
