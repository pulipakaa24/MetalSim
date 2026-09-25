# SolverXPBD: revolute joints whose angle crosses ±π (limit range reaching π, or limit overshoot) receive a ~2π "limit correction" and explode

**Newton** 1.7.0.dev0 (`newton-physics/newton@45458023f24629c1feffb1f336594808f537db14`), **Warp** 1.18.0.dev; CPU and Metal.

## Summary

`solve_body_joints` measures joint angles from the relative quaternion with a swing-twist decomposition whose angles live
in (−π, π] (twist `2 asin(qtwist.x)`, swing `2 acos(qswing.w)` after forcing the two joint-frame quaternions into the
same hemisphere). A revolute joint that rotates past π therefore reads an angle near −π: if its limit range includes
angles beyond π (Unitree G1 elbow pitch: −0.227..3.421 rad), or a limit near π is overshot during a hard impact, the
solver sees a violation of almost 2π at the *other* limit and applies a correction of that size in one step. Energy
jumps by orders of magnitude; more iterations do not help.

## Reproducer

`python repro_angle_wrap.py` (newton + warp only, CPU): one revolute link, zero gravity, no drive, spun at 3 rad/s
towards the upper limit (lower limit −0.5 rad), dt 1.25 ms, 1.5 s.

```
  upper limit 3.0 rad,  4 iterations: peak |qd|      3.0 rad/s, final energy / initial      0.024, final q +2.767
  upper limit 3.1 rad,  4 iterations: peak |qd|     11.3 rad/s, final energy / initial      0.396, final q -0.344
  upper limit 3.1 rad, 16 iterations: peak |qd|     14.6 rad/s, final energy / initial      0.009, final q -0.438
  upper limit 3.4 rad,  4 iterations: peak |qd| 746531.5 rad/s, final energy / initial 61923253388.028, final q +5.721
  upper limit 3.4 rad, 16 iterations: peak |qd| 491629.7 rad/s, final energy / initial 26855527736.816, final q +5.792
```

With 3.1 rad the overshoot past the limit crosses π and the link is thrown to the lower limit (final q −0.34 / −0.44 rad);
with 3.4 rad the motion explodes. On a 44-body humanoid (G1, Isaac Lab gains) the same mechanism produced energy ×925
in 2 s of free floating with random joint velocities (joints hitting limits) and most of the blow-ups under large random
position targets (a knee driven 0.33 rad past its 2.54 rad limit read −3.41 rad and reached 878 rad/s in 20 ms).

## Cause

`newton/_src/solvers/xpbd/kernels.py`, `solve_body_joints`, angular part (from line 1889): the joint-frame quaternions are
flipped into one hemisphere (`if wp.dot(q_p, q_c) < 0.0: q_c *= -1.0`), then `err_0 = 2 asin(qtwist[0])`, and the swing
errors are rescaled with `theta = 2 acos(qswing[3])`; all of these are principal values. Limits are compared against
these errors with no knowledge of the previous angle.

## Suggested fix

Unwrap the measured angle of each limited/driven revolute axis against a reference that is continuous in time, e.g. the
angle at the start of the step (from `eval_ik`/the previous `body_q`), or compute the error relative to the middle of the
limit range (rotate the parent frame by `(lower + upper) / 2`, so the wrap point sits π away from the centre of the
range). At minimum, validate at `finalize()` that revolute limits lie inside (−π, π) and warn.

## Workaround

Keep revolute limits inside ±(π − margin) (we use 0.15 rad) and/or re-centre each joint's zero at the middle of its
limit range (rotate `joint_X_p` about the axis by the midpoint, shift limits, coordinates and targets); both remove the
failure for limits and reduce it for overshoot. Smaller substeps reduce overshoot.
