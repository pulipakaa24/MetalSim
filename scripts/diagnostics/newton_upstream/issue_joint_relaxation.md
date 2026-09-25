# SolverXPBD: unequal `joint_linear_relaxation` / `joint_angular_relaxation` (the defaults 0.7 / 0.4) make joints transmit torque and gravity wrongly

**Newton** 1.7.0.dev0 (`newton-physics/newton@45458023f24629c1feffb1f336594808f537db14`), **Warp** 1.18.0.dev; reproduced on the CPU device and on Metal (same numbers).

## Summary

With the default `SolverXPBD(joint_linear_relaxation=0.7, joint_angular_relaxation=0.4)`, a single revolute
pendulum accelerates **43 % too fast under a joint torque** and **18 % too slowly under gravity**, independent of
the iteration count. With equal relaxation factors (0.4/0.4, 0.7/0.7 or 1.0/1.0) the result matches the analytic
value and `SolverFeatherstone`. The error carries over to anything driven through joints: joint torques
(`Control.joint_f`), and PD drives built on them, hold the wrong static equilibrium (the pendulum in
`repro_drive_bias.py` sags 0.0303 rad instead of 0.0529 rad with relaxation 0.7/0.4 and exactly 0.0522 / 0.0522 with 0.4/0.4).

## Reproducer

`python repro_relaxation.py` (newton + warp only, CPU). A 6 kg body, COM 0.208 m from a revolute pivot to the
world, no drive, `dt` 2.5 ms; the angular acceleration over the first 50 ms is compared with the analytic value
(inertia about the pivot). A small-angle finite difference over 50 ms reads about 1.05 when exact.

```
measured / analytic angular acceleration
  Featherstone                                      joint torque 1.050   gravity 1.050
  XPBD it 2 relaxation lin 0.7 / ang 0.4 (defaults) joint torque 1.430   gravity 0.815
  XPBD it 2 relaxation lin 0.4 / ang 0.4            joint torque 1.055   gravity 1.047
  XPBD it 2 relaxation lin 0.7 / ang 0.7            joint torque 1.051   gravity 1.049
  XPBD it 2 relaxation lin 1.0 / ang 1.0            joint torque 1.050   gravity 1.050
  XPBD it 8 relaxation lin 0.7 / ang 0.4 (defaults) joint torque 1.431   gravity 0.817
  XPBD it 8 relaxation lin 0.4 / ang 0.4            joint torque 1.052   gravity 1.050
```

## Cause

In `solve_body_joints` (`newton/_src/solvers/xpbd/kernels.py`, lines 1728-1731 for DISTANCE joints and 1884-1887 for
the positional rows of all other joints), one positional constraint impulse `d_lambda` is applied as

```python
lin_delta_p += linear_p * (d_lambda * linear_relaxation)
ang_delta_p += angular_p * (d_lambda * angular_relaxation)   # angular_p = -cross(r_p, n): the moment of the SAME impulse
lin_delta_c += linear_c * (d_lambda * linear_relaxation)
ang_delta_c += angular_c * (d_lambda * angular_relaxation)
```

`angular_p` / `angular_c` are the moments of the linear impulse about each body's COM, so scaling them by a
different factor than the linear part applies a force whose moment is 0.4 / 0.7 = 57 % of what it should be. For a
body hanging on a pivot this under-transmits the pivot's reaction moment: gravity (acting at the COM) is partly
"absorbed" and a joint torque rotates the body as if about a point between its COM and the pivot. More iterations
do not help because every correction has the same inconsistency.

## Suggested fix

Scale both parts of a positional row by one factor (`linear_relaxation`), and both parts of an angular row by
`angular_relaxation`, e.g. lines 1729/1731 and 1885/1887 use `linear_relaxation`. Equal factors are what the reproducer
shows to be exact. (Separately: on a 44-body humanoid, angular relaxation above ~0.4 diverges because joint
corrections are summed per body (Jacobi); so simply raising the angular factor is not a substitute.)

## Workaround

`SolverXPBD(model, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)`.
