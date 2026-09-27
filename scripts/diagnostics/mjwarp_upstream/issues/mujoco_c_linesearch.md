### Intro

Hi! I develop MetalSim (https://github.com/pulipakaa24/MetalSim), which runs MuJoCo Warp on Apple GPUs and uses MuJoCo C as its reference for physics checks.

### My setup

MuJoCo 3.14.0 from pip, Python API, float64, CPU, macOS 26 on Apple M4 Max (arm64). The solver code of 3.14.0 (`engine_solver.c`) is the same as `main` today.

### What's happening? What did you expect?

With an elliptic cone at `impratio="10"` and `ls_iterations="20"`, the Newton solver sometimes returns a `qacc` that is not the minimizer: the line search exhausts its budget and the solver stops with a gradient of about 1e2 (the restart below indicates that the next line search finds no improvement, i.e. the `alpha == 0` exit of `mj_solPrimal`). With `ls_iterations` of 23 or more, the same state converges in one more iteration to a gradient of 6e-13. I understand that 20 is below the default of 50. I'm reporting it because values of 20 or below are common in practice (mjlab's velocity, tracking and lift tasks use 20; the MJX variants of Menagerie models use 4 to 20) and such models are often checked against MuJoCo C, and because the solver gives no signal: `solver_niter` is small and nothing warns.

The documentation says that lowering `ls_iterations` too far can prevent convergence, so the unconverged result by itself may be expected. What surprised me is that the solver ends up at a point it cannot leave: restarting from the returned `qacc` performs 0 iterations, while the gradient is still about 1e2 against a tolerance of 1e-8. From the outside, the solve looks converged.

### Steps for reproduction

1. Run the code below (the model is included as a string; it is also reproduced under "Minimal model").
2. Compare the `ls_iterations` 20 to 22 lines with 23 and 50.

Output:

```
ls_iterations 20: niter 2, final gradient 124, neval [7, 20], max |qacc - qacc(ls 200)| 59.2
ls_iterations 21: niter 2, final gradient 124, neval [7, 22], max |qacc - qacc(ls 200)| 59.2
ls_iterations 22: niter 2, final gradient 124, neval [7, 22], max |qacc - qacc(ls 200)| 59.2
ls_iterations 23: niter 3, final gradient 6.09e-13, neval [7, 23, 2], max |qacc - qacc(ls 200)| 0
ls_iterations 50: niter 3, final gradient 6.09e-13, neval [7, 23, 2], max |qacc - qacc(ls 200)| 0
restart from the ls 20 result: niter 0, max |qacc - qacc(ls 200)| 59.2
```

The restart line shows that the returned point is a fixed point of the solver at this budget: starting from it, the first line search along the Newton direction finds no improvement within 20 evaluations and the solve ends with 0 iterations, although the converged solution is 3.1e2 lower in the scaled cost (summed per-iteration improvements: 21,771 at `ls_iterations=20`, 22,079 at 200). The gradient and improvement tests are not what stops it (gradient 124, tolerance 1e-8).

Frequency. Free box, random velocity kicks, 40,000 solves: 28 solves differ from the `ls_iterations=200` result by more than 1 in `qacc` (up to 59). Menagerie `unitree_g1/scene.xml` with random position targets (it falls and flails), 20,000 solves: 16 differ from the `ls_iterations=100` result by more than 1, up to 1.5e3, with exit gradients of 3 to 16. For the same two setups there were no such solves with elliptic cones at `impratio="1"`, with pyramidal cones at `impratio="10"`, or with elliptic `impratio="10"` at `ls_iterations` 30 or 50. On a G1 humanoid walking-policy dataset (another model, 2,304 control steps of 8 substeps), `ls_iterations=20` led to 6 blow-ups (|qvel| up to 2,144 rad/s within one substep, in a state with a single torso-ground contact and seven joint limits); with 50 or 100 there were none.

Two questions. Is a line search that exhausts its budget expected to end the solve (through `alpha == 0` at the next iteration) without any indication, or would it be reasonable to report it (for example via `solver_niter` or a warning) or to keep iterating? And does the 1-D search need that many evaluations here because the cost along the search direction changes curvature where the contact crosses the cone boundary (the curvature ratio is about impratio)? Genesis hit a similar pattern in its MuJoCo-style Newton solver (for its `signorini` contact mode: the line search spent its budget bisecting stick/slip transitions and the solve exited with a force residual of the order of a body's weight) and changed the search to take the next such kink as a candidate instead of the bracket midpoint and to floor its gradient tolerance at the rounding noise of the derivative (Genesis-Embodied-AI/genesis-world#3382). google-deepmind/mujoco_warp#1700 and #3619 changed the MuJoCo Warp / MJX acceptance test and do not touch `PrimalSearch`.

### Minimal model for reproduction

<details>
<summary>minimal XML</summary>

```XML
<mujoco>
  <option timestep="0.0025" cone="elliptic" impratio="10" integrator="implicitfast" iterations="100" tolerance="1e-8"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .3"><freejoint/><geom type="box" size=".2 .1 .05" mass="5"/></body>
  </worldbody>
</mujoco>
```
</details>

### Code required for reproduction

```python
import numpy as np, mujoco

XML = """
<mujoco>
  <option timestep="0.0025" cone="elliptic" impratio="10" integrator="implicitfast" iterations="100" tolerance="1e-8"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .3"><freejoint/><geom type="box" size=".2 .1 .05" mass="5"/></body>
  </worldbody>
</mujoco>"""
QPOS = [-0.38228398123211904, -1.5617372196006747, 0.14420735808017446, -0.3039178177313174, 0.6782960782869877,
        -0.6545711847046525, -0.1381483058176044]
QVEL = [-2.5320433011146646, 0.4435273997725518, -3.833483970428372, 20.58784610148961, -4.891237201965733,
        -1.521169201653485]
QACC_WARMSTART = [196.2178511861834, 185.09984762860262, 456.16231234940415, -2622.9781161564592, 1990.857671015227,
                  -729.4643685817738]

def solve(ls, warmstart=QACC_WARMSTART):
  m = mujoco.MjModel.from_xml_string(XML); m.opt.ls_iterations = ls
  d = mujoco.MjData(m); d.qpos[:] = QPOS; d.qvel[:] = QVEL; d.qacc_warmstart[:] = warmstart
  mujoco.mj_forward(m, d)
  return d

ref = solve(200).qacc
for ls in (20, 21, 22, 23, 50):
  d = solve(ls); n = d.solver_niter[0]
  print(f"ls_iterations {ls}: niter {n}, final gradient {d.solver[n - 1].gradient:.3g}, "
        f"neval {[d.solver[i].neval for i in range(n)]}, max |qacc - qacc(ls 200)| {np.abs(d.qacc - ref).max():.3g}")
d20 = solve(20)
d = solve(20, warmstart=d20.qacc)
print(f"restart from the ls 20 result: niter {d.solver_niter[0]}, max |qacc - qacc(ls 200)| {np.abs(d.qacc - ref).max():.3g}")
```

### Confirmations

I read the `ls_iterations` / `ls_tolerance` documentation and the solver section of the MJX and MuJoCo Warp docs, and searched previous issues and discussions for line search, `ls_iterations`, elliptic cone and impratio. The closest issue I found is #2313 (MJX float32 early termination from improvement overshoots), which has a different mechanism. #3619 is related in spirit but concerns MJX.
