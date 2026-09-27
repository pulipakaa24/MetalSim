"""Reproducer posted in google-deepmind/mujoco#3628: MuJoCo C Newton solver, elliptic cone, impratio 10, ls_iterations 20;
an exhausted line search ends the solve at a non-converged qacc. pip mujoco 3.14.0, CPU."""
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
