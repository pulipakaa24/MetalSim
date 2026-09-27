# Free-box scan used for google-deepmind/mujoco#3628 (the reproducer state is the worst single-contact case)
import numpy as np, mujoco, sys
XML = """<mujoco><option timestep="0.0025" cone="%s" impratio="%s" integrator="implicitfast" iterations="100" tolerance="1e-8"/>
<worldbody><geom type="plane" size="5 5 .1"/>
<body pos="0 0 .3"><freejoint/><geom type="box" size=".2 .1 .05" mass="5"/></body></worldbody></mujoco>"""
def run(cone, imp, ls, ref, keep=False):
  rng = np.random.default_rng(1)
  m = mujoco.MjModel.from_xml_string(XML % (cone, imp)); m.opt.ls_iterations = ls
  mr = mujoco.MjModel.from_xml_string(XML % (cone, imp)); mr.opt.ls_iterations = ref
  d = mujoco.MjData(m); dr = mujoco.MjData(mr); cases = []; n = 0
  for ep in range(50):
    mujoco.mj_resetData(m, d); d.qvel[:] = rng.normal(0, 3, 6)
    for t in range(800):
      if t % 50 == 0: d.qvel[:] += rng.normal(0, 3, 6)
      mujoco.mj_forward(m, d); mujoco.mj_copyData(dr, mr, d); mujoco.mj_forward(mr, dr); n += 1
      dq = np.abs(d.qacc - dr.qacc).max()
      if dq > 1.0: cases.append((dq, d.ncon, int(d.solver_niter[0]), d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy()))
      mujoco.mj_step(m, d)
  print(f"cone {cone} impratio {imp} ls_iterations {ls} vs {ref}: {n} solves, {len(cases)} with max|dqacc| > 1")
  return cases
c = run("elliptic", 10, 20, 200)
run("elliptic", 1, 20, 200); run("pyramidal", 10, 20, 200); run("elliptic", 10, 50, 200); run("elliptic", 10, 30, 200)
one = [x for x in c if x[1] == 1]
print("single-contact cases", len(one), "max dqacc", max(x[0] for x in one) if one else None)
best = max(one or c, key=lambda x: x[0])
np.set_printoptions(precision=17, floatmode="unique")
print("dq", best[0], "ncon", best[1], "niter", best[2]); print("qpos", repr(best[3])); print("qvel", repr(best[4])); print("qacc_warmstart", repr(best[5]))
