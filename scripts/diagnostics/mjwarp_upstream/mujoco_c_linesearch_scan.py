# Scan used for google-deepmind/mujoco#3628 (Menagerie G1; run from mujoco_menagerie/unitree_g1: python this.py scene.xml 20 10; env CONE/IMP/REF)
"""Scan MuJoCo C (float64) for Newton solves whose result at ls_iterations=20 differs from ls_iterations=100.
Menagerie G1 (scene.xml or scene_with_hands.xml), elliptic cones, impratio 10, random position targets."""
import os, sys, numpy as np, mujoco
CONE=int(os.environ.get('CONE',1)); IMP=float(os.environ.get('IMP',10)); REF=int(os.environ.get('REF',100))
scene = sys.argv[1]; ls = int(sys.argv[2]) if len(sys.argv) > 2 else 20; nep = int(sys.argv[3]) if len(sys.argv) > 3 else 20
def load(lsi):
  m = mujoco.MjModel.from_xml_path(scene)
  m.opt.cone = CONE; m.opt.impratio = IMP; m.opt.timestep = 0.0025
  m.opt.iterations = 100; m.opt.ls_iterations = lsi; m.opt.tolerance = 1e-8
  return m
m = load(ls); mref = load(REF); d = mujoco.MjData(m); dref = mujoco.MjData(mref)
rng = np.random.default_rng(0); lo, hi = m.actuator_ctrlrange.T
nsolve = 0; bad = []; blow = 0
for ep in range(nep):
  mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[7:] += rng.normal(0, 0.2, m.nq - 7)
  for t in range(2000):
    if t % 40 == 0: d.ctrl[:] = rng.uniform(lo, hi)
    mujoco.mj_forward(m, d)
    mujoco.mj_copyData(dref, mref, d); mujoco.mj_forward(mref, dref)
    nsolve += 1
    dq = np.abs(d.qacc - dref.qacc).max()
    if dq > 1.0:
      n = int(d.solver_niter[0]); s = d.solver[n - 1]
      bad.append((ep, t, dq, n, d.ncon, d.nefc, s.gradient, [int(d.solver[i].neval) for i in range(n)]))
    mujoco.mj_step(m, d)
    if np.abs(d.qvel).max() > 200: blow += 1; break
print(f"{scene.split('/')[-1]} cone {CONE} imp {IMP} ls {ls} ref {REF}: {nsolve} solves, {len(bad)} with max|dqacc| > 1 vs ls 100, episodes ending |qvel|>200: {blow}")
for b in bad[:8]: print("  ep %d t %d dqacc %.3g niter %d ncon %d nefc %d final gradient %.3g neval %s" % b)
