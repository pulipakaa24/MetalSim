"""The checks of test_epa_nonloop_horizon (collision_gjk_test.py, epa-fixes) as a script, so they run on trees whose
test module does not import (93e5ce1 has no _horizon_is_loop / epa_horizon_size) and on mutants.

Usage: PYTHONPATH=<mjw tree> python -I nonloop_check.py   (with -I, PYTHONPATH is ignored: run without -I)
Prints per pose: MJW deepest dist, C deepest dist, |diff|, overflow bits, nacon, and PASS/FAIL against the test's
tolerances (2e-8 boxes, 1e-6 ellipsoids)."""
import os

import warp as wp

wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402
from mujoco_warp import test_data  # noqa: E402

BOX = '<geom type="box" size=".025 .025 .025"/><body><freejoint/><geom type="box" size=".025 .025 .025"/></body>'
ELL = '<geom type="ellipsoid" size=".05 .03 .02"/><body><freejoint/><geom type="ellipsoid" size=".04 .025 .03"/></body>'
POSES = [
  ("box 1531", BOX, "-2.1158824352e-02 -6.6878882263e-03 4.9974132254e-02 1.3436272695e-01 2.9754414064e-05 "
   "-5.2466821423e-04 9.9093207711e-01", 2e-8),
  ("box 3635", BOX, "1.1042433995e-02 1.7353867029e-02 4.9951583291e-02 2.6547766364e-01 -4.7347918626e-04 "
   "1.0911244443e-04 9.6411688815e-01", 2e-8),
  ("ellip 320", ELL, "0.0502827088 0.028716553 -0.0339722746 -0.049520108 0.2793360898 0.8799608347 0.3810354803", 1e-6),
]
CYL = '<geom type="cylinder" size=".03 .02"/><body><freejoint/><geom type="cylinder" size=".02 .04"/></body>'
if os.environ.get("EXTRA"):  # candidates whose restart restores a valid face (search/, instr bit 25)
  POSES = [
    ("box s12 3681", BOX, "5.2688197441e-03 -4.5746638777e-03 4.9999595380e-02 9.6642667520e-01 1.4869133534e-03 "
     "3.8914768146e-04 2.5693796744e-01", 2e-8),
    ("ellip s12 55", ELL, "-7.3897864161e-04 -2.0439472508e-02 -4.5535543848e-02 -2.4062015535e-01 -8.1050973059e-01 "
     "5.2155101089e-01 -1.1471905031e-01", 1e-6),
    ("ellip s14 80", ELL, "-1.9735966572e-02 5.5930757965e-02 4.7976683187e-03 -3.1546613422e-01 -6.9086514123e-01 "
     "-3.7602842153e-02 6.4944014432e-01", 1e-6),
    ("cyl s12 1714", CYL, "-1.3028226098e-02 2.1945445595e-02 -4.0578555027e-02 -6.0507750643e-01 6.8689649464e-01 "
     "1.8976949674e-01 3.5502387944e-01", 1e-6),
  ]
print("mujoco_warp", mjw.__file__)
with wp.ScopedDevice("cpu"):
  for name, bodies, qpos, tol in POSES:
    mjm, mjd, m, d = test_data.fixture(
      xml=f"<mujoco><worldbody>{bodies}</worldbody><keyframe><key qpos='{qpos}'/></keyframe></mujoco>", keyframe=0
    )
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    n = int(d.nacon.numpy()[0])
    ov = int(d.overflow.numpy()[0])
    cd = float(mjd.contact.dist[: mjd.ncon].min())
    wd = float(d.contact.dist.numpy()[:n].min()) if n else float("nan")
    ok = n > 0 and ov == 0 and abs(wd - cd) <= tol
    bits = [b for b in range(32) if ov >> b & 1]
    print(f"{name:10s} MJW {wd: .9e} C {cd: .9e} |diff| {abs(wd - cd):.2e} tol {tol:.0e} nacon {n} overflow bits {bits} "
          f"{'PASS' if ok else 'FAIL'}")
