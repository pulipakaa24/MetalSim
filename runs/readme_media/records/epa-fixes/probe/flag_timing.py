"""When are the fork's heightfield flags read? (stream epa-fixes, for the DECISIONS.md wording)

A 5 mm sphere 1 mm above a flat heightfield, gap 0.01 per geom (pair 0.02), Warp CPU. HFIELD_TRUE_DISTANCE changes the
reported dist (True: the separation, 0.001; False: MuJoCo C 3.14's convention, about -0.015 here, as C 3.14).
Usage: PYTHONPATH=<mjw tree> python flag_timing.py <mode>
  default        no assignment
  before         collision_convex.HFIELD_TRUE_DISTANCE = False after import, before the first collision call
  after          the same assignment after a first collision call (then a second call)
  gjk_before     collision_gjk.HFIELD_PRISM_POLYTOPE = False after import, before the first collision call (its
                 effect on this pose is none at the defaults; printed for completeness)"""
import sys

import mujoco
import numpy as np
import warp as wp

wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402
from mujoco_warp._src import collision_convex as CCV  # noqa: E402

mode = sys.argv[1]
xml = """<mujoco><asset><hfield name="h" nrow="5" ncol="5" size="1 1 .1 .1"/></asset><worldbody>
<geom type="hfield" hfield="h" gap="0.01"/><body pos="0 0 .006"><freejoint/><geom type="sphere" size=".005" gap="0.01"/></body>
</worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)


def run():
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m)
    dd = mjw.put_data(m, d, nconmax=16, njmax=128)
    mjw.kinematics(mm, dd)
    mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0])
    return np.round(dd.contact.dist.numpy()[:n], 6).tolist()


if mode == "before":
  CCV.HFIELD_TRUE_DISTANCE = False
if mode == "gjk_before":
  from mujoco_warp._src import collision_gjk as G
  G.HFIELD_PRISM_POLYTOPE = False
r1 = run()
if mode == "after":
  CCV.HFIELD_TRUE_DISTANCE = False
  r2 = run()
  print(mode, "first call", r1, "second call", r2)
else:
  print(mode, r1)
print("MuJoCo C", m.opt.ccd_iterations, np.round(d.contact.dist[: d.ncon], 6).tolist())
