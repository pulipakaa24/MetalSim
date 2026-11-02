"""MuJoCo C main's MjcConvexTest.HFieldMarginAndGap (8626d3c) on MuJoCo Warp (copy on PYTHONPATH, CPU): 3x3 heightfield
size 1 1 0.1 0.1, terrain margin 0.02 gap 0.01, sphere r 0.05 at z 0.065 (15 mm clear): expect dist 0.015, pos z 0.0075.
Also a box and the box mesh at the same clearance, and both mesh paths."""
import importlib, os, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import collision_convex as CCX
print(mjw.__file__)
mesh = '<mesh name="b" vertex="-.05 -.05 -.05  .05 -.05 -.05  -.05 .05 -.05  .05 .05 -.05  -.05 -.05 .05  .05 -.05 .05  -.05 .05 .05  .05 .05 .05"/>'
for plane in (False, True):
  CCX.HFIELD_PLANE_CONTACTS = plane
  for typ, geom in (("sphere", 'type="sphere" size=".05"'), ("box", 'type="box" size=".05 .05 .05"'), ("mesh", 'type="mesh" mesh="b"'),
                    ("capsule", 'type="capsule" size=".05 .05" euler="0 90 0"')):
    if plane and typ != "mesh": continue
    for z in (0.065, 0.075, 0.085):
      xml = f"""<mujoco><asset><hfield name="h" nrow="3" ncol="3" size="1 1 0.1 0.1"/>{mesh}</asset><worldbody>
      <geom type="hfield" hfield="h" margin="0.02" gap="0.01"/><body pos="0.03 0.04 {z}"><freejoint/><geom {geom}/></body></worldbody></mujoco>"""
      m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
      with wp.ScopedDevice("cpu"):
        mm = mjw.put_model(m); dd = mjw.put_data(m, d); mjw.kinematics(mm, dd); mjw.collision(mm, dd)
        n = int(dd.nacon.numpy()[0]); dist = dd.contact.dist.numpy()[:n]; pz = dd.contact.pos.numpy()[:n, 2]
      print(f"{'plane' if plane else 'gjk  '} {typ:7s} clearance {z - 0.05:.3f}: MJW n {n} dist {np.round(dist, 5).tolist()} pos z {np.round(pz, 5).tolist()} | C 3.14 n {d.ncon} dist {np.round(d.contact.dist[:d.ncon], 5).tolist()}")
