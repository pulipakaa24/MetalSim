"""PARITY 1.6's minimal repro: a box mesh pressed 3 cm into a flat heightfield (base 1.9 m deep, as Isaac's rough
terrain), MuJoCo Warp's GJK/EPA path (HFIELD_PLANE_CONTACTS False), CPU device; prints the contacts. Several yaw angles."""
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import collision_convex
collision_convex.HFIELD_PLANE_CONTACTS = False
print(mjw.__file__)
for yaw in (0, 17, 33, 61):
  xml = f"""<mujoco><asset><hfield name="h" nrow="21" ncol="21" size="1 1 0.1 1.9"/>
  <mesh name="b" vertex="-.1 -.033 -.01  .1 -.033 -.01  -.1 .033 -.01  .1 .033 -.01  -.1 -.033 .01  .1 -.033 .01  -.1 .033 .01  .1 .033 .01"/></asset>
  <worldbody><geom type="hfield" hfield="h"/><body pos="0.013 0.021 -0.02" euler="0 0 {yaw}"><freejoint/><geom type="mesh" mesh="b"/></body></worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); dd = mjw.put_data(m, d); mjw.kinematics(mm, dd); mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0])
    dist, fr = dd.contact.dist.numpy()[:n], dd.contact.frame.numpy()[:n, 0]
  print(f"yaw {yaw:2d}: C {d.ncon} contacts, depth {(-d.contact.dist[:d.ncon]).round(4).tolist()} | MJW {n}: depth {(-dist).round(4).tolist()} normals {fr.round(2).tolist()}")
