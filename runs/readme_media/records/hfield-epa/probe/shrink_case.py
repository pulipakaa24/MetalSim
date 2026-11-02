"""A box mesh corner 4 mm above a flat heightfield with gap 0.01 per geom (pair margin + gap 0.02): contacts on CPU."""
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
print(mjw.__file__)
xml = """<mujoco><asset><hfield name="h" nrow="11" ncol="11" size="0.5 0.5 0.1 0.1"/>
<mesh name="b" vertex="-.05 -.03 -.02  .05 -.03 -.02  -.05 .03 -.02  .05 .03 -.02  -.05 -.03 .02  .05 -.03 .02  -.05 .03 .02  .05 .03 .02"/></asset>
<worldbody><geom type="hfield" hfield="h" gap="0.01"/><body pos="0.013 0.021 0.1" euler="20 10 30"><freejoint/><geom type="mesh" mesh="b" gap="0.01"/></body></worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_kinematics(m, d)
R = d.geom_xmat[1].reshape(3, 3); V = m.mesh_vert @ R.T + d.geom_xpos[1]
d.qpos[2] += 0.004 - V[:, 2].min()  # lowest corner 4 mm above z = 0
for k in range(3):
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); dd = mjw.put_data(m, d); mjw.kinematics(mm, dd); mjw.collision(mm, dd)
    n = int(dd.nacon.numpy()[0]); dist = dd.contact.dist.numpy()[:n]; fr = dd.contact.frame.numpy()[:n, 0]
  print(f"shift {k}: {n} contacts, dist {np.round(dist, 9).tolist()}, normals {np.round(fr, 6).tolist()}")
  d.qpos[0] += 0.0371; d.qpos[1] += 0.0213
