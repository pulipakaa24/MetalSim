"""Margin convention on heightfields: sphere (r 0.1) at gap delta above a flat heightfield, geom margin m on both
geoms (pair margin = max). Prints contact dist of MuJoCo C and MuJoCo Warp (whatever is on PYTHONPATH), CPU device."""
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
print(mjw.__file__)
for typ, size in (("sphere", ".1"), ("box", ".1 .1 .1"), ("capsule", ".1 .1")):
  for m_ in (0.0, 0.02):
    for delta in (-0.005, 0.005, 0.015):
      xml = f"""<mujoco><asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.1 0.1"/></asset><worldbody>
      <geom type="hfield" hfield="h" margin="{m_}"/><body pos="0.03 0.04 {0.1 + delta}"><freejoint/>
      <geom type="{typ}" size="{size}" margin="{m_}"/></body></worldbody></mujoco>"""
      if typ == "capsule":
        xml = xml.replace('<freejoint/>', '<freejoint/>').replace(f'pos="0.03 0.04 {0.1 + delta}"', f'pos="0.03 0.04 {0.1 + delta}" euler="0 90 0"')
      mm = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(mm); mujoco.mj_forward(mm, dd)
      cd = sorted(dd.contact.dist[:dd.ncon].round(5).tolist())
      with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mm); d = mjw.put_data(mm, dd); mjw.forward(m, d)
        n = int(d.nacon.numpy()[0]); wd = sorted(d.contact.dist.numpy()[:n].round(5).tolist())
      print(f"{typ:8s} margin {m_} gap {delta:+.3f}: C ncon {dd.ncon} dist {cd[:4]} | MJW ncon {n} dist {wd[:4]}")
