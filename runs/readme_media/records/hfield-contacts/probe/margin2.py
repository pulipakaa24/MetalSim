"""Margin convention on heightfields: a geom at gap g above a flat heightfield (heights 0: surface at z = 0), margin m on both geoms
(pair margin = sum, MuJoCo 3.14), gap 0. Prints the deepest contact dist of MuJoCo C and MuJoCo Warp (copy on PYTHONPATH,
CPU device; MJW_SET as in cc2) next to the plane convention (dist = g) and C's heightfield value g - 1.5 * m_pair.
Shapes: sphere r 0.1, box 0.1, capsule r 0.1 (horizontal), box mesh 0.2 m (8 vertices)."""
import os, importlib, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
for kv in filter(None, os.environ.get("MJW_SET", "").split(",")):
  k, v = kv.split("="); mod, attr = k.rsplit(".", 1); M = importlib.import_module(f"mujoco_warp._src.{mod}")
  setattr(M, attr, type(getattr(M, attr))(eval(v)))
print(mjw.__file__, os.environ.get("MJW_SET", ""))
mesh = '<asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.1 0.1"/><mesh name="b" vertex="-.1 -.1 -.1  .1 -.1 -.1  -.1 .1 -.1  .1 .1 -.1  -.1 -.1 .1  .1 -.1 .1  -.1 .1 .1  .1 .1 .1"/></asset>'
for typ, geom in (("sphere", 'type="sphere" size=".1"'), ("box", 'type="box" size=".1 .1 .1"'),
                  ("capsule", 'type="capsule" size=".1 .1" euler="0 90 0"'), ("mesh", 'type="mesh" mesh="b"')):
  for m_ in (0.0, 0.01):
    for g in (-0.005, 0.005, 0.015, 0.025, 0.035):
      xml = f"""<mujoco>{mesh}<worldbody><geom type="hfield" hfield="h" margin="{m_}"/><body pos="0.03 0.04 {0.1 + g}">
      <freejoint/><geom {geom} margin="{m_}"/></body></worldbody></mujoco>"""
      mm = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(mm); mujoco.mj_forward(mm, dd)
      cd = min(dd.contact.dist[:dd.ncon]) if dd.ncon else None
      with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mm); d = mjw.put_data(mm, dd); mjw.kinematics(m, d); mjw.collision(m, d)
        n = int(d.nacon.numpy()[0]); wd = min(d.contact.dist.numpy()[:n]) if n else None
      mp = 2 * m_
      f = lambda x: "  none " if x is None else f"{x:+.4f}"
      print(f"{typ:8s} margin {m_:.2f} (pair {mp:.2f}) gap {g:+.3f}: C {f(cd)} (ncon {dd.ncon}) | MJW {f(wd)} (ncon {n}) | g - 1.5 m_pair {g - 1.5 * mp:+.4f}")
