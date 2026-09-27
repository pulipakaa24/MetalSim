"""Self-contained flex reproducers as posted in the google-deepmind/mujoco_warp issues (plain MuJoCo + MuJoCo Warp API).
MuJoCo C (float64) is stepped into contact, MuJoCo Warp (CPU device) gets that exact state; printed: contact counts,
whether the (geom, flex, elem, vert) contact sets are equal, and max |qvel_Warp - qvel_C| after one step.
usage: python flex_issue_harness.py [case ...]"""
import sys, numpy as np, mujoco, warp as wp
import mujoco_warp as mjw

HEAD = '<mujoco><option timestep="{dt}" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/><worldbody>'
CLOTH3 = ('<flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" '
          'radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>')
CASES = {
  "second_flex": (HEAD + '<geom type="sphere" size="0.15" pos="0 0 0.15"/><flexcomp name="rope" type="grid" count="3 1 1" '
                  'spacing="0.05 0.05 0.05" pos="1 1 0.02" dim="1" radius="0.005" mass="0.05"><edge equality="true"/>'
                  '</flexcomp>' + CLOTH3 + '</worldbody></mujoco>', 0.002, 5),
  "first_flex": (HEAD + '<geom type="sphere" size="0.15" pos="0 0 0.15"/>' + CLOTH3 + '</worldbody></mujoco>', 0.002, 5),
  "box_cloth_3x3": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/>' + CLOTH3 + '</worldbody></mujoco>',
                    0.002, 5),
  "plane_cap": (HEAD + '<geom type="plane" size="2 2 0.1"/><flexcomp name="cloth" type="grid" count="10 10 1" '
                'spacing="0.04 0.04 0.04" pos="0 0 0.004" dim="2" radius="0.005" mass="0.2"><edge equality="true"/>'
                '<contact selfcollide="none"/></flexcomp></worldbody></mujoco>', 0.002, 3),
  "box_cloth": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/><flexcomp name="cloth" type="grid" '
                'count="10 10 1" spacing="0.04 0.04 0.04" pos="0 0 0.305" dim="2" radius="0.005" mass="0.2">'
                '<edge equality="true"/><contact selfcollide="none"/></flexcomp></worldbody></mujoco>', 0.002, 12),
  "cable_box": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/><flexcomp name="rope" type="grid" '
                'count="12 1 1" spacing="0.03 0.03 0.03" pos="-0.15 0 0.31" dim="1" radius="0.006" mass="0.05">'
                '<edge equality="true"/></flexcomp></worldbody></mujoco>', 0.002, 30),
  "soft_cube": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/><flexcomp name="soft" type="grid" '
                'count="4 4 4" spacing="0.05 0.05 0.05" pos="0.02 0.01 0.31" dim="3" radius="0.003" mass="0.5" '
                'dof="trilinear"><elasticity young="5e3" poisson="0.3" damping="0.01"/><contact selfcollide="none"/>'
                '</flexcomp></worldbody></mujoco>', 0.001, 40),
}


def contact_set(c, n):
  return sorted(zip(c.geom[:n, 0].tolist(), c.flex[:n, 1].tolist(), c.elem[:n, 1].tolist(), c.vert[:n, 1].tolist()))


def compare(name):
  xml, dt, nsteps = CASES[name]
  m = mujoco.MjModel.from_xml_string(xml.format(dt=dt))
  d = mujoco.MjData(m)
  d.qpos[:] += np.random.default_rng(0).normal(0, 1e-4, m.nq)  # avoid exact ties
  for _ in range(nsteps):
    mujoco.mj_step(m, d)
  mujoco.mj_forward(m, d)
  with wp.ScopedDevice("cpu"):
    mw = mjw.put_model(m)
    mw.opt.warn_overflow = 0
    nv = m.nflexvert
    dw = mjw.put_data(m, d, nworld=1, nconmax=12 * nv + 64, njmax=20 * nv + 256, njmax_nnz=400 * nv + 4096)
    mjw.forward(mw, dw)
    dwc = mujoco.MjData(m)
    mjw.get_data_into(dwc, m, dw)
    mjw.step(mw, dw)
    qvel_w = dw.qvel.numpy()[0]
  ncon_c, set_c = d.ncon, contact_set(d.contact, d.ncon)
  set_w = contact_set(dwc.contact, dwc.ncon)
  mujoco.mj_step(m, d)
  print(f"{name}: ncon Warp {dwc.ncon} / C {ncon_c}, contact sets equal: {set_w == set_c}, "
        f"one-step max |dqvel| {np.abs(qvel_w - d.qvel).max():.2e} m/s")


if __name__ == "__main__":
  print("mujoco", mujoco.__version__, "mujoco_warp", mjw.__file__, "warp", wp.__version__)
  for name in sys.argv[1:] or CASES:
    compare(name)
