MuJoCo C builds each flex's BVH from its active elements (`flex_elemlayer < flex_activelayers`, by default the surface layer of a tetrahedral volume), so interior tetrahedra never produce geom contacts. `_flex_narrowphase_elem_detect` tests every element; the existing `_elem_active` helper is used for self and flex-flex collision but not there.

Versions: MuJoCo Warp `main` at cc97eea, MuJoCo 3.14.0 (pip), warp-lang 1.15.0, Warp CPU device. The reproducer steps MuJoCo C (float64) into contact, gives MuJoCo Warp that exact state, and compares the (geom, flex, elem, vert) contact sets after `forward` and the velocities after one `step`; float32 against float64, so about 1e-5 m/s is agreement.

A 4x4x4 trilinear soft cube resting on a box:

```
soft_cube: ncon Warp 54 / C 34, contact sets equal: False, one-step max |dqvel| 9.65e-08 m/s
```

In this state the interior contacts duplicate surface ones, so the one-step dynamics agree, but the contacts (and their constraint rows) grow by 59 %. On my fork (based on v3.14.0), over a 0.4 s drop of the same cube, filtering with `_elem_active` took the contact sets from 0/73 to 45/73 steps equal to C and the one-step velocity error median from 3.1e-4 to 6.5e-5 m/s. The change is in [6234396](https://github.com/pulipakaa24/mujoco_warp/commit/6234396186) on my fork, behind a module flag `FLEX_ACTIVE_LAYERS_ONLY`.

<details><summary>Reproducer</summary>

```python
import sys, numpy as np, mujoco, warp as wp
import mujoco_warp as mjw

HEAD = '<mujoco><option timestep="{dt}" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/><worldbody>'
CLOTH3 = ('<flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" '
          'radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>')
CASES = {
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
```
</details>
