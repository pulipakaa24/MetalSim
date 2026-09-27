`collision_primitive_core.box_triangle` (the port of `mjraw_BoxTriangle`) returns at most two contacts per triangle and skips C's margin tests. MuJoCo C returns every triangle vertex inside the box inflated by radius + margin and every box corner touching the triangle, up to mjMAXCONPAIR per element (at most 11 in practice). A cloth on a box therefore gets fewer, differently placed contacts than in MuJoCo C, even where no per-pair cap is involved.

Versions: MuJoCo Warp `main` at cc97eea, MuJoCo 3.14.0 (pip), warp-lang 1.15.0, Warp CPU device. The reproducer steps MuJoCo C (float64) into contact, gives MuJoCo Warp that exact state, and compares the (geom, flex, elem, vert) contact sets after `forward` and the velocities after one `step`; float32 against float64, so about 1e-5 m/s is agreement.

A 3x3 cloth lying on a box (24 contacts in C, well under any cap):

```
box_cloth_3x3: ncon Warp 16 / C 24, contact sets equal: False, one-step max |dqvel| 1.56e-02 m/s
```

On my fork (based on v3.14.0), generating all of C's candidates in C's order with C's tests gives 24/24, equal sets and 3.2e-7 m/s; for a draped 10x10 cloth the raw candidate count goes from 286 to 384, which equals an uncapped MuJoCo C build. The change is in [b2e9ea5](https://github.com/pulipakaa24/mujoco_warp/commit/b2e9ea5c74) on my fork, behind a module flag `BOX_TRIANGLE_ALL_CONTACTS` (that commit also contains the per-pair cap work discussed in #1709).

<details><summary>Reproducer</summary>

```python
import sys, numpy as np, mujoco, warp as wp
import mujoco_warp as mjw

HEAD = '<mujoco><option timestep="{dt}" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/><worldbody>'
CLOTH3 = ('<flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" '
          'radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>')
CASES = {
  "box_cloth_3x3": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/>' + CLOTH3 + '</worldbody></mujoco>',
                    0.002, 5),
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
