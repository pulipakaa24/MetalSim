For `dim=1` flexes, `_flex_narrowphase_elem_detect` returns early (`if flex_dim[flexid] < 2`, collision_flex.py) and only the vertices collide, as spheres. MuJoCo C collides each 1D element as a capsule (`mj_collideGeomElem` -> `makeCapsule` -> `mjraw_SphereCapsule`, `mjraw_CapsuleCapsule`, `mjraw_CapsuleBox`, otherwise `mjc_ConvexElem`), so a rope lying across an edge is supported along its segments. `test_sphere_rope_collision` currently asserts Warp's 1 contact where C produces 2. Supporting segment contacts also needs a dim-1 branch in `_flex_contact_bodies_weights`, which has none today (the rows would be empty).

Versions: MuJoCo Warp `main` at cc97eea, MuJoCo 3.14.0 (pip), warp-lang 1.15.0, Warp CPU device. The reproducer steps MuJoCo C (float64) into contact, gives MuJoCo Warp that exact state, and compares the (geom, flex, elem, vert) contact sets after `forward` and the velocities after one `step`; float32 against float64, so about 1e-5 m/s is agreement.

A 12-vertex rope lying across a box:

```
cable_box: ncon Warp 6 / C 12, contact sets equal: False, one-step max |dqvel| 3.56e-02 m/s
```

On my fork (based on v3.14.0), with dim-1 elements in the element narrowphase using the raw primitives (normal from geom to flex, as in C; zero-radius capsules inflated by the flex radius for the CCD geoms) plus inverse-distance vertex weights for dim-1 element contacts, the case gives 12/12, equal sets and 2.7e-6 m/s. The change is in [a277152](https://github.com/pulipakaa24/mujoco_warp/commit/a2771520f5) on my fork, behind a module flag `CABLE_CAPSULE_ELEMENTS`.

<details><summary>Reproducer</summary>

```python
import sys, numpy as np, mujoco, warp as wp
import mujoco_warp as mjw

HEAD = '<mujoco><option timestep="{dt}" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/><worldbody>'
CLOTH3 = ('<flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" '
          'radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>')
CASES = {
  "cable_box": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/><flexcomp name="rope" type="grid" '
                'count="12 1 1" spacing="0.03 0.03 0.03" pos="-0.15 0 0.31" dim="1" radius="0.006" mass="0.05">'
                '<edge equality="true"/></flexcomp></worldbody></mujoco>', 0.002, 30),
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
