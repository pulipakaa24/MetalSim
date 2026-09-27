MuJoCo C filters the contacts of every geom-flex pair (per body and flex), flex-flex pair and self-collision to mjMAXCONPAIR = 50 (`filterFlexContacts` in 3.14). MuJoCo Warp applies its sampling only to flex-flex and self groups, and `test_plane_cloth_no_fps_limit` asserts the uncapped count. I saw that #1668 deliberately uses proximity deduplication instead of an FPS limited to mjMAXCONPAIR for flex-geom pairs, with a TODO for a configurable limit, so this is meant as data for that decision rather than a request to copy C's cap as is.

Versions: MuJoCo Warp `main` at cc97eea, MuJoCo 3.14.0 (pip), warp-lang 1.15.0, Warp CPU device. The reproducer steps MuJoCo C (float64) into contact, gives MuJoCo Warp that exact state, and compares the (geom, flex, elem, vert) contact sets after `forward` and the velocities after one `step`; float32 against float64, so about 1e-5 m/s is agreement.

```
plane_cap (10x10 cloth on a plane): ncon Warp 100 / C 50,  contact sets equal: False, one-step max |dqvel| 3.71e-02 m/s
box_cloth (10x10 cloth on a box):   ncon Warp 265 / C 50,  contact sets equal: False, one-step max |dqvel| 6.34e-02 m/s
```

(`box_cloth` also includes the box-triangle difference of #1706.) On my fork I grouped flex-geom candidates per (world, body, flex) as C's midphase does, kept coincident candidates eligible, broke ties in C's order, and reproduced C 3.14's selection exactly (it swaps each chosen contact to the front without updating its bookkeeping, so it is not a plain FPS: a plain FPS matched 0 of 83 capped sets in a cloth-and-cable scene, the emulation 83 of 83; settled mean height bias 2.3 mm -> 0.03 mm). Commits [b2e9ea5](https://github.com/pulipakaa24/mujoco_warp/commit/b2e9ea5c74) and [bbe19bb](https://github.com/pulipakaa24/mujoco_warp/commit/bbe19bb100), behind `ENABLE_GEOM_FLEX_FPS` and `FLEX_FPS_MODE`.

The case for making the limit configurable (as in the TODO): in our measurements on 2026-09-25, a 1 m, 33 x 33 continuum cloth (E 1e6, nu 0.45, thickness 0.01) dropped on a 0.4 m box sank through the box in MuJoCo C 3.14 itself, because 50 contacts per pair cannot hold it; with MuJoCo C rebuilt with the cap at 4000 it has about 290 contacts at impact and about 80 at rest and rests on the box at 0.422 m. MuJoCo Warp with the C-style cap reproduced both outcomes, and with the cap raised to 400 the cloth rests at 0.420 m ([64a1ea5](https://github.com/pulipakaa24/mujoco_warp/commit/64a1ea55cb), `FLEX_MAXCONPAIR`). I have not re-run that large-cloth case against `main`.

<details><summary>Reproducer</summary>

```python
import sys, numpy as np, mujoco, warp as wp
import mujoco_warp as mjw

HEAD = '<mujoco><option timestep="{dt}" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/><worldbody>'
CLOTH3 = ('<flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" '
          'radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>')
CASES = {
  "plane_cap": (HEAD + '<geom type="plane" size="2 2 0.1"/><flexcomp name="cloth" type="grid" count="10 10 1" '
                'spacing="0.04 0.04 0.04" pos="0 0 0.004" dim="2" radius="0.005" mass="0.2"><edge equality="true"/>'
                '<contact selfcollide="none"/></flexcomp></worldbody></mujoco>', 0.002, 3),
  "box_cloth": (HEAD + '<geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/><flexcomp name="cloth" type="grid" '
                'count="10 10 1" spacing="0.04 0.04 0.04" pos="0 0 0.305" dim="2" radius="0.005" mass="0.2">'
                '<edge equality="true"/><contact selfcollide="none"/></flexcomp></worldbody></mujoco>', 0.002, 12),
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
