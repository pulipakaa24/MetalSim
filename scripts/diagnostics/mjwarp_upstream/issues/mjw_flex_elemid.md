`_flex_narrowphase_elem_detect` (collision_flex.py) launches over all flex elements and passes the global `elemid` to the contact writer (around line 2071), so `contact.elem` holds the global element index. The contact Jacobian (`_efc_contact_*_flex` in constraint.py, `flex_elemdataadr[f] + e * (dim + 1)`) and MuJoCo C interpret `contact.elem` as the index local to the flex. The two coincide for flex 0, which is why the existing tests pass; for any later flex the contact force is applied to the vertices of a different element (or read out of range). The kernel already computes `local_elemid = elemid - flex_elemadr[flexid]` a few lines above and uses it for the element data.

Versions: MuJoCo Warp `main` at cc97eea, MuJoCo 3.14.0 (pip), warp-lang 1.15.0, Warp CPU device. The reproducer steps MuJoCo C (float64) into contact, gives MuJoCo Warp that exact state, and compares the (geom, flex, elem, vert) contact sets after `forward` and the velocities after one `step`; float32 against float64, so about 1e-5 m/s is agreement.

A 3x3 cloth resting on a sphere, once as the only flex (control) and once as the second flex after a small rope placed far away:

```
second_flex: ncon Warp 2 / C 2, contact sets equal: False, one-step max |dqvel| 3.17e-02 m/s
first_flex:  ncon Warp 1 / C 1, contact sets equal: True,  one-step max |dqvel| 2.47e-06 m/s
```

On my fork (based on v3.14.0, where the code is the same), passing `local_elemid` instead of `elemid` gives equal sets and 4.6e-6 m/s. In a larger scene a 10x10 cloth that was the second flex fell through a box (0.76 m vertex error after 0.8 s against MuJoCo C). The change is in [a277152](https://github.com/pulipakaa24/mujoco_warp/commit/a2771520f5) on my fork (that commit also contains the cable change of #1707; this part is the three `elemid` -> `local_elemid` call sites).

<details><summary>Reproducer</summary>

```python
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
