"""Minimal reproducers for the MuJoCo Warp flex fixes proposed upstream (UPSTREAM.md). Each case: MuJoCo C (float64) is
stepped to a state in contact; MuJoCo Warp gets that state, runs forward (contact set) and one step (velocities).
Printed: contact count W/C, whether the (geom, flex, elem, vert) contact sets are equal, max |qvel_W - qvel_C|.
Run once with the unmodified MuJoCo Warp and once with the fixed one; plain MuJoCo Warp API only (no MetalSim).
usage: python upstream_repro.py [cpu|metal:0] [case ...]"""
import sys, json, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw

CASES = {
  # 1. element ids of a flex that is not the first: a 3x3 cloth (flex 1) resting on a sphere (one contact per triangle, no
  #    cap involved), a 3-vertex rope (flex 0) far away. Control: the same cloth as the only flex ("first_flex").
  "second_flex": ("""<mujoco><option timestep="0.002" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="sphere" size="0.15" pos="0 0 0.15"/>
    <flexcomp name="rope" type="grid" count="3 1 1" spacing="0.05 0.05 0.05" pos="1 1 0.02" dim="1" radius="0.005" mass="0.05"><edge equality="true"/></flexcomp>
    <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>
    </worldbody></mujoco>""", 5),
  "first_flex": ("""<mujoco><option timestep="0.002" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="sphere" size="0.15" pos="0 0 0.15"/>
    <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>
    </worldbody></mujoco>""", 5),
  # 3a. box-triangle contacts without the cap: a 3x3 cloth on a box (C: all vertices and corners per triangle, < 50 total)
  "box_cloth_3x3": ("""<mujoco><option timestep="0.002" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/>
    <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.05 0.05 0.05" pos="0.01 0.02 0.305" dim="2" radius="0.005" mass="0.1"><edge equality="true"/><contact selfcollide="none"/></flexcomp>
    </worldbody></mujoco>""", 5),
  # 2. per-pair contact cap: a 10x10 cloth lying on a plane (100 vertex contacts; MuJoCo C keeps mjMAXCONPAIR = 50)
  "plane_cap": ("""<mujoco><option timestep="0.002" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="plane" size="2 2 0.1"/>
    <flexcomp name="cloth" type="grid" count="10 10 1" spacing="0.04 0.04 0.04" pos="0 0 0.004" dim="2" radius="0.005" mass="0.2"><edge equality="true"/><contact selfcollide="none"/></flexcomp>
    </worldbody></mujoco>""", 3),
  # 3b. box-triangle contacts and the cap together: a 10x10 cloth lying on a box
  "box_cloth": ("""<mujoco><option timestep="0.002" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/>
    <flexcomp name="cloth" type="grid" count="10 10 1" spacing="0.04 0.04 0.04" pos="0 0 0.305" dim="2" radius="0.005" mass="0.2"><edge equality="true"/><contact selfcollide="none"/></flexcomp>
    </worldbody></mujoco>""", 12),
  # 4. cable segments as capsules: a 12-vertex rope lying across a box edge (C: capsule-box per segment)
  "cable_box": ("""<mujoco><option timestep="0.002" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/>
    <flexcomp name="rope" type="grid" count="12 1 1" spacing="0.03 0.03 0.03" pos="-0.15 0 0.31" dim="1" radius="0.006" mass="0.05"><edge equality="true"/></flexcomp>
    </worldbody></mujoco>""", 30),
  # 5. volumes: only surface-layer elements collide (C's flex BVH holds active elements only): a trilinear 4x4x4 cube on a box
  "soft_cube": ("""<mujoco><option timestep="0.001" solver="CG" iterations="100" ls_iterations="50" jacobian="sparse"/>
    <worldbody><geom type="box" size="0.15 0.15 0.15" pos="0 0 0.15"/>
    <flexcomp name="soft" type="grid" count="4 4 4" spacing="0.05 0.05 0.05" pos="0.02 0.01 0.31" dim="3" radius="0.003" mass="0.5" dof="trilinear">
      <elasticity young="5e3" poisson="0.3" damping="0.01"/><contact selfcollide="none"/></flexcomp>
    </worldbody></mujoco>""", 40),
}


def keys(c, n):
  if n == 0:
    return []
  return sorted(map(tuple, np.stack([c.geom[:n, 0], c.flex[:n, 1], c.elem[:n, 1], c.vert[:n, 1]], 1).tolist()))


def run(name, dev):
  xml, nsteps = CASES[name]
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)
  d.qpos[:] += np.random.default_rng(0).normal(0, 1e-4, m.nq)  # no exact ties
  mujoco.mj_forward(m, d)
  for _ in range(nsteps):
    mujoco.mj_step(m, d)
  mujoco.mj_forward(m, d)
  with wp.ScopedDevice(dev):
    M = mjw.put_model(m)
    M.opt.warn_overflow = 0
    nv3 = 3 * m.nflexvert
    D = mjw.put_data(m, d, nworld=1, nconmax=12 * m.nflexvert + 64, njmax=20 * m.nflexvert + 256, njmax_nnz=400 * m.nflexvert + 4096)
    mjw.forward(M, D)
    wd = mujoco.MjData(m)
    mjw.get_data_into(wd, m, D)
    mjw.step(M, D)
    qv = D.qvel.numpy()[0]
  n_w = wd.ncon
  c_contacts = keys(d.contact, d.ncon)
  w_contacts = keys(wd.contact, n_w)
  mujoco.mj_step(m, d)
  return {"case": name, "device": dev, "mujoco_warp": mjw.__file__.split("/")[-3], "ncon_warp": int(n_w), "ncon_c": int(len(c_contacts)),
          "contact_sets_equal": w_contacts == c_contacts, "onestep_max_dqvel": float(np.abs(qv - d.qvel).max()),
          "max_abs_qvel_c": float(np.abs(d.qvel).max())}


if __name__ == "__main__":
  dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
  for name in (sys.argv[2:] or CASES):
    try:
      print(json.dumps(run(name, dev)), flush=True)
    except Exception as e:  # noqa: BLE001
      print(json.dumps({"case": name, "device": dev, "error": repr(e)[:300]}), flush=True)
