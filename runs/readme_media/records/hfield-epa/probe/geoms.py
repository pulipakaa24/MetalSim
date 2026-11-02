"""List the G1 rough task's geoms that can collide with the heightfield (stream hfield-epa)."""
import numpy as np, mujoco
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.learn.terrain import isaac_rough_terrain
hf = isaac_rough_terrain(seed=0, collision="hfield")
m, _ = build_g1_model("rough", hf, effort_limit="actuator")
print("ccd_iterations", m.opt.ccd_iterations, "ccd_tol", m.opt.ccd_tolerance, "nhfield", m.nhfield, "hfield nrow/ncol", m.hfield_nrow, m.hfield_ncol, "size", m.hfield_size)
g0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
for g in range(m.ngeom):
  if g == g0: continue
  ok = (m.geom_contype[g] & m.geom_conaffinity[g0]) or (m.geom_contype[g0] & m.geom_conaffinity[g])
  if not ok: continue
  t = mujoco.mjtGeom(m.geom_type[g]).name
  nv = m.mesh_vertnum[m.geom_dataid[g]] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else 0
  print(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]), t, "nvert", nv, "rbound", round(float(m.geom_rbound[g]), 3), "size", np.round(m.geom_size[g], 3))
