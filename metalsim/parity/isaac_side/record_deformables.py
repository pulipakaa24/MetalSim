"""Isaac side of MetalSim's deformable protocol (runs inside Isaac Sim 5.1 / Isaac Lab v2.3.2 on the parity VM).

Three scenes in one stage, 10 m apart, PhysX GPU, 200 Hz physics, 5 s, per-step node positions and velocities:
  (a) cloth: 1 m x 1 m PhysX particle cloth (Isaac Lab v2.3.2 has no cloth asset; omni.physx particleUtils, the
      parameters of PhysX's ParticleClothDemo), 21 x 21 vertices, released flat at z = 0.5 m over a static 0.4 m box
  (b) rope: 0.5 m x 2 cm x 2 cm Isaac Lab DeformableObject (PhysX FEM), horizontal at z = 1.0 m, nodes within 1 cm of
      its -x end held by kinematic targets, released
  (c) soft cube: 0.2 m Isaac Lab DeformableObject (PhysX FEM), centre at z = 0.5 m, dropped onto the ground plane
Materials: Isaac Lab's DeformableBodyMaterialCfg defaults except youngs_modulus 1e5 / poissons_ratio 0.4 (Isaac Lab's
deformable tutorial values; the default 5e7 makes a 2 cm rod a stiff cantilever) and density 1000 (default None =
solver decides, which leaves the mass unknown). Everything is written to --out/meta.json and --out/record.npz.

    python record_deformables.py --headless --out ~/deformable_out
"""
import argparse, json, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, Vt
from omni.physx.scripts import particleUtils, physicsUtils

os.makedirs(args.out, exist_ok=True)
DT = 1.0 / 200.0
meta = {"physics_dt": DT, "seconds": args.seconds, "device": "cuda:0", "gravity": [0, 0, -9.81],
        "isaac_lab": "v2.3.2", "isaac_sim": "5.1", "scenes": {}}

sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
stage = omni.usd.get_context().get_stage()

# ground (Isaac Lab GroundPlaneCfg default physics material: static/dynamic friction 0.5, restitution 0)
gp = sim_utils.GroundPlaneCfg()
gp.func("/World/ground", gp)
meta["ground"] = {"static_friction": gp.physics_material.static_friction, "dynamic_friction": gp.physics_material.dynamic_friction,
                  "restitution": gp.physics_material.restitution}
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

# ---------------- (a) cloth over a box ----------------
CLOTH_ORIGIN = (0.0, 0.0)
BOX = 0.4
box_cfg = sim_utils.CuboidCfg(size=(BOX, BOX, BOX), collision_props=sim_utils.CollisionPropertiesCfg(),
                              physics_material=sim_utils.RigidBodyMaterialCfg())
box_cfg.func("/World/cloth_scene/box", box_cfg, translation=(CLOTH_ORIGIN[0], CLOTH_ORIGIN[1], BOX / 2))
N, W, Z0 = 21, 1.0, 0.5
xs = np.linspace(-W / 2, W / 2, N)
pts = [Gf.Vec3f(float(CLOTH_ORIGIN[0] + x), float(CLOTH_ORIGIN[1] + y), Z0) for y in xs for x in xs]
tris = []
for j in range(N - 1):
    for i in range(N - 1):
        a, b, c, d = j * N + i, j * N + i + 1, (j + 1) * N + i, (j + 1) * N + i + 1
        tris += [a, b, d, a, d, c]
cloth_path = Sdf.Path("/World/cloth_scene/cloth")
mesh = UsdGeom.Mesh.Define(stage, cloth_path)
mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts))
mesh.GetFaceVertexCountsAttr().Set([3] * (len(tris) // 3))
mesh.GetFaceVertexIndicesAttr().Set(tris)
radius = 0.5 * W / (N - 1)                           # ParticleClothDemo: particles just touching at rest
cloth = {"verts_per_side": N, "size_m": W, "z0": Z0, "particle_rest_offset": radius, "contact_offset": 1.5 * radius,
         "solver_position_iterations": 16, "spring_stretch_stiffness": 10000.0, "spring_bend_stiffness": 200.0,
         "spring_shear_stiffness": 100.0, "spring_damping": 0.2, "particle_mass": 0.02, "pbd_friction": 0.6,
         "pbd_drag": 0.0, "pbd_lift": 0.0, "self_collision": True, "box_size": BOX,
         "source": "omni.physx ParticleClothDemo values (drag/lift set to 0: no aerodynamics in MuJoCo)"}
psys = Sdf.Path("/World/cloth_scene/particleSystem")
particleUtils.add_physx_particle_system(stage=stage, particle_system_path=psys, contact_offset=cloth["contact_offset"],
                                       rest_offset=radius, particle_contact_offset=cloth["contact_offset"],
                                       solid_rest_offset=radius, fluid_rest_offset=0.0, solver_position_iterations=16)
pmat = Sdf.Path("/World/cloth_scene/particleMaterial")
particleUtils.add_pbd_particle_material(stage, pmat, friction=cloth["pbd_friction"], drag=0.0, lift=0.0)
physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(psys), pmat)
particleUtils.add_physx_particle_cloth(stage=stage, path=cloth_path, dynamic_mesh_path=None, particle_system_path=psys,
                                       spring_stretch_stiffness=cloth["spring_stretch_stiffness"],
                                       spring_bend_stiffness=cloth["spring_bend_stiffness"],
                                       spring_shear_stiffness=cloth["spring_shear_stiffness"],
                                       spring_damping=cloth["spring_damping"], self_collision=True, self_collision_filter=True)
cloth["mass"] = cloth["particle_mass"] * N * N
UsdPhysics.MassAPI.Apply(mesh.GetPrim()).GetMassAttr().Set(cloth["mass"])
meta["scenes"]["cloth"] = cloth

# ---------------- (b) rope, (c) soft cube: Isaac Lab DeformableObject ----------------
MAT = dict(youngs_modulus=1e5, poissons_ratio=0.4, density=1000.0)      # others: Isaac Lab defaults
mat_cfg = sim_utils.DeformableBodyMaterialCfg(**MAT)
props = sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001)   # Isaac Lab tutorial values
ROPE_L, ROPE_T, ROPE_Z, ROPE_X0 = 0.5, 0.02, 1.0, 10.0
rope = DeformableObject(DeformableObjectCfg(
    prim_path="/World/rope", spawn=sim_utils.MeshCuboidCfg(size=(ROPE_L, ROPE_T, ROPE_T), deformable_props=props, physics_material=mat_cfg),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(ROPE_X0 + ROPE_L / 2, 0.0, ROPE_Z))))
CUBE, CUBE_Z, CUBE_X0 = 0.2, 0.5, 20.0
cube = DeformableObject(DeformableObjectCfg(
    prim_path="/World/cube", spawn=sim_utils.MeshCuboidCfg(size=(CUBE, CUBE, CUBE), deformable_props=props, physics_material=mat_cfg),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(CUBE_X0, 0.0, CUBE_Z))))

def mat_meta():
    m = {k: getattr(mat_cfg, k) for k in ("density", "dynamic_friction", "youngs_modulus", "poissons_ratio", "elasticity_damping", "damping_scale")}
    p = {k: getattr(props, k) for k in ("rest_offset", "contact_offset", "simulation_hexahedral_resolution", "solver_position_iteration_count",
                                        "vertex_velocity_damping", "self_collision")}
    return m, p

m_, p_ = mat_meta()
meta["scenes"]["rope"] = {"length": ROPE_L, "thickness": ROPE_T, "z0": ROPE_Z, "x0": ROPE_X0, "pinned_end": "-x, nodes within 0.01 m",
                          "mass": MAT["density"] * ROPE_L * ROPE_T * ROPE_T, "material": m_, "deformable_props": p_}
meta["scenes"]["cube"] = {"size": CUBE, "z0": CUBE_Z, "x0": CUBE_X0, "mass": MAT["density"] * CUBE ** 3, "material": m_, "deformable_props": p_}

sim.reset()
import omni.physics.tensors as tensors
view = tensors.create_simulation_view("torch")
view.set_subspace_roots("/")
cloth_view = view.create_particle_cloth_view(str(cloth_path))

# pin the rope's -x end
tgt = rope.data.nodal_kinematic_target.clone()
p0 = rope.data.default_nodal_state_w[..., :3].clone()
tgt[..., :3] = p0
tgt[..., 3] = 1.0
pinned = p0[0, :, 0] < p0[0, :, 0].min() + 0.01
tgt[0, pinned, 3] = 0.0
rope.write_nodal_kinematic_target_to_sim(tgt)
meta["scenes"]["rope"]["n_pinned"] = int(pinned.sum())

nsteps = int(round(args.seconds / DT))
rec = {k: [] for k in ("cloth_pos", "cloth_vel", "rope_pos", "rope_vel", "cube_pos", "cube_vel")}
t0 = time.time()
for k in range(nsteps + 1):
    if k:
        rope.write_data_to_sim(); cube.write_data_to_sim()
        sim.step(render=False)
        rope.update(DT); cube.update(DT)
    rec["cloth_pos"].append(cloth_view.get_positions().reshape(-1, 3).cpu().numpy().copy())
    rec["cloth_vel"].append(cloth_view.get_velocities().reshape(-1, 3).cpu().numpy().copy())
    rec["rope_pos"].append(rope.data.nodal_pos_w[0].cpu().numpy().copy()); rec["rope_vel"].append(rope.data.nodal_vel_w[0].cpu().numpy().copy())
    rec["cube_pos"].append(cube.data.nodal_pos_w[0].cpu().numpy().copy()); rec["cube_vel"].append(cube.data.nodal_vel_w[0].cpu().numpy().copy())
wall = time.time() - t0
meta["wall_s_including_readback"] = wall
meta["counts"] = {"cloth_particles": int(rec["cloth_pos"][0].shape[0]), "rope_nodes": int(rec["rope_pos"][0].shape[0]),
                  "cube_nodes": int(rec["cube_pos"][0].shape[0])}
np.savez_compressed(os.path.join(args.out, "record.npz"), **{k: np.array(v, np.float32) for k, v in rec.items()},
                    cloth_rest=np.array([[p[0], p[1], p[2]] for p in pts], np.float32), cloth_tris=np.array(tris, np.int32).reshape(-1, 3),
                    pinned=pinned.cpu().numpy())
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1, default=str)
print("DEFORMABLE_RECORD_DONE", json.dumps(meta["counts"]), f"{wall:.1f}s")
simulation_app.close()
