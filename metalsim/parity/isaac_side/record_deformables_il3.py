"""Isaac side of MetalSim's deformable protocol for Isaac Lab 3.0-EA (Isaac Sim 6.1), both deformable backends.

    python record_deformables_il3.py --physics isaacsim_physx --out ~/deformable_il3/physx --headless
    python record_deformables_il3.py --physics newton_vbd     --out ~/deformable_il3/newton_vbd --headless

Protocols (200 Hz physics, 5 s, per-step node positions/velocities; the three scenes 10 m apart in one stage):
  (a) cloth: 1 m x 1 m surface deformable (MeshRectangleCfg, edge_refinement 21), released flat at z = 0.5 m over a
      static 0.4 m box (top at 0.4 m) standing on the ground
  (b) rope: 0.5 m long, released from horizontal at z = 1.0 m with its -x end held
      - physx: 0.5 x 0.02 x 0.02 m volume deformable, nodes within 1 cm of the -x end held by kinematic targets
      - newton_vbd: Isaac Lab 3.0 CableObject (VBD rod, 20 segments), segment 0 re-posed every step (no kinematic
        API on cables); plus the same volume rod as physx (VBD volume, kinematic targets) for a like-for-like row
  (c) soft cube: 0.2 m volume deformable, centre at z = 0.5 m, dropped onto the ground
  Mesh-resolution sweep (--resolutions, default "d,2,4,8"): rope and cube repeated side by side (2 m apart in y) at
  Isaac Lab's default mesh and at 2, 4, 8 cells across the rod's cross-section / along the cube's edge, to test the
  shear-locking explanation of the rope's short period (docs/research/deformables_2026-09-25.md section 11): the
  period should approach the thin-rod value (~1.2-1.3 s at this amplitude) as the mesh refines.
Materials: each backend's Isaac Lab 3.0 config defaults (PhysxDeformableBodyMaterialCfg /
PhysxSurfaceDeformableBodyMaterialCfg; NewtonDeformableBodyMaterialCfg / NewtonSurfaceDeformableBodyMaterialCfg /
CableMaterialCfg), except where --override says otherwise; every value actually used goes to meta.json.
Newton solver settings follow scripts/demos/deformables.py (VBD 20 iterations, 4 substeps, soft contact ke 1e5 kd 1),
with the soft-contact friction set to the ground/box friction (the demo's 0.01 would make every drape frictionless).
Every scene is wrapped: a failure is recorded in meta.json ("errors") and the other scenes still run. After sim.reset()
the prims under each scene (applied schemas and all attributes, including what the PhysX cooker wrote: simulation and
collision tet meshes, springs, rest shapes) go to cooked_prims.json / cooked.npz.

Comparison plan on the Mac (scripts/diagnostics/deformable/physx_protocol.py): the Newton-backend recording first
against the same solver on Metal (Newton 1.5.2 VBD / SolverCoupledProxy in .venv-newton152), then MuJoCo Warp flex,
MetalSim XPBDSim and the physx_cloth prototype as alternatives; the PhysX-backend recording against XPBD (PBD cloth)
and flex (FEM volumes).
"""
import argparse, json, os, time, traceback
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx", "newton_vbd"])
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--scenes", default="cloth,rope,cube")
# mesh-resolution sweep (shear-locking test): cells across the rod's 2 cm cross-section / along the cube's 0.2 m edge.
# "d" = Isaac Lab's default edge_refinement (4.0). edge_refinement = bounding-box diagonal / max edge length.
parser.add_argument("--resolutions", default="1,2,4,8")
# pre-tetrahedralized structured meshes (make_tetmeshes.py): rod_n{n}.usda, cube_n{n}.usda. Isaac Lab's automatic
# tetrahedralization (pytetwild) aborts the Kit process on the parity VM ("double free or corruption").
parser.add_argument("--tetdir", default=os.path.expanduser("~/tetmeshes"))
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.physics import PhysicsCfg

NEWTON = args.physics == "newton_vbd"
if NEWTON:
    from isaaclab_newton.physics import NewtonSoftContactCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg as PropsCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg as VolMatCfg
    from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg as SurfMatCfg
else:
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg as PropsCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg as VolMatCfg
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg as SurfMatCfg

DT = 1.0 / 200.0
FRICTION = 0.5          # ground / box (Isaac Lab GroundPlaneCfg default rigid material)
os.makedirs(args.out, exist_ok=True)
meta = {"physics": args.physics, "physics_dt": DT, "seconds": args.seconds, "gravity": [0, 0, -9.81],
        "isaac_lab": "3.0.0-EA", "scenes": {}, "errors": {}}
scenes = set(args.scenes.split(","))


def cfg_dict(c):
    try:
        return {k: (v if isinstance(v, (int, float, str, bool, type(None), list, tuple)) else str(v)) for k, v in c.to_dict().items()}
    except Exception:  # noqa: BLE001
        return str(c)


with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
    if NEWTON:
        physics_cfg.solver_cfg.iterations = 20
        physics_cfg.num_substeps = 4
        physics_cfg.collision_decimation = 1
        physics_cfg.soft_contact_cfg = NewtonSoftContactCfg(soft_contact_ke=1.0e5, soft_contact_kd=1.0, soft_contact_mu=FRICTION)
    meta["physics_cfg"] = cfg_dict(physics_cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device, physics=physics_cfg))

    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/ground", gp)
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    meta["ground"] = cfg_dict(gp)

    objs = {}
    # (a) cloth over a static box
    if "cloth" in scenes:
        try:
            box = sim_utils.CuboidCfg(size=(0.4, 0.4, 0.4), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION))
            box.func("/World/cloth_scene/box", box, translation=(0.0, 0.0, 0.2))
            surf = SurfMatCfg()
            cloth_cfg = DeformableObjectCfg(
                prim_path="/World/cloth_scene/cloth",
                spawn=sim_utils.MeshRectangleCfg(size=(1.0, 1.0), edge_refinement=21, deformable_props=PropsCfg(),
                                                 physics_material=surf),
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)))
            objs["cloth"] = cloth_cfg.class_type(cloth_cfg)
            meta["scenes"]["cloth"] = {"size": [1.0, 1.0], "edge_refinement": 21, "z0": 0.5, "box": 0.4,
                                       "material": cfg_dict(surf), "props": cfg_dict(PropsCfg())}
        except Exception:  # noqa: BLE001
            meta["errors"]["cloth_setup"] = traceback.format_exc()

    vol = VolMatCfg()
    if not NEWTON:
        vol.dynamic_friction = vol.static_friction = FRICTION
    # (b) rope: volume rod (both backends) and, on Newton, the Isaac Lab cable
    if "rope" in scenes:
        for ri, res in enumerate(args.resolutions.split(",")):
            try:
                name = f"rod_n{res}"
                if res == "d":
                    refine = 4.0
                    spawn = sim_utils.MeshCuboidCfg(size=(0.5, 0.02, 0.02), deformable_props=PropsCfg(), physics_material=vol,
                                                    edge_refinement=refine)
                else:
                    refine = None
                    spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(args.tetdir, f"rod_n{res}.usda"), deformable_props=PropsCfg(),
                                                 physics_material=vol)
                rod_cfg = DeformableObjectCfg(
                    prim_path=f"/World/rope_scene/{name}", spawn=spawn,
                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(10.25, -2.0 * ri, 1.0)))
                objs[name] = rod_cfg.class_type(rod_cfg)
                meta["scenes"][name] = {"size": [0.5, 0.02, 0.02], "z0": 1.0, "x0": 10.0, "y0": -2.0 * ri,
                                        "cells_across": res, "edge_refinement": refine, "mesh": "structured hex->6 tets, cell aspect 2.5" if res != "d" else "pytetwild",
                                        "pinned": "-x end, nodes within 0.01 m", "material": cfg_dict(vol), "props": cfg_dict(PropsCfg())}
            except Exception:  # noqa: BLE001
                meta["errors"][f"rod_{res}_setup"] = traceback.format_exc()
        if NEWTON:
            try:
                from isaaclab.assets import CableObjectCfg
                nseg = 20
                cmat = sim_utils.CableMaterialCfg(thickness=0.02, density=1000.0)
                cable_cfg = CableObjectCfg(
                    prim_path="/World/rope_scene/cable",
                    spawn=sim_utils.CableCfg(positions=[(0.5 * i / nseg, 0.0, 0.0) for i in range(nseg + 1)], physics_material=cmat,
                                             collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]),
                    init_state=CableObjectCfg.InitialStateCfg(pos=(10.0, 3.0, 1.0)))
                objs["cable"] = cable_cfg.class_type(cable_cfg)
                meta["scenes"]["cable"] = {"length": 0.5, "segments": nseg, "z0": 1.0, "x0": 10.0, "y0": 3.0,
                                           "pinned": "segment 0 re-posed each step", "material": cfg_dict(cmat)}
            except Exception:  # noqa: BLE001
                meta["errors"]["cable_setup"] = traceback.format_exc()
    # (c) soft cube
    if "cube" in scenes:
        for ci, res in enumerate([r for r in args.resolutions.split(",") if r != "1"]):
            try:
                name = f"cube_n{res}"
                if res == "d":
                    refine = 4.0
                    spawn = sim_utils.MeshCuboidCfg(size=(0.2, 0.2, 0.2), deformable_props=PropsCfg(), physics_material=vol, edge_refinement=refine)
                else:
                    refine = None
                    spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(args.tetdir, f"cube_n{res}.usda"), deformable_props=PropsCfg(),
                                                 physics_material=vol)
                cube_cfg = DeformableObjectCfg(
                    prim_path=f"/World/cube_scene/{name}", spawn=spawn,
                    init_state=DeformableObjectCfg.InitialStateCfg(pos=(20.0, -2.0 * ci, 0.5)))
                objs[name] = cube_cfg.class_type(cube_cfg)
                meta["scenes"][name] = {"size": 0.2, "z0": 0.5, "x0": 20.0, "y0": -2.0 * ci, "cells_per_edge": res,
                                        "edge_refinement": refine, "material": cfg_dict(vol), "props": cfg_dict(PropsCfg())}
            except Exception:  # noqa: BLE001
                meta["errors"][f"cube_{res}_setup"] = traceback.format_exc()

    sim.reset()

    # the cooked deformable data Isaac/PhysX writes onto the prims (simulation/collision tet meshes, springs, rest
    # shapes, applied schemas and every attribute): the cooker is closed source, so the inputs are dumped as written
    def dump_prims(root_paths):
        from pxr import Usd
        try:
            stage = sim_utils.get_current_stage()
        except Exception:  # noqa: BLE001
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        attrs, arrays = {}, {}
        for root in root_paths:
            prim = stage.GetPrimAtPath(root)
            if not prim or not prim.IsValid():
                continue
            for p in Usd.PrimRange(prim):
                entry = {"type": str(p.GetTypeName()), "schemas": [str(x) for x in p.GetAppliedSchemas()], "attrs": {}}
                for a in p.GetAttributes():
                    try:
                        v = a.Get()
                    except Exception:  # noqa: BLE001
                        continue
                    if v is None:
                        continue
                    try:
                        arr = np.asarray(v)
                    except Exception:  # noqa: BLE001
                        arr = None
                    if arr is not None and arr.dtype != object and arr.size > 16:
                        key = (str(p.GetPath()) + ":" + a.GetName()).replace("/", "|")
                        arrays[key] = arr
                        entry["attrs"][a.GetName()] = f"<array {arr.shape} {arr.dtype} in cooked.npz[{key}]>"
                    else:
                        entry["attrs"][a.GetName()] = str(v)
                attrs[str(p.GetPath())] = entry
        return attrs, arrays

    try:
        cooked, cooked_arrays = dump_prims(["/World/cloth_scene", "/World/rope_scene", "/World/cube_scene"])
        json.dump(cooked, open(os.path.join(args.out, "cooked_prims.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(args.out, "cooked.npz"), **cooked_arrays)
        meta["cooked_prims"] = len(cooked)
    except Exception:  # noqa: BLE001
        meta["errors"]["cooked_dump"] = traceback.format_exc()

    def T(a):
        return (a.torch if hasattr(a, "torch") else a).detach().cpu().numpy().copy()

    # pin each rod's -x end
    for rname in [k for k in objs if k.startswith("rod")]:
        try:
            rod = objs[rname]
            p0 = T(rod.data.default_nodal_state_w)[..., :3]
            kt = rod.data.nodal_kinematic_target
            tgt = kt.torch.clone() if hasattr(kt, "torch") else kt.clone()
            tgt[..., :3] = torch.as_tensor(p0, device=tgt.device)
            tgt[..., 3] = 1.0
            pinned = p0[0, :, 0] < p0[0, :, 0].min() + 1.0e-4          # the -x end face
            tgt[0, torch.as_tensor(pinned, device=tgt.device), 3] = 0.0
            rod.write_nodal_kinematic_target_to_sim_index(tgt)
            meta["scenes"][rname]["n_pinned"] = int(pinned.sum())
            meta["scenes"][rname]["pinned_mask_key"] = rname + "_pinned"
            rec_static = globals().setdefault("_pinned", {}); rec_static[rname + "_pinned"] = pinned
        except Exception:  # noqa: BLE001
            meta["errors"][rname + "_pin"] = traceback.format_exc()
    cable_pose0 = T(objs["cable"].data.segment_pose_w) if "cable" in objs else None

    rec = {}
    def grab():
        for name, o in objs.items():
            try:
                if name == "cable":
                    rec.setdefault("cable_pose", []).append(T(o.data.segment_pose_w)[0])
                    rec.setdefault("cable_vel", []).append(T(o.data.segment_velocity_w)[0])
                else:
                    rec.setdefault(name + "_pos", []).append(T(o.data.nodal_pos_w)[0])
                    rec.setdefault(name + "_vel", []).append(T(o.data.nodal_vel_w)[0])
            except Exception:  # noqa: BLE001
                meta["errors"].setdefault(name + "_read", traceback.format_exc())

    nsteps = int(round(args.seconds / DT))
    grab()
    t0 = time.time()
    for k in range(nsteps):
        if cable_pose0 is not None:
            try:  # hold segment 0 (position and orientation) and zero its velocity
                c = objs["cable"]
                pose = c.data.segment_pose_w.torch.clone(); vel = c.data.segment_velocity_w.torch.clone()
                pose[:, 0] = torch.as_tensor(cable_pose0[:, 0], device=pose.device); vel[:, 0] = 0.0
                c.write_segment_pose_to_sim_index(segment_pose=pose); c.write_segment_velocity_to_sim_index(segment_velocity=vel)
            except Exception:  # noqa: BLE001
                meta["errors"].setdefault("cable_pin", traceback.format_exc()); cable_pose0 = None
        for o in objs.values():
            o.write_data_to_sim()
        sim.step()
        for o in objs.values():
            o.update(DT)
        grab()
    meta["wall_s_including_readback"] = time.time() - t0
    meta["counts"] = {k: list(np.array(v[0]).shape) for k, v in rec.items()}
    np.savez_compressed(os.path.join(args.out, "record.npz"), **{k: np.array(v, np.float32) for k, v in rec.items()},
                        **globals().get("_pinned", {}))
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1, default=str)
    print("DEFORMABLE_IL3_DONE", args.physics, json.dumps(meta["counts"]), "errors:", list(meta["errors"]))
