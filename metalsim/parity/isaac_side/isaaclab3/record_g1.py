"""Isaac Lab 3.0 (v3.0.0-EA, Isaac Sim 6.1) port of `isaac_side/record_g1.py`: the MetalSim fidelity protocol
on Isaac-Velocity-Flat-G1, on either physics backend (PhysX or Newton/MuJoCo Warp), with RTX frames.

Same protocol as the 2.3.2 recorder (resets and terminations disabled, fixed command (0.5, 0, 0), one env):
  A_hold  : zero actions, 150 control steps (3 s)
  B_random: seeded N(0,1) action sequence per joint (rng seed 0, same draw order as 2.3.2), 250 steps (5 s)
  C_drop  : root raised to 1.0 m, zero actions, 150 steps
per control step: joint pos/vel, root pose (quaternion written as WXYZ, converted from 3.0's XYZW so files
compare directly with the 2.3.2 recordings), root velocity (body frame), applied torques, contact forces
(total and normal where the backend provides them); RTX RGB + depth every --frame_every steps.

3.0 G1 task differences neutralised here so the protocol stays deterministic and equal to the 2.3.2 one:
push_robot, add_base_mass (now active on torso_link in 3.0) and the ±0.5 reset velocity range are disabled.

On Newton the exact MuJoCo Warp settings actually used (mjw_model.opt, per-geom solref/solimp/friction/margin,
njmax/nconmax, substeps, collision pipeline) are written to meta.json, the generated MJCF is saved, and the
solver's per-substep iteration counts (mjw_data.solver_niter) are recorded per control step.

    python record_g1.py --out ~/parity3/fidelity/physx  physics=isaacsim_physx
    python record_g1.py --out ~/parity3/fidelity/newton physics=newton_mjwarp
"""
import argparse, os, sys, json, math, contextlib

import warp as wp
wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
parser.add_argument("--frame_every", type=int, default=5)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_camera", action="store_true")
add_launcher_args(parser)
from isaaclab_tasks.utils import setup_preset_cli, resolve_task_config  # noqa: E402
args, remaining = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + remaining

import numpy as np, torch, gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402

os.makedirs(args.out, exist_ok=True)
cfg, _ = resolve_task_config(args.task, "")
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
# deterministic protocol (as 2.3.2): no reset randomization, no terminations, long episodes, fixed command
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.events.push_robot = None          # active in 3.0's G1 task, removed in 2.3.2's
cfg.events.add_base_mass = None       # active in 3.0's G1 task (torso x[0.8, 1.25]), removed in 2.3.2's
cfg.terminations.base_contact = None
cfg.episode_length_s = 1000.0
cmd = cfg.commands.base_velocity
cmd.heading_command = False; cmd.rel_standing_envs = 0.0; cmd.rel_heading_envs = 0.0
cmd.ranges.lin_vel_x = (0.5, 0.5); cmd.ranges.lin_vel_y = (0.0, 0.0); cmd.ranges.ang_vel_z = (0.0, 0.0)
cmd.debug_vis = False
cfg.observations.policy.enable_corruption = False
cfg.scene.env_spacing = 20.0

CAM_POS, CAM_TARGET, VFOV, W, H = (3.2, -2.4, 1.4), (0.3, 0.0, 0.6), 42.0, 1024, 576


def lookat_quat_wxyz(pos, target):
    f = np.array(target, float) - np.array(pos, float); f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def wxyz_to_xyzw(q):
    return (q[1], q[2], q[3], q[0])


SUN_ROT_WXYZ = (0.9063, 0.0, 0.4226, 0.0)
if not args.no_camera:
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.7)
    cfg.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=400.0, color=(0.75, 0.8, 0.9)))
    cfg.scene.sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95), angle=0.53),
                                 init_state=AssetBaseCfg.InitialStateCfg(rot=wxyz_to_xyzw(SUN_ROT_WXYZ)))
    hfov = 2 * math.degrees(math.atan(math.tan(math.radians(VFOV / 2)) * W / H))
    focal = 20.955 / 2 / math.tan(math.radians(hfov / 2))
    cfg.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", update_period=0.0, width=W, height=H,
                                 data_types=["rgb", "distance_to_image_plane"],
                                 spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
                                 offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=wxyz_to_xyzw(lookat_quat_wxyz(CAM_POS, CAM_TARGET)), convention="opengl"),
                                 renderer_cfg=IsaacRtxRendererCfg())


def grey_ground(stage_path="/World/ground"):
    """Bind a plain grey PreviewSurface on every ground mesh, stronger than descendants (the plane terrain ignores
    `visual_material`; same fix as the 2.3.2 recorder)."""
    from pxr import Usd
    from isaaclab.sim.utils.stage import get_current_stage
    stage = get_current_stage()
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.7)
    from isaaclab.sim.spawners.materials.visual_materials import spawn_preview_surface
    spawn_preview_surface("/World/Looks/ground_grey", mat)
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(stage_path), Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() in ("Mesh", "Plane") and not prim.IsInstanceProxy():
            sim_utils.bind_visual_material(prim.GetPath().pathString, "/World/Looks/ground_grey", stronger_than_descendants=True); n += 1
    sim_utils.bind_visual_material(stage_path, "/World/Looks/ground_grey", stronger_than_descendants=True)
    print(f"[scene] grey ground bound on {stage_path} (+{n} meshes)", flush=True)


def T(x):
    """ProxyArray / tensor -> numpy."""
    if x is None:
        return None
    if hasattr(x, "torch"):
        x = x.torch
    return x.detach().cpu().numpy()


def newton_settings(out_dir):
    """Exact MuJoCo Warp settings of the running Newton model (None on PhysX)."""
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception:
        return None
    solver = getattr(NewtonManager, "_solver", None)
    if solver is None or not hasattr(solver, "mjw_model"):
        return None
    m = solver.mjw_model
    opt = {}
    for k in dir(m.opt):
        if k.startswith("_"):
            continue
        try:
            v = getattr(m.opt, k)   # some attributes are removed-API stubs that raise (e.g. ls_parallel)
            if callable(v):
                continue
            a = v.numpy() if hasattr(v, "numpy") else v
            a = np.asarray(a)
            opt[k] = a.tolist() if a.size <= 16 else f"array{a.shape}"
        except Exception:
            opt[k] = repr(v)[:200]
    geom = {}
    for k in ("geom_solref", "geom_solimp", "geom_friction", "geom_margin", "geom_gap", "geom_condim", "geom_priority", "geom_solmix", "geom_type"):
        if hasattr(m, k):
            try:
                a = getattr(m, k).numpy()
                a = a.reshape(-1, *a.shape[2:]) if a.ndim >= 2 and a.shape[0] == 1 else a
                uniq = np.unique(np.round(a.reshape(a.shape[0] if a.ndim > 1 else -1, -1), 8), axis=0)
                geom[k] = {"shape": list(a.shape), "unique_rows": uniq.tolist()[:20]}
            except Exception as e:
                geom[k] = repr(e)
    other = {}
    for k in ("jnt_solref", "jnt_solimp", "dof_armature", "dof_damping", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "jnt_range", "jnt_margin"):
        if hasattr(m, k):
            try:
                a = getattr(m, k).numpy()
                other[k] = {"shape": list(a.shape), "first_world": np.asarray(a[0] if a.ndim >= 2 and a.shape[0] == 1 else a).round(6).tolist()}
            except Exception as e:
                other[k] = repr(e)
    sizes = {k: int(getattr(m, k)) for k in ("nq", "nv", "nu", "nbody", "ngeom", "njnt") if hasattr(m, k)}
    d = solver.mjw_data
    for k in ("njmax", "nconmax", "naconmax", "nworld"):
        if hasattr(d, k):
            try:
                sizes["data_" + k] = int(getattr(d, k))
            except Exception:
                pass
    pcfg = cfg.sim.physics
    ncfg = {k: getattr(pcfg, k) for k in ("num_substeps", "collision_decimation", "use_cuda_graph", "debug_mode", "deterministic_mode") if hasattr(pcfg, k)}
    ncfg["solver_cfg"] = {k: getattr(pcfg.solver_cfg, k) for k in pcfg.solver_cfg.__dataclass_fields__ if k != "class_type"} if hasattr(pcfg.solver_cfg, "__dataclass_fields__") else repr(pcfg.solver_cfg)
    ncfg["default_shape_cfg"] = {k: getattr(pcfg.default_shape_cfg, k) for k in pcfg.default_shape_cfg.__dataclass_fields__} if hasattr(pcfg.default_shape_cfg, "__dataclass_fields__") else repr(pcfg.default_shape_cfg)
    ncfg["collision_cfg"] = ({k: repr(getattr(pcfg.collision_cfg, k)) for k in pcfg.collision_cfg.__dataclass_fields__} if pcfg.collision_cfg is not None and hasattr(pcfg.collision_cfg, "__dataclass_fields__") else repr(pcfg.collision_cfg))
    try:
        import newton, mujoco, mujoco_warp
        ver = {"newton": newton.__version__, "mujoco": mujoco.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", "?"), "warp": wp.__version__}
    except Exception as e:
        ver = repr(e)
    # Newton model contact parameters as authored (before the solref conversion)
    nm = {}
    try:
        model = NewtonManager._model
        for k in ("shape_material_ke", "shape_material_kd", "shape_material_mu", "shape_margin", "shape_gap", "rigid_contact_max"):
            if hasattr(model, k):
                v = getattr(model, k)
                a = np.asarray(v.numpy() if hasattr(v, "numpy") else v)
                nm[k] = np.unique(a.round(8)).tolist()[:10] if a.size > 1 else a.tolist()
    except Exception as e:
        nm["error"] = repr(e)
    return {"versions": ver, "isaaclab_newton_cfg": json.loads(json.dumps(ncfg, default=repr)), "mjw_model_opt": opt, "mjw_model_geom": geom,
            "mjw_model_joint_actuator": other, "sizes": sizes, "newton_model_contact": nm,
            "solver_timestep_s": float(cfg.sim.dt) / float(getattr(pcfg, "num_substeps", 1))}


with launch_simulation(cfg, args) as physics_cfg:
    # the physics= preset is resolved by Hydra (resolve_task_config) or by launch_simulation; check both
    print(f"[record] cfg.sim.physics={type(cfg.sim.physics).__name__} resolved={type(physics_cfg).__name__}", flush=True)
    is_newton = "Newton" in type(cfg.sim.physics).__name__
    if is_newton and hasattr(cfg.sim.physics, "solver_cfg"):
        cfg.sim.physics.solver_cfg.save_to_mjcf = os.path.join(args.out, "newton_generated.xml")
    env = gym.make(args.task, cfg=cfg)
    if not args.no_camera:
        try:
            grey_ground()
            for _ in range(2):
                env.unwrapped.sim.render()
        except Exception as e:  # never lose a recording over the ground colour
            print(f"[scene] grey ground failed: {e!r}", flush=True)
    u = env.unwrapped
    robot = u.scene["robot"]; contacts = u.scene["contact_forces"]
    camera = u.scene["camera"] if not args.no_camera else None
    joint_names = list(robot.joint_names); body_names = list(robot.body_names); contact_bodies = list(contacts.body_names)
    nj = len(joint_names)
    meta = {"isaaclab": "3.0.0-EA", "task": args.task, "physics_cfg": type(cfg.sim.physics).__name__, "joint_names": joint_names, "body_names": body_names,
            "contact_bodies": contact_bodies, "control_dt": float(u.step_dt), "physics_dt": float(u.physics_dt), "decimation": int(cfg.decimation),
            "quaternion_convention_in_files": "wxyz (converted from Isaac Lab 3.0 xyzw)",
            "camera": None if args.no_camera else {"pos": CAM_POS, "target": CAM_TARGET, "vfov_deg": VFOV, "width": W, "height": H,
                                                  "quat_wxyz": list(map(float, lookat_quat_wxyz(CAM_POS, CAM_TARGET)))},
            "render": "rt",
            "default_joint_pos": T(robot.data.default_joint_pos)[0].tolist(),
            "stiffness": {n: float(v) for n, v in zip(joint_names, T(robot.data.joint_stiffness)[0].tolist())},
            "damping": {n: float(v) for n, v in zip(joint_names, T(robot.data.joint_damping)[0].tolist())},
            "lights": {"sun": {"intensity": 3000.0, "color": [1.0, 0.98, 0.95], "rot_wxyz": list(SUN_ROT_WXYZ)}, "dome": {"intensity": 400.0, "color": [0.75, 0.8, 0.9]}},
            "events_disabled": ["push_robot", "add_base_mass", "reset_base velocity range", "terminations.base_contact"]}
    if is_newton:
        try:
            meta["newton"] = newton_settings(args.out)
        except Exception as e:  # never lose the recording over the settings dump
            import traceback; traceback.print_exc(); meta["newton"] = {"error": repr(e)}
    else:
        p = cfg.sim.physics
        meta["physx_cfg"] = {k: repr(getattr(p, k)) for k in getattr(p, "__dataclass_fields__", {})}
        meta["physx_articulation"] = {"solver_position_iteration_count": 8, "solver_velocity_iteration_count": 4, "source": "G1_CFG articulation_props"}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1, default=repr)
    print("[record] meta written", flush=True)

    rng = np.random.default_rng(args.seed)
    seq_B = rng.normal(size=(250, nj)).astype(np.float32)
    np.savez(os.path.join(args.out, "action_sequence_B.npz"), actions=seq_B, joint_names=np.array(joint_names))
    zeros = lambda t: np.zeros((args.num_envs, nj), np.float32)  # noqa: E731
    protocols = {"A_hold": (150, zeros, None), "B_random": (250, lambda t: np.repeat(seq_B[t][None], args.num_envs, 0), None), "C_drop": (150, zeros, 1.0)}

    def save_img(tag, t):
        import imageio.v2 as iio
        rgb = T(camera.data.output["rgb"])[0][..., :3]
        depth = T(camera.data.output["distance_to_image_plane"])[0].squeeze()
        iio.imwrite(os.path.join(args.out, f"{tag}_{t:04d}_rgb.png"), rgb.astype(np.uint8))
        np.save(os.path.join(args.out, f"{tag}_{t:04d}_depth.npy"), depth.astype(np.float32))

    def niter():
        try:
            from isaaclab_newton.physics import NewtonManager
            return NewtonManager._solver.mjw_data.solver_niter.numpy().copy()
        except Exception:
            return None

    for tag, (Tn, act_fn, drop_z) in protocols.items():
        env.reset(seed=args.seed)
        if drop_z is not None:
            pose = T(robot.data.default_root_pose).copy(); pose[:, :3] += T(u.scene.env_origins); pose[:, 2] = drop_z
            vel = np.zeros((args.num_envs, 6), np.float32)
            robot.write_root_pose_to_sim_index(root_pose=torch.as_tensor(pose, device=u.device))
            robot.write_root_velocity_to_sim_index(root_velocity=torch.as_tensor(vel, device=u.device))
            u.sim.step(render=False); u.scene.update(u.physics_dt)
        keys = ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_b", "root_ang_vel_b", "torque", "contact", "contact_normal", "solver_niter")
        rec = {k: [] for k in keys}
        for t in range(Tn):
            with torch.inference_mode():
                env.step(torch.as_tensor(act_fn(t), device=u.device))
            q = T(robot.data.root_quat_w)                       # xyzw
            rec["joint_pos"].append(T(robot.data.joint_pos)); rec["joint_vel"].append(T(robot.data.joint_vel))
            rec["root_pos"].append(T(robot.data.root_pos_w) - T(u.scene.env_origins)); rec["root_quat"].append(q[:, [3, 0, 1, 2]])
            rec["root_lin_vel_b"].append(T(robot.data.root_lin_vel_b)); rec["root_ang_vel_b"].append(T(robot.data.root_ang_vel_b))
            rec["torque"].append(T(robot.data.applied_torque))
            with contextlib.suppress(Exception):
                rec["contact"].append(T(contacts.data.net_forces_w))
            with contextlib.suppress(Exception):
                rec["contact_normal"].append(T(contacts.data.net_normal_forces_w))
            n = niter() if is_newton else None
            if n is not None:
                rec["solver_niter"].append(n)
            if camera is not None and t % args.frame_every == 0:
                save_img(tag, t)
        np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), **{k: np.stack(v) for k, v in rec.items() if len(v)})
        extra = f", solver_niter mean {np.mean(rec['solver_niter']):.1f} max {np.max(rec['solver_niter'])}" if rec["solver_niter"] else ""
        print(f"[record] {tag}: {Tn} steps, final root z {rec['root_pos'][-1][:, 2]}{extra}", flush=True)
    open(os.path.join(args.out, "DONE"), "w").write("ok")
    env.close()
