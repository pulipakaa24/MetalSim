"""Isaac Lab 3.0-EA / Isaac Sim 6.1 port of isaac_side/record_penetration.py: PhysX per-contact separation after every
physics step of the G1 C_drop protocol, via omni.physics.tensors RigidContactView.get_contact_data(dt) (forces,
points, normals, separations, counts, start indices) on the PhysX backend (physics=isaacsim_physx).
Same protocol as the 2.3.2 script (push_robot / add_base_mass / reset velocity range disabled, as in record_g1.py).

    python record_penetration.py --out ~/parity3/penetration_il3 --viz none physics=isaacsim_physx
"""
import argparse, os, sys, json
import warp as wp
wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_per_prim", type=int, default=16)
add_launcher_args(parser)
from isaaclab_tasks.utils import setup_preset_cli, resolve_task_config  # noqa: E402
args, remaining = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + remaining
import numpy as np, torch, gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402

os.makedirs(args.out, exist_ok=True)
cfg, _ = resolve_task_config(args.task, "")
cfg.scene.num_envs = 1; cfg.seed = args.seed
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.events.push_robot = None; cfg.events.add_base_mass = None
cfg.terminations.base_contact = None
cfg.episode_length_s = 1000.0
cmd = cfg.commands.base_velocity
cmd.heading_command = False; cmd.rel_standing_envs = 0.0; cmd.rel_heading_envs = 0.0
cmd.ranges.lin_vel_x = (0.5, 0.5); cmd.ranges.lin_vel_y = (0.0, 0.0); cmd.ranges.ang_vel_z = (0.0, 0.0); cmd.debug_vis = False
cfg.observations.policy.enable_corruption = False


def T(x):
    if hasattr(x, "torch"): x = x.torch
    if hasattr(x, "detach"): return x.detach().cpu().numpy()
    if hasattr(x, "numpy"): return x.numpy()
    return np.asarray(x)


with launch_simulation(cfg, args) as physics_cfg:
    print(f"[pen] physics {type(cfg.sim.physics).__name__}", flush=True)
    env = gym.make(args.task, cfg=cfg)
    E = env.unwrapped; robot = E.scene["robot"]
    from pxr import Usd, UsdPhysics, PhysxSchema
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_physx.physics import PhysxManager
    stage = get_current_stage()
    sensor_paths = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
                    if p.HasAPI(PhysxSchema.PhysxContactReportAPI) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    ground = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath("/World/ground"), Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    print("[pen] sensors", len(sensor_paths), "ground", ground, flush=True)
    view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(sensor_paths, filter_patterns=[ground] * len(sensor_paths),
                                                                          max_contact_data_count=args.max_per_prim * len(sensor_paths))
    body_names = [p.rsplit("/", 1)[-1] for p in sensor_paths]
    dt = float(E.physics_dt); dec = int(E.cfg.decimation)
    rows = []; root_z = []
    def read(k):
        f, pts, nrm, sep, cnt, start = [T(x) for x in view.get_contact_data(dt)]
        f = f.reshape(-1); pts = pts.reshape(-1, 3); nrm = nrm.reshape(-1, 3); sep = sep.reshape(-1)
        cnt = cnt.reshape(view.sensor_count, -1); start = start.reshape(view.sensor_count, -1)
        for s in range(cnt.shape[0]):
            for fi in range(cnt.shape[1]):
                for j in range(int(start[s, fi]), int(start[s, fi] + cnt[s, fi])):
                    rows.append((k, s, sep[j], f[j], *pts[j], *nrm[j]))
    env.reset(seed=args.seed)
    pose = T(robot.data.default_root_pose).copy(); pose[:, :3] += T(E.scene.env_origins); pose[:, 2] = 1.0
    robot.write_root_pose_to_sim_index(root_pose=torch.as_tensor(pose, device=E.device))
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=E.device))
    k = 0
    def phys():
        global k
        E.sim.step(render=False); E.scene.update(dt); k += 1; read(k)
        root_z.append(float(T(robot.data.root_pos_w)[0, 2] - T(E.scene.env_origins)[0, 2]))
    phys()
    a = torch.zeros((1, robot.num_joints), device=E.device)
    with torch.inference_mode():
        for t in range(150):
            E.action_manager.process_action(a)
            for _ in range(dec):
                E.action_manager.apply_action(); E.scene.write_data_to_sim(); phys()
    r = np.array(rows, np.float64).reshape(-1, 10)
    np.savez_compressed(os.path.join(args.out, "C_drop_penetration.npz"), tensor_step=r[:, 0].astype(np.int32), tensor_body=r[:, 1].astype(np.int32),
                        tensor_separation=r[:, 2], tensor_normal_force=r[:, 3], tensor_point=r[:, 4:7], tensor_normal=r[:, 7:10],
                        report_step=np.zeros(0, np.int32), report_body=np.zeros(0, np.int32), report_separation=np.zeros(0), report_impulse=np.zeros(0), report_point=np.zeros((0, 3)),
                        root_z=np.array(root_z), body_names=np.array(body_names))
    mins = {}
    for row in r: mins[int(row[0])] = min(mins.get(int(row[0]), np.inf), row[2])
    ks = np.array(sorted(mins)); mv = np.array([mins[x] for x in ks]); i = int(np.argmin(mv))
    meta = {"isaaclab": "3.0.0-EA", "physics_dt": dt, "decimation": dec, "n_phys_steps": k, "sensor_paths": sensor_paths, "ground_colliders": ground,
            "tensor_rows": int(len(r)), "report_rows": 0, "collider_offsets": {}, "final_root_z_ctrl": root_z[-1],
            "summary_tensor_all": {"min_separation_m": float(mv[i]), "at_phys_step": int(ks[i]), "at_t_s": float(ks[i] * dt), "first_contact_step": int(ks[0]),
                                   "settle_mean_of_step_minima_last_0p5s_m": float(mv[ks >= k - 100].mean())}}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print("[pen] summary", json.dumps(meta["summary_tensor_all"]), "final root z", root_z[-1], flush=True)
    open(os.path.join(args.out, "DONE"), "w").write("ok")
    sys.stdout.flush(); os._exit(0)
