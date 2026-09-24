"""Isaac Lab side of the MetalSim fidelity protocol (runs inside Isaac Sim; Isaac Lab v2.3.x).

Records, on Isaac-Velocity-Flat-G1-v0 with resets and terminations disabled and a fixed command:
  A. PD hold: zero actions for 3 s
  B. open-loop random targets: a seeded N(0,1) action sequence (per joint name) for 5 s
  C. drop: root raised to 1.0 m, zero actions, 3 s
per control step: joint pos/vel (Isaac joint order + names), root pose/velocity, applied torques,
net contact forces per body; and camera frames (RGB + distance-to-image-plane) from a camera at a
fixed pose relative to the env origin, every `--frame_every` steps, under the requested render mode.
Everything is written to --out as .npz / .png so the same protocol can be replayed on MetalSim.

    python record_g1.py --headless --out ~/parity_out/rt --render rt
    python record_g1.py --headless --out ~/parity_out/pt --render pt --frame_every 25
"""
import argparse, os, json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--render", default="rt", choices=["rt", "pt"])
parser.add_argument("--frame_every", type=int, default=5)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np, torch, gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-v0", device="cuda:0", num_envs=args.num_envs)
# deterministic protocol: no reset randomization, no terminations, long episodes, fixed command
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.terminations.base_contact = None
cfg.episode_length_s = 1000.0
cmd = cfg.commands.base_velocity
cmd.heading_command = False; cmd.rel_standing_envs = 0.0; cmd.rel_heading_envs = 0.0
cmd.ranges.lin_vel_x = (0.5, 0.5); cmd.ranges.lin_vel_y = (0.0, 0.0); cmd.ranges.ang_vel_z = (0.0, 0.0)
cfg.observations.policy.enable_corruption = False
cfg.scene.env_spacing = 6.0
# lighting that MetalSim can reproduce: one sun + a uniform sky (the default is an HDR dome)
cfg.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=400.0, color=(0.75, 0.8, 0.9)))
cfg.scene.sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95), angle=0.53),
                             init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9063, 0.0, 0.4226, 0.0)))  # tilt about y: sun from +x/-z ... recorded below
# camera at MetalSim's hero pose relative to the env origin, OpenGL convention (MuJoCo's camera frame)
import math
def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w); y = (R[0, 2] - R[2, 0]) / (4 * w); z = (R[1, 0] - R[0, 1]) / (4 * w)
    return (w, x, y, z)
CAM_POS, CAM_TARGET, VFOV = (3.2, -2.4, 1.4), (0.3, 0.0, 0.6), 42.0
W, H = 1024, 576
hfov = 2 * math.degrees(math.atan(math.tan(math.radians(VFOV / 2)) * W / H))
focal = 20.955 / 2 / math.tan(math.radians(hfov / 2))
cfg.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", update_period=0.0, width=W, height=H,
                             data_types=["rgb", "distance_to_image_plane"],
                             spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
                             offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=lookat_quat(CAM_POS, CAM_TARGET), convention="opengl"))
if args.render == "pt":
    cfg.sim.render.carb_settings = {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 64, "/rtx/pathtracing/totalSpp": 64, "/rtx/pathtracing/clampSpp": 64}
else:
    cfg.sim.render.rendering_mode = "quality"

env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=cfg)
robot = env.unwrapped.scene["robot"]; contacts = env.unwrapped.scene["contact_forces"]; camera = env.unwrapped.scene["camera"]
joint_names = list(robot.joint_names); body_names = list(robot.body_names); contact_bodies = list(contacts.body_names)
nj = len(joint_names)
meta = {"joint_names": joint_names, "body_names": body_names, "contact_bodies": contact_bodies, "control_dt": float(env.unwrapped.step_dt),
        "physics_dt": float(env.unwrapped.physics_dt), "camera": {"pos": CAM_POS, "target": CAM_TARGET, "vfov_deg": VFOV, "width": W, "height": H,
        "quat_wxyz": list(map(float, lookat_quat(CAM_POS, CAM_TARGET)))}, "render": args.render,
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "stiffness": {n: float(v) for n, v in zip(joint_names, robot.data.joint_stiffness[0].cpu().tolist())},
        "damping": {n: float(v) for n, v in zip(joint_names, robot.data.joint_damping[0].cpu().tolist())},
        "lights": {"sun": {"intensity": 3000.0, "color": [1.0, 0.98, 0.95], "rot_wxyz": [0.9063, 0.0, 0.4226, 0.0]}, "dome": {"intensity": 400.0, "color": [0.75, 0.8, 0.9]}}}
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
rng = np.random.default_rng(args.seed)
protocols = {"A_hold": (150, lambda t: np.zeros((args.num_envs, nj), np.float32), None),
             "B_random": (250, None, None),
             "C_drop": (150, lambda t: np.zeros((args.num_envs, nj), np.float32), 1.0)}
seq_B = rng.normal(size=(250, nj)).astype(np.float32)          # same sequence for every env, indexed by joint NAME order below
np.savez(os.path.join(args.out, "action_sequence_B.npz"), actions=seq_B, joint_names=np.array(joint_names))
protocols["B_random"] = (250, lambda t: np.repeat(seq_B[t][None], args.num_envs, 0), None)

def save_img(tag, t):
    rgb = camera.data.output["rgb"][0].cpu().numpy()[..., :3]
    depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
    import imageio.v2 as iio
    iio.imwrite(os.path.join(args.out, f"{tag}_{t:04d}_rgb.png"), rgb.astype(np.uint8))
    np.save(os.path.join(args.out, f"{tag}_{t:04d}_depth.npy"), depth.astype(np.float32))

for tag, (T, act_fn, drop_z) in protocols.items():
    env.reset(seed=args.seed)
    if drop_z is not None:
        root = robot.data.default_root_state.clone(); root[:, :3] += env.unwrapped.scene.env_origins; root[:, 2] = drop_z
        robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:])
        env.unwrapped.sim.step(render=False); env.unwrapped.scene.update(env.unwrapped.physics_dt)
    rec = {k: [] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_b", "root_ang_vel_b", "torque", "contact")}
    for t in range(T):
        a = torch.as_tensor(act_fn(t), device="cuda:0")
        env.step(a)
        rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy()); rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
        rec["root_pos"].append((robot.data.root_pos_w - env.unwrapped.scene.env_origins).cpu().numpy()); rec["root_quat"].append(robot.data.root_quat_w.cpu().numpy())
        rec["root_lin_vel_b"].append(robot.data.root_lin_vel_b.cpu().numpy()); rec["root_ang_vel_b"].append(robot.data.root_ang_vel_b.cpu().numpy())
        rec["torque"].append(robot.data.applied_torque.cpu().numpy()); rec["contact"].append(contacts.data.net_forces_w.cpu().numpy())
        if t % args.frame_every == 0:
            save_img(tag, t)
    np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), **{k: np.stack(v) for k, v in rec.items()})
    print(f"[record] {tag}: {T} steps, final root z {rec['root_pos'][-1][:, 2]}", flush=True)
open(os.path.join(args.out, "DONE"), "w").write("ok")
env.close()
simulation_app.close()
