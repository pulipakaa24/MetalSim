"""Isaac side: roll a policy in Isaac-Velocity-Flat-G1-v0 from the deterministic protocol state and
record camera frames + trajectory, so the same policy and initial state can be compared against the
MetalSim rollout side by side. Accepts rsl_rl checkpoints (model_*.pt, Isaac's own training) or a
MetalSim export ({"actor": state_dict of Sequential Linear/ELU, "joint_names": [...]}).

    python play_policy.py --headless --ckpt logs/rsl_rl/g1_flat/<run>/model_500.pt --out ~/parity_out/play/isaac_it500 --render rt
"""
import argparse, os, json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True); parser.add_argument("--out", required=True)
parser.add_argument("--render", default="rt", choices=["rt", "pt"]); parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--frame_every", type=int, default=2); parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.enable_cameras = True
app_launcher = AppLauncher(args); simulation_app = app_launcher.app

import math, numpy as np, torch, torch.nn as nn, gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import imageio.v2 as iio

os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-v0", device="cuda:0", num_envs=args.num_envs)
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.terminations.base_contact = None; cfg.episode_length_s = 1000.0
cmd = cfg.commands.base_velocity; cmd.heading_command = False; cmd.rel_standing_envs = 0.0; cmd.rel_heading_envs = 0.0
cmd.ranges.lin_vel_x = (0.5, 0.5); cmd.ranges.lin_vel_y = (0.0, 0.0); cmd.ranges.ang_vel_z = (0.0, 0.0)
cfg.observations.policy.enable_corruption = False
cfg.scene.env_spacing = 20.0
cmd.debug_vis = False                                     # no command arrow in the frames
if hasattr(cfg.scene, "height_scanner") and cfg.scene.height_scanner is not None: cfg.scene.height_scanner.debug_vis = False
cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.7)   # same plain grey ground as MetalSim's replay
cfg.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=400.0, color=(0.75, 0.8, 0.9)))
cfg.scene.sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95), angle=0.53),
                             init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9063, 0.0, 0.4226, 0.0)))
def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))
CAM_POS, CAM_TARGET, VFOV, W, H = (3.2, -2.4, 1.4), (0.3, 0.0, 0.6), 42.0, 1024, 576
hfov = 2 * math.degrees(math.atan(math.tan(math.radians(VFOV / 2)) * W / H)); focal = 20.955 / 2 / math.tan(math.radians(hfov / 2))
cfg.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", update_period=0.0, width=W, height=H, data_types=["rgb"],
                             spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
                             offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=lookat_quat(CAM_POS, CAM_TARGET), convention="opengl"))
if args.render == "pt":
    cfg.sim.render.carb_settings = {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 32, "/rtx/pathtracing/totalSpp": 32}
else:
    cfg.sim.render.rendering_mode = "quality"
env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=cfg)
robot = env.unwrapped.scene["robot"]; camera = env.unwrapped.scene["camera"]
joint_names = list(robot.joint_names); nj = len(joint_names)

# policy: rsl_rl checkpoint (actor.* keys, Isaac joint order) or MetalSim export (actor state dict + its joint order)
ck = torch.load(args.ckpt, map_location="cuda:0", weights_only=False)
def build_actor(sd, prefix):
    keys = sorted({k[len(prefix):].split(".")[0] for k in sd if k.startswith(prefix)}, key=int)   # "actor.0.weight" or "0.weight"
    layers = []
    for i, k in enumerate(keys):
        w = sd[f"{prefix}{k}.weight"]; layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(keys) - 1: layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    net.load_state_dict({f"{2 * i}.weight": sd[f"{prefix}{k}.weight"] for i, k in enumerate(keys)} | {f"{2 * i}.bias": sd[f"{prefix}{k}.bias"] for i, k in enumerate(keys)})
    return net.cuda().eval()
if "model_state_dict" in ck:                      # rsl_rl
    actor = build_actor(ck["model_state_dict"], "actor."); perm_obs = None; perm_act = None; src = "rsl_rl"
else:                                             # MetalSim export
    actor = build_actor(ck["actor"], ""); ms_joints = ck["joint_names"]
    ms_of_isaac = [ms_joints.index(n) for n in joint_names]              # MetalSim index for each Isaac joint
    perm_act = torch.tensor(ms_of_isaac, device="cuda:0")               # a_isaac[j] = a_ms[ms_of_isaac[j]]
    isaac_of_ms = [joint_names.index(n) for n in ms_joints]
    perm_obs = torch.tensor(isaac_of_ms, device="cuda:0"); src = "metalsim"
print(f"[play] {src} policy from {args.ckpt}", flush=True)

obs, _ = env.reset(seed=0)
rec = {"joint_pos": [], "root_pos": [], "root_quat": []}
frames = []
for t in range(args.steps):
    o = obs["policy"]
    if perm_obs is not None:      # reorder joint blocks (pos, vel, last action) from Isaac's joint order into the MetalSim policy's order
        head = o[:, :12]; jp = o[:, 12:12 + nj][:, perm_obs]; jv = o[:, 12 + nj:12 + 2 * nj][:, perm_obs]; la = o[:, 12 + 2 * nj:12 + 3 * nj][:, perm_obs]
        o = torch.cat([head, jp, jv, la], 1)
    with torch.no_grad():
        a = actor(o)
    if perm_act is not None:
        a = a[:, perm_act]
    obs, *_ = env.step(a)
    rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy()); rec["root_pos"].append((robot.data.root_pos_w - env.unwrapped.scene.env_origins).cpu().numpy()); rec["root_quat"].append(robot.data.root_quat_w.cpu().numpy())
    if t % args.frame_every == 0:
        frames.append(camera.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8))
np.savez_compressed(os.path.join(args.out, "traj.npz"), joint_names=np.array(joint_names), **{k: np.stack(v) for k, v in rec.items()})
iio.mimwrite(os.path.join(args.out, "isaac.mp4"), frames, fps=int(50 / args.frame_every), codec="libx264", quality=8, macro_block_size=None)
json.dump({"ckpt": args.ckpt, "source": src, "steps": args.steps, "frame_every": args.frame_every, "final_root_z": rec["root_pos"][-1][:, 2].tolist()}, open(os.path.join(args.out, "meta.json"), "w"))
print(f"[play] done: final root z {rec['root_pos'][-1][:, 2]}", flush=True)
env.close(); simulation_app.close()
