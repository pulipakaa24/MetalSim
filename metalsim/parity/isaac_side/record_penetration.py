"""PhysX per-contact penetration during the G1 1 m drop (protocol C_drop of record_g1.py), Isaac Sim 5.1 + Isaac Lab 2.3.2.

Same scene and protocol as record_g1.py C_drop (Isaac-Velocity-Flat-G1-v0, no reset randomization, zero actions,
root placed at z = 1.0 m, 150 control steps x decimation 4 = 600 physics steps at 5 ms), but the control step is
unrolled so that contacts are read after EVERY physics step, from two independent PhysX outputs:

  1. omni.physics.tensors RigidContactView.get_contact_data(dt) -> (forces, points, normals, separations, counts,
     start_indices), "the detailed contact data between sensors and filter prims per patch ... contact forces,
     contact points, contact normals, and separation distances" (omni.physics.tensors 107.3 impl/api.py). Sensors =
     every robot body carrying PhysxContactReportAPI (the Isaac Lab ContactSensor's body set), filter = every
     collider under /World/ground.  Isaac Lab's ContactSensor calls the same function but only unpacks the points.
  2. omni.physx get_physx_simulation_interface().get_full_contact_report() -> (headers, contact data) with
     ContactData.separation, .impulse, .position, .normal per point ("Get contact report data for current simulation
     step directly", omni.physx 107.3 bindings/_physx.pyi).  May be empty under the GPU pipeline; recorded if not.

PhysX convention: separation < 0 is penetration; contacts are generated once shapes are within contactOffset (so
positive separations appear before touchdown) and the solver drives separation toward restOffset. Both offsets of
the foot colliders are read from the stage and written to meta.

    python record_penetration.py --headless --out ~/parity3/penetration_il2 [--device cuda:0]
"""
import argparse, os, json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_per_prim", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

dev = args.device if args.device else "cuda:0"
os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-v0", device=dev, num_envs=1)
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.terminations.base_contact = None
cfg.episode_length_s = 1000.0
cmd = cfg.commands.base_velocity
cmd.heading_command = False; cmd.rel_standing_envs = 0.0; cmd.rel_heading_envs = 0.0
cmd.ranges.lin_vel_x = (0.5, 0.5); cmd.ranges.lin_vel_y = (0.0, 0.0); cmd.ranges.ang_vel_z = (0.0, 0.0)
cmd.debug_vis = False
cfg.observations.policy.enable_corruption = False
env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=cfg)
E = env.unwrapped
robot = E.scene["robot"]; contacts = E.scene["contact_forces"]

import omni.usd
from pxr import Usd, UsdPhysics, PhysxSchema, PhysicsSchemaTools
from isaacsim.core.simulation_manager import SimulationManager
stage = omni.usd.get_context().get_stage()
# sensors: robot bodies with the contact report API; filters: colliders under /World/ground
sensor_paths = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
                if p.HasAPI(PhysxSchema.PhysxContactReportAPI) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
ground_colliders = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath("/World/ground"), Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.CollisionAPI)]
print("[pen] sensors", len(sensor_paths), "ground colliders", ground_colliders, flush=True)
offsets = {}
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"), Usd.TraverseInstanceProxies()):
    if p.HasAPI(UsdPhysics.CollisionAPI) and "ankle_roll" in p.GetPath().pathString:
        a = PhysxSchema.PhysxCollisionAPI(p)
        ro = a.GetRestOffsetAttr().Get() if a else None; co = a.GetContactOffsetAttr().Get() if a else None
        offsets[p.GetPath().pathString] = {"type": p.GetTypeName(), "restOffset": None if ro is None else float(ro), "contactOffset": None if co is None else float(co)}
for g in ground_colliders:
    a = PhysxSchema.PhysxCollisionAPI(stage.GetPrimAtPath(g))
    ro = a.GetRestOffsetAttr().Get() if a else None; co = a.GetContactOffsetAttr().Get() if a else None
    offsets[g] = {"type": stage.GetPrimAtPath(g).GetTypeName(), "restOffset": None if ro is None else float(ro), "contactOffset": None if co is None else float(co)}
print("[pen] offsets", json.dumps(offsets), flush=True)

view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
    sensor_paths, filter_patterns=[ground_colliders] * len(sensor_paths) if ground_colliders else [["/World/ground"]] * len(sensor_paths),
    max_contact_data_count=args.max_per_prim * len(sensor_paths))
print("[pen] contact view sensors", view.sensor_count, "filters", view.filter_count, "max data", view.max_contact_data_count, flush=True)
from omni.physx import get_physx_simulation_interface
physx_sim = get_physx_simulation_interface()
body_names = [p.rsplit("/", 1)[-1] for p in sensor_paths]
lab_body_names = list(robot.body_names)
dt = float(E.physics_dt); dec = int(E.cfg.decimation)

rows_t = []   # tensor API: (phys_step, body_idx, separation, normal_force, px, py, pz, nx, ny, nz)
rows_r = []   # full report: (phys_step, body_idx, separation, impulse_norm, px, py, pz)
per_step = {"t": [], "root_z": [], "foot_z": [], "net_force_z": []}
foot_ids = [lab_body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
path_to_idx = {p: i for i, p in enumerate(sensor_paths)}

def read_contacts(k):
    f, pts, nrm, sep, cnt, start = view.get_contact_data(dt)
    conv = lambda a: (a.detach().cpu().numpy() if hasattr(a, "detach") else (a.numpy() if hasattr(a, "numpy") else np.asarray(a)))
    f, pts, nrm, sep, cnt, start = [conv(x) for x in (f, pts, nrm, sep, cnt, start)]
    f = f.reshape(-1); pts = pts.reshape(-1, 3); nrm = nrm.reshape(-1, 3); sep = sep.reshape(-1)
    cnt = cnt.reshape(view.sensor_count, -1); start = start.reshape(view.sensor_count, -1)
    for s in range(cnt.shape[0]):
        for fi in range(cnt.shape[1]):
            for j in range(int(start[s, fi]), int(start[s, fi] + cnt[s, fi])):
                rows_t.append((k, s, sep[j], f[j], *pts[j], *nrm[j]))
    try:
        headers, data = physx_sim.get_full_contact_report()[:2]
        for h in headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0)); a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            b = path_to_idx.get(a0, path_to_idx.get(a1, -1))
            for j in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                c = data[j]
                rows_r.append((k, b, c.separation, float(np.linalg.norm([c.impulse[0], c.impulse[1], c.impulse[2]])), c.position[0], c.position[1], c.position[2]))
    except Exception as e:
        if k < 3: print(f"[pen] full contact report failed: {e!r}", flush=True)

env.reset(seed=args.seed)
root = robot.data.default_root_state.clone(); root[:, :3] += E.scene.env_origins; root[:, 2] = 1.0
robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:])
k = 0
def phys_step():
    global k
    E.sim.step(render=False); E.scene.update(dt); k += 1
    read_contacts(k)
    per_step["t"].append(k * dt); per_step["root_z"].append(float(robot.data.root_pos_w[0, 2] - E.scene.env_origins[0, 2]))
    per_step["foot_z"].append(robot.data.body_pos_w[0, foot_ids, 2].cpu().numpy().tolist())
    per_step["net_force_z"].append(contacts.data.net_forces_w[0, :, 2].cpu().numpy().tolist())
phys_step()   # the settle step record_g1.py takes after placing the root (physics step 1)
nj = robot.num_joints
a = torch.zeros((1, nj), device=dev)
root_z_ctrl = []
for t in range(150):   # env.step unrolled: process action once, apply + write + step + update per substep
    E.action_manager.process_action(a)
    for _ in range(dec):
        E.action_manager.apply_action(); E.scene.write_data_to_sim()
        phys_step()
    root_z_ctrl.append(float(robot.data.root_pos_w[0, 2] - E.scene.env_origins[0, 2]))
rt = np.array(rows_t, np.float64).reshape(-1, 10); rr = np.array(rows_r, np.float64).reshape(-1, 7)
np.savez_compressed(os.path.join(args.out, "C_drop_penetration.npz"),
    tensor_step=rt[:, 0].astype(np.int32), tensor_body=rt[:, 1].astype(np.int32), tensor_separation=rt[:, 2], tensor_normal_force=rt[:, 3],
    tensor_point=rt[:, 4:7], tensor_normal=rt[:, 7:10],
    report_step=rr[:, 0].astype(np.int32), report_body=rr[:, 1].astype(np.int32), report_separation=rr[:, 2], report_impulse=rr[:, 3], report_point=rr[:, 4:7],
    t=np.array(per_step["t"]), root_z=np.array(per_step["root_z"]), foot_z=np.array(per_step["foot_z"]), net_force_z=np.array(per_step["net_force_z"]),
    root_z_ctrl=np.array(root_z_ctrl), body_names=np.array(body_names), lab_body_names=np.array(lab_body_names))
meta = {"physics_dt": dt, "decimation": dec, "n_phys_steps": k, "device": dev, "sensor_paths": sensor_paths, "ground_colliders": ground_colliders,
        "collider_offsets": offsets, "tensor_rows": int(len(rt)), "report_rows": int(len(rr)),
        "physx_scene": {kk: str(v) for kk, v in vars(E.cfg.sim.physx).items()} if hasattr(E.cfg.sim, "physx") else {}}
def summ(step, sep, force=None):
    if len(sep) == 0: return None
    act = np.ones_like(sep, bool) if force is None else force > 0
    s = np.where(act, sep, np.inf); per = {}
    for st in np.unique(step):
        m = step == st; per[int(st)] = float(np.min(s[m]))
    ks = np.array(sorted(per)); mins = np.array([per[x] for x in ks])
    fin = np.isfinite(mins)
    i = int(np.argmin(np.where(fin, mins, np.inf)))
    last = ks >= k - 100   # last 0.5 s
    return {"min_separation_m": float(mins[i]), "at_phys_step": int(ks[i]), "at_t_s": float(ks[i] * dt),
            "settle_min_separation_last_0p5s_m": float(np.min(mins[last & fin])) if np.any(last & fin) else None,
            "settle_mean_of_step_minima_last_0p5s_m": float(np.mean(mins[last & fin])) if np.any(last & fin) else None,
            "first_contact_step": int(ks[0])}
meta["summary_tensor_all"] = summ(rt[:, 0], rt[:, 2])
meta["summary_tensor_loaded"] = summ(rt[:, 0], rt[:, 2], rt[:, 3])
meta["summary_report_all"] = summ(rr[:, 0], rr[:, 2])
meta["summary_report_loaded"] = summ(rr[:, 0], rr[:, 2], rr[:, 3])
meta["final_root_z_ctrl"] = root_z_ctrl[-1]
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print("[pen] summary", json.dumps({kk: meta[kk] for kk in meta if kk.startswith("summary") or kk.startswith("final")}), flush=True)
open(os.path.join(args.out, "DONE"), "w").write("ok")
env.close(); simulation_app.close()
