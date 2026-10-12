"""R2: the column-collapse scene through Isaac Lab 3.0-EA's own MPM path (MPMObject + MPMSolverCfg + NewtonMPMManager).

    python record_mpm_il3.py --viz none --aspect 2 --out ~/parity3/mpm/R2_a2

Same scene as scenes.py ``column`` (R1): the Isaac Lab cell_center lattice with Isaac Lab's material defaults, a
frictional back wall and floor (mu 0.5), frictionless side walls, voxel 0.05, dense grid, Gauss-Seidel 250 iterations
tol 1e-4, 60 Hz, one solver step per frame, project_outside after every step (Isaac Lab's default is off; Newton's
examples and R1 project). Colliders are USD cuboids with contact_margin 0 (Newton's builder default) and equal static
and dynamic friction. The manager runs Newton's own conditional-graph loop, so R2 is compared with R0_a2_stock (same
loop): equality says Isaac Lab's wrapper adds nothing to the physics.

Writes init.npz / traj.npz / meta.json in record_mpm.py's format (no per-solve log: the manager's loop is not patched).
"""
import argparse, hashlib, json, os, time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--aspect", type=float, default=2.0)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args = parser.parse_args()

LENGTH, WIDTH, VOXEL, PPC, FPS = 0.2, 0.1, 0.05, 3.0, 60.0
H = args.aspect * LENGTH
T = 0.25
WALL_H = 0.5 * H + 0.5
REACH = 0.5 * LENGTH * (2.0 + 4.0 * args.aspect) + 1.0


def sim_cfg():
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg
    import isaaclab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=args.device,
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(voxel_size=VOXEL, grid_type="dense", solver="gs", max_iterations=250,
                                    tolerance=1.0e-4, project_outside_colliders=True),
            num_substeps=1,
        ),
    )


def scene_cfg():
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    def box(path, center, half, mu):
        return AssetBaseCfg(
            prim_path=path,
            spawn=sim_utils.CuboidCfg(
                size=(2 * half[0], 2 * half[1], 2 * half[2]),
                collision_props=NewtonCollisionPropertiesCfg(collision_enabled=True, contact_margin=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=center),
        )

    @configclass
    class ColumnSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5)))
        back = box("/World/Back", (-T, 0.5 * WIDTH, WALL_H), (T, 0.5 * WIDTH + 2 * T, WALL_H), 0.5)
        side0 = box("/World/Side0", (0.5 * REACH, -T, WALL_H), (0.5 * REACH + T, T, WALL_H), 0.0)
        side1 = box("/World/Side1", (0.5 * REACH, WIDTH + T, WALL_H), (0.5 * REACH + T, T, WALL_H), 0.0)
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Column",
            spawn=MPMGridCfg(lower=(0.0, 0.0, 0.0), upper=(LENGTH, WIDTH, H), voxel_size=VOXEL,
                             particles_per_cell=PPC, particle_placement="cell_center", jitter=0.0),
        )

    return ColumnSceneCfg(num_envs=1, env_spacing=0.0)


os.makedirs(args.out, exist_ok=True)
cfg = sim_cfg()
with launch_simulation(cfg, args):
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    sim = sim_utils.SimulationContext(cfg)
    scene = InteractiveScene(scene_cfg())
    sim.reset()
    media = scene["media"]

    def grab(a):
        return (a.torch if hasattr(a, "torch") else a).detach().cpu().numpy().reshape(-1, 3).astype(np.float32).copy()

    q0, qd0 = grab(media.data.particle_pos_w), grab(media.data.particle_vel_w)
    np.savez_compressed(os.path.join(args.out, "init.npz"), q=q0, qd=qd0, mass=np.zeros(0), radius=np.zeros(0))
    frames = int(round(args.seconds * FPS))
    dt = sim.get_physics_dt()
    pos, vel, wall = [], [], []
    for f in range(frames):
        t0 = time.perf_counter()
        sim.step(render=False)
        scene.update(dt)
        p, v = grab(media.data.particle_pos_w), grab(media.data.particle_vel_w)
        wall.append(time.perf_counter() - t0)
        pos.append(p)
        vel.append(v)
    np.savez_compressed(os.path.join(args.out, "traj.npz"), pos=np.stack(pos), vel=np.stack(vel),
                        frame=np.arange(1, frames + 1, dtype=np.int32))
    import newton, warp

    meta = {"scene": "column(isaaclab)", "aspect": args.aspect, "frames": frames, "frame_dt": dt,
            "init_md5": hashlib.md5(q0.tobytes() + qd0.tobytes()).hexdigest(), "particle_count": int(q0.shape[0]),
            "wall_s": wall, "versions": {"newton": newton.__version__, "warp": warp.__version__},
            "info": {"scene": "column", "spec": {"aspect": args.aspect, "length": LENGTH, "width": WIDTH,
                                                  "voxel_size": VOXEL, "particles_per_cell": PPC},
                     "initial_height": H}}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"done: {frames} frames, {q0.shape[0]} particles")
