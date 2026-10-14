# Scenes from a config file

One small YAML file declares a scene: a robot, movable objects, camera mounts, a lidar, a background and the env count.
`python -m metalsim.run` runs it, and `--film` makes an offline video with a doc.

```bash
python -m metalsim.run configs/scenes/go2_boxes.yaml --check            # parse + build the model (CPU)
python -m metalsim.run configs/scenes/go2_boxes.yaml --steps 200 --save runs/scenes/go2_boxes
python -m metalsim.run configs/scenes/go2_boxes.yaml --film             # tier-2 film + doc (metalsim.render.film)
mjpython -m metalsim.run configs/scenes/go2_boxes.yaml --viewer         # interactive, env 0 (macOS needs mjpython)
python -m metalsim.run configs/scenes/franka_pour.yaml --film           # a registered task (its own scene builder)
```

The Python API does the same:

```python
from metalsim.scene import cfg as C, compose
scene = compose.compose(C.load("configs/scenes/go2_boxes.yaml"))   # MjSpec -> MjModel -> BatchSim, cameras, lidar
scene.reset()                                                        # per-env randomization (reset_root_state_uniform)
scene.step(actions)                                                  # (N, nu) numpy / torch, or None = the scene policy
obs = scene.render()       # {camera: {data_type: (N, H, W, ...)}}, host path
hits = scene.sense()       # {lidar: {"range": (N, beams), ...}}
```

Training code uses the GPU paths directly: `scene.cameras[name].renderer.render(scene.sim, v)`, `scene.sim.t.qpos`.

## Schema

Field names follow Isaac Lab 3.0 (`upstream/IsaacLab3` at `ae37b028`), so an Isaac scene config maps across almost
field for field. The classes are in `metalsim/scene/cfg.py`.

| Isaac Lab | here | notes |
|---|---|---|
| `InteractiveSceneCfg(num_envs, env_spacing)` | top level `num_envs`, `env_spacing` | envs are separate MuJoCo Warp worlds, each in its own frame |
| `ArticulationCfg(spawn, init_state)` | `robot:` | `MjcfFileCfg` / `UrdfFileCfg` / `UsdFileCfg`; `init_state.joint_pos` {regex: value}; the MJCF's `home` keyframe is the default |
| `RigidObjectCfg` | `objects: {name: ...}` | `CuboidCfg` / `SphereCfg` / `CylinderCfg` / `CapsuleCfg` / `MeshFileCfg`; `mass_props`, `visual_material.diffuse_color`, `physics_material`; `count:` replicates |
| `init_state.rot` | same | **(x, y, z, w)**: Isaac Lab 3.0's order (2.x was w-first) |
| `EventTermCfg(reset_root_state_uniform, pose_range, velocity_range)` | `randomize: {pose_range, velocity_range}` | same keys (x y z roll pitch yaw), sampled per env at every reset |
| `CameraCfg(prim_path, offset, width, height, data_types, spawn=PinholeCameraCfg)` | `cameras: {name: ...}` | `prim_path` ".../Robot/<link>/<name>" mounts on the link; `offset.convention` ros / opengl / world; data types rgb, distance_to_image_plane, distance_to_camera, semantic_segmentation, instance_id_segmentation_fast, normals; plus `renderer.tier` (0, 1, 2) |
| `RayCasterCfg(pattern_cfg=LidarPatternCfg, max_distance)` | `sensors: {name: ...}` | metalsim's hardware ray-traced lidar |
| `GroundPlaneCfg`, `DomeLightCfg(texture_file)`, `DistantLightCfg` | `background: {ground, dome, lights}` | the dome HDRI lights tier 2 |
| NuRec / splat background | `background.splat` | below |

Not yet covered: several articulations in one scene, contact sensors and IMUs as config entries, deformables and
terrain generators (the tasks build those themselves).

## Splat backgrounds

`background.splat` puts a Gaussian-splat scene behind the robot. It has a file path, a file-to-sim transform (`up`,
`scale`, `yaw_deg`, `translate`, the same as `metalsim.render.splat.make_transform`) and a collision proxy. The splat
is only seen; the proxy is what the physics collides with. It sits in geom group 4, which no renderer draws over the
splat.

Each camera then renders through `SplatRenderer` with a tier-0 mesh pass composited in front. The robot and objects
occlude the splats by depth.

| `collision.type` | what collides | where it comes from |
|---|---|---|
| `plane` | a plane at `plane_z` | Isaac Lab's NuRec recipe: the floor aligned at z = 0, a ground plane added (`docs/source/features/imitation-learning/humanoids_imitation.rst`) |
| `mesh` | a supplied mesh (.obj / .stl / .usd, e.g. a NuRec scene's `mesh.usd`) | NVIDIA's NuRec scenes reference an nvblox mesh as the volume's `proxy`; World Labs Marble exports a GLB collider |
| `structure` | floor plane + every wall of the floor plan as a box | InteriorGS `structure.json`; doors are closed |
| `occupancy` | a height field from the 2D occupancy map, furniture included | InteriorGS `occupancy.png` / `occupancy.json` |

Research behind this (2026-10-14):
- **Isaac Lab 3.0.** It loads a splat as a plain `AssetBaseCfg(spawn=UsdFileCfg(...usdz))` with no collision
  (`scripts/demos/sensors/ppisp_camera.py`), or adds a ground plane for NuRec scenes.
- **Isaac Sim's NuRec volumes.** These take up to four proxy meshes for shadows and physics.
- **Research systems.** They build a mesh from the reconstruction:
  - GaussGym: NKSR.
  - Vid2Sim: TSDF fusion of rendered depth.
  - SplatGym: an octree of points.
  - SplatSim: keeps the simulator's own meshes and moves robot Gaussians by forward kinematics.

  None of them reports how accurate the proxy is; they report policy success.
- **InteriorGS.** It ships no mesh, only the floor plan and the occupancy map. Hence the two proxy types above.

Building a proxy automatically from the splat itself (depth fusion) is not implemented.

The reference splat is InteriorGS scene `0204_840158` (`configs/scenes/go2_interiorgs.yaml`), fetched by
`scripts/fetch_splat_scenes.sh interiorgs`:
- **Access:** gated on Hugging Face.
- **Licence:** non-commercial research and education only, no redistribution. Nothing from it is committed.
- **Frame:** z-up metres with the floor at 0, so the transform is the identity.
- **Placement:** `--rooms` lists the room centres for placing the robot.

Only public sources are used; no Niantic scans.

Filming a splat background with the offline path tracer is not supported yet, because tier 2 does not draw splats.
`--save` records the composited cameras instead.
